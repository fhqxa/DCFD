import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()

    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.text_features = None
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None


    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(
                        lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=cfg.micro_batch_size, shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
                                            batch_size=64, sampler=init_sampler, shuffle=False,
                                            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
                                            batch_size=64, shuffle=False,
                                            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
                                      batch_size=64, shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=True)

        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            if cfg.tll_projection:
                self.model = DCFD(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.init_text_feat()
                self.model.text_features = self.text_features
                self.model.text_features_raw = self.text_features_raw
                self.tuner = self.model.tuner
                self.head = self.model.head

            else:
                self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.tuner = self.model.tuner
                self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")

            torch.cuda.empty_cache()

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        # 首先冻结整个模型的所有参数
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        print("Turning on gradients in the tuner")
        for name, param in self.model.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.model.head.named_parameters():
            param.requires_grad_(True)
        for name, param in self.model.fusion_module.named_parameters():
            param.requires_grad_(True)
        for name, param in self.model.dino_model.named_parameters():
            param.requires_grad_(True)

        # 打印参数统计信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")

        # 收集可训练参数用于优化器
        trainable_param_groups = [
            {"params": self.model.tuner.parameters()},
            {"params": self.model.head.parameters()},
            {"params": self.model.fusion_module.parameters()},
            {"params": self.model.dino_model.parameters()}
        ]

        # 创建优化器
        self.optim = torch.optim.SGD(
            trainable_param_groups,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal":  # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM":  # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB":  # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW":  # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS":  # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA":  # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE":  # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "FUSION":  # 复杂特征融合损失
            self.criterion = ComplexFusionLoss(
                cls_num_list=cls_num_list,
                device=self.device,
                lambda_standard=1.0,
                lambda_patch=1.0,
                lambda_total=0.5
            )


    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_text_feat(self):
        cfg = self.cfg
        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(self.classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
            self.text_features_raw = text_features
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(self.classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of {}."
            prompts = self.get_tokenized_prompts(self.classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
            self.text_features_raw = text_features

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.text_features = text_features.to(self.device)

    def init_head_text_feat(self):
        if self.text_features is None:
            self.init_text_feat()
        if self.cfg.classifier == "SeparatedLearnableClassifier":
            self.head.apply_weight(self.text_features_raw)
        else:
            self.head.apply_weight(self.text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx + cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _, feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features,
                                                                                                          all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        best_acc = 0.0
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            self.head.train()
            self.model.dino_model.train()
            end = time.time()


            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    # 处理output可能是字典的情况（ComplexFusionLoss）
                    if isinstance(output, dict):
                        # 使用logits_total进行预测，如果不存在则使用logits_global
                        pred_logits = output.get('logits')
                        pred = pred_logits.argmax(dim=1)
                    else:
                        pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                                         num_epochs - epoch_idx - 1
                                 ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    # 添加低不确定性样本的准确率
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)

                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()
            if (epoch_idx + 1) % 5 == 0 and epoch_idx > 5:
                test_acc = self.test()
                print(f"Epoch {epoch_idx + 1}: Test Accuracy = {test_acc:.4f}")

                # 检查是否为最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model(cfg.output_dir)
                    print(f"best model saved with accuracy {best_acc:.4f}")

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        # self.save_model(cfg.output_dir)

        # self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        self.model.eval()

        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                # output = self.model(image)['logits_total']
                output = self.model(image)
                if isinstance(output, dict):
                    output = output.get('logits')
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k])['logits'])
                    # output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def visualization(self, directory="./test_image"):
        """
        可视化目录下的图片，生成基于DINO注意力的热力图掩码效果。

        Args:
            directory (str): 包含图片的文件夹路径
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import glob

        # 1. 设置模型为评估模式
        self.model.eval()

        # 2. 获取图片列表 (支持常见图片格式)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))

        if not image_paths:
            print(f"No images found in {directory}")
            return

        # 创建输出目录
        output_dir = os.path.join(directory, "vis_results")
        os.makedirs(output_dir, exist_ok=True)

        # 获取预处理变换 (复用测试时的中心裁剪变换，但去掉Lambda堆叠，因为我们要处理单张图)
        # 注意：这里需要重新构建一个简单的transform，因为self.test_loader用的transform包含Lambda堆叠用于TTA
        cfg = self.cfg
        resolution = cfg.resolution

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # 用于可视化的Transform：Resize -> CenterCrop -> ToTensor -> Normalize
        # 保持原始比例可能更好，但为了匹配ViT Patch，通常直接Resize或CenterCrop到固定大小
        vis_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # 直接Resize到模型输入大小，方便对应Patch
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 用于显示的原图Transform (不Normalize，转为PIL或Numpy)
        raw_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
        ])

        thresholds = [0.3, 0.5, 0.7, 0.9]

        print(f"Starting visualization for {len(image_paths)} images...")

        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="Visualizing"):
                try:
                    # 1. 加载图像
                    original_img_pil = Image.open(img_path).convert("RGB")
                    img_name = os.path.basename(img_path)

                    # 2. 预处理用于模型输入
                    input_tensor = vis_transform(original_img_pil).unsqueeze(0).to(self.device)  # [1, C, H, W]

                    # 3. 获取中间特征 (需要修改模型调用方式或临时挂钩，这里我们手动复现DCFD forward中的关键部分)
                    # 注意：为了获取 attn_map，我们需要访问 dino_model 和 image_encoder

                    # A. 获取 CLIP Image Features
                    # 假设 self.model.image_encoder 是 Peft_ViT 或类似结构，它应该能返回 cls 和 patch
                    # 如果 self.model.image_encoder 没有直接返回 patch 的接口，可能需要调整
                    # 这里假设我们可以调用内部模块。如果封装太严，可能需要临时修改 DCFD.forward

                    # 为了稳健性，我们直接调用 DCFD 内部的组件
                    model_module = self.model.module if hasattr(self.model, 'module') else self.model

                    # 获取 CLIP 特征
                    # Peft_ViT 的 forward 通常返回 (cls, patch) 或者只返回 cls
                    # 查看 DCFD.py line 135: image_cls, image_patch = self.image_encoder(image, self.tuner)
                    image_cls, image_patch = model_module.image_encoder(input_tensor, model_module.tuner)

                    # B. 获取 DINO 特征
                    # DINO model is frozen and eval
                    dino_tokens = model_module.dino_model.forward_features(input_tensor)  # [B, 197, 768]
                    dino_cls = dino_tokens[:, 0:1, :]  # [B, 1, 768]
                    dino_patch = dino_tokens[:, 1:, :]  # [B, 196, 768]

                    # C. 计算 Attention Map (复现 DCFD.py line 146-149)
                    # attn_map = torch.bmm(dino_cls, dino_patch.transpose(1, 2)).squeeze(1)
                    # [B, 1, 768] x [B, 768, 196] -> [B, 1, 196] -> [B, 196]
                    attn_map_raw = torch.bmm(dino_cls, dino_patch.transpose(1, 2)).squeeze(1)  # [B, 196]

                    # 归一化到 0-1 (复现 DCFD.py line 148-149)
                    attn_min = attn_map_raw.min(dim=-1, keepdim=True)[0]
                    attn_max = attn_map_raw.max(dim=-1, keepdim=True)[0]
                    attn_map_norm = (attn_map_raw - attn_min) / (attn_max - attn_min + 1e-6)  # [B, 196]

                    # 取出第一个样本的 attn map
                    attn_vec = attn_map_norm[0].cpu().numpy()  # [196]

                    # Reshape to 14x14 (假设是 ViT-B/16, 224x224)
                    # 如果分辨率不同，patch grid 可能不同，这里硬编码 14x14，可根据 cfg.resolution 动态计算
                    grid_size = int(np.sqrt(attn_vec.shape[0]))
                    if grid_size * grid_size != attn_vec.shape[0]:
                        # 如果不是正方形，尝试强制 reshape 或报错，通常 ViT 是正方形
                        print(f"Warning: Attention map size {attn_vec.shape[0]} is not a perfect square.")
                        continue

                    attn_matrix = attn_vec.reshape(grid_size, grid_size)  # [14, 14]

                    # 4. 准备原图用于绘制
                    # 将原始 PIL 图像 resize 到同样大小以便像素级对应，或者我们将 mask upsample
                    # 这里选择将 mask upsample 到原图大小
                    np_img = np.array(original_img_pil.resize((resolution, resolution)))  # [H, W, 3]

                    # Upsample attention map to image size for visualization smoothing
                    # 使用双线性插值上采样
                    attn_tensor = torch.from_numpy(attn_matrix).unsqueeze(0).unsqueeze(0)  # [1, 1, 14, 14]
                    attn_upsampled = F.interpolate(attn_tensor, size=(resolution, resolution), mode='bilinear',
                                                   align_corners=False)
                    attn_upsampled = attn_upsampled.squeeze().cpu().numpy()  # [H, W]

                    # 5. 绘制不同阈值的结果
                    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(4 * (len(thresholds) + 1), 4))

                    # 显示原图
                    axes[0].imshow(np_img)
                    axes[0].set_title("Original")
                    axes[0].axis('off')

                    for i, tau in enumerate(thresholds):
                        ax = axes[i + 1]

                        # 生成掩码: 保留 >= tau 的区域
                        # mask == 1 表示保留，mask == 0 表示掩盖
                        mask = (attn_upsampled >= tau).astype(np.float32)

                        # 创建覆盖层：灰色半透明
                        # 我们希望：保留区域显示原图，去除区域显示灰色
                        # 方法：创建一个灰色背景，然后在上面叠加原图 * mask
                        # 或者：原图 * mask + 灰色 * (1 - mask)

                        gray_overlay = np.ones_like(np_img) * 128  # 灰色背景 (0-255)

                        # 混合
                        # result = img * mask_expanded + gray * (1 - mask_expanded)
                        mask_expanded = np.stack([mask] * 3, axis=-1)  # [H, W, 3]

                        blended_img = (np_img.astype(np.float32) * mask_expanded +
                                       gray_overlay.astype(np.float32) * (1 - mask_expanded)).astype(np.uint8)

                        ax.imshow(blended_img)
                        ax.set_title(f"Tau={tau}")
                        ax.axis('off')

                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f"vis_{img_name}")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"Visualization completed. Results saved to {output_dir}")


    def save_model(self, directory):
        # 收集所有参与训练或可能改变的子模块
        checkpoint = {
            "tuner": self.model.tuner.state_dict(),
            "head": self.model.head.state_dict(),
            "fusion_module": self.model.fusion_module.state_dict(),
            "dino_model": self.model.dino_model.state_dict(),
        }

        # 统一移除 module. 前缀
        for key in checkpoint.keys():
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'Checkpoint not found at "{load_path}"')

        checkpoint = torch.load(load_path, map_location=self.device)

        # 定义一个内部辅助函数来处理加载逻辑
        def load_sub_module(module, checkpoint_key, name):
            if checkpoint_key in checkpoint:
                state_dict = checkpoint[checkpoint_key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    new_state_dict[k] = v

                # 严格检查核心参数形状（防止分类头类别数对不上等问题）
                try:
                    module.load_state_dict(new_state_dict, strict=True)
                    print(f"Successfully loaded {name} weights (Strict: True)")
                except RuntimeError as e:
                    print(f"Warning: Loading {name} with strict=False due to: {e}")
                    module.load_state_dict(new_state_dict, strict=False)

        # 依次加载各个部分
        load_sub_module(self.model.tuner, "tuner", "Tuner")
        load_sub_module(self.model.head, "head", "Head")
        load_sub_module(self.model.fusion_module, "fusion_module", "Fusion Module")
        load_sub_module(self.model.dino_model, "dino_model", "DINO Model")

        print(f"Finished loading weights from {load_path}")





