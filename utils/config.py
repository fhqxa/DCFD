from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.imb_factor = None  # for long-tailed cifar dataset
_C.tll_projection = False # 是否是在LIFT上扩展新模型

_C.backbone = ""
_C.resolution = 224

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.gpu = None  # assign a single gpu 
_C.num_workers = 20
_C.prec = "fp16"  # fp16, fp32, amp

_C.num_epochs = 10
_C.batch_size = 128
_C.micro_batch_size = 128  # for gradient accumulation, must be a divisor of batch size
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.loss_type = "LA"  # "CE" / "Focal" / "LDAM" / "CB" / "GRW" / "BS" / "LA" / "LADE"
_C.new_loss = 0.0

_C.classifier = "CosineClassifier"
_C.scale = 25  # for cosine classifier

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias 
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = False
_C.lora = False
_C.lora_mlp = False
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.mask = False   # fine-tuning a specific proportion of all parameters
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
_C.adaptformer_scale = "learnable"  # "learnable" or scalar
_C.mask_ratio = None
_C.mask_seed = None

_C.init_head = None  # "text_feat" (only for CLIP) / "class_mean" / "1_shot" / "10_shot" / "100_shot" / "linear_probe"
_C.prompt = "default" # "classname" / "default" / "ensemble" / "descriptor"
_C.tte = False  # test-time ensemble
_C.expand = 24 # expand the width and height of images for test-time ensemble
_C.tte_mode = "fivecrop" # "fivecrop" / "tencrop" / "randaug"
_C.randaug_times = 1

_C.zero_shot = False  # zero-shot CLIP (only for CLIP)
_C.test_only = False  # load model and test
_C.test_logits = False
_C.test_dual = False
_C.dual_model_dir = None
_C.test_train = False  # load model and test on the training set
_C.model_dir = None
_C.note = ""


def load_config_from_yaml(config_path):
    """
    从YAML文件自动加载配置，自动创建不存在的参数

    Args:
        config_path: str - YAML配置文件路径

    Returns:
        cfg: CfgNode - 合并了yaml配置的配置对象

    示例:
        # 在yaml文件中添加新参数，无需修改config.py
        cfg = load_config_from_yaml("config.yaml")
        print(cfg.new_param)  # 自动可用
    """
    import yaml
    import os

    # 读取yaml文件
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    # 创建新的配置节点
    cfg = _C.clone()

    # 递归设置所有yaml参数
    def _set_config(cfg_node, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 如果是字典，递归处理
                if key not in cfg_node:
                    cfg_node[key] = CN()
                _set_config(cfg_node[key], value)
            else:
                # 直接设置值
                cfg_node[key] = value

    _set_config(cfg, yaml_config)

    return cfg

def auto_update_config(cfg, yaml_file):
    import yaml

    with open(yaml_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    def update_dict(yaml_dict, cfg_node):
        for k, v in yaml_dict.items():
            if isinstance(v, dict):
                if not hasattr(cfg_node, k):
                    cfg_node[k] = CN()
                update_dict(v, cfg_node[k])
            else:
                # 自动类型转换：字符串转float或int
                if isinstance(v, str):
                    try:
                        if '.' in v or 'e' in v or 'E' in v:
                            v = float(v)
                        elif v.isdigit():
                            v = int(v)
                    except ValueError:
                        pass
                cfg_node[k] = v

    update_dict(yaml_cfg, cfg)
    return cfg