import os
import json
from .lt_data import LT_Dataset


class ImageNet_LT(LT_Dataset):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    train_txt = "./datasets/ImageNet_LT/ImageNet_LT_train.txt"
    test_txt = "./datasets/ImageNet_LT/ImageNet_LT_test.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])
        self.coarse_names, self.fine_to_coarse = self.load_coarse_labels_mapping(
            './datasets/ImageNet_LT/fine2coarse.json')

    def load_coarse_labels_mapping(self, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取粗标签名称字段
        coarse_label_names = data.get('coarse_label_names', [])

        # 提取细标签到粗标签的映射字段
        fine2coarse = data.get('fine2coarse', [])

        return coarse_label_names, fine2coarse

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames
