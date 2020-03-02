# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : celebA_dataset.py
import os

from PIL import Image
from torchvision.transforms import transforms
from .registry import Dataset
from config import *
from .base_dataset import BaseDataset
import numpy as np

__all__ = ["CelebADataset"]


def get_img_attr(attr_file):
    attr = np.zeros((202600, 40))
    with open(attr_file) as f:
        f.readline()
        f.readline()
        lines = f.readlines()
        id = 0
        for line in lines:
            vals = line.split()
            id += 1
            for j in range(40):
                # change the labels
                # self.attr[id, j] = int(vals[j + 1])
                if int(vals[j + 1]) == -1:
                    attr[id, j] = 0
                else:
                    attr[id, j] = 1
    return attr


def get_img_name_by_partition(part_dir, partition_flag):
    img = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.split()
            if num == partition_flag:
                img.append(pic_dir)
    return img


@Dataset.register("celebA")
class CelebADataset(BaseDataset):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        self.config = config
        self.attr_file = os.path.join(config.dataset_config.data_root_dir, config.dataset_config.attr_file)
        self.attr = get_img_attr(self.attr_file)
        self.partition_file = os.path.join(config.dataset_config.data_root_dir, config.dataset_config.part_file)
        if config.isTrain:
            self.image_names = get_img_name_by_partition(self.partition_file, "0")
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            if self.config.isTest:
                self.image_names = get_img_name_by_partition(self.partition_file, "2")
            else:
                self.image_names = get_img_name_by_partition(self.partition_file, "1")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        whole_dir = os.path.join(self.config.dataset_config.data_root_dir, self.config.dataset_config.face_whole)

        face_whole = Image.open(os.path.join(whole_dir, self.image_names[index])).convert('RGB')

        face_whole = self.transform(face_whole)

        idx = int(self.image_names[index].split(".")[0])

        return face_whole, self.attr[idx, :]
