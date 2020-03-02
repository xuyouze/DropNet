# coding:utf-8
# @Time         : 2019/9/16
# @Author       : xuyouze
# @File Name    : lfwa_dataset.py

import os

from torchvision.transforms import transforms

from PIL import Image

from config import *
from dataset.registry import Dataset
from .base_dataset import BaseDataset
import numpy as np

__all__ = ["LFWADataset"]

def save_original_attr_and_img(train_dir, test_dir, attr_dir, proportion_of_testset=1 / 10):
    # train_dir = '/home/xuyouze/Documents/LFW/anno/train.txt'
    # test_dir = '/home/xuyouze/Documents/LFW/anno/test.txt'
    # attr_dir = '/home/xuyouze/Documents/LFW/anno/lfw_attributes.txt'

    with open(attr_dir) as f:
        f.readline()
        f.readline()
        lines = f.readlines()

        total_size = len(lines)

        test_size = int(proportion_of_testset * total_size)
        train_size = total_size - test_size

        train_imgs = []
        test_imgs = []
        train_attr = np.zeros([train_size, 40])
        test_attr = np.zeros([test_size, 40])
        test_idx = np.random.choice(total_size, test_size, replace=False)
        train_cur = test_cur = 0
        for i, line in enumerate(lines):
            # 每一行有可能有76-79个，其中前2个或者3个是人名，后边依次是每个人图片的数量和73个属性
            # 把图像的路径加入到self.imgs
            vals = line.split()
            len_name = len(vals) - 74  # 73个属性加上 1个num(这个人的第几张图片)

            person_name = vals[0]
            for j in range(1, len_name):
                person_name += '_' + vals[j]
            img_name = person_name + '/' + person_name + '_' + str(vals[len_name]).zfill(4) + '.jpg'

            # 把图像的标签加入到self.attr
            index = [65, 36, 56, 60, 13, 30, 41, 39, 10, 11,
                     21, 12, 35, 20, 49, 15, 47, 59, 61, 69,
                     1, 43, 17, 37, 46, 51, 64, 40, 29, 62,
                     31, 18, 28, 27, 71, 50, 67, 73, 72, 7]

            whole_attr = np.array([float(s) for s in vals[-74:]])

            temp = np.where(whole_attr > 0, 1, 0)[index]

            # 由于56和57是男性女性有魅力，所以只要有一个为1就认为attractive属性为1
            if whole_attr[57] >= 0:
                temp[2] = 1
            if i in test_idx:
                test_attr[test_cur, :] = temp
                test_imgs.append(img_name)
                test_cur += 1
            else:
                train_attr[train_cur, :] = temp
                train_imgs.append(img_name)
                train_cur += 1

    with open(train_dir, "w") as f:
        for (name, attr_img) in zip(train_imgs, train_attr):
            f.writelines(name + " ")
            f.writelines(" ".join([str(i) for i in attr_img.tolist()]))
            f.write("\n")
    with open(test_dir, "w") as f:
        for (name, attr_img) in zip(test_imgs, test_attr):
            f.writelines(name + " ")
            f.writelines(" ".join([str(i) for i in attr_img.tolist()]))
            f.write("\n")


def get_train_img_attr(attr_file, img_dir, augment_dir):
    with open(attr_file) as f:
        lines = f.readlines()
        set_size = len(lines)
        imgs = []
        attr = np.zeros((set_size, 40))

        for i, line in enumerate(lines):
            vals = line.split()
            imgs.append(vals[0])
            attr[i, :] = [vals[j + 1] for j in range(40)]
        # attr = np.tile(attr, (2, 1))
        # imgs = [os.path.join(img_dir, i) for i in imgs] + [os.path.join(augment_dir, i) for i in imgs]
        imgs = [os.path.join(augment_dir, i) for i in imgs]
    return imgs, attr


def get_test_img_attr(attr_file, img_dir):
    with open(attr_file) as f:
        lines = f.readlines()
        set_size = len(lines)
        imgs = []
        attr = np.zeros((set_size, 40))

        for i, line in enumerate(lines):
            vals = line.split()
            imgs.append(os.path.join(img_dir, vals[0]))
            attr[i, :] = [vals[j + 1] for j in range(40)]
    return imgs, attr

@Dataset.register("lfwa")

class LFWADataset(BaseDataset):

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.config = config
        self.img_dir = os.path.join(self.config.dataset_config.data_root_dir, self.config.dataset_config.img_dir)
        self.img_augment_dir = os.path.join(self.config.dataset_config.data_root_dir,
                                            self.config.dataset_config.img_augment_dir)

        if config.isTrain:
            self.attr_file = os.path.join(config.dataset_config.data_root_dir, config.dataset_config.attr_train_file)
            self.imgs, self.attr = get_train_img_attr(self.attr_file, self.img_dir, self.img_augment_dir)

            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        else:

            self.attr_file = os.path.join(config.dataset_config.data_root_dir, config.dataset_config.attr_test_file)
            self.imgs, self.attr = get_test_img_attr(self.attr_file, self.img_dir)

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.length = len(self.imgs)

    def __getitem__(self, index):

        images = Image.open(self.imgs[index]).convert('RGB')
        images = self.transform(images)

        return images, self.attr[index, :]

    def __len__(self):
        return self.length
