# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : base_dataset.py

from abc import ABC, abstractmethod

from torch.utils import data

from config import *


class BaseDataset(data.Dataset, ABC):
    def __init__(self, config: BaseConfig):
        self.config = config

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass
