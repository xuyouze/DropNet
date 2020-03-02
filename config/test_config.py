# coding:utf-8
# @Time         : 2019/9/16 
# @Author       : xuyouze
# @File Name    : test_config.py

from .base_config import BaseConfig

__all__ = ["TestConfig"]


class TestConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.eval = True
        self.batch_size = 512
        self.isTrain = False
        self.isTest = False
