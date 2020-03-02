# coding:utf-8
# @Time         : 2019/9/16 
# @Author       : xuyouze
# @File Name    : lfwa_config.py

from .registry import DatasetConfig
__all__ = ["LFWAConfig"]


@DatasetConfig.register("lfwa")
class LFWAConfig(object):

    def __init__(self):
        super(LFWAConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/LFW"
        self.attr_train_file = 'anno/train.txt'
        self.attr_test_file = 'anno/test.txt'
        self.img_dir = "img"
        self.img_augment_dir = "lfw-deepfunneled"
        self.attribute_num =40
