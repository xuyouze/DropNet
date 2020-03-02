# coding:utf-8
# @Time         : 2019/10/9 
# @Author       : xuyouze
# @File Name    : dukemtmc_config.py

from .registry import DatasetConfig

__all__ = ["DukeMTMCConfig"]

@DatasetConfig.register("duke")
class DukeMTMCConfig(object):

    def __init__(self):
        super(DukeMTMCConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/DukeMTMC-reID"
        self.data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']
        self.attr_file = "attribute/duke_attribute.mat"
        self.attribute_num = 23
