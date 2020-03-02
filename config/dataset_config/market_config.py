# coding:utf-8
# @Time         : 2019/10/17 
# @Author       : xuyouze
# @File Name    : market_config.py

from .registry import DatasetConfig

__all__ = ["MarketConfig"]


@DatasetConfig.register("market")
class MarketConfig(object):

    def __init__(self):
        super(MarketConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/Market-1501"
        self.data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']
        self.attr_file = "attribute/market_attribute.mat"
        self.attribute_num = 30
