# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : build.py

from .registry import DatasetConfig

import importlib
import os
import glob


def build_dataset_config(dataset_name):
    [importlib.import_module("config.dataset_config." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_config.py"))]

    # print(__all__)
    assert dataset_name in DatasetConfig, \
        f'dataset config name {dataset_name} is not registered in registry :{DatasetConfig.keys()}'
    return DatasetConfig[dataset_name]()
