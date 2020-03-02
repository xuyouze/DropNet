# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : build.py


from .registry import Dataset

import importlib
import os
import glob


def build_dataset(dataset_name):
    [importlib.import_module("dataset." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_dataset.py"))]

    assert dataset_name in Dataset, \
        f'dataset name {dataset_name} is not registered in registry :{Dataset.keys()}'
    return Dataset[dataset_name]
