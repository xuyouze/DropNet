# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : build.py
from loss.registry import Loss

import importlib
import os
import glob


def build_loss(loss_name):
    [importlib.import_module("loss." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_loss.py"))]

    assert loss_name in Loss, \
        f'loss name {loss_name} is not registered in registry :{Loss.keys()}'
    return Loss[loss_name]
