# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : build.py

from models.registry import Model

import importlib
import os
import glob


def build_model(model_name):
    [importlib.import_module("models." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_model.py"))]

    assert model_name in Model, \
        f'model name {model_name} is not registered in registry :{Model.keys()}'
    return Model[model_name]
