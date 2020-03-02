# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : __init__.py


import importlib

from config.base_config import BaseConfig
from models.base_model import BaseModel
from models.build import build_model

__all__ = ["create_model"]

#
# def find_model_using_name(model_name: str):
#     model_filename = "models." + model_name + "_model"
#     modellib = importlib.import_module(model_filename)
#
#     model = None
#     target_model_name = model_name.replace("_", "") + "model"
#     target_model_name = target_model_name.replace("-", "")
#     for name, cls in modellib.__dict__.items():
#         if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
#             model = cls
#             break
#
#     if not model:
#         raise NotImplementedError(
#             "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
#                 model_filename, target_model_name))
#     return model


def create_model(config: BaseConfig):
    # model = find_model_using_name(config.model_name)
    model = build_model(config.model_name)
    instance = model(config)
    config.logger.info("{0} model has been created".format(config.model_name))
    return instance
