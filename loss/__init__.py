# coding:utf-8
# @Time         : 2019/7/2 
# @Author       : xuyouze
# @File Name    : __init__.py.py
import importlib

from torch import nn

from config import BaseConfig
from loss.build import build_loss

__all__ = ["create_loss"]


# def find_loss_using_name(loss_name: str):
#     model_filename = "loss." + loss_name + "_loss"
#     modellib = importlib.import_module(model_filename)
#
#     model = None
#     target_model_name = loss_name.replace("_", "").replace("-", "") + "loss"
#     for name, cls in modellib.__dict__.items():
#         if name.lower() == target_model_name.lower() and issubclass(cls, nn.Module):
#             model = cls
#             break
#
#     if not model:
#         raise NotImplementedError(
#             "In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (
#                 model_filename, target_model_name))
#     return model


def create_loss(config: BaseConfig):
    # loss = find_loss_using_name(config.loss_name)
    loss = build_loss(config.loss_name)
    instance = loss(config)
    config.logger.info("{0} loss has been created".format(config.loss_name))
    return instance
