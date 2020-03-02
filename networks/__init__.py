# coding:utf-8
# @Time         : 2019/6/27 
# @Author       : xuyouze
# @File Name    : __init__.py.py
import importlib

from networks.build import build_network
from .model_component import get_scheduler, init_net

__all__ = ["create_network_model", "get_scheduler"]


def create_network_model(config):
    network = build_network(config.network_name)

    config.logger.info("{0} network has been created".format(config.network_name))

    return network(config)

#
# def find_network_using_name(config):
#     model_filename = "networks." + config.network_name + "_model_component"
#     modellib = importlib.import_module(model_filename)
#     network = None
#     for name, cls in modellib.__dict__.items():
#         if name == "define_net":
#             network = cls
#
#     if not network:
#         raise NotImplementedError(
#             "In %s.py, there should be a method named %s." % (
#                 model_filename, "define_net"))
#
#     return network
