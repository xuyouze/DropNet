# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : __init__.py.py

from .test_config import TestConfig
from .train_config import TrainConfig

from .base_config import BaseConfig

__all__ = ["TrainConfig", "BaseConfig", "TestConfig"]
