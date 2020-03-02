# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : base_config.py

import importlib
import os
import sys
import torch

import logging

from .dataset_config import build_dataset_config
from .logger_config import config

__all__ = ["BaseConfig"]


class BaseConfig(object):
	def __init__(self):
		# model component parameters
		self.checkpoints_dir = "ckp"

		# dataset name [celebA | lfwa | duke | market]
		self.dataset_name = "celebA"
		# self.dataset_name = "lfwa"
		# self.dataset_name = "duke"
		# self.dataset_name = "market"

		# model name [common]
		self.model_name = "common"

		# model name [resnet]
		self.network_name = "resnet"

		# loss name [focal | ghm-c | drop | bce]
		# self.loss_name = "drop"
		# self.loss_name = "focal"
		# self.loss_name = "ghmc"
		self.loss_name = "bce"

		# network initialization type [normal]
		self.init_type = "normal"
		self.init_gain = 0.2  # scaling factor for normal

		# global saving and loading parameters
		self.batch_size = 100
		self.num_threads = 4
		self.last_epoch = "last"
		self.load_iter = 0

		self.isTrain = None

		# dataset parameters
		self.dataset_config = build_dataset_config(self.dataset_name)
		self.balance_attr_pos_prop = torch.FloatTensor([0.5] * self.dataset_config.attribute_num)

		# logging config
		logging.config.dictConfig(config)
		self.logger = logging.getLogger("TrainLogger")
		self.test_logger = logging.getLogger("TestLogger")

		if not os.path.exists(self.checkpoints_dir):
			os.makedirs(self.checkpoints_dir)
