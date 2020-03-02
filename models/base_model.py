# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : base_model.py
import importlib
import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn

from config.base_config import BaseConfig
from networks import *


class BaseModel(ABC):

	def __init__(self, config: BaseConfig) -> None:
		super().__init__()
		self.config = config

		self.net_names = []
		self.optimizers = []
		self.schedulers = []
		self.save_path = config.checkpoints_dir
		self.correct = None
		self.output = None
		self.attr = None
		self.validate_size = None
		self.pos_num = None
		self.tnr = None
		self.tpr = None

	@abstractmethod
	def set_input(self, x):
		pass

	@abstractmethod
	def forward(self):
		pass

	def optimize_parameters(self):
		self.forward()
		for optimizer in self.optimizers:
			optimizer.zero_grad()
		self.backward()
		for optimizer in self.optimizers:
			optimizer.step()

	def backward(self):
		for name in self.net_names:
			setattr(self, "loss_%s" % name,
					getattr(self, "criterion_%s" % name)(getattr(self, "output_%s" % name),
														 getattr(self,
																 "attr_%s" % name)).cuda())
			getattr(self, "loss_%s" % name).backward()

	def setup(self):
		"""
		setup the network
		if Train:
			set the optimizer
		else:
			load the pre-training models
		:return:
		"""
		print('-----------------------------------------------')
		if self.config.isTrain:
			self.schedulers = [get_scheduler(optimizer, self.config) for optimizer in self.optimizers]
		if not self.config.isTrain or self.config.continue_train:
			load_prefix = "iter_%d" % self.config.load_iter if self.config.load_iter > 0 else self.config.last_epoch
			self.load_networks(load_prefix)

		self.print_networks()

	def update_learning_rate(self):
		for scheduler in self.schedulers:
			scheduler.step()

	def get_current_loss(self):
		errors_map = OrderedDict()
		for name in self.net_names:
			if isinstance(name, str):
				errors_map[name] = float(getattr(self, "loss_" + name))

		return errors_map

	def eval(self):
		for name in self.net_names:
			if isinstance(name, str):
				net = getattr(self, "net_" + name)
				net.eval()

	def train(self):
		for name in self.net_names:
			if isinstance(name, str):
				net = getattr(self, "net_" + name)
				net.train()

	def save_networks(self, epoch_prefix):
		for name in self.net_names:
			if isinstance(name, str):
				save_filename = "%s_net_%s.pth" % (epoch_prefix, name)
				save_path = os.path.join(self.save_path, save_filename)
				net = getattr(self, "net_" + name)
				torch.save(net.module.cpu().state_dict(), save_path)
				net.cuda()

	def load_networks(self, epoch_prefix):
		for name in self.net_names:
			if isinstance(name, str):
				load_filename = "%s_net_%s.pth" % (epoch_prefix, name)
				load_path = os.path.join(self.save_path, load_filename)
				net = getattr(self, "net_" + name)
				if isinstance(net, nn.DataParallel):
					net = net.module
				print('loading the model from %s' % load_path)
				state_dict = torch.load(load_path)
				net.load_state_dict(state_dict)

	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def print_networks(self):
		for name in self.net_names:
			if isinstance(name, str):
				net = getattr(self, "net_" + name)
				num_params = 0
				for param in net.parameters():
					num_params += param.numel()
				self.config.logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))

	def get_learning_rate(self):
		return self.optimizers[0].param_groups[0]["lr"]

	def test(self):
		with torch.no_grad():
			self.forward()
			self.output = torch.zeros(self.attr.size(0), self.config.dataset_config.attribute_num)
			for name in self.net_names:
				self.output[:, getattr(self, "attr_%s_index" % name)] = getattr(self, "output_%s" % name).cpu()
			com1 = self.output > 0.5
			com2 = self.attr > 0

			# class_balance accuracy
			accuracy = com1.eq(com2)
			self.pos_num.add_(com2.sum(0).float())
			tpr = (accuracy & (com2 > 0)).sum(0).float()
			tnr = (accuracy & (com2 < 1)).sum(0).float()
			self.tpr.add_(tpr)
			self.tnr.add_(tnr)
			# mean accuracy
			mean_accuracy = accuracy.sum(0).float()
			self.correct.add_(mean_accuracy)

	def get_model_precision(self):
		return self.correct / self.validate_size

	def get_model_class_balance_precision(self):
		return 1 / 2 * (self.tpr / self.pos_num + self.tnr / (self.get_validate_size() - self.pos_num))

	def clear_precision(self):
		self.correct = torch.FloatTensor(self.config.dataset_config.attribute_num).fill_(0)
		self.tpr = torch.FloatTensor(self.config.dataset_config.attribute_num).fill_(0)
		self.tnr = torch.FloatTensor(self.config.dataset_config.attribute_num).fill_(0)
		self.pos_num = torch.FloatTensor(self.config.dataset_config.attribute_num).fill_(0)

	def create_network_model(self):
		return create_network_model(self.config)

	def set_validate_size(self, validate_size: int):
		self.validate_size = validate_size

	def get_validate_size(self):
		if self.validate_size:
			return self.validate_size
		else:
			return 0

