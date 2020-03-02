# coding:utf-8
# @Time         : 2019/7/2 
# @Author       : xuyouze
# @File Name    : focal_loss.py
import torch
from torch import nn
from torch.autograd import Variable

from config import TrainConfig
from loss.registry import Loss

__all__ = ["FocalLoss"]


@Loss.register("focal")
class FocalLoss(nn.Module):
	"""
		This is a multi-attribute implementation of focal loss
		The href of  focal loss :https://github.com/unsky/focal-loss
	"""

	def __init__(self, config: TrainConfig):

		super(FocalLoss, self).__init__()
		self.config = config
		self.gamma = config.gamma
		self.alpha = config.alpha
		self.size_average = config.size_average
		self.weight = None

	def forward(self, input, target):
		"""

		:param input: the shape is
		:param target:
		:return:
		"""
		if self.alpha is None or isinstance(self.alpha, (float, int)):
			self.alpha = (self.alpha * torch.ones((target.size(1), 2))).cuda()
		if isinstance(self.alpha, list):
			self.alpha = torch.stack((torch.tensor(self.alpha), 1 - torch.tensor(self.alpha)), dim=1).cuda()
		pt = Variable(torch.sigmoid(input)).cuda()
		# loss = nn.BCELoss(reduction="none")(pt, target)
		loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)
		loss = target * torch.pow(1 - pt, self.gamma) * loss + (1 - target) * torch.pow(pt, self.gamma) * loss

		if self.weight is not None:
			loss = loss * self.weight
		assert loss.shape == target.shape

		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()

	def set_weight(self, weight):
		self.weight = weight
