# coding:utf-8
# @Time         : 2019/7/30
# @Author       : xuyouze
# @File Name    : ghmc_Loss.py

import torch
import torch.nn as nn

from config import TrainConfig
from loss.registry import Loss

__all__ = ["GHMCLoss"]


@Loss.register("ghmc")
class GHMCLoss(nn.Module):
	"""
		This is a multi-attribute implementation of ghm-c loss
		The paper of ghm-c : https://www.aaai.org/ojs/index.php/AAAI/article/view/4877
	"""

	def __init__(self, config: TrainConfig):
		super(GHMCLoss, self).__init__()

		bins = 100
		momentum = 0.6

		self.bins = 100
		self.momentum = momentum
		self.edges = [float(x) / bins for x in range(bins + 1)]
		self.edges[-1] += 1e-6
		self.acc_sum = None
		self.class_loss_weight = None
		self.weight_num = torch.zeros(40, bins)

	def forward(self, pred, target, *args, **kwargs):
		""" Args:
		pred [batch_num, class_num]:
			The direct prediction of classification fc layer.
		target [batch_num, class_num]:
			multi class target(index) for each sample.
		"""

		edges = self.edges
		mmt = self.momentum
		weights = torch.zeros_like(pred)
		batch_size, class_num = target.shape

		# gradient length
		g = torch.abs(pred.sigmoid() - target).detach()

		# tot = torch.numel(pred)
		num_in_bin = torch.zeros((class_num, self.bins))
		for i in range(self.bins):
			# idxs = N * C
			# weight = N * C
			# num_in_bins = C * bins

			idxs = (g >= edges[i]) & (g < edges[i + 1])
			# num_in_bin[:, i] = idxs.sum(dim=0)

			for j in range(class_num):
				if num_in_bin[j, i] > 0:
					if mmt > 0:
						if self.acc_sum is None:
							self.acc_sum = torch.zeros_like(num_in_bin)
						self.acc_sum[j, i] = mmt * self.acc_sum[j, i] + (1 - mmt) * num_in_bin[j, i]
						weights[:, j] = weights[:, j] + batch_size / self.acc_sum[j, i] * idxs[:, j].to(
							dtype=torch.float32)

					else:
						weights[:, j] = weights[:, j] + batch_size / num_in_bin[j, i] * idxs[:, j].to(
							dtype=torch.float32)
		self.weight_num = self.weight_num + num_in_bin

		# weights = torch.pow(weights, 0.5)
		# loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
		loss = nn.BCEWithLogitsLoss()

		# if self.class_loss_weight is not None:
		#     loss = loss * self.class_loss_weight
		# return loss.mean()
		return loss.forward(pred, target)

	def set_weight(self, weight):
		self.class_loss_weight = weight

	def get_weight_num(self):
		a = self.weight_num
		self.weight_num = torch.zeros(40, self.bins)
		return a
