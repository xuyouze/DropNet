# coding:utf-8
# @Time         : 2019/9/16 
# @Author       : xuyouze
# @File Name    : train_config.py


from .base_config import BaseConfig

__all__ = ["TrainConfig"]


class TrainConfig(BaseConfig):
	def __init__(self):
		super(TrainConfig, self).__init__()

		# network saving and print parameters
		self.save_latest_freq = 256 * 1024  # frequency of saving the latest results
		self.print_freq = 200  # frequency of showing training results on console
		self.continue_train = False  # continue training: load the latest model
		self.isTrain = True

		# training parameters
		self.epoch_start = 0  # the starting epoch count
		self.niter = 0  # of iter at starting learning rate used for linear learning rate policy
		self.niter_decay = 60  # of iter to linearly decay learning rate to zero,

		# momentum term of adam
		self.beta1 = 0.9

		# the batch size of image
		self.batch_size = 128

		# lr parameters
		self.lr = 0.001  # initial learning rate for adam
		# self.lr = 0.00001
		# learning rate policy. [linear | warm_up | cosine | step]
		self.lr_policy = "linear"
		# self.lr_policy = "warm_up"

		# loss parameters for focal loss
		self.gamma = 2
		self.alpha = 1
		self.size_average = True

		# loss parameters for ghm-c loss
		self.bins = 100

		# loss parameters for drop loss
		self.dropout_scope = 30
		self.dropout_scope_decay = 20
		self.dropout_scope_lowest = 30
