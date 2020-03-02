# coding:utf-8
# @Time         : 2019/5/14 
# @Author       : xuyouze
# @File Name    : test.py

import os

import torch

from config import TestConfig
from dataset import create_dataset
from models import create_model


def test_method(gpu):
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu
	config = TestConfig()
	# config.checkpoints_dir = "/media/data2/xyz_data/CelebA_full/full_third_2019-6-19_0.9135_ckp"

	print("{} model was initialized".format(config.model_name))
	# dataset is test set or val
	config.isTest = True
	dataset = create_dataset(config=config)
	model = create_model(config)
	for j in range(0, 102, 1):
		config.load_iter = j
		model.setup()
		model.clear_precision()
		if config.eval:
			model.eval()
		dataset_size = len(dataset)

		print("test dataset len: %d " % dataset_size)
		total_iter = int(dataset_size / config.batch_size)
		model.set_validate_size(dataset_size)
		# fc_feature = np.zeros((dataset_size, 2048))
		# label = np.zeros((dataset_size, 40))
		for i, data in enumerate(dataset):
			model.set_input(data)
			print("[%s/%s]" % (i, total_iter))
			model.test()

		print(model.get_model_precision())
		print(model.get_model_class_balance_precision())
		print("mean accuracy: {}".format(torch.mean(model.get_model_precision())))
		print("class_balance accuracy: {}".format(torch.mean(model.get_model_class_balance_precision())))


test_method("0")
