# coding:utf-8
# @Time         : 2019/10/17 
# @Author       : xuyouze
# @File Name    : market_dataset.py


import os

import numpy as np
from torchvision.transforms import transforms
import scipy.io

from PIL import Image

from config import *
from dataset.dukemtmc_dataset import get_img, get_data
from dataset.registry import Dataset
from .base_dataset import BaseDataset

__all__ = ["MarketDataset"]


def get_img_attr(dataset_config):
	train, query, test = get_data(dataset_config.data_root_dir, dataset_config.data_group_suffix)
	train_label = ['age',
				   'backpack',
				   'bag',
				   'handbag',
				   'downblack',
				   'downblue',
				   'downbrown',
				   'downgray',
				   'downgreen',
				   'downpink',
				   'downpurple',
				   'downwhite',
				   'downyellow',
				   'upblack',
				   'upblue',
				   'upgreen',
				   'upgray',
				   'uppurple',
				   'upred',
				   'upwhite',
				   'upyellow',
				   'clothes',
				   'down',
				   'up',
				   'hair',
				   'hat',
				   'gender']

	test_label = ['age',
				  'backpack',
				  'bag',
				  'handbag',
				  'clothes',
				  'down',
				  'up',
				  'hair',
				  'hat',
				  'gender',
				  'upblack',
				  'upwhite',
				  'upred',
				  'uppurple',
				  'upyellow',
				  'upgray',
				  'upblue',
				  'upgreen',
				  'downblack',
				  'downwhite',
				  'downpink',
				  'downpurple',
				  'downyellow',
				  'downgray',
				  'downblue',
				  'downgreen',
				  'downbrown'
				  ]

	train_person_id = []
	for personid in train:
		train_person_id.append(personid)
	train_person_id.sort(key=int)

	test_person_id = []
	for personid in test:
		test_person_id.append(personid)
	test_person_id.sort(key=int)
	test_person_id.remove('-1')
	test_person_id.remove('0000')

	f = scipy.io.loadmat(os.path.join(dataset_config.data_root_dir, dataset_config.attr_file))

	test_attribute = {}
	train_attribute = {}
	for test_train in range(len(f['market_attribute'][0][0])):
		if test_train == 0:
			id_list_name = 'test_person_id'
			group_name = 'test_attribute'
		else:
			id_list_name = 'train_person_id'
			group_name = 'train_attribute'
		for attribute_id in range(len(f['market_attribute'][0][0][test_train][0][0])):
			if isinstance(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
				continue
			for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][attribute_id][0])):
				id = locals()[id_list_name][person_id]
				if id not in locals()[group_name]:
					locals()[group_name][id] = []
				locals()[group_name][id].append(
					f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])

	unified_train_atr = {}
	for k, v in train_attribute.items():
		temp_atr = [0] * len(test_label)
		for i in range(len(test_label)):
			temp_atr[i] = v[train_label.index(test_label[i])]
		unified_train_atr[k] = temp_atr

	return unified_train_atr, test_attribute, test_label


def get_attr_binary(dataset_config):
	train_market_attr, test_market_attr, label = get_img_attr(dataset_config)

	for id in train_market_attr:
		train_market_attr[id][:] = [x - 1 for x in train_market_attr[id]]
		if train_market_attr[id][0] == 0:
			train_market_attr[id].pop(0)
			train_market_attr[id].insert(0, 1)
			train_market_attr[id].insert(1, 0)
			train_market_attr[id].insert(2, 0)
			train_market_attr[id].insert(3, 0)
		elif train_market_attr[id][0] == 1:
			train_market_attr[id].pop(0)
			train_market_attr[id].insert(0, 0)
			train_market_attr[id].insert(1, 1)
			train_market_attr[id].insert(2, 0)
			train_market_attr[id].insert(3, 0)
		elif train_market_attr[id][0] == 2:
			train_market_attr[id].pop(0)
			train_market_attr[id].insert(0, 0)
			train_market_attr[id].insert(1, 0)
			train_market_attr[id].insert(2, 1)
			train_market_attr[id].insert(3, 0)
		elif train_market_attr[id][0] == 3:
			train_market_attr[id].pop(0)
			train_market_attr[id].insert(0, 0)
			train_market_attr[id].insert(1, 0)
			train_market_attr[id].insert(2, 0)
			train_market_attr[id].insert(3, 1)

	for id in test_market_attr:
		test_market_attr[id][:] = [x - 1 for x in test_market_attr[id]]
		if test_market_attr[id][0] == 0:
			test_market_attr[id].pop(0)
			test_market_attr[id].insert(0, 1)
			test_market_attr[id].insert(1, 0)
			test_market_attr[id].insert(2, 0)
			test_market_attr[id].insert(3, 0)
		elif test_market_attr[id][0] == 1:
			test_market_attr[id].pop(0)
			test_market_attr[id].insert(0, 0)
			test_market_attr[id].insert(1, 1)
			test_market_attr[id].insert(2, 0)
			test_market_attr[id].insert(3, 0)
		elif test_market_attr[id][0] == 2:
			test_market_attr[id].pop(0)
			test_market_attr[id].insert(0, 0)
			test_market_attr[id].insert(1, 0)
			test_market_attr[id].insert(2, 1)
			test_market_attr[id].insert(3, 0)
		elif test_market_attr[id][0] == 3:
			test_market_attr[id].pop(0)
			test_market_attr[id].insert(0, 0)
			test_market_attr[id].insert(1, 0)
			test_market_attr[id].insert(2, 0)
			test_market_attr[id].insert(3, 1)

	label.pop(0)
	label.insert(0, 'young')
	label.insert(1, 'teenager')
	label.insert(2, 'adult')
	label.insert(3, 'old')

	return train_market_attr, test_market_attr, label


@Dataset.register("market")
class MarketDataset(BaseDataset):
	"""
	the code for market-1501 and dukemtmc dataset are cite from https://github.com/hyk1996/Person-Attribute-Recognition-MarketDuke
	"""

	def __init__(self, config: BaseConfig):
		super().__init__(config)
		self.config = config
		# self.img_dir = os.path.join(self.config.dataset_config.data_root_dir, self.config.dataset_config.img_dir)
		train, val, test = get_img(self.config.dataset_config)
		train_attr, test_attr, self.label = get_attr_binary(self.config.dataset_config)

		if config.isTrain:

			# distribution:每个属性的正样本占比
			distribution = np.zeros(self.config.dataset_config.attribute_num)
			for k, v in train_attr.items():
				distribution += np.array(v)
			self.distribution = distribution / len(train_attr)
			self.img = train['data']
			self.img_ids = train['ids']
			self.img_attr = train_attr

			self.transforms = transforms.Compose([
				transforms.Resize(size=(288, 144)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

		else:
			self.transforms = transforms.Compose([
				transforms.Resize(size=(288, 144)),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
			self.img_attr = test_attr

			if self.config.isTest:
				# test set
				self.img = test['data']
				self.img_ids = test['ids']
			else:
				# val set
				self.img = val['data']
				self.img_ids = val['ids']

		self.length = len(self.img)

	def __getitem__(self, index):

		img_path = self.img[index][0]

		i = self.img[index][1]
		id = self.img[index][2]
		cam = self.img[index][3]
		name = self.img[index][4]

		label = np.asarray(self.img_attr[id])
		images = Image.open(img_path).convert('RGB')

		images = self.transforms(images)

		return images, label

	# return images, i, label, id, cam, name

	def __len__(self):
		return self.length
