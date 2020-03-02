# coding:utf-8
# @Time         : 2019/10/9 
# @Author       : xuyouze
# @File Name    : dukemtmc_dataset.py


import os

import numpy as np
from torchvision.transforms import transforms
import scipy.io
from config import TrainConfig, TestConfig
import time
from PIL import Image

from config import *
from dataset.registry import Dataset
from .base_dataset import BaseDataset

__all__ = ["DukeMTMCDataset"]


def get_img(dataset_config):
	data_group = [{"data": [], "ids": []} for _ in range(3)]
	# data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']

	for group, suffix in zip(data_group, dataset_config.data_group_suffix):
		name_dir = os.path.join(dataset_config.data_root_dir, suffix)
		file_list = sorted(os.listdir(name_dir))
		for name in file_list:
			if name[-3:] == 'jpg':
				id = name.split('_')[0]
				cam = int(name.split('_')[1][1])
				images = os.path.join(name_dir, name)
				if id != '0000' and id != '-1':
					if id not in group['ids']:
						group['ids'].append(id)
					group['data'].append(
						[images, group['ids'].index(id), id, cam, name.split('.')[0]])

	return data_group


def get_data(data_dir, data_group_suffix):
	# train = query = test = {}
	data_group = [{} for _ in range(3)]
	# data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']

	for group, suffix in zip(data_group, data_group_suffix):
		name_dir = os.path.join(data_dir, suffix)
		file_list = os.listdir(name_dir)
		for name in file_list:
			if name[-3:] == 'jpg':
				idx = name.split('_')[0]
				if idx not in group:
					group[idx] = [[] for _ in range(9)]
				cam_n = int(name.split('_')[1][1]) - 1
				group[idx][cam_n].append(os.path.join(name_dir, name))
	return data_group


def get_img_attr(dataset_config):
	train, query, test = get_data(dataset_config.data_root_dir, dataset_config.data_group_suffix)
	train_label = ['backpack',
				   'bag',
				   'handbag',
				   'boots',
				   'gender',
				   'hat',
				   'shoes',
				   'top',
				   'downblack',
				   'downwhite',
				   'downred',
				   'downgray',
				   'downblue',
				   'downgreen',
				   'downbrown',
				   'upblack',
				   'upwhite',
				   'upred',
				   'uppurple',
				   'upgray',
				   'upblue',
				   'upgreen',
				   'upbrown']

	test_label = ['boots',
				  'shoes',
				  'top',
				  'gender',
				  'hat',
				  'backpack',
				  'bag',
				  'handbag',
				  'downblack',
				  'downwhite',
				  'downred',
				  'downgray',
				  'downblue',
				  'downgreen',
				  'downbrown',
				  'upblack',
				  'upwhite',
				  'upred',
				  'upgray',
				  'upblue',
				  'upgreen',
				  'uppurple',
				  'upbrown']

	train_person_id = []
	for personid in train:
		train_person_id.append(personid)
	train_person_id.sort(key=int)

	test_person_id = []
	for personid in test:
		test_person_id.append(personid)
	test_person_id.sort(key=int)

	f = scipy.io.loadmat(os.path.join(dataset_config.data_root_dir, dataset_config.attr_file))

	test_attribute = {}
	train_attribute = {}
	for test_train in range(len(f['duke_attribute'][0][0])):
		if test_train == 1:
			id_list_name = 'test_person_id'
			group_name = 'test_attribute'
		else:
			id_list_name = 'train_person_id'
			group_name = 'train_attribute'
		for attribute_id in range(len(f['duke_attribute'][0][0][test_train][0][0])):
			if isinstance(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
				continue
			for person_id in range(len(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0])):
				id = locals()[id_list_name][person_id]
				if id not in locals()[group_name]:
					locals()[group_name][id] = []
				locals()[group_name][id].append(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])

	for i in range(8):
		train_label.insert(8, train_label[-1])
		train_label.pop(-1)

	unified_train_atr = {}
	for k, v in train_attribute.items():
		temp_atr = list(v)
		for i in range(8):
			temp_atr.insert(8, temp_atr[-1])
			temp_atr.pop(-1)
		unified_train_atr[k] = temp_atr

	unified_test_atr = {}
	for k, v in test_attribute.items():
		temp_atr = [0] * len(train_label)
		for i in range(len(train_label)):
			temp_atr[i] = v[test_label.index(train_label[i])]
		unified_test_atr[k] = temp_atr
	# two zero appear in train '0370' '0679'
	# zero_check=[]
	# for id in train_attribute:
	#    if 0 in train_attribute[id]:
	#        zero_check.append(id)
	# for i in range(len(zero_check)):
	#    train_attribute[zero_check[i]] = [1 if x==0 else x for x in train_attribute[zero_check[i]]]
	unified_train_atr['0370'][7] = 1
	unified_train_atr['0679'][7] = 2

	return unified_train_atr, unified_test_atr, train_label


def get_attr_binary(dataset_config):
	train_duke_attr, test_duke_attr, label = get_img_attr(dataset_config)
	for idx in train_duke_attr:
		train_duke_attr[idx][:] = [x - 1 for x in train_duke_attr[idx]]
	for idx in test_duke_attr:
		test_duke_attr[idx][:] = [x - 1 for x in test_duke_attr[idx]]
	return train_duke_attr, test_duke_attr, label


@Dataset.register("duke")
class DukeMTMCDataset(BaseDataset):
	"""
	the code for market-1501 and dukemtmc dataset are cite from https://github.com/hyk1996/Person-Attribute-Recognition-MarketDuke
	"""

	def __init__(self, config: BaseConfig):
		super().__init__(config)
		self.config = config
		# self.img_dir = os.path.join(self.config.dataset_config.data_root_dir, self.config.dataset_config.img_dir)
		train_attr, test_attr, self.label = get_attr_binary(self.config.dataset_config)
		train, val, test = get_img(self.config.dataset_config)

		if config.isTrain:

			# distribution:每个属性的正样本占比

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
