# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : __init__.py


import importlib

from torch.utils import data

from config.base_config import BaseConfig
from dataset.base_dataset import BaseDataset
from .build import build_dataset

__all__ = ["create_dataset"]

#
# def find_dataset_using_name(dataset_name: str):
#     dataset_filename = "dataset." + dataset_name + "_dataset"
#     datasetlib = importlib.import_module(dataset_filename)
#
#     dataset = None
#     target_dataset_name = dataset_name.replace("_", "") + "dataset"
#     for name, cls in datasetlib.__dict__.items():
#         if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
#             dataset = cls
#             break
#
#     if not dataset:
#         raise NotImplementedError(
#             "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
#                 dataset_filename, target_dataset_name))
#     return dataset


class CustomDatasetDataLoader(object):

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        # dataset_class = find_dataset_using_name(config.dataset_name)
        dataset_class = build_dataset(config.dataset_name)
        self.dataset = dataset_class(config)
        config.logger.info("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=not config.isTrain,
            num_workers=int(config.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data


def create_dataset(config):
    data_loader = CustomDatasetDataLoader(config)
    dataset = data_loader.load_data()
    return dataset
