# coding:utf-8
# @Time         : 2019/5/9
# @Author       : xuyouze
# @File Name    : celebA_config.py

from .registry import DatasetConfig

__all__ = ["CelebAConfig"]


@DatasetConfig.register("celebA")
class CelebAConfig(object):

    def __init__(self):
        super(CelebAConfig, self).__init__()

        self.data_root_dir = "/media/data1/xuyouze/CelebA_full"
        self.part_file = 'Eval/list_eval_partition.txt'
        self.attr_file = 'Anno/list_attr_celeba.txt'
        self.landmark_file = "Anno/list_landmarks_align_celeba.txt"

        self.img_dir = "img_align_celeba"
        self.face_whole = "face_whole"
        self.attribute_num = 40
