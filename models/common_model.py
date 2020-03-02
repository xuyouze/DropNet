# coding:utf-8
# @Time         : 2019/6/27 
# @Author       : xuyouze
# @File Name    : common_model.py


import torch
from torch import nn, optim
from torch.autograd import Variable

from config import BaseConfig
from models.registry import Model
from .base_model import BaseModel
from loss import create_loss

__all__ = ["CommonModel"]

@Model.register("common")
class CommonModel(BaseModel):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)
        # define the net, such as net_%s % net_names
        # self.net_face_upper =
        self.config = config
        self.net_names = ["whole"]

        # define the net
        self.net_whole = self.create_network_model()
        self.attr_whole_index = [i for i in range(config.dataset_config.attribute_num)]

        # define input and output
        for name in self.net_names:
            setattr(self, "img_%s" % name, None)
            setattr(self, "output_%s" % name, None)
            setattr(self, "attr_%s" % name, None)

        # define optimizer and loss
        if config.isTrain:
            for name in self.net_names:
                if self.config.loss_name == "bce":
                    setattr(self, "criterion_%s" % name, nn.BCEWithLogitsLoss())
                else:
                    setattr(self, "criterion_%s" % name, create_loss(self.config))

                setattr(self, "optimizer_%s" % name,
                        optim.Adam(getattr(self, "net_%s" % name).parameters(), lr=config.lr,
                                   betas=(config.beta1, 0.999)))
                setattr(self, "loss_%s" % name, None)
                self.optimizers.append(getattr(self, "optimizer_%s" % name))

    def set_input(self, x):
        self.img_whole, self.attr = x
        for name in self.net_names:
            setattr(self, "img_%s" % name, Variable(getattr(self, "img_%s" % name)).cuda())
            setattr(self, "attr_%s" % name,
                    Variable(self.attr[:, getattr(self, "attr_%s_index" % name)].cuda()).type(torch.cuda.FloatTensor))

    def forward(self):
        for name in self.net_names:
            setattr(self, "output_%s" % name, getattr(self, "net_%s" % name)(getattr(self, "img_%s" % name)))

    def update_loss_dropout(self, epoch):
        for name in self.net_names:
            getattr(self, "criterion_%s" % name).update_loss_dropout(epoch)

    def get_criterion_bins(self):
        return getattr(self, "criterion_%s" % "whole").get_weight_num()
