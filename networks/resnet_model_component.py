# coding:utf-8
# @Time         : 2019/7/29 
# @Author       : xuyouze
# @File Name    : resnet_model_component.py
import torch
from torch import nn
from torchvision.models import resnet50

from networks.registry import Network

__all__ = ["define_net"]


@Network.register("resnet")
def define_net(config):
    net_whole = Resnet50(True, config.dataset_config.attribute_num)
    return torch.nn.DataParallel(net_whole).cuda()


class Resnet50(nn.Module):
    def __init__(self, pre_train, output_num=40):
        super().__init__()
        pre_model = resnet50(pretrained=pre_train)
        self.resnet_layer = nn.Sequential(*list(pre_model.children())[:-1])
        self.Linear_layer = nn.Linear(2048, output_num, bias=False)
        self.BN_layer = nn.BatchNorm2d(2048)

    def forward(self, x):
        x = self.resnet_layer(x)
        # nn.Conv2d(kernel_size=3, stride=1, padding=0, bias=False)
        x = self.BN_layer(x)

        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
