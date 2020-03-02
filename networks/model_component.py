# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : model_component.py
import torch
from torch import nn
from torch.optim import lr_scheduler

__all__ = ["get_scheduler", "init_net"]


def get_scheduler(optimizer, config):
    """

    :param optimizer:
    :param config:
    :return: define the scheduler
    """
    if config.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.epoch_start - config.niter) / float(config.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'warm_up':

        def lambda_warm_up(epoch):
            epoch = epoch + config.epoch_start - config.niter

            if epoch <= 10:
                lr_l = 3.5 * (epoch + 1) / 10
            elif 10 < epoch <= 30:
                lr_l = 0.2

            elif 30 < epoch <= 60:
                lr_l = 0.1
            else:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warm_up)

    elif config.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.niter + config.niter_decay, eta_min=4e-08)

    else:
        raise NotImplementedError("learning rate policy [%s] is not implemented" % config.lr_policy)
    return scheduler


def init_net(net, init_type="normal", init_gain=0.02):
    net = torch.nn.DataParallel(net).cuda()

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net
