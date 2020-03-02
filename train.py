# coding:utf-8
# @Time         : 2019/5/14 
# @Author       : xuyouze
# @File Name    : train.py
import os
import time

import torch

from config import TrainConfig, TestConfig
from models import *
from dataset import *


def validate(model):
	validate_config = TestConfig()
	validate_config.isTest = True

	logger = validate_config.test_logger
	logger.info("--------------------------------------------------------")
	logger.debug("test the model using the validate dataset")

	validate_dataset = create_dataset(validate_config)
	model.eval()
	model.clear_precision()
	model.set_validate_size(len(validate_dataset))
	logger.info("validate dataset len: %d " % len(validate_dataset))
	validate_total_iter = int(len(validate_dataset) / validate_config.batch_size)

	for j, valida_data in enumerate(validate_dataset):
		model.set_input(valida_data)
		logger.debug("[%s/%s]" % (j, validate_total_iter))
		model.test()
	# output the precision
	logger.debug(model.get_model_precision())
	logger.debug(model.get_model_class_balance_precision())
	logger.info("mean accuracy: {}".format(torch.mean(model.get_model_precision())))
	logger.info(
		"class_balance accuracy: {}".format(torch.mean(model.get_model_class_balance_precision())))
	logger.debug("validate mode end")
	logger.info("--------------------------------------------------------")


if __name__ == '__main__':

	# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	# set_seed(2019)

	config = TrainConfig()
	config.continue_train = False  # continue training: load the latest model
	# config.continue_train = True  # continue training: load the latest model
	# config.load_iter = 16
	logger = config.logger

	config.batch_size = 80

	logger.info("current batch size is {}".format(config.batch_size))
	dataset = create_dataset(config)  # create a dataset given opt.dataset_mode and other options
	model = create_model(config)
	model.setup()
	config.print_freq = int(len(dataset) / config.batch_size / 10)
	total_iters = 0
	logger.info("dataset len: %d " % len(dataset))
	logger.info("----------------------- model build complete ----------------------------")

	# train begin
	for epoch in range(config.epoch_start, config.niter + config.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		logger.info("epoch [{}/{}] begin at: {} ,learning rate : {}".format(epoch, config.niter + config.niter_decay,
																			time.strftime('%Y-%m-%d %H:%M:%S',
																						  time.localtime(
																							  epoch_start_time)),
																			model.get_learning_rate()))

		for i, data in enumerate(dataset):
			iter_start_time = time.time()

			total_iters += config.batch_size
			epoch_iter += config.batch_size
			model.set_input(data)
			model.optimize_parameters()

			if i % config.print_freq == 0:
				losses = model.get_current_loss()
				t_comp = (time.time() - iter_start_time) / config.batch_size
				logger.debug("epoch[%d/%d], iter[%d/%d],current loss=%s,Time consuming: %s sec" % (
					epoch, config.niter + config.niter_decay, epoch_iter, len(dataset), losses, t_comp))

			if total_iters % config.save_latest_freq == 0:
				logger.debug("saving the last model (epoch %d, total iters %d)" % (epoch, total_iters))
				model.save_networks(config.last_epoch)

		logger.debug('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
		# save weight
		model.save_networks(config.last_epoch)
		model.save_networks("iter_%d" % epoch)
		# get the model precision
		validate(model)
		model.update_loss_dropout(epoch)

		model.train()

		logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (
			epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()  # update learning rates at the end of every epoch.
