import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from train import metric
from train.target_model import target_model
from data import iterator_factory as fac
from train.lr_scheduler import MultiFactorScheduler as MFS


def train_target_model(net_full, model_prefix, num_classes, tgt_dataset, input_conf, clip_length=8, segments=3, frame_per_seg=1, 
				train_frame_interval=2, val_frame_interval=2, resume_epoch=-1, batch_size=4, save_freq=1, lr_base=0.01, lr_factor=0.1, lr_steps=[4000, 8000], 
				end_epoch=1000, fine_tune=False, data_parallel=False, use_im=False, use_cluster=False, use_consistency=False, use_contrastive=False, **kwargs):

	assert torch.cuda.is_available(), "Currently, we only support CUDA version"
	torch.multiprocessing.set_sharing_strategy('file_system')

	# split network in net_full
	net_feat = net_full[0]
	net_fc = net_full[1]
	net_cls = net_full[2]

	# data iterator
	arid_mean = [0.079612, 0.073888, 0.072454]
	arid_std = [0.100459, 0.09705, 0.089911]
	tgt_mean = input_conf['mean']
	tgt_std = input_conf['std']
	if tgt_dataset == 'ARID':
		tgt_mean = arid_mean
		tgt_std = arid_std
	iter_seed = torch.initial_seed() + 100 + max(0, resume_epoch) * 100
	tgt_train_iter, tgt_eval_iter = fac.creat(name=tgt_dataset, batch_size=batch_size, clip_length=clip_length, segments=segments, frame_per_seg=frame_per_seg, 
										train_interval=train_frame_interval, val_interval=val_frame_interval, mean=tgt_mean, std=tgt_std, seed=iter_seed)

	# wapper (dynamic model)
	step_callback_freq = 50
	if tgt_dataset.upper() == 'ARID':
		step_callback_freq = 20
	elif tgt_dataset.upper() == 'HMDB51':
		step_callback_freq = 10
	elif tgt_dataset.upper() == 'KINETICS-SPORTS':
		step_callback_freq = 200
	logging.info("Actual learning step is {}".format(lr_steps))
	net = target_model(net_feat=net_feat,net_fc=net_fc,net_cls=net_cls,model_prefix=model_prefix,num_classes=num_classes,
						step_callback_freq=step_callback_freq,save_freq=save_freq,batch_size=batch_size,
						use_im=use_im,use_cluster=use_cluster,use_consistency=use_consistency,use_contrastive=use_contrastive)
	net.net_feat.cuda()
	net.net_fc.cuda()
	net.net_cls.cuda()

	# config optimization
	param_base_layers = []
	param_fc_layers = []
	param_class_layers = []
	# param_freeze_layers = []
	name_base_layers = []
	for name, param in net.net_feat.named_parameters():
		if fine_tune:
			param_base_layers.append(param)
			name_base_layers.append(name)
		else: 
			param_base_layers.append(param)
	for name, param in net.net_fc.named_parameters():
		if fine_tune:
			param_fc_layers.append(param)
		else:
			param_fc_layers.append(param)
	for name, param in net.net_cls.named_parameters():
		if fine_tune:
			param_class_layers.append(param)
		else:
			param_class_layers.append(param)

	if name_base_layers:
		out = "[\'" + '\', \''.join(name_base_layers) + "\']"
		logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers), out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

	if data_parallel:
		net.net_feat = torch.nn.DataParallel(net.net_feat).cuda()
		net.net_fc = torch.nn.DataParallel(net.net_fc).cuda()
		net.net_cls = torch.nn.DataParallel(net.net_cls).cuda()
	else:
		net.net_feat = net.net_feat.cuda()
		net.net_fc = net.net_fc.cuda()
		net.net_cls = net.net_cls.cuda()

	optimizer_feat = torch.optim.SGD([{'params':param_base_layers,'lr_mult':0.1}],lr=lr_base,momentum=0.9,weight_decay=0.0001,nesterov=True)
	optimizer_fc = torch.optim.SGD([{'params':param_fc_layers,'lr_mult':1.0}],lr=lr_base,momentum=0.9,weight_decay=0.0001,nesterov=True)
	optimizer_cls = torch.optim.SGD([{'params':param_class_layers,'lr_mult':0.0}],lr=lr_base,momentum=0.9,weight_decay=0.0001,nesterov=True)

	# resume training: model and optimizer
	if resume_epoch < 0:
		net.load_checkpoint(epoch=-1, optimizer_feat=optimizer_feat, optimizer_fc=optimizer_fc, optimizer_cls=optimizer_cls)
		epoch_start = 0
		step_counter = 0
	else:
		net.load_checkpoint(epoch=resume_epoch, optimizer_feat=optimizer_feat, optimizer_fc=optimizer_fc, optimizer_cls=optimizer_cls)
		epoch_start = resume_epoch
		step_counter = epoch_start * tgt_train_iter.__len__()

	# set learning rate scheduler
	num_worker = 1
	lr_scheduler = MFS(base_lr=lr_base, steps=[int(x/(batch_size*num_worker)) for x in lr_steps], factor=lr_factor, step_counter=step_counter)
	
	# define evaluation metric
	metrics_tgt = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(name="top1", topk=1), metric.Accuracy(name="top5", topk=5),)
	# enable cudnn tune
	cudnn.benchmark = True

	net.fit(train_iter=tgt_train_iter,eval_iter=tgt_eval_iter,optimizer_feat=optimizer_feat,optimizer_fc=optimizer_fc,optimizer_cls=optimizer_cls,
			lr_scheduler=lr_scheduler,metrics_tgt=metrics_tgt,epoch_start=epoch_start,epoch_end=end_epoch,)
