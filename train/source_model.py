import os
import time
import socket
import logging
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import metric
from . import callback

"""
Static Model
"""
class static_model(object):

	def __init__(self, net_feat, net_fc, net_cls, model_prefix='', end_epoch=50, **kwargs):

		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		# init params
		self.net_feat = net_feat
		self.net_fc = net_fc
		self.net_cls = net_cls
		self.model_prefix = model_prefix
		self.end_epoch = end_epoch
		self.smooth_par = 0.1

	def load_state(self, state_dict_feat, state_dict_fc, state_dict_cls, strict=False):

		if strict:
			self.net_feat.load_state_dict(state_dict=state_dict_feat)
			self.net_fc.load_state_dict(state_dict=state_dict_fc)
			self.net_cls.load_state_dict(state_dict=state_dict_cls)
		else:
			# customized partialy load function
			net_state_keys_feat = list(self.net_feat.state_dict().keys())
			for name, param in state_dict_feat.items():
				if name in self.net_feat.state_dict().keys():
					dst_param_shape = self.net_feat.state_dict()[name].shape
					if param.shape == dst_param_shape:
						self.net_feat.state_dict()[name].copy_(param.view(dst_param_shape))
						net_state_keys_feat.remove(name)
			net_state_keys_fc = list(self.net_fc.state_dict().keys())
			for name, param in state_dict_fc.items():
				if name in self.net_fc.state_dict().keys():
					dst_param_shape = self.net_fc.state_dict()[name].shape
					if param.shape == dst_param_shape:
						self.net_fc.state_dict()[name].copy_(param.view(dst_param_shape))
						net_state_keys_fc.remove(name)
			net_state_keys_cls = list(self.net_cls.state_dict().keys())
			for name, param in state_dict_cls.items():
				if name in self.net_cls.state_dict().keys():
					dst_param_shape = self.net_cls.state_dict()[name].shape
					if param.shape == dst_param_shape:
						self.net_cls.state_dict()[name].copy_(param.view(dst_param_shape))
						net_state_keys_cls.remove(name)
			# indicating missed keys
			if net_state_keys_feat:
				num_batches_list = []
				for i in range(len(net_state_keys_feat)):
					if 'num_batches_tracked' in net_state_keys_feat[i]:
						num_batches_list.append(net_state_keys_feat[i])
				pruned_additional_states = [x for x in net_state_keys_feat if x not in num_batches_list]
				if pruned_additional_states:
					logging.info("There are layers in current network not initialized by pretrained")
					pruned = "[\'" + '\', \''.join(pruned_additional_states) + "\']"
					logging.warning(">> Failed to load for feature network: {}".format(pruned[0:150] + " ... " + pruned[-150:]))
				return False
			if net_state_keys_fc:
				num_batches_list = []
				for i in range(len(net_state_keys_fc)):
					if 'num_batches_tracked' in net_state_keys_fc[i]:
						num_batches_list.append(net_state_keys_fc[i])
				pruned_additional_states = [x for x in net_state_keys_fc if x not in num_batches_list]
				if pruned_additional_states:
					logging.info("There are layers in current network not initialized by pretrained")
					pruned = "[\'" + '\', \''.join(pruned_additional_states) + "\']"
					logging.warning(">> Failed to load for fcbn network: {}".format(pruned[0:150] + " ... " + pruned[-150:]))
				return False
			if net_state_keys_cls:
				num_batches_list = []
				for i in range(len(net_state_keys_cls)):
					if 'num_batches_tracked' in net_state_keys_cls[i]:
						num_batches_list.append(net_state_keys_cls[i])
				pruned_additional_states = [x for x in net_state_keys_cls if x not in num_batches_list]
				if pruned_additional_states:
					logging.info("There are layers in current network not initialized by pretrained")
					pruned = "[\'" + '\', \''.join(pruned_additional_states) + "\']"
					logging.warning(">> Failed to load for classifier network: {}".format(pruned[0:150] + " ... " + pruned[-150:]))
				return False
		return True

	def get_checkpoint_path(self, epoch):

		assert self.model_prefix, "model_prefix undefined!"
		if epoch > 0 and not epoch == end_epoch:
			checkpoint_path = "{}_ep-source-{:04d}.pth".format(self.model_prefix, epoch)
		elif epoch == self.end_epoch or not epoch > 0:
			checkpoint_path = "{}_ep-source.pth".format(self.model_prefix)
		return checkpoint_path

	def load_checkpoint(self, epoch, optimizer_feat=None, optimizer_fc=None, optimizer_cls=None):

		load_path = self.get_checkpoint_path(epoch)
		assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

		checkpoint = torch.load(load_path)
		all_params_matched = self.load_state(checkpoint['state_dict_feat'], checkpoint['state_dict_fc'], checkpoint['state_dict_cls'], strict=False)

		if optimizer_feat and optimizer_fc and optimizer_cls:
			if 'optimizer_feat' in checkpoint.keys() and 'optimizer_fc' in checkpoint.keys() and 'optimizer_cls' in checkpoint.keys()\
				and all_params_matched:
				optimizer_feat.load_state_dict(checkpoint['optimizer_feat'])
				optimizer_fc.load_state_dict(checkpoint['optimizer_fc'])
				optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
				logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
			else:
				logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
		else:
			logging.info("Only model state resumed from: `{}'".format(load_path))

		if 'epoch' in checkpoint.keys():
			if checkpoint['epoch'] != epoch:
				logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

	def save_checkpoint(self, epoch, optimizer_state_feat=None, optimizer_state_fc=None, optimizer_state_cls=None):

		save_path = self.get_checkpoint_path(epoch)
		save_folder = os.path.dirname(save_path)

		if not os.path.exists(save_folder):
			logging.debug("mkdir {}".format(save_folder))
			os.makedirs(save_folder)

		if not (optimizer_state_feat and optimizer_state_fc and optimizer_state_cls):
			torch.save({'epoch': epoch, 'state_dict_feat': self.net_feat.state_dict(), 'state_dict_fc': self.net_fc.state_dict(),
						'state_dict_cls': self.net_cls.state_dict()}, save_path)
			logging.info("Checkpoint (only model) saved to: {}".format(save_path))
		else:
			torch.save({'epoch': epoch, 'state_dict_feat': self.net_feat.state_dict(), 'optimizer_feat': optimizer_state_feat, 
						'state_dict_fc': self.net_fc.state_dict(), 'optimizer_fc': optimizer_state_fc,
						'state_dict_cls': self.net_cls.state_dict(), 'optimizer_cls': optimizer_state_cls}, save_path)
			logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


	def forward(self, src_data, src_label):
		""" typical forward function with:
			single output and single loss
		"""
		if 'TRN_BASE' in self.net_feat.__class__.__name__.upper():
			src_data = src_data.float().cuda()
			src_label = src_label.cuda()
			if self.net_feat.training:
				torch.set_grad_enabled(True)
			else:
				torch.set_grad_enabled(False)
			video_feat, scale_feats = self.net_feat(src_data)
			video_feat, scale_feats = self.net_fc(video_feat, scale_feats)
			video_feat, scale_feats, pred = self.net_cls(video_feat, scale_feats)
			output = pred

		else:
			logging.error("network '{}'' not implemented".format(self.net_feat.__class__.__name__))
			raise NotImplementedError()

		if src_label is not None:
			if self.net_feat.training and self.net_fc.training and self.net_cls.training:
				# source data class classification
				class_criterion = metric.CrossEntropyLabelSmooth(num_classes=pred.shape[-1], epsilon=self.smooth_par)
				if torch.cuda.is_available():
					class_criterion = class_criterion.cuda()
				loss = class_criterion(output, src_label)
			else:
				with torch.no_grad():
					class_criterion = nn.CrossEntropyLoss()
					if torch.cuda.is_available():
						class_criterion = class_criterion.cuda()
					loss = class_criterion(output, src_label)
		else:
			loss = None

		return [output], [loss], [video_feat], scale_feats


"""
Dynamic model that is able to update itself
"""
class source_model(static_model):

	def __init__(self,net_feat,net_fc,net_cls,model_prefix='',num_classes=400,step_callback=None,step_callback_freq=10,
				 epoch_callback=None,save_freq=1,batch_size=None,end_epoch=50,**kwargs):

		# load parameters
		if kwargs:
			logging.warning("Unknown kwargs in source_model: {}".format(kwargs))

		super(source_model, self).__init__(net_feat,net_fc,net_cls,model_prefix=model_prefix,end_epoch=end_epoch)

		# load optional arguments
		# - callbacks
		self.callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None, 'epoch_elapse': None, 
								'namevals': None, 'optimizer_dict_feat': None, 'optimizer_dict_fc': None, 'optimizer_dict_cls': None,}

		if not step_callback:
			step_callback = callback.CallbackList(callback.SpeedMonitor(), callback.MetricPrinter())

		if not epoch_callback:
			epoch_callback = (lambda **kwargs: None)

		self.num_classes = num_classes
		self.step_callback = step_callback
		self.step_callback_freq = step_callback_freq
		self.epoch_callback = epoch_callback
		self.save_freq = save_freq
		self.batch_size = batch_size

	def step_end_callback(self):
		# logging.debug("Step {} finished!".format(self.i_step))
		self.step_callback(**(self.callback_kwargs))

	def epoch_end_callback(self):
		self.epoch_callback(**(self.callback_kwargs))

		if self.callback_kwargs['epoch_elapse'] is not None:
			logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
					self.callback_kwargs['epoch'], self.callback_kwargs['epoch_elapse'], self.callback_kwargs['epoch_elapse']/3600.))

		if self.callback_kwargs['epoch'] == 0 or ((self.callback_kwargs['epoch']+1) % self.save_freq) == 0:
			self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1, 
								 optimizer_state_feat=self.callback_kwargs['optimizer_dict_feat'],
								 optimizer_state_fc=self.callback_kwargs['optimizer_dict_fc'],
								 optimizer_state_cls=self.callback_kwargs['optimizer_dict_cls'])

	"""
	Learning rate
	"""
	def adjust_learning_rate(self, lr, optimizer_feat, optimizer_fc, optimizer_cls):
		for param_group_feat in optimizer_feat.param_groups:
			if 'lr_mult' in param_group_feat:
				lr_mult = param_group_feat['lr_mult']
			else:
				lr_mult = 1.0
			param_group_feat['lr'] = lr * lr_mult

		for param_group_fc in optimizer_fc.param_groups:
			if 'lr_mult' in param_group_fc:
				lr_mult = param_group_fc['lr_mult']
			else:
				lr_mult = 1.0
			param_group_fc['lr'] = lr * lr_mult

		for param_group_cls in optimizer_cls.param_groups:
			if 'lr_mult' in param_group_cls:
				lr_mult = param_group_cls['lr_mult']
			else:
				lr_mult = 1.0
			param_group_cls['lr'] = lr * lr_mult

	"""
	Optimization
	"""
	def fit(self, train_iter, eval_iter, optimizer_feat, optimizer_fc, optimizer_cls, lr_scheduler,
			metrics_src=metric.Accuracy(topk=1), epoch_start=0, epoch_end=1000, **kwargs):

		"""
		checking
		"""
		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		assert torch.cuda.is_available(), "only support GPU version"

		"""
		start the main loop
		"""
		class_weight = torch.from_numpy(np.array([1.0] * self.num_classes)).float()
		if torch.cuda.is_available():
			class_weight = class_weight.cuda()
		pause_sec = 0.
		first_dynamic_weight=True
		for i_epoch in range(epoch_start, epoch_end):
			self.callback_kwargs['epoch'] = i_epoch
			epoch_start_time = time.time()

			###########
			# 1] TRAINING
			###########
			metrics_src.reset()
			self.net_feat.train()
			self.net_fc.train()
			self.net_cls.train()
			sum_sample_inst_src = 0
			sum_sample_elapse = 0.
			sum_update_elapse = 0
			batch_start_time = time.time()

			logging.info("Start epoch {:d}:".format(i_epoch))
			for i_batch, (src_data, src_label, _) in enumerate(train_iter):
				self.callback_kwargs['batch'] = i_batch
				update_start_time = time.time()
				# [forward] making next step
				if 'TRN_BASE' in self.net_feat.__class__.__name__.upper():
					src_outputs, losses, src_feats, _ = self.forward(src_data, src_label)

				# [backward]
				optimizer_feat.zero_grad()
				optimizer_fc.zero_grad()
				optimizer_cls.zero_grad()
				for loss in losses: loss.backward()
				self.adjust_learning_rate(optimizer_feat=optimizer_feat, optimizer_fc=optimizer_fc, optimizer_cls=optimizer_cls, lr=lr_scheduler.update())
				optimizer_feat.step()
				optimizer_fc.step()
				optimizer_cls.step()

				# [evaluation] update train metric
				metrics_src.update([output.data.cpu() for output in src_outputs], src_label.cpu(), [loss.data.cpu() for loss in losses])

				# timing each batch
				sum_sample_elapse += time.time() - batch_start_time
				sum_update_elapse += time.time() - update_start_time
				batch_start_time = time.time()
				sum_sample_inst_src += src_data.shape[0]

				if (i_batch % self.step_callback_freq) == 0:
					# retrive eval results and reset metic
					self.callback_kwargs['namevals'] = metrics_src.get_name_value()
					metrics_src.reset()
					# speed monitor
					self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst_src
					self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst_src
					# callbacks
					self.step_end_callback()
					sum_update_elapse = 0
					sum_sample_elapse = 0
					sum_sample_inst_src = 0
				# torch.cuda.empty_cache()
			###########
			# 2] END OF EPOCH
			###########
			self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
			self.callback_kwargs['optimizer_dict_feat'] = optimizer_feat.state_dict()
			self.callback_kwargs['optimizer_dict_fc'] = optimizer_fc.state_dict()
			self.callback_kwargs['optimizer_dict_cls'] = optimizer_cls.state_dict()
			self.epoch_end_callback()

			###########
			# 3] Evaluation
			###########
			if (eval_iter is not None) and ((i_epoch+1) % max(1, int(self.save_freq/2))) == 0:

				logging.info("Start evaluating epoch {:d}:".format(i_epoch))
				# torch.cuda.empty_cache()
				metrics_src.reset()
				self.net_feat.eval()
				self.net_fc.eval()
				self.net_cls.eval()
				sum_sample_elapse = 0.
				sum_sample_inst_tgt = 0
				sum_forward_elapse = 0.
				batch_start_time = time.time()

				for i_batch_eval, (src_data_eval, src_label_eval, _) in enumerate(eval_iter):
					self.callback_kwargs['batch'] = i_batch_eval

					forward_start_time = time.time()

					if 'TRN_BASE' in self.net_feat.__class__.__name__.upper():
						eval_outputs, eval_losses, _, _ = self.forward(src_data_eval, src_label_eval)

					metrics_src.update([output.data.cpu() for output in eval_outputs], src_label_eval.cpu(), [loss.data.cpu() for loss in eval_losses])

					sum_forward_elapse += time.time() - forward_start_time
					sum_sample_elapse += time.time() - batch_start_time
					batch_start_time = time.time()
					sum_sample_inst_tgt += src_data_eval.shape[0]

				# evaluation callbacks
				self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst_tgt
				self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_sample_inst_tgt
				self.callback_kwargs['namevals'] = metrics_src.get_name_value()
				self.step_end_callback()

		logging.info("Optimization done!")
