"""
Metric function library
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations as c

class EvalMetric(object):

	def __init__(self, name, **kwargs):
		self.name = str(name)
		self.reset()

	def update(self, preds, labels, losses):
		raise NotImplementedError()

	def reset(self):
		self.num_inst = 0
		self.sum_metric = 0.0

	def get(self):
		if self.num_inst == 0:
			return (self.name, float('nan'))
		else:
			return (self.name, self.sum_metric / self.num_inst)

	def get_name_value(self):
		name, value = self.get()
		if not isinstance(name, list):
			name = [name]
		if not isinstance(value, list):
			value = [value]
		return list(zip(name, value))

	def check_label_shapes(self, preds, labels):
		# raise if the shape is inconsistent
		if (type(labels) is list) and (type(preds) is list):
			label_shape, pred_shape = len(labels), len(preds)
		else:
			label_shape, pred_shape = labels.shape[0], preds.shape[0]

		if label_shape != pred_shape:
			raise NotImplementedError("")


class MetricList(EvalMetric):
	"""Handle multiple evaluation metric
	"""
	def __init__(self, *args, name="metric_list"):
		assert all([issubclass(type(x), EvalMetric) for x in args]), \
			"MetricList input is illegal: {}".format(args)
		self.metrics = [metric for metric in args]
		super(MetricList, self).__init__(name=name)

	def update(self, preds, labels, losses=None):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels
		losses = [losses] if type(losses) is not list else losses

		for metric in self.metrics:
			metric.update(preds, labels, losses)

	def reset(self):
		if hasattr(self, 'metrics'):
			for metric in self.metrics:
				metric.reset()
		else:
			logging.warning("No metric defined.")

	def get(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get())
		return ouputs

	def get_name_value(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get_name_value())
		return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
	"""Computes accuracy classification score.
	"""
	def __init__(self, name='accuracy', topk=1):
		super(Accuracy, self).__init__(name)
		self.topk = topk

	def update(self, preds, labels, losses):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels

		self.check_label_shapes(preds, labels)
		for pred, label in zip(preds, labels):
			assert self.topk <= pred.shape[1], \
				"topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
			_, pred_topk = pred.topk(self.topk, 1, True, True)

			pred_topk = pred_topk.t()
			correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))

			self.sum_metric += float(correct.contiguous().view(-1).float().sum(0, keepdim=True).numpy())
			self.num_inst += label.shape[0]


class Loss(EvalMetric):
	"""Dummy metric for directly printing loss.
	"""
	def __init__(self, name='loss'):
		super(Loss, self).__init__(name)

	def update(self, preds, labels, losses):
		assert losses is not None, "Loss undefined."
		for loss in losses:
			self.sum_metric += float(loss.numpy().sum())
			self.num_inst += loss.numpy().size


def Entropy(input_):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	return entropy


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""
	def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.reduction = reduction
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
		if self.use_gpu: targets = targets.cuda()
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).sum(dim=1)
		if self.reduction:
			return loss.mean()
		else:
			return loss
		return loss


def weighted_consistency_loss(logits, video_logit, lbd=10.0, eta=0.5, loss='kl', k_mom=1, sfm_par=1):
	m = len(logits)
	softmaxs = [F.softmax(logit, dim=1) for logit in logits]
	video_softmax = F.softmax(video_logit, dim=1)
	avg_softmax = sum(softmaxs) / m
	avg_logit = sum(logits) / m
	weights = [class_wise_entropy(logit, sfm_par) for logit in logits]

	loss_kl = [weight * kl_div(logit, avg_softmax) for (weight, logit) in zip(weights, logits)]
	loss_kl = sum(loss_kl) / m

	if loss == 'kl':
		consistency = lbd * loss_kl
	elif loss == 'mse':
		sm1, sm2 = avg_softmax, video_softmax.unsqueeze(-1)
		loss_mse = ((sm2 - sm1) ** k_mom).sum(1)
		consistency = lbd * loss_mse
	elif loss == 'mse-kl':
		sm1, sm2 = avg_softmax, video_softmax.unsqueeze(-1)
		loss_mse = ((sm2 - sm1) ** k_mom).sum(1)
		consistency = lbd/10.0 * loss_mse * 2.0
		consistency += lbd * loss_kl * 1.8
	else:
		raise NotImplementedError()
	return consistency.mean()


def kl_div(input, targets):
	return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def class_wise_entropy(logit, sfm_par=5):
	sfm = F.softmax(logit * sfm_par, dim=1)
	lgsfm = F.log_softmax(logit * sfm_par, dim=1)
	weight = 1 + sfm * lgsfm  # additive inverse entropy
	weight = torch.sum(weight, dim=1)
	# weight = F.softmax(weight, dim=1) / torch.mean(F.softmax(weight, dim=1))
	weight = weight / torch.mean(weight)
	return weight


def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_ssrl_loss(feature_list): # mod from reference
	index_list = [k for k in range(len(feature_list))]

	loss = 0.0
	for (i,j) in list(c(index_list,2)):
		feat_1 = feature_list[i]
		feat_2 = feature_list[j]
		corr = torch.mm(feat_1.T, feat_2)
		corr.div_(feat_1.shape[0])

		on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(corr).pow_(2).sum()
		loss += 0.5 * (on_diag + 3.9e-3 * off_diag)

	loss = loss / len(list(c(index_list, 2)))
	return loss



if __name__ == "__main__":

	import torch

	# Test Accuracy Metric
	predicts = [torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]]))]
	labels   = [torch.from_numpy(np.array([   0,            1,          1 ]))]
	losses   = [torch.from_numpy(np.array([   0.3,       0.4,       0.5   ]))]

	logging.getLogger().setLevel(logging.DEBUG)
	logging.debug("input pred:  {}".format(predicts))
	logging.debug("input label: {}".format(labels))
	logging.debug("input loss: {}".format(labels))

	acc = Accuracy()

	acc.update(preds=predicts, labels=labels, losses=losses)

	logging.info(acc.get())

	# Test MetricList
	metrics = MetricList(Loss(name="ce-loss"),
						 Accuracy(topk=1, name="acc-top1"), 
						 Accuracy(topk=2, name="acc-top2"), 
						 )
	metrics.update(preds=predicts, labels=labels, losses=losses)

	logging.info("------------")
	logging.info(metrics.get())
	acc.get_name_value()
