import logging
import os
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.utils.weight_norm as weightNorm


def weight_from_entropy(logit, sfm_par=5):
	"""
	:param logit: tensor [b, classes]
	:return: tensor [b, 1]
	"""
	sfm = F.softmax(logit * sfm_par, dim=1)
	lgsfm = F.log_softmax(logit * sfm_par, dim=1)
	weight = 1 + sfm * lgsfm  # additive inverse entropy
	weight = torch.sum(weight, dim=1)
	# weight = (F.softmax(weight.unsqueeze(1), dim=1) / torch.mean(F.softmax(weight.unsqueeze(1), dim=1))).squeeze(1)
	weight = weight / torch.mean(weight)
	return weight



class BN_AC_CONV3D(nn.Module):

	def __init__(self, num_in, num_filter,
				 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
		super(BN_AC_CONV3D, self).__init__()
		self.bn = nn.BatchNorm3d(num_in)
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
							   stride=stride, groups=g, bias=bias)

	def forward(self, x):
		h = self.relu(self.bn(x))
		h = self.conv(h)
		return h


# Multi-fiber unit (for constructing MFNet)
class MF_UNIT(nn.Module):

	def __init__(self, num_in, num_mid, num_out, g=1, stride=(1,1,1), first_block=False, use_3d=True):
		super(MF_UNIT, self).__init__()
		num_ix = int(num_mid/4)
		kt,pt = (3,1) if use_3d else (1,0)
		# prepare input
		self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0))
		self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0))
		# main part
		self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride, g=g)
		if first_block:
			self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
		else:
			self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1), g=g)
		# adapter
		if first_block:
			self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)

	def forward(self, x):

		h = self.conv_i1(x)
		x_in = x + self.conv_i2(h)

		h = self.conv_m1(x_in)
		h = self.conv_m2(h)

		if hasattr(self, 'conv_w1'):
			x = self.conv_w1(x)

		return h + x


# Load pretrained into network
def load_state(network, state_dict):
	# customized partialy load function
	net_state_keys = list(network.state_dict().keys())
	net_state_keys_copy = net_state_keys.copy()
	sup_string = ""
	for key in state_dict.keys():
		if "backbone" in key:
			sup_string = "backbone."
		elif "module" in key:
			sup_string = "module."

	for i, _ in enumerate(net_state_keys_copy):
		name = net_state_keys_copy[i]
		if name.startswith('classifier') or name.startswith('fc') or name.startswith('cls'):
			continue

		if not sup_string:
			name_pretrained = name
		else:
			name_pretrained = sup_string + name

		if name_pretrained in state_dict.keys():
			dst_param_shape = network.state_dict()[name].shape
			if state_dict[name_pretrained].shape == dst_param_shape:
				network.state_dict()[name].copy_(state_dict[name_pretrained].view(dst_param_shape))
				net_state_keys.remove(name)

	# indicating missed keys
	if net_state_keys:
		num_batches_list = []
		for i in range(len(net_state_keys)):
			if 'num_batches_tracked' in net_state_keys[i]:
				num_batches_list.append(net_state_keys[i])
		pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]

		if pruned_additional_states:
			logging.info("There are layers in current network not initialized by pretrained")
			logging.warning(">> Failed to load: {}".format(pruned_additional_states))

		return False
	return True


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
	"3x3 convolution with padding"
	return nn.Conv2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		dilation=dilation,
		bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self,
				 inplanes,
				 planes,
				 stride=1,
				 dilation=1,
				 downsample=None,
				 style='pytorch',
				 with_cp=False):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride, dilation)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.dilation = dilation
		assert not with_cp

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self,
				 inplanes,
				 planes,
				 stride=1,
				 dilation=1,
				 downsample=None,
				 style='pytorch',
				 with_cp=False):
		"""Bottleneck block for ResNet.
		If style is "pytorch", the stride-two layer is the 3x3 conv layer,
		if it is "caffe", the stride-two layer is the first 1x1 conv layer.
		"""
		super(Bottleneck, self).__init__()
		assert style in ['pytorch', 'caffe']
		self.inplanes = inplanes
		self.planes = planes
		if style == 'pytorch':
			self.conv1_stride = 1
			self.conv2_stride = stride
		else:
			self.conv1_stride = stride
			self.conv2_stride = 1
		self.conv1 = nn.Conv2d(
			inplanes,
			planes,
			kernel_size=1,
			stride=self.conv1_stride,
			bias=False)
		self.conv2 = nn.Conv2d(
			planes,
			planes,
			kernel_size=3,
			stride=self.conv2_stride,
			padding=dilation,
			dilation=dilation,
			bias=False)

		self.bn1 = nn.BatchNorm2d(planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(
			planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.dilation = dilation
		self.with_cp = with_cp

	def forward(self, x):

		def _inner_forward(x):
			identity = x

			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bn2(out)
			out = self.relu(out)

			out = self.conv3(out)
			out = self.bn3(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out += identity

			return out

		if self.with_cp and x.requires_grad:
			out = cp.checkpoint(_inner_forward, x)
		else:
			out = _inner_forward(x)

		out = self.relu(out)

		return out


def make_res_layer(block,
				   inplanes,
				   planes,
				   blocks,
				   stride=1,
				   dilation=1,
				   style='pytorch',
				   with_cp=False):
	downsample = None
	if stride != 1 or inplanes != planes * block.expansion:
		downsample = nn.Sequential(
			nn.Conv2d(
				inplanes,
				planes * block.expansion,
				kernel_size=1,
				stride=stride,
				bias=False),
			nn.BatchNorm2d(planes * block.expansion),
		)

	layers = []
	layers.append(
		block(
			inplanes,
			planes,
			stride,
			dilation,
			downsample,
			style=style,
			with_cp=with_cp))
	inplanes = planes * block.expansion
	for i in range(1, blocks):
		layers.append(
			block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp))

	return nn.Sequential(*layers)


class SimpleSpatialTemporalModule(nn.Module):
	def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
		super(SimpleSpatialTemporalModule, self).__init__()

		assert spatial_type in ['avg']
		self.spatial_type = spatial_type

		self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
		self.temporal_size = temporal_size
		self.pool_size = (self.temporal_size, ) + self.spatial_size

		if self.spatial_type == 'avg':
			self.op = nn.AvgPool3d(self.pool_size, stride=1, padding=0)

	def init_weights(self):
		pass

	def forward(self, input):
		return self.op(input)


class ClsHead(nn.Module):
	"""Simplest classification head"""

	def __init__(self,
				 with_avg_pool=True,
				 temp_feat_size=1,
				 sp_feat_size=7,
				 dp_ratio=0.8,
				 in_channels=2048,
				 num_classes=101,
				 classifier_type='ori',
				 init_std=0.01):

		super(ClsHead, self).__init__()

		self.with_avg_pool = with_avg_pool
		self.dp_ratio = dp_ratio
		self.in_channels = in_channels
		self.dp_ratio = dp_ratio
		self.temp_feat_size = temp_feat_size
		self.sp_feat_size = sp_feat_size
		self.init_std = init_std
		self.classifier_type = classifier_type

		if self.dp_ratio != 0:
			self.dropout = nn.Dropout(p=self.dp_ratio)
		else:
			self.dropout = None
		if self.with_avg_pool:
			self.avg_pool = nn.AvgPool3d((temp_feat_size, sp_feat_size, sp_feat_size))
		if self.classifier_type == 'wn':
			self.fc_cls = weightNorm(nn.Linear(in_channels, num_classes))
		else:
			self.fc_cls = nn.Linear(in_channels, num_classes)

	def init_weights(self):
		nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
		nn.init.constant_(self.fc_cls.bias, 0)

	def forward(self, x):
		if x.ndimension() == 4:
			x = x.unsqueeze(2)
		assert x.shape[1] == self.in_channels
		assert x.shape[2] == self.temp_feat_size
		assert x.shape[3] == self.sp_feat_size
		assert x.shape[4] == self.sp_feat_size
		if self.with_avg_pool:
			x = self.avg_pool(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = x.view(x.size(0), -1)

		cls_score = self.fc_cls(x)
		return cls_score

	def loss(self,
			 cls_score,
			 labels):
		losses = dict()
		losses['loss_cls'] = F.cross_entropy(cls_score, labels)

		return losses


class _SimpleConsensus(torch.autograd.Function):
	"""Simplest segmental consensus module"""

	'''def __init__(self,
				 consensus_type='avg',
				 dim=1):
		super(_SimpleConsensus, self).__init__()
		assert consensus_type in ['avg']
		self.consensus_type = consensus_type
		self.dim = dim
		self.shape = None'''

	@staticmethod
	def forward(ctx, x, consensus_type='avg', dim=1):
		ctx.shape = x.size()
		ctx.consensus_type = consensus_type
		ctx.dim = dim
		if ctx.consensus_type == 'avg':
			output = x.mean(dim=ctx.dim, keepdim=True)
		else:
			output = None
		return output

	@staticmethod
	def backward(ctx, grad_output):
		if ctx.consensus_type == 'avg':
			grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
		else:
			grad_in = None
		return grad_in, None, None


class SimpleConsensus(nn.Module):
	def __init__(self, consensus_type, dim=1):
		super(SimpleConsensus, self).__init__()

		assert consensus_type in ['avg']
		self.consensus_type = consensus_type
		self.dim = dim

	def init_weights(self):
		pass

	def forward(self, input):
		return _SimpleConsensus.apply(input, self.consensus_type, self.dim)


class SimpleSpatialModule(nn.Module):
	def __init__(self, spatial_type='avg', spatial_size=7):
		super(SimpleSpatialModule, self).__init__()

		assert spatial_type in ['avg']
		self.spatial_type = spatial_type

		self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)

		if self.spatial_type == 'avg':
			self.op = nn.AvgPool2d(self.spatial_size, stride=1, padding=0)


	def init_weights(self):
		pass

	def forward(self, input):
		return self.op(input)



if __name__ == "__main__":

	net = AdversarialNetwork(2048)
	input_data = torch.randn(5,2048)
	if torch.cuda.is_available():
		net = net.cuda()
		input_data = input_data.cuda()
	output_data = net(input_data)
	print (output_data.shape)
