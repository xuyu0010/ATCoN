import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.init import xavier_uniform_, normal, constant_

try:
	from . import TRNmodule
	from .util import load_state
	from .util import conv3x3, BasicBlock, Bottleneck, make_res_layer
	from .util import SimpleSpatialModule, SimpleConsensus, ClsHead
except:
	import TRNmodule
	from util import load_state
	from util import conv3x3, BasicBlock, Bottleneck, make_res_layer
	from util import SimpleSpatialModule, SimpleConsensus, ClsHead


class TRN_base(nn.Module):
	"""ResNet backbone for spatial stream in TRN.

	Args:
		depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
		num_stages (int): Resnet stages, normally 4.
		strides (Sequence[int]): Strides of the first block of each stage.
		dilations (Sequence[int]): Dilation of each stage.
		out_indices (Sequence[int]): Output from which stages.
		style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
			layer is the 3x3 conv layer, otherwise the stride-two layer is
			the first 1x1 conv layer.
		frozen_stages (int): Stages to be frozen (all param fixed). -1 means
			not freezing any parameters.
		bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
			running stats (mean and var).
		bn_frozen (bool): Whether to freeze weight and bias of BN layers.
		partial_bn (bool): Whether to freeze weight and bias of **all but the first** BN layers.
		with_cp (bool): Use checkpoint or not. Using checkpoint will save some
			memory while slowing down the training speed.
	"""

	arch_settings = {
		18: (BasicBlock, (2, 2, 2, 2)),
		34: (BasicBlock, (3, 4, 6, 3)),
		50: (Bottleneck, (3, 4, 6, 3)),
		101: (Bottleneck, (3, 4, 23, 3)),
		152: (Bottleneck, (3, 8, 36, 3))
	}

	def __init__(self,
				 depth=50,
				 pretrained=None,
				 num_stages=4,
				 strides=(1, 2, 2, 2),
				 dilations=(1, 1, 1, 1),
				 out_indices=(0, 1, 2, 3),
				 style='pytorch',
				 frozen_stages=-1,
				 bn_eval=True,
				 bn_frozen=False,
				 partial_bn=False,
				 with_cp=False,
				 segments=3,
				 consensus_type='avg',
				 fcbn_type='ori',
				 classifier_type='ori',
				 dynamic_reverse=False,
				 num_classes=400):
		super(TRN_base, self).__init__()
		if depth not in self.arch_settings:
			raise KeyError('invalid depth {} for resnet'.format(depth))
		self.depth = depth
		self.pretrained = pretrained
		self.num_stages = num_stages
		assert num_stages >= 1 and num_stages <= 4
		self.strides = strides
		self.dilations = dilations
		assert len(strides) == len(dilations) == num_stages
		self.out_indices = out_indices
		assert max(out_indices) < num_stages
		self.style = style
		self.frozen_stages = frozen_stages
		self.bn_eval = bn_eval
		self.bn_frozen = bn_frozen
		self.partial_bn = partial_bn
		self.with_cp = with_cp
		self.segments = segments
		self.consensus_type = consensus_type
		self.dynamic_reverse = dynamic_reverse

		self.block, stage_blocks = self.arch_settings[depth]
		self.stage_blocks = stage_blocks[:num_stages]
		self.inplanes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.res_layers = []
		for i, num_blocks in enumerate(self.stage_blocks):
			stride = strides[i]
			dilation = dilations[i]
			planes = 64 * 2**i
			res_layer = make_res_layer(
				self.block,
				self.inplanes,
				planes,
				num_blocks,
				stride=stride,
				dilation=dilation,
				style=self.style,
				with_cp=with_cp)
			self.inplanes = planes * self.block.expansion
			layer_name = 'layer{}'.format(i + 1)
			self.add_module(layer_name, res_layer)
			self.res_layers.append(layer_name)

		self.feat_dim = self.block.expansion * 64 * 2**(len(self.stage_blocks) - 1)
		self.avgpool = SimpleSpatialModule(spatial_type='avg', spatial_size=7)
		if self.consensus_type == 'avg':
			self.num_bottleneck = 2048
			self.img_feature_dim = 2048
			self.consensus = SimpleConsensus(self.consensus_type)
		elif self.consensus_type == 'trn':
			logging.info('TRNNetwork:: Utilizing consensus type {}'.format(self.consensus_type))
			self.num_bottleneck = 512
			self.img_feature_dim = 1024
			self.new_fc = nn.Conv2d(2048, self.img_feature_dim, 1)
			self.consensus = TRNmodule.RelationModule(self.img_feature_dim, self.num_bottleneck, self.segments)
			xavier_uniform_(self.new_fc.weight)
			constant_(self.new_fc.bias, 0)
		elif self.consensus_type == 'trn-m':
			logging.info('TRNNetwork:: Utilizing consensus type {}'.format(self.consensus_type))
			self.num_bottleneck = 512
			self.img_feature_dim = 1024
			self.new_fc = nn.Conv2d(2048, self.img_feature_dim, 1)
			self.consensus = TRNmodule.RelationModuleMultiScale(self.img_feature_dim, self.num_bottleneck, self.segments, rand_relation_sample=False)
			xavier_uniform_(self.new_fc.weight)
			constant_(self.new_fc.bias, 0)

		#############
		# Initialization
		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/tsn2d_rgb_r50.pth')
			logging.info("TRNNetwork:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self, pretrained['state_dict'])
		else:
			logging.info("TRNNetwork:: graph initialized, use random inilization!")

	def forward(self, x):
		outs = []
		assert x.shape[2] == self.segments, ValueError("input shape {} not match segments {}".format(x.shape[2], self.segments))
		for i in range(x.shape[2]):
			out = x[:, :, 0, :, :]
			out = self.conv1(out)
			out = self.bn1(out)
			out = self.relu(out)
			out = self.maxpool(out)

			for i, layer_name in enumerate(self.res_layers):
				res_layer = getattr(self, layer_name)
				out = res_layer(out)
			out = self.avgpool(out)
			if 'trn' in self.consensus_type:
				out = self.new_fc(out)
			outs.append(out)

		x = torch.stack(outs, dim=2)
		x = x.permute(0, 2, 1, 3, 4)
		if self.consensus_type == 'avg':
			x = self.consensus(x).squeeze(1)
			x_list = []
		elif self.consensus_type == 'trn':
			x = self.consensus(x.contiguous())
			x_list = []
		elif self.consensus_type == 'trn-m':
			x, x_list = self.consensus(x.contiguous())
			x = torch.sum(x, 1).unsqueeze(-1).unsqueeze(-1)

		return x, x_list


class TRN_fcbn(nn.Module):

	def __init__(self, consensus_type='avg', fcbn_type='ori', num_classes=400, pretrained=None):
		super(TRN_fcbn, self).__init__()
		self.consensus_type = consensus_type
		self.fcbn_type = fcbn_type
		logging.info('TRNNetwork:: fcbn_type: {}'.format(fcbn_type))
		if self.consensus_type == 'avg':
			self.num_bottleneck = 2048
		elif self.consensus_type == 'trn' or self.consensus_type == 'trn-m':
			self.num_bottleneck = 512
		self.fc_full = nn.Conv2d(self.num_bottleneck, self.num_bottleneck, 1)
		self.bn_full = nn.BatchNorm2d(self.num_bottleneck, affine=True)

	def forward(self, x, x_list):
		x = self.fc_full(x)
		if self.fcbn_type == 'bn':
			x = self.bn_full(x)
		for i in range(len(x_list)):
			x_list[i] = self.fc_full(x_list[i].unsqueeze(-1).unsqueeze(-1))
			if self.fcbn_type == 'bn':
				x_list[i] = self.bn_full(x_list[i])

		return x, x_list


class TRN_cls(nn.Module):

	def __init__(self, consensus_type='avg', classifier_type='ori', num_classes=400, pretrained=None):
		super(TRN_cls, self).__init__()
		self.consensus_type = consensus_type
		self.classifier_type = classifier_type
		logging.info('TRNNetwork:: classifier_type: {}'.format(classifier_type))
		logging.info('TRNNetwork:: number of classes: {}'.format(num_classes))
		if self.consensus_type == 'avg':
			self.num_bottleneck = 2048
		elif self.consensus_type == 'trn' or self.consensus_type == 'trn-m':
			self.num_bottleneck = 512
		self.cls_head_full = ClsHead(with_avg_pool=False, temp_feat_size=1, sp_feat_size=1, dp_ratio=0.5, 
								in_channels=self.num_bottleneck, num_classes=num_classes, classifier_type=self.classifier_type)

	def forward(self, x, x_list):
		pred_full = self.cls_head_full(x)
		x = x.squeeze(-1).squeeze(-1)
		pred_list = []
		for i in range(len(x_list)):
			pred_scale = self.cls_head_full(x_list[i])
			x_list[i] = x_list[i].squeeze(-1).squeeze(-1)
			pred_list.append(pred_scale.unsqueeze(-1))
		pred = torch.cat(pred_list, dim=-1)
		pred = torch.mean(pred, dim=-1)
		pred = pred + pred_full

		return x, x_list, pred


if __name__ == "__main__":

	net_feat = TRN_base(pretrained=False, segments=4, consensus_type='trn-m')
	net_fc = TRN_fcbn(consensus_type='trn-m', fcbn_type='bn')
	net_cls = TRN_cls(consensus_type='trn-m', classifier_type='wn', num_classes=23)
	data = torch.randn(4,3,4,224,224) # [bs,c,t,h,w]
	if torch.cuda.is_available():
		net_feat = net_feat.cuda()
		net_fc = net_fc.cuda()
		net_cls = net_cls.cuda()
		data = data.cuda()
	video_feat, scale_feats = net_feat(data)
	video_feat, scale_feats = net_fc(video_feat, scale_feats)
	video_feat, scale_feats, pred = net_cls(video_feat, scale_feats)
	print(video_feat.shape)
	print(pred.shape)