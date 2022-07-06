import logging
import numpy as np
from math import ceil

import torch
import torch.nn as nn

class RelationModule(torch.nn.Module):
	# this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
	def __init__(self, img_feature_dim, num_bottleneck, num_frames):
		super(RelationModule, self).__init__()
		self.num_frames = num_frames
		self.img_feature_dim = img_feature_dim
		self.num_bottleneck = num_bottleneck
		self.classifier = self.fc_fusion()
	def fc_fusion(self):
		# naive concatenate
		classifier = nn.Sequential(
				nn.ReLU(),
				nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck),
				nn.ReLU(),
				)
		return classifier
	def forward(self, input):
		input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
		input = self.classifier(input)
		input = input.unsqueeze(1)
		return input

class RelationModuleMultiScale(torch.nn.Module):
	# Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

	def __init__(self, img_feature_dim, num_bottleneck, num_frames, rand_relation_sample=False):
		super(RelationModuleMultiScale, self).__init__()
		self.subsample_num = 9 # how many relations selected to sum up
		self.img_feature_dim = img_feature_dim
		self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

		self.relations_scales = []
		self.subsample_scales = []
		for scale in self.scales:
			relations_scale = self.return_relationset(num_frames, scale)
			self.relations_scales.append(relations_scale)
			self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

		# self.num_class = num_class
		self.num_frames = num_frames
		self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
		for i in range(len(self.scales)):
			scale = self.scales[i]
			fc_fusion = nn.Sequential(
						nn.ReLU(),
						nn.Linear(scale * self.img_feature_dim, num_bottleneck),
						nn.ReLU(),
						)

			self.fc_fusion_scales += [fc_fusion]

		print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])
		self.rand_relation_sample = rand_relation_sample

	def forward(self, input):
		# the first one is the largest scale
		act_scale_1 = input[:, self.relations_scales[0][0] , :]
		act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
		act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
		act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
		act_all = act_scale_1.clone()
		act_all_list = [act_all.squeeze(1)]

		for scaleID in range(1, len(self.scales)):
			act_relation_all = torch.zeros_like(act_scale_1)
			# iterate over the scales
			num_total_relations = len(self.relations_scales[scaleID])
			num_select_relations = self.subsample_scales[scaleID]
			idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]
			idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False).tolist()

			if self.rand_relation_sample:
				idx_relations = idx_relations_randomsample
			else:
				idx_relations = idx_relations_evensample

			for idx in idx_relations:
				act_relation = input[:, self.relations_scales[scaleID][idx], :]
				act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
				act_relation = self.fc_fusion_scales[scaleID](act_relation)
				act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
				act_relation_all += act_relation

			act_all = torch.cat((act_all, act_relation_all), 1)
			act_all_list.append(act_relation_all.squeeze(1))
		return act_all, act_all_list

	def return_relationset(self, num_frames, num_frames_relation):
		import itertools
		return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

if __name__ == "__main__":

	batch_size = 3
	num_frames = 5
	num_bottleneck = 24
	img_feature_dim = 12
	input_var = torch.randn(batch_size, num_frames, img_feature_dim, 1, 1)
	print(input_var.shape)
	model1 = RelationModuleMultiScale(img_feature_dim, num_bottleneck, num_frames, rand_relation_sample=True)
	output1, output1_list = model1(input_var)
	print(output1.shape)
	print(output1_list[0].shape)
