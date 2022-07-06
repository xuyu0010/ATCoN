import os
import numpy as np
import math
import logging

from video_iterator import Video
import video_sampler as sampler

def get_video_list(video_prefix, txt_list):
	# formate:
	# [vid, label, video_subpath]
	assert os.path.exists(video_prefix), "MeanStdExtractor:: failed to locate: `{}'".format(video_prefix)
	assert os.path.exists(txt_list), "MeanStdExtractor:: failed to locate: `{}'".format(txt_list)

	video_list = []
	new_video_info = {}
	logging_interval = 100
	with open(txt_list) as f:
		lines = f.read().splitlines()
		# logging.info("MeanStdExtractor:: found {} videos in `{}'".format(len(lines), txt_list))
		print("MeanStdExtractor:: found {} videos in `{}'".format(len(lines), txt_list))
		for i, line in enumerate(lines):
			v_id, label, video_subpath = line.split()
			video_path = os.path.join(video_prefix, video_subpath)
			if not os.path.exists(video_path):
				# logging.warning("MeanStdExtractor:: >> cannot locate `{}'".format(video_path))
				print("MeanStdExtractor:: >> cannot locate `{}'".format(video_path))
				continue
			info = [int(v_id), int(label), video_subpath]
			video_list.append(info)

	return video_list

if __name__ == "__main__":

	# dataset = 'HMDB51'
	# dataset = 'ARID'
	dataset = 'UCF101'
	dataset_root='../dataset'
	data_root = os.path.join(dataset_root, dataset)

	interval = 8
	if dataset == 'ARID': # Set 1 for only the original AID11
		interval = 4

	video_prefix = os.path.join(data_root, 'raw', 'data')
	# txt_list = os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_arid_train_da.txt')
	# txt_list = os.path.join(data_root, 'raw', 'list_cvt', 'arid_train_da.txt')
	txt_list = os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_train_da.txt')
	
	video_list = get_video_list(video_prefix=video_prefix, txt_list=txt_list)
	R_list = []
	G_list = []
	B_list = []
	pixels = 0
	count = 0

	# for index in range(0, len(video_list), 3):
	for index in range(0, len(video_list), 2):
	# for index in range(1): # For testing only

		v_id, label, vid_subpath = video_list[index]
		video_path = os.path.join(video_prefix, vid_subpath)
		# logging.info("MeanStdExtractor:: processing video {}/{} in `{}'".format(index, len(video_list), video_path))
		print("MeanStdExtractor:: processing video {}/{} in `{}'".format(index, len(video_list), video_path))
		
		video = Video(vid_path=video_path)
		total_frames = video.count_frames(check_validity=False)
		frameid_extractor = sampler.SequentialSampling(num=total_frames, interval=interval, fix_cursor=True, shuffle=False)
		sampled_idxs = frameid_extractor.sampling(range_max=total_frames, v_id=v_id)
		sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=True)

		if sampled_frames is None:
			count = count + 1
			continue
		
		for frame in range(len(sampled_frames)):
			
			R_val = sampled_frames[frame][:, :, 0]
			if not R_val.shape == (240, 320):
				count = count + 1
				break
			# R_val = R_val.tolist()
			G_val = sampled_frames[frame][:, :, 1]
			B_val = sampled_frames[frame][:, :, 2]
			R_list.append(R_val)
			G_list.append(G_val)
			B_list.append(B_val)
			pix_num = R_val.size
			# pix_num = len(R_val)

			pixels = pixels + pix_num

	R_sum = np.sum(R_list)
	G_sum = np.sum(G_list)
	B_sum = np.sum(B_list)
	R_mean = R_sum / pixels / 255
	G_mean = G_sum / pixels / 255
	B_mean = B_sum / pixels / 255
	R_std = np.std(R_list, dtype=np.float64) / 255
	G_std = np.std(G_list, dtype=np.float64) / 255
	B_std = np.std(B_list, dtype=np.float64) / 255

	# logging.info("Mean is [{}, {}, {}] and Std is [{}, {}, {}]".format(R_mean, G_mean, B_mean, R_std, G_std, B_std))
	print("Mean is [{}, {}, {}] and Std is [{}, {}, {}]".format(R_mean, G_mean, B_mean, R_std, G_std, B_std))
	print("Computation Done with {} videos ignored!".format(count))
