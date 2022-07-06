import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for hmdb51
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_msda_train.txt'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_ucf101_train_da.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_msda_test.txt'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_ucf101_val_da.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )

	return (train, val)


def get_arid(data_root='./dataset/ARID',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for arid
	"""
	logging.debug("VideoIter:: clip_length = {}, segments = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, segments, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'arid_msda_train.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'arid_msda_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )

	return (train, val)


def get_mit(data_root='./dataset/MIT',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for moments-in-time
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'mit_msda_train.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'mit_msda_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )

	return (train, val)


def get_kinetics_daily(data_root='./dataset/Kinetics',
				 clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
				 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
				 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for kinetics-600 (daily)
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_daily_msda_train.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[256, 320]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_daily_msda_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
	return (train, val)


def get_ucf101(data_root='./dataset/UCF101',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for ucf-101
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_msda_train.txt'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_hmdb51_train_da.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_msda_test.txt'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_hmdb51_val_da.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )

	return (train, val)


def get_kinetics_sports(data_root='./dataset/Kinetics',
				 clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
				 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
				 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for kinetics-600 (sports)
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_sports_msda_train.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[256, 320]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_sports_msda_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
	return (train, val)


def get_sports1m(data_root='./dataset/Sports1M',
				 clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
				 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
				 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for kinetics-600 (sports)
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'sports1m_msda_train.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[256, 320]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  return_item_subpath=True,
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'sports1m_msda_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
	return (train, val)


def creat(name, batch_size, num_workers=4, **kwargs):

	if name.upper() == 'HMDB51':
		train, val = get_hmdb51(**kwargs)
	elif name.upper() == 'ARID':
		train, val = get_arid(**kwargs)
	elif name.upper() == 'MIT':
		train, val = get_mit(**kwargs)
	elif name.upper() == 'KINETICS-DAILY':
		train, val = get_kinetics_daily(**kwargs)
	elif name.upper() == 'UCF101':
		train, val = get_ucf101(**kwargs)
	elif name.upper() == 'KINETICS-SPORTS':
		train, val = get_kinetics_sports(**kwargs)
	elif name.upper() == 'SPORTS1M':
		train, val = get_sports1m(**kwargs)
	else:
		assert NotImplementedError("iter {} not found".format(name))


	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
	val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

	return (train_loader, val_loader)
