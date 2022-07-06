import os
import json
import socket
import logging
import argparse
from datetime import date

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_target_model import train_target_model
from network.target_symbol_builder import get_target_symbol

torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="Video PDA Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True, help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='UH', choices=['Daily', 'Sports', 'UH'])
parser.add_argument('--tgt-dataset', default='HMDB51')
parser.add_argument('--clip-length', default=16, help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=2,help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",help="set logging file.")
parser.add_argument('--log-file', type=str, default="",help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0",help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='ATCoN',help="choose the base network")
parser.add_argument('--pretrained_2d', type=bool, default=True,help="load default 2D pretrained model.")
parser.add_argument('--resume-epoch', type=int, default=-1, help="resume train")
parser.add_argument('--segments', type=int, default=5, help="resume train")
parser.add_argument('--frame_per_seg', type=int, default=1, help="frames sampled per segment")
parser.add_argument('--consensus_type', type=str, default='trn-m', help="tsn consensus type, choose from avg, trn and trn-m")
parser.add_argument('--fcbn_type', type=str, default='bn', help="tsn fcbn type, choose from ori and bn")
parser.add_argument('--classifier_type', type=str, default='wn', help="tsn classifier type, choose from ori and wn")
# optimization
parser.add_argument('--fine-tune', type=bool, default=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr-base', type=float, default=0.005)
parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [1.2,1.8]], help="# of samples before changing lr")
parser.add_argument('--lr-factor', type=float, default=0.1, help="reduce the lr with factor")
parser.add_argument('--save-freq', type=float, default=5)
parser.add_argument('--end-epoch', type=int, default=30)
parser.add_argument('--random-seed', type=int, default=1, help='random seed (default: 1)')
# adaptation method
parser.add_argument('--method', type=str, default='atcon', help="type of adaptation method")

def autofill(args):
	# customized
	today = date.today()
	today = today.strftime("%m%d")
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	if not args.log_file:
		if os.path.exists("./exps/logs"):
			args.log_file = "./exps/logs/{}-{}_at-{}-target-1.log".format(args.task_name, today, socket.gethostname())
		else:
			args.log_file = ".{}-{}_at-{}-target-1.log".format(args.task_name, today, socket.gethostname())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)

	gpu_num = len(args.gpus.split(','))
	assert gpu_num == 1, "Currently, we do not support multiple gpus."
	args.data_parallel = False
	if gpu_num == 1:
		args.data_parallel = False
	else:
		args.data_parallel = True
	print("Main::>>>>>>>Data Parallel status is {}".format(args.data_parallel))

	args.use_im = False
	args.use_cluster = False
	args.use_consistency = False
	args.use_contrastive = False

	if args.method.upper() == 'ATCON':
		assert args.network.upper() == 'ATCON', "ATCoN method only supports ATCoN network."
		args.use_im = True
		args.use_cluster = True
		args.use_consistency = True
		args.use_contrastive = True
	elif args.method.upper() == 'SHOT':
		assert args.network.upper() == 'TRN', "SHOT method only supports TRN."
		args.use_im = True
		args.use_cluster = True		

	return args

def set_logger(log_file='', debug_mode=True):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		while os.path.exists(log_file):
			log_id = int(log_file.split('-')[-1].split('.')[0])
			new_log_id = log_id + 1
			k = log_file.rfind(str(log_id))
			log_file = log_file[:k] + str(new_log_id) + log_file[k+1:]
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	""" add '%(filename)s:%(lineno)d %(levelname)s:' to format show target file """
	logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers = handlers)

if __name__ == "__main__":

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=True)
	logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))

	logging.info("Start training with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))

	# set device states
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	assert torch.cuda.is_available(), "CUDA is not available"
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)

	# load dataset related configuration
	dataset_cfg = dataset.get_config(name=args.dataset)
	net_full, input_conf = get_target_symbol(name=args.network, pretrained=args.pretrained_2d if args.resume_epoch<0 else None, segments=args.segments, 
								consensus_type=args.consensus_type, fcbn_type=args.fcbn_type, classifier_type=args.classifier_type,
								batch_size=args.batch_size, **dataset_cfg)

	base_num = 840.0 # HMDB for U-H
	train_num_dict = {'HMDB51':840.0,'UCF101':1438.0,\
					'ARID':2776.0,'MIT':4000.0,'KINETICS-DAILY':8959.0,\
					'KINETICS-SPORTS':19104.0,'SPORTS-1M':14754.0} 
	# HMDB:840 for UH, UCF:1438 for UH, HMDB:560 for Daily, UCF:2145 for Sports
	lr_expansion = round(float(train_num_dict[args.tgt_dataset.upper()] / base_num), 2)
	args.lr_steps = [int(lr_expansion*step) for step in args.lr_steps]

	# training
	kwargs = {}
	kwargs.update(dataset_cfg)
	kwargs.update({'input_conf': input_conf})
	kwargs.update(vars(args))
	train_target_model(net_full=net_full, **kwargs)
