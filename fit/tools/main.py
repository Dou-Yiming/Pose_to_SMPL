import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import time
import inspect
import sys
import os
import logging

import argparse
import json
from tqdm import tqdm
sys.path.append(os.getcwd())
from display_utils import display_model
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_pic,save_params
torch.backends.cudnn.benchmark=True

def parse_args():
    parser = argparse.ArgumentParser(description='Fit SMPL')
    parser.add_argument('--exp', dest='exp',
                        help='Define exp name',
                        default=time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), type=str)
    parser.add_argument('--config_path', dest='config_path',
                        help='Select configuration file',
                        default='fit/configs/config.json', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='select dataset',
                        default='', type=str)
    args = parser.parse_args()
    return args

def get_config(args):
    with open(args.config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    return cfg

def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(os.path.join(cur_path, 'tb'))

    return logger, writer

if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
    assert not os.path.exists(cur_path), 'Duplicate exp name'
    os.mkdir(cur_path)

    cfg = get_config(args)
    json.dump(dict(cfg), open(os.path.join(cur_path, 'config.json'), 'w'))

    logger, writer = get_logger(cur_path)
    logger.info("Start print log")

    device = set_device(USE_GPU=cfg.USE_GPU)
    logger.info('using device: {}'.format(device))
    
    smpl_layer = SMPL_Layer(
        center_idx = 0,
        gender='neutral',
        model_root='smplpytorch/native/models')
    
    for root,dirs,files in os.walk(cfg.DATASET_PATH):
        for file in files:
            logger.info('Processing file: {}'.format(file))
            target_path=os.path.join(root,file)
    
            target = np.array(transform(np.load(target_path)))
            logger.info('File shape: {}'.format(target.shape))
            target = torch.from_numpy(target).float()
            
            res = train(smpl_layer,target,
                logger,writer,device,
                args,cfg)
            
            # save_pic(target,res,smpl_layer,file,logger)
            save_params(res,file,logger)
        