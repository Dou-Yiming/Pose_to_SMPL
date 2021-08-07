from fit.tools.save import save_pic
import matplotlib as plt
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import scipy.io
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
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
from map import mapping

   
class Early_Stop:
    def __init__(self, eps = -1e-3, stop_threshold = 10) -> None:
        self.min_loss=float('inf')
        self.eps=eps
        self.stop_threshold=stop_threshold
        self.satis_num=0
        
    def update(self, loss):
        delta = (loss - self.min_loss) / self.min_loss
        if float(loss) < self.min_loss:
            self.min_loss = float(loss)
            update_res=True
        else:
            update_res=False
        if delta >= self.eps:
            self.satis_num += 1
        else:
            self.satis_num = max(0,self.satis_num-1)
        return update_res, self.satis_num >= self.stop_threshold


def init(smpl_layer, target, device, cfg):
    params={}
    params["pose_params"] = torch.rand(target.shape[0], 72) * 0.0
    params["shape_params"] = torch.rand(target.shape[0], 10) * 0.03
    params["scale"] = torch.ones([1])
    
    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)
    
    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = True
    params["scale"].requires_grad = False
    
    optimizer = optim.Adam([params["pose_params"], params["shape_params"]],
                           lr=cfg.TRAIN.LEARNING_RATE)
    
    index={}
    smpl_index=[]
    dataset_index=[]
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])
        
    index["smpl_index"]=torch.tensor(smpl_index).to(device)
    index["dataset_index"]=torch.tensor(dataset_index).to(device)
    
    return smpl_layer, params,target, optimizer, index


def train(smpl_layer, target,
          logger, writer, device,
          args, cfg):
    res = []
    smpl_layer, params,target, optimizer, index = \
        init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]
    
    early_stop = Early_Stop()
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
    # for epoch in range(cfg.TRAIN.MAX_EPOCH):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss = F.smooth_l1_loss(Jtr.index_select(1, index["smpl_index"]) * 100,
                                target.index_select(1, index["dataset_index"]) * 100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        update_res, stop = early_stop.update(float(loss))
        if update_res:
            res = [pose_params, shape_params, verts, Jtr]
        if stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break
        
        if epoch % cfg.TRAIN.WRITE == 0:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, EarlyStopSatis: {}".format(
            #         epoch, float(loss), early_stop.satis_num))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
    
    
    logger.info('Train ended, min_loss = {:.9f}'.format(float(early_stop.min_loss)))
    return res
