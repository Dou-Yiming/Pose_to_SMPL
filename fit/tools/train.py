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

def train(smpl_layer, target,
          logger, writer, device,
          args, cfg):
    res = []
    pose_params = torch.rand(target.shape[0], 72) * 0.0
    shape_params = torch.rand(target.shape[0], 10) * 0.03
    scale = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    pose_params = pose_params.to(device)
    shape_params = shape_params.to(device)
    target = target.to(device)
    scale = scale.to(device)

    pose_params.requires_grad = True
    shape_params.requires_grad = True
    scale.requires_grad = False
    smpl_layer.requires_grad = False

    optimizer = optim.Adam([pose_params, shape_params],
                           lr=cfg.TRAIN.LEARNING_RATE)
    
    min_loss = float('inf')
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss = F.smooth_l1_loss(Jtr * 100, target * 100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if float(loss) < min_loss:
            min_loss = float(loss)
            res = [pose_params, shape_params, verts, Jtr]
        if epoch % cfg.TRAIN.WRITE == 0:
            # logger.info("Epoch {}, lossPerBatch={:.9f}, scale={:.6f}".format(
            #         epoch, float(loss), float(scale)))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
    logger.info('Train ended, min_loss = {:.9f}'.format(float(min_loss)))
    return res
