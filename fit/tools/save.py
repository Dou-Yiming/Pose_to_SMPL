import sys
import os
import re
from tqdm import tqdm
import numpy as np
import json


sys.path.append(os.getcwd())
from display_utils import display_model


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_pic(target, res, smpl_layer, file, logger):
    pose_params, shape_params, verts, Jtr = res
    name = re.split('[/.]', file)[-2]
    gt_path = "fit/output/HumanAct12/picture/gt/{}".format(name)
    fit_path = "fit/output/HumanAct12/picture/fit/{}".format(name)
    create_dir_not_exist(gt_path)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {} and {}'.format(gt_path, fit_path))
    for i in tqdm(range(target.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': target.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(gt_path+"/frame_{}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=True)
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{}".format(i)),
            batch_idx=i,
            show=False)
    logger.info('Pictures saved')


def save_params(res, file, logger):
    pose_params, shape_params, verts, Jtr = res
    name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/HumanAct12/params/"
    create_dir_not_exist(fit_path)
    logger.info('Saving params at {}'.format(fit_path))
    pose_params = pose_params.cpu().detach()
    pose_params = pose_params.numpy().tolist()
    shape_params = shape_params.cpu().detach()
    shape_params = shape_params.numpy().tolist()
    Jtr = Jtr.cpu().detach()
    Jtr = Jtr.numpy().tolist()
    params = {}
    params["pose_params"] = pose_params
    params["shape_params"] = shape_params
    params["Jtr"] = Jtr
    f = open(os.path.join((fit_path),
                          "{}_params.json".format(name)), 'w')
    json.dump(params, f)
    logger.info('Params saved')
