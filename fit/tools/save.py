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


def save_pic(res, smpl_layer, file, logger, dataset_name):
    _, _, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/picture/fit/{}".format(dataset_name,file_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    for i in tqdm(range(Jtr.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=False)
    logger.info('Pictures saved')


def save_params(res, file, logger, dataset_name):
    pose_params, shape_params, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/params/".format(dataset_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving params at {}'.format(fit_path))
    pose_params = (pose_params.cpu().detach()).numpy().tolist()
    shape_params = (shape_params.cpu().detach()).numpy().tolist()
    Jtr = (Jtr.cpu().detach()).numpy().tolist()
    verts = (verts.cpu().detach()).numpy().tolist()
    params = {}
    params["pose_params"] = pose_params
    params["shape_params"] = shape_params
    params["Jtr"] = Jtr
    params["mesh"] = verts
    f = open(os.path.join((fit_path),
                          "{}_params.json".format(file_name)), 'w')
    json.dump(params, f)
