from display_utils import display_model
from label import get_label
import sys
import os
import re
from tqdm import tqdm
import numpy as np
import pickle

sys.path.append(os.getcwd())


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_pic(res, smpl_layer, file, logger, dataset_name, target):
    _, _, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/picture/{}".format(dataset_name, file_name)
    os.makedirs(fit_path,exist_ok=True)
    logger.info('Saving pictures at {}'.format(fit_path))
    for i in tqdm(range(Jtr.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{:0>4d}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=True)
    logger.info('Pictures saved')


def save_params(res, file, logger, dataset_name):
    pose_params, shape_params, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/".format(dataset_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving params at {}'.format(fit_path))
    label = get_label(file_name, dataset_name)
    pose_params = (pose_params.cpu().detach()).numpy().tolist()
    shape_params = (shape_params.cpu().detach()).numpy().tolist()
    Jtr = (Jtr.cpu().detach()).numpy().tolist()
    verts = (verts.cpu().detach()).numpy().tolist()
    params = {}
    params["label"] = label
    params["pose_params"] = pose_params
    params["shape_params"] = shape_params
    params["Jtr"] = Jtr
    print("label:{}".format(label))
    with open(os.path.join((fit_path),
                           "{}_params.pkl".format(file_name)), 'wb') as f:
        pickle.dump(params, f)


def save_single_pic(res, smpl_layer, epoch, logger, dataset_name, target):
    _, _, verts, Jtr = res
    fit_path = "fit/output/{}/picture".format(dataset_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    display_model(
        {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath=fit_path+"/epoch_{:0>4d}".format(epoch),
        batch_idx=60,
        show=False,
        only_joint=False)
    logger.info('Picture saved')