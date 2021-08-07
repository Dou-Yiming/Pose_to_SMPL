import numpy as np

def mapping(Jtr,cfg):
    name=cfg.DATASET.NAME
    if not name=='HumanAct12':
        mapped_joint=cfg.DATASET.DATA_MAP.UTD_MHAD
        Jtr_mapped=np.zeros([Jtr.shape[0],len(mapped_joint),Jtr.shape[2]])
        for i in range(Jtr.shape[0]):
            for j in range(len(mapped_joint)):
                for k in range(Jtr.shape[2]):
                    Jtr_mapped[i][j][k]=Jtr[i][mapped_joint[j]][k]
        return Jtr_mapped
    return Jtr

