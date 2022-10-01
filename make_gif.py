import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('D:/OneDrive - sjtu.edu.cn/MVIG/Action-Dataset/Pose_to_SMPL/fit/output/NTU/picture') )
for filename in filenames:
    images.append(imageio.imread('D:/OneDrive - sjtu.edu.cn/MVIG/Action-Dataset/Pose_to_SMPL/fit/output/NTU/picture/'+filename))
imageio.mimsave('clapping_example.gif', images, duration=0.2)