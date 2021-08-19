import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./fit/output/Human3.6M/picture/fit/s_01_act_09_subact_02_ca_02') )
for filename in filenames:
    images.append(imageio.imread('./fit/output/Human3.6M/picture/fit/s_01_act_09_subact_02_ca_02/'+filename))
imageio.mimsave('fit_mesh.gif', images, duration=0.2)