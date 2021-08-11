import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./fit/output/CMU_Mocap/picture/fit/01_01') )
for filename in filenames:
    images.append(imageio.imread('./fit/output/CMU_Mocap/picture/fit/01_01/'+filename))
imageio.mimsave('fit.gif', images, duration=0.2)