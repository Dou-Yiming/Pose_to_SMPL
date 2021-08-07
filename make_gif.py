import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./fit/output/UTD_MHAD/picture/') )
for filename in filenames:
    images.append(imageio.imread('./fit/output/UTD_MHAD/picture/fit/a10_s1_t1_skeleton/'+filename))
imageio.mimsave('./fit.gif', images, duration=0.3)