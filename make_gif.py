import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./output/') )
for filename in filenames:
    images.append(imageio.imread('./output/'+filename))
imageio.mimsave('./output/gif.gif', images, duration=0.5)