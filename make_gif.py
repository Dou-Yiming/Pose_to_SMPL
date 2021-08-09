import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./fit/output/HumanAct12/picture/fit/P01G01R01F0001T0064A0101') )
for filename in filenames:
    images.append(imageio.imread('./fit/output/HumanAct12/picture/fit/P01G01R01F0001T0064A0101/'+filename))
imageio.mimsave('./assets/fit.gif', images, duration=0.25)