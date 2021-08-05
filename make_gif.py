import matplotlib.pyplot as plt
import imageio, os
images = []
filenames = sorted(fn for fn in os.listdir('./fit/output/HumanAct12/picture/fit/P01G01R01F0449T0505A0201') )
for filename in filenames:
    images.append(imageio.imread('./fit/output/HumanAct12/picture/fit/P01G01R01F0449T0505A0201/'+filename))
imageio.mimsave('./fit.gif', images, duration=0.3)