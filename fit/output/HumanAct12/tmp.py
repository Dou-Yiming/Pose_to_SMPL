import pickle
import numpy as np
a=pickle.load(open("./P01G01R01F0001T0064A0101_params.pkl",'rb'))
for key,value in a.items():
    if key =='label':
        print("{}: {}".format(key,value))
    else:
        print("{}: {}".format(key,np.array(value).shape))