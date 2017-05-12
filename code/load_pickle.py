import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


var_dict= dict()
for root, dirs, files in os.walk("../data/pickled_vars"):
        path = root.split(os.sep)
        for file in files:
            if '.p' in str(file).lower():
                var_dict[os.path.splitext(os.path.basename(file))[0]] = pickle.load(open(os.path.join(root,file),'rb'))

print(var_dict.keys())
