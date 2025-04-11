import pickle
from visualiser import visualise_dfs_time

import numpy as np

with open('dfs.pickle', 'rb') as f:
    dfs = pickle.load(f)
    print(len(dfs.F_time == np.inf))
    visualise_dfs_time(dfs, 1600)            
