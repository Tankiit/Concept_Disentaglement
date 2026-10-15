import numpy as np
from oss_from_cache import group_loaders


gl = group_loaders(batch_size=512, shuffle=True, seed=0)   # kuc=0, uuc_val=2, uuc_test=5

for i,b in enumerate(gl['train']['kkc']):
    print(i,b['X'].shape,b['y'].shape,b['index'].shape,b['group'].shape)
    

