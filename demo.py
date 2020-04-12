#%%
%load_ext autoreload
%autoreload 2

import numpy as np
from src.viGMM_full import VariationalGaussianMixture

#%%
X = np.loadtxt('data/faithful.txt')
X = (X - X.mean(axis=0)) / X.std(axis=0)
#%%
model = VariationalGaussianMixture(10)
model.fit(X)

# %%
