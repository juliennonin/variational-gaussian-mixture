#%%
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.viGMM_full import VariationalGaussianMixture
from src.viGMM_CB import VariationalGaussianMixtureCB
from scipy.stats import multivariate_normal
from utils import plot_confidence_ellipse



#%% Synthetic 1
m_true = np.array([[0, 0], [3, -3], [3, 3], [-3, 3], [-3, -3]])
covs_true = np.array([[[1, 0], [0, 1]], [[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]])
X = np.concatenate([np.random.multivariate_normal(m_true[k], covs_true[k], 100) for k in range(len(m_true))])

#%% Synthetic 2
m_true = np.array([[0, -2],[0, 0], [0, 2]])
covs_true = np.array([[[2, 0], [0, 0.2]], [[2, 0], [0, 0.2]], [[2, 0], [0, 0.2]]])
X = np.concatenate([np.random.multivariate_normal(m_true[k], covs_true[k], 100) for k in range(len(m_true))])

#%% Synthetic 3
m_true = np.array([[0, 0], [0, 0], [0, 0]])
covs_true = np.array([[[1, 0], [0, 0.2]], [[0.02, -0.08], [-0.08, 1.5]], [[0.5, 0.4], [0.4, 0.5]]])
X = np.concatenate([np.random.multivariate_normal(m_true[k], covs_true[k], 100) for k in range(len(m_true))])

#%% Olf Faithful data set
X = np.loadtxt('data/faithful.txt')


#%%
# Standardize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# model = VariationalGaussianMixture(K=10, display=True, max_iter=200, plot_period=5, init_param="kmeans")
model = VariationalGaussianMixtureCB(K=10, display=True, max_iter=201, plot_period=20, init_param="kmeans")
model.fit(X)

#%% Display ELBO
plt.plot(model.elbo, 'k')
plt.margins(x=0)
plt.xlabel('iteration')
plt.ylabel('Variational lower bound')
plt.show()

#%% Display final GM
model._display_2D(X, n_levels=21, show_components=True)
plt.show()


# %%
