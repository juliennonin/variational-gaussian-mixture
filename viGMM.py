#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.special import digamma, logsumexp
from utils import plot_confidence_ellipse

#%% Load and standardize data
X = np.loadtxt('data/faithful.txt')
X = (X - X.mean(axis=0)) / X.std(axis=0)
N, D = X.shape  # n_samples, n_features

#%% Prior Params
K = 10  # n components

alpha0 = 1/K  # weight concentration prior (Ditichlet)
m0 = np.zeros(D)  # mean prior
beta0 = 1  # mean precision prior
v0 = D  # degrees of freedom prior
invW0 = v0 * (X - m0).T.dot(X - m0) / N  # covariance prior

alpha = alpha0 * np.ones(K)  # weight concentration
m = np.array([np.random.multivariate_normal(m0, invW0 / (v0 * beta0)) for _ in range(K)])  # means
beta = beta0 * np.ones(K)  # mean precision
invW = np.array([invW0 for _ in range(K)])  # covariances
v = v0 * np.ones(K) # degrees of freedom
    
#%% Initialization
# _initialize_parameters (random)
# r = np.random.rand(N, K)
# r /= r.sum(axis=1)[:, np.newaxis]

# _initialize_parameters (kmeans)
# r = np.zeros((N, K))
# label = KMeans(n_clusters=K, n_init=1).fit(X).labels_
# r[np.arange(N), label] = 1


# m_step(X, np.log(r))


plt.figure()
plt.plot(*X.T, 'o', c='dimgrey', alpha = 0.5)
ax = plt.gca()
for k in range(K):
    plot_confidence_ellipse(m[k], invW[k] / v[k], 0.9, ax=ax, ec='red')
plt.show()

#%% Display
for _ in range(100):
    log_resp = e_step(X)
    m_step(X, log_resp)

    if _%10 == 0:
        plt.figure()
        plt.plot(*X.T, 'o', c='dimgrey', alpha = 0.5)
        ax = plt.gca()
        w, covs = map_estimate()
        for k in range(K):
            if not(np.allclose(m[k], [0, 0], atol=1e-3) and w[k] < 1e-3):
             plot_confidence_ellipse(m[k], invW[k] / v[k], 0.9, ax=ax, ec='teal')
        plt.show()



#%% E-step
def e_step(X):  # _e_step [_base.py]
    """Compute responsabilities"""
    global alpha, beta, m, invW, v
    E = np.zeros((N, K))
    W = np.linalg.inv(invW)
    for k in range(K):
        Xc = X - m[k]
        E[:,k] = D / beta[k] + v[k] * np.sum(Xc @ W[k] * Xc, axis=1)  # 10.64  [TO CHECK]
    logLambdaTilde = np.sum(digamma(0.5 * (v - np.c_[np.arange(0, D)])), axis=0) + D * np.log(2) + np.log(np.linalg.det(W))   # 10.65
    logPiTilde = digamma(alpha) - digamma(np.sum(alpha))  # 10.66, _estimate_log_weights
    logRho = logPiTilde + 0.5*logLambdaTilde - 0.5 * (E + D * np.log(2 * np.pi)) # 10.46
    logR = logRho - np.c_[logsumexp(logRho, axis=1)]  # 10.49, log_resp
    return logR


def compute_Ess(X, r): # _estimate_gaussian_parameters
    Nk = r.sum(axis=0) + 10*np.finfo(r.dtype).eps  # 10.51
    x_bar = (r.T @ X)  / np.vstack(Nk)  # 10.52
    S = np.zeros((K, D, D))
    # _estimate_gaussian_covariance_full
    for k in range(K):
        Xc = X - x_bar[k]
        S[k] = ((r[:,k] * Xc.T) @ Xc) / Nk[k]  # 10.53
        S[k].flat[::D+1] += 1e-6  # regularization added to the diag. Assure that the covariance matrices are all positive
    return Nk, x_bar, S


def compute_lower_bound(logR):
    pass

def m_step(X, logR):  # _m_step
    global alpha, beta, m, invW, v
    Nk, xbar, S = compute_Ess(X, np.exp(logR))
    # _estimate_weights
    alpha = alpha0 + Nk  # 10.58, _estimate_weigths
    # _estimate_means
    beta = beta0 + Nk  # 10.60, _estimate_means [498]
    m = (beta0*m0 + xbar*np.c_[Nk]) / np.c_[beta]  # 10.61, _estimate_means [499]
    # _estimate_precisions
    v = v0 + Nk  # 10.63, _estimate_precisions_ [546]
    invW = np.zeros((K, D, D))
    for k in range(K):
        xc = xbar[k] - m0
        invW[k] = invW0 + Nk[k]*S[k] + (beta0 * Nk[k]) * np.outer(xc, xc) / beta[k]  # 10.62, _estimate_precisions [553]
    #[TODO ????] normalize covariance
    # invW /= (v[:, np.newaxis, np.newaxis])
    # W = np.linalg.inv(W)  #  

def map_estimate():
    global alpha, invW
    weigths = alpha / np.sum(alpha)
    covs = invW / v[:, np.newaxis, np.newaxis]
    return weigths, covs

#%%
model = BayesianGaussianMixture(n_components=K, covariance_type='full', max_iter=2)
labels = model.fit_predict(X)
#%%
ax = plt.gca()
for i, l in enumerate(set(labels)):
    plt.plot(*X[labels==l].T, 'o')
    plot_confidence_ellipse(model.means_[l], model.covariances_[l], 0.9, ax, ec=f'C{i}')

# %%
