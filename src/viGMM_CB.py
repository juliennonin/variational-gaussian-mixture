#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.special import digamma, logsumexp
from utils import plot_confidence_ellipse

#%% Load and standardize data
X = np.loadtxt('data/gaussian.txt')
X = (X - X.mean(axis=0)) / X.std(axis=0)
N, D = X.shape  # n_samples, n_features

#%% Prior Params
def init():
    global invW0, m0, beta0, S0, v0, pi, resp
    m0 = np.mean(X, axis=0)
    invW0 = np.cov(X.T)
    beta0 = 1
    S0 = np.linalg.inv(beta0 * np.eye(D))
    v0 = D
    pi = np.ones(K) / K
    # resp = np.random.rand(N, K)
    # resp /= resp.sum(axis=1)[:, np.newaxis]
    resp = np.loadtxt('data/resp.txt')

#%% 
# Init
K = 10
init()
esp_T = np.array([v0 * np.linalg.inv(invW0) for _ in range(K)])
esp_mu = np.zeros((K, D))
esp_mu_muT = np.zeros((K, D, D))
update_params()
display()
for i in range(200):
    compute_esp()
    compute_resp()
    update_params()
    m_step()
    if i % 10 == 0:
        print(i)
        display()


#%% E-step

def compute_esp():
    global esp_T, esp_log_det_T, esp_mu, esp_mu_muT
    W = np.linalg.inv(invW)
    invS = np.linalg.inv(S)
    esp_T = np.zeros_like(W)
    esp_log_det_T = np.zeros(K)
    esp_mu = np.copy(m)
    esp_mu_muT = np.zeros_like(S)
    for k in range(K):
        esp_T[k] = v[k] * W[k]
        esp_log_det_T[k] = digamma(0.5*(v[k] - np.arange(D))).sum() + D * np.log(2) - np.log(np.linalg.det(invW[k]))
        esp_mu_muT[k] = invS[k] + np.outer(m[k], m[k])

def compute_resp():
    global resp
    log_rho = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            log_rho[n, k] = np.log(pi[k]+1e-15) + 0.5 * esp_log_det_T[k] - 0.5 * np.trace(
                esp_T[k] @ (np.outer(X[n], X[n]) - np.outer(X[n], esp_mu[k]) - np.outer(esp_mu[k], X[n]) + esp_mu_muT[k])
            )
    log_resp = log_rho - logsumexp(log_rho, axis=1)[:, np.newaxis]
    resp = np.exp(log_resp)

def update_params():
    global m, S, v, invW
    # m = np.zeros_like(m)
    S = np.zeros((K, D, D))
    # S_old = np.copy(S)
    # invS = np.zeros_like(invS)
    # v = np.zeros_like(v)
    invW = np.zeros((K, D, D))
    m = np.zeros((K, D))
    
    eta = resp.sum(axis=0) + 10*np.finfo(resp.dtype).eps
    v = v0 + eta
    S = esp_T * eta[:,np.newaxis,np.newaxis] + beta0 * np.eye(D)
    invS = np.linalg.inv(S)
    

    for k in range(K):
        # S[k] = beta0 * np.eye(D) + esp_T[k] * eta[k]

        m[k] = invS[k] @ esp_T[k] @ (resp[:, k] @ X)

        s = np.zeros((D, D))
        for n in range(N):
            s += resp[n, k] * (np.outer(X[n], X[n]) - np.outer(X[n], esp_mu[k]) - np.outer(esp_mu[k], X[n]) + esp_mu_muT[k])
        invW[k] = invW0 + s

def m_step():
    global pi
    pi = resp.sum(axis=0) / resp.sum()

def compute_lower_bound(logR):
    pass

def display():
    plt.figure(figsize=(6,6))
    plt.plot(*X.T, 'o', c='dimgrey', alpha = 0.5)
    ax = plt.gca()
    for k in range(K):
        if pi[k] >= 1/(2*K):
            plot_confidence_ellipse(m[k], invW[k]/v[k], 0.9, ax=ax, ec='teal')
    ax.set_aspect('equal')
    plt.show()

# %%
