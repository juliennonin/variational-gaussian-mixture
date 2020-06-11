import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.special import digamma, logsumexp
from utils import plot_confidence_ellipse


class VariationalGaussianMixture():
    """Variational Bayesian estimation of a Gaussian mixture

    References
    ----------
       [2] Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer."""

    def __init__(self, K, init_param="random", seed=2208, max_iter=200,
                 alpha0=None, m0=None, beta0=None, invW0=None, nu0=None,
                 display=False, plot_period=None):
        self.K = K
        self.init_param = init_param
        self.rd = np.random.RandomState(seed)
        self.max_iter = max_iter
        self.display = display
        self.plot_period = plot_period or max_iter // 10

        self.alpha0 = alpha0
        self.beta0 = beta0
        self.m0 = m0
        self.nu0 = nu0
        self.invW0 = invW0

    def _initialize_parameters(self, X):
        n_samples, D = X.shape
        self.alpha0 = self.alpha0 or (1. / self.K)
        self.beta0 = self.beta0 or 1.
        self.m0 = self.m0 or X.sum(axis=0)
        self.nu0 = self.nu0 or D
        self.invW0 = self.invW0 or np.atleast_2d(np.cov(X.T))

        if self.init_param == "random":
            resp = self.rd.rand(n_samples, self.K)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        elif self.init_param == "kmeans":
            resp = np.zeros((n_samples, self.K))
            label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        
        else:
            raise ValueError("Correct values for 'init_param' are ['random', 'kmeans']")
        # np.savetxt('data/_resp.txt', resp)
        self._m_step(X, np.log(resp))
    
    def fit(self, X):
        self.fit_predict(X)
        return self

    def fit_predict(self, X):
        """Estimate model parameters"""
        _, D = X.shape
        self._initialize_parameters(X)
        for i in range(self.max_iter):
            log_resp = self._e_step(X)
            self._m_step(X, log_resp)

            if self.display and D == 2 and i % self.plot_period == 0:
                self._get_final_parameters()
                self._display_2D(X, i)

        self._get_final_parameters()
        if self.display and D == 2:
            self._display_2D(X, i)
        # Final e-step to guarantee that the labels are consistent
        log_resp = self._e_step(X)
        return log_resp.argmax(axis=1)

    def _get_final_parameters(self):
        self.weigths = self.alpha / np.sum(self.alpha)
        self.covs = self.invW / self.nu[:, np.newaxis, np.newaxis]

    def _compute_statististics(self, X, resp):
        _, D = X.shape
        N = resp.sum(axis=0) + 10*np.finfo(resp.dtype).eps  # (10.51)
        x_bar = (resp.T @ X)  / N[:, np.newaxis]  # (10.52)
        S = np.zeros((self.K, D, D))
        for k in range(self.K):
            Xc = X - x_bar[k]
            S[k] = ((resp[:,k] * Xc.T) @ Xc) / N[k]  # (10.53)
            S[k].flat[::D+1] += 1e-6  # regularization added to the diag. Assure that the covariance matrices are all positive
        return N, x_bar, S

    def _m_step(self, X, log_resp):
        _, D = X.shape
        N, x_bar, S = self._compute_statististics(X, np.exp(log_resp))

        self.alpha = self.alpha0 + N  # (10.58), weight concentration
        self.beta = self.beta0 + N  # (10.60), mean precision
        self.nu = self.nu0 + N  # (10.63), degrees of freedom
        self.m = (self.beta0 * self.m0 + x_bar * N[:,np.newaxis]) / self.beta[:,np.newaxis]  # (10.61), means
        
        self.invW = np.zeros((self.K, D, D))  # covariances
        for k in range(self.K):
            xc = x_bar[k] - self.m0
            self.invW[k] = self.invW0 + N[k] * S[k] + (self.beta0 * N[k]) * (
                np.outer(xc, xc) / self.beta[k])  # (10.62)

    def _e_step(self, X):
        n_samples, D = X.shape
        W = np.linalg.inv(self.invW)
        
        E = np.zeros((n_samples, self.K))
        for k in range(self.K):
            Xc = X - self.m[k]
            E[:,k] = D / self.beta[k] + self.nu[k] * np.sum(Xc @ W[k] * Xc, axis=1)  # (10.64)
        log_lambda_tilde = np.sum(digamma(0.5 * (self.nu - np.arange(0, D)[:,np.newaxis])), axis=0) \
             + D * np.log(2) + np.log(np.linalg.det(W))   # (10.65)
        log_pi_tilde = digamma(self.alpha) - digamma(np.sum(self.alpha))  # (10.66)
        
        log_rho = log_pi_tilde + 0.5*log_lambda_tilde - 0.5 * (E + D * np.log(2 * np.pi)) # (10.46)
        log_resp = log_rho - np.c_[logsumexp(log_rho, axis=1)]  # (10.49)
        return log_resp


    def _display_2D(self, X, i=None, **kwargs):
        assert X.shape[1] == 2, "Only 2D display is available"
        plt.figure()
        plt.plot(*X.T, 'o', c='dimgrey', alpha=0.5)
        ax = plt.gca()
        for k in range(self.K):
            if not(np.allclose(self.m[k], [0, 0], atol=1e-3) and self.weigths[k] < 1e-3):
             plot_confidence_ellipse(self.m[k], self.covs[k], 0.9, ax=ax, ec='teal')
        if i: plt.title(f'iteration {i}')
        plt.show()
