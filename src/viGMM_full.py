import numpy as np
import matplotlib.pyplot as plt
from .base import BaseGaussianMixture, log_wishart_B, log_dirichlet_C
from scipy.special import digamma, gammaln, logsumexp
from utils import plot_confidence_ellipse


class VariationalGaussianMixture(BaseGaussianMixture):
    """Variational Bayesian estimation of a Gaussian mixture

    References
    ----------
       [2] Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer."""

    def __init__(self, K, init_param="random", seed=2208, max_iter=200,
                 alpha0=None, m0=None, beta0=None, invW0=None, nu0=None,
                 display=False, plot_period=None):
        super().__init__(
            K, init_param=init_param, seed=seed, max_iter=max_iter,
            display=display, plot_period=plot_period)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.m0 = m0
        self.nu0 = nu0
        self.invW0 = invW0

    def _initialize(self, X, resp):
        n_samples, D = X.shape
        self.alpha0 = self.alpha0 or (1. / self.K)
        self.beta0 = self.beta0 or 1.
        self.m0 = self.m0 or X.sum(axis=0)
        self.nu0 = self.nu0 or D
        self.invW0 = self.invW0 or np.atleast_2d(np.cov(X.T))
                
        self._m_step(X, np.log(resp))

    def fit_predict(self, X):
        """Estimate model parameters"""
        _, D = X.shape
        self._initialize_parameters(X)

        self.elbo = np.empty(self.max_iter)
        for i in range(self.max_iter):
            ln_resp, ln_lambda_tilde, ln_pi_tilde = self._e_step(X)
            self._m_step(X, ln_resp)
            self.elbo[i] = self._compute_lower_bound(X, ln_resp, ln_lambda_tilde, ln_pi_tilde)

            if self.display and D == 2 and i % self.plot_period == 0:
                self._get_final_parameters()
                self.display_2D(X)
                plt.title(f'iteration {i}')
                plt.show()

        self._get_final_parameters()
        if self.display and D == 2:
            self.display_2D(X)
            plt.title(f'iteration {i}')
            plt.show()
        # Final e-step to guarantee that the labels are consistent
        ln_resp, *_ = self._e_step(X)
        return ln_resp.argmax(axis=1)

    def _get_final_parameters(self):
        self.weights = self.alpha / np.sum(self.alpha)
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

    
    def _e_step(self, X):
        n_samples, D = X.shape
        W = np.linalg.inv(self.invW)
        
        E = np.zeros((n_samples, self.K))
        for k in range(self.K):
            Xc = X - self.m[k]
            E[:,k] = D / self.beta[k] + self.nu[k] * np.sum(Xc @ W[k] * Xc, axis=1)  # (10.64)
        ln_lambda_tilde = np.sum(digamma(0.5 * (self.nu - np.arange(0, D)[:,np.newaxis])), axis=0) \
             + D * np.log(2) + np.log(np.linalg.det(W))   # (10.65)
        ln_pi_tilde = digamma(self.alpha) - digamma(np.sum(self.alpha))  # (10.66)
        
        ln_rho = ln_pi_tilde + 0.5*ln_lambda_tilde - 0.5 * (E + D * np.log(2 * np.pi)) # (10.46)
        ln_resp = ln_rho - np.c_[logsumexp(ln_rho, axis=1)]  # (10.49)
        return ln_resp, ln_lambda_tilde, ln_pi_tilde

    def _m_step(self, X, ln_resp):
        _, D = X.shape
        N, x_bar, S = self._compute_statististics(X, np.exp(ln_resp))

        self.alpha = self.alpha0 + N  # (10.58), weight concentration
        self.beta = self.beta0 + N  # (10.60), mean precision
        self.nu = self.nu0 + N  # (10.63), degrees of freedom
        self.m = (self.beta0 * self.m0 + x_bar * N[:,np.newaxis]) / self.beta[:,np.newaxis]  # (10.61), means
        
        self.invW = np.zeros((self.K, D, D))  # covariances
        for k in range(self.K):
            xc = x_bar[k] - self.m0
            self.invW[k] = self.invW0 + N[k] * S[k] + (self.beta0 * N[k]) * (
                np.outer(xc, xc) / self.beta[k])  # (10.62)

    def _compute_lower_bound(self, X, ln_resp, ln_lambda_tilde, ln_pi_tilde):
        _, D = X.shape
        resp = np.exp(ln_resp)
        N, x_bar, S = self._compute_statististics(X, resp)
        W = np.linalg.inv(self.invW)

        ln_p_x = 0.5 * np.sum(N * ln_lambda_tilde) \
            - 0.5 * D * np.sum(N / self.beta) \
            - 0.5 * np.sum(N * self.nu * np.trace(S @ W, axis1=1, axis2=2)) \
            - 0.5 * np.sum([N[k] * self.nu[k] * (x_bar[k] - self.m[k]) @ W[k] @ (x_bar[k] - self.m[k]) for k in range(self.K)]) \
            - 0.5 * N.sum() * D * np.log(2 * np.pi)#  (10.71)
        ln_p_z = np.sum(resp * ln_pi_tilde) # (10.72)
        ln_p_pi = (self.alpha0 - 1) * ln_pi_tilde.sum() + log_dirichlet_C([self.alpha0] * self.K) # (10.73)
        ln_p_mu_lambda = 0.5 * np.sum(ln_lambda_tilde) \
            + 0.5 * self.K * D * np.log(0.5 * self.beta0 / np.pi) \
            - 0.5 * D * (self.beta0 / self.beta).sum() \
            - 0.5 * self.beta0 * np.sum([self.nu[k] * (self.m[k] - self.m0) @ W[k] @ (self.m[k] - self.m0) for k in range(self.K)]) \
            + self.K * log_wishart_B(self.invW0, self.nu0) \
            + 0.5 * (self.nu0 - D - 1) * ln_lambda_tilde.sum() \
            - 0.5 * np.sum(self.nu * np.trace(self.invW0 @ W, axis1=1, axis2=2)) # (10.74)

        ln_q_z = np.sum(resp * ln_resp) # (10.75)
        ln_q_pi = np.sum((self.alpha - 1) * ln_pi_tilde) + log_dirichlet_C(self.alpha) # (10.76)
        ln_q_mu_lambda = 0.5 * np.sum(ln_lambda_tilde) - 0.5 * self.K * D \
            + np.sum(0.5 * D * np.log(0.5 * self.beta / np.pi)) \
            + np.sum([log_wishart_B(self.invW[k], self.nu[k]) for k in range(self.K)]) \
            + np.sum(0.5 * (self.nu - D - 1) * ln_lambda_tilde) \
            - np.sum(0.5 * self.nu * D)  # (10.77)

        return ln_p_x + ln_p_z + ln_p_pi + ln_p_mu_lambda - ln_q_z - ln_q_pi - ln_q_mu_lambda


    # def _display_2D(self, X, i=None, **kwargs):
    #     assert X.shape[1] == 2, "Only 2D display is available"
    #     plt.figure()
    #     plt.plot(*X.T, 'o', c='dimgrey', alpha=0.5)
    #     ax = plt.gca()
    #     for k in range(self.K):
    #         if not(np.allclose(self.m[k], [0, 0], atol=1e-3) and self.weights[k] < 1e-3):
    #          plot_confidence_ellipse(self.m[k], self.covs[k], 0.9, ax=ax, ec='teal')
    #     if i: plt.title(f'iteration {i}')
    #     plt.show()
