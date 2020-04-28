import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.special import digamma, gammaln, logsumexp
from utils import plot_confidence_ellipse


def log_wishart(invW, nu):
    D = len(invW)
    return + 0.5 * nu * np.log(np.linalg.det(invW)) \
           - 0.5 * nu * D * np.log(2) \
           - 0.25 * D * (D-1) * np.log(np.pi) \
           - gammaln(0.5 * (nu - np.arange(D))).sum()

class VariationalGaussianMixtureCB():
    """Variarional Bayesian estimation of a Gaussian mixture

    References
    ----------
        [2] Corduneanu, Adrian and Bishop, Christopher M. (2001), "Variational 
        Bayesian Model Selection for Mixture Distributions", in Proc. AI and 
        Statistics Conf., pp. 27-34."""

    def __init__(self, K, init_param="random", seed=2208, max_iter=200, 
                 beta0=None, nu0=None, invW0=None,
                 display=False, plot_period=None):
        self.K = K
        self.init_param = init_param
        self.rd = np.random.RandomState(seed)
        self.max_iter = max_iter
        self.display = display
        self.plot_period = plot_period or max_iter // 10

        self.beta0 = beta0
        self.nu0 = nu0
        self.invW0 = invW0

    def _initialize_parameters(self, X):
        n_samples, D = X.shape
        self.nu0 = self.nu0 or D
        self.invW0 = self.invW0 or np.atleast_2d(np.cov(X.T))
        self.beta0 = self.beta0 or 1.

        self.weights = np.ones(self.K) / self.K

        if self.init_param == "random":
            resp = self.rd.rand(n_samples, self.K)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        elif self.init_param == "kmeans":
            resp = np.zeros((n_samples, self.K))
            label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        
        elif self.init_param == "_debug":
            resp = np.loadtxt('data/resp.txt')
        
        else:
            raise ValueError("Correct values for 'init_param' are ['random', 'kmeans']")
        
        
        ## Init expectations
        expect = {"T" : np.array([self.nu0 * np.linalg.inv(self.invW0) for _ in range(self.K)]),
                  "log_det_T" : np.zeros(self.K),
                  "mu" : np.zeros((self.K, D)),  # (23)
                  "mu_muT" : np.zeros((self.K, D, D))}

        self._update_params(X, resp, expect)

    def fit(self, X):
        self.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        _, D = X.shape
        self._initialize_parameters(X)

        self.elbo = np.empty(self.max_iter)
        for i in range(self.max_iter):
            print(i)
            expect = self._compute_expectations(D)
            log_resp, log_rho_tilde = self._compute_resp(X, expect)
            resp = np.exp(log_resp)
            self._update_params(X, resp, expect)
            self._m_step(resp)
            self.elbo[i] = self._compute_lower_bound(X, log_resp, resp, log_rho_tilde, expect)

            if self.display and D == 2 and i % self.plot_period == 0:
                self._get_final_parameters()
                self._display_2D(X, i)
        
        self._get_final_parameters()
        if self.display and D == 2:
            self._display_2D(X, i)

    def _compute_expectations(self, D):
        exp_T = self.nu[:, np.newaxis, np.newaxis] * np.linalg.inv(self.invW)  # (25)
        exp_log_det_T = np.sum(digamma(0.5 * (self.nu - np.arange(D)[:,np.newaxis])), axis=0) \
            + D * np.log(2) - np.log(np.linalg.det(self.invW))  # (26)

        expect = {"T" : exp_T,
                  "log_det_T" : exp_log_det_T,
                  "mu" : np.copy(self.m),  # (23)
                  "mu_muT" : np.zeros_like(self.S),
                  "muT_mu": np.zeros(self.K)}
        
        invS = np.linalg.inv(self.S)
        for k in range(self.K):
            expect["mu_muT"][k] = invS[k] + np.outer(self.m[k], self.m[k])  # (24)
            expect["muT_mu"][k] = np.trace(invS[k]) + self.m[k] @ self.m[k]
        return expect
        
    def _compute_resp(self, X, expect):
        N, D = X.shape
        log_rho_tilde = np.zeros((N, self.K))
        for n in range(N):
            for k in range(self.K):
                log_rho_tilde[n, k] = 0.5 * expect["log_det_T"][k] - 0.5 * np.trace(
                    expect["T"][k] @ (np.outer(X[n], X[n]) - np.outer(X[n], expect["mu"][k]) \
                        - np.outer(expect["mu"][k], X[n]) + expect["mu_muT"][k])
                )  # (17)
        log_rho = log_rho_tilde + np.log(self.weights + 10 * np.finfo(self.weights.dtype).eps)
        log_resp = log_rho - logsumexp(log_rho, axis=1)[:, np.newaxis]  # (16)
        return log_resp, log_rho_tilde

    def _update_params(self, X, resp, expect):
        N, D = X.shape
        eta = resp.sum(axis=0) + 10*np.finfo(resp.dtype).eps  # sum_n (z_nk)
        self.nu = self.nu0 + eta  # (20)
        self.S = self.beta0 * np.eye(D) + expect["T"] * eta[:,np.newaxis,np.newaxis]  # (18)
        invS = np.linalg.inv(self.S)
        
        self.invW = np.zeros((self.K, D, D))  # init (21)
        self.m = np.zeros((self.K, D))  # init (19)
        
        for k in range(self.K):
            self.m[k] = invS[k] @ expect["T"][k] @ (resp[:, k] @ X)  # (19)

            s = np.zeros((D, D))
            for n in range(N):
                s += resp[n, k] * (np.outer(X[n], X[n]) - np.outer(X[n], expect["mu"][k]) \
                    - np.outer(expect["mu"][k], X[n]) +  expect["mu_muT"][k])  # (21 --)
            self.invW[k] = self.invW0 + s  # (-- 21)

    def _m_step(self, resp):
        self.weights = resp.sum(axis=0) / resp.sum()

    def _compute_lower_bound(self, X, log_resp, resp, log_rho_tilde, expect):
        N, D = X.shape

        ln_p_x = np.sum(resp * (log_rho_tilde))  # (28)
        ln_p_z = np.sum(resp * np.log(self.weights))  # (29)
        ln_p_mu = self.K * D * np.log(0.5 * self.beta0 / np.pi) \
            - 0.5 * self.beta0 * np.sum(expect["muT_mu"])  # (30)
        ln_p_T = self.K * log_wishart(self.invW0, self.nu0) \
            + 0.5 * (self.nu0 - D - 1) * expect["log_det_T"].sum() \
            - 0.5 * np.trace(self.invW0 * expect["T"].sum())  # (31)
        
        ln_q_z = np.sum(resp * np.log(resp))  # (32)
        ln_q_mu = - 0.5 * self.K * D * (1 + np.log(2 * np.pi)) \
            + 0.5 * np.sum(np.log(np.linalg.det(self.S)))  # (33)
        ln_q_T  = np.sum([log_wishart(self.invW[k], self.nu[k]) for k in range(self.K)]) \
            + np.sum(0.5 * (self.nu - D - 1) * expect['log_det_T']) \
            - np.sum(0.5 * np.trace(self.invW @ expect['T'], axis1=1, axis2=2))  # (34)
        return ln_p_x + ln_p_z + ln_p_mu + ln_p_T + ln_q_z + ln_q_mu + ln_q_T

    def _get_final_parameters(self):
        self.covs = self.invW / self.nu[:, np.newaxis, np.newaxis]
    
    def _display_2D(self, X, i=None, eps=1e-5):
        assert X.shape[1] == 2, "Only 2D display is available"
        plt.figure()
        plt.plot(*X.T, 'o', c='dimgrey', alpha = 0.5)
        ax = plt.gca()
        for k in range(self.K):
            if self.weights[k] >= eps:
                plot_confidence_ellipse(self.m[k], self.covs[k], 0.9, ax=ax, ec='teal')
        # ax.set_aspect('equal')
        if i: plt.title(f'iteration {i}')
        plt.show()
