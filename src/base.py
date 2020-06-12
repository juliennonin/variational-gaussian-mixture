import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import plot_confidence_ellipse
from scipy.stats import multivariate_normal
from scipy.special import digamma, gammaln, logsumexp
from matplotlib.colors import LogNorm
from abc import abstractmethod

def log_wishart_B(invW, nu):
    D = len(invW)
    return + 0.5 * nu * np.log(np.linalg.det(invW)) \
           - 0.5 * nu * D * np.log(2) \
           - 0.25 * D * (D-1) * np.log(np.pi) \
           - gammaln(0.5 * (nu - np.arange(D))).sum()

def log_dirichlet_C(alpha):
    return gammaln(np.sum(alpha)) - gammaln(alpha).sum()

class BaseGaussianMixture():
    """Abstract class for mixture models."""

    def __init__(self, K, init_param, seed, max_iter, display, plot_period):
        self.K = K
        self.init_param = init_param
        self.rd = np.random.RandomState(seed)
        self.max_iter = max_iter
        self.display = display
        self.plot_period = plot_period or max_iter // 10

    def _initialize_parameters(self, X):
        n_samples, D = X.shape
        if self.init_param == "random":
            resp = self.rd.rand(n_samples, self.K)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        elif self.init_param == "kmeans":
            resp = np.zeros((n_samples, self.K))
            label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_
            resp[np.arange(n_samples), label] = 1

        elif self.init_param == "_debug":
            resp = np.loadtxt('data/_resp.txt')
        
        else:
            raise ValueError("Correct values for 'init_param' are ['random', 'kmeans']")
        # np.savetxt('data/_resp.txt', resp)

        self._initialize(X, resp)
    
    @abstractmethod
    def _initalize(self, X, resp):
        """Initialize the model parameters and hyperparameters"""
        pass

    def fit(self, X):
        self.fit_predict(X)
        return self

    @abstractmethod
    def fit_predict(self, X):
        pass

    @abstractmethod
    def _get_final_parameters(self):
        pass

    def display_2D(self, X, n_levels=21, show_components=True):
        assert X.shape[1] == 2, "Only 2D display is available"
        ## Grid
        xmin, xmax, ymin, ymax = X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()
        mx, my =.1 * (xmax - xmin), .1 * (ymax - ymin)  # margins
        xmin, xmax, ymin, ymax = xmin - mx, xmax + mx, ymin - my, ymax + my
        # xmin, xmax = min(X[:,0].min(), X[:,1].min()), max(X[:,0].max(), X[:,1].max())
        # ymin, ymax = xmin, xmax
        x = np.linspace(xmin, xmax, 200)
        y = np.linspace(ymin, ymax, 200)
        x, y = np.meshgrid(x, y)
        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x; pos[:,:,1] = y
        rvs = [multivariate_normal(self.m[k], self.covs[k]) for k in range(self.K)]
        Z = sum([self.weights[k] * rvs[k].pdf(pos) for k in range(self.K)])

        plt.figure(figsize=(6,6))
        
        ## Heatmap
        scale = np.amax(np.abs(Z))
        plt.imshow(Z, interpolation='bilinear', origin='lower', vmin=-scale,vmax=scale,
            cmap="RdBu", extent=(xmin, xmax, ymin, ymax))
        
        ## Log-contours
        levels_exp = np.linspace(np.floor(np.log10(Z.min())-1), np.ceil(np.log10(Z.max())+1), n_levels)
        levels = np.power(10, levels_exp)
        plt.contour(x, y, Z, linewidths=1., colors="#93b8db",
            levels=levels, norm=LogNorm())  # A1B5C8
        
        ## Scatter plot of the dataset
        plt.plot(*X.T, '.', c='k', alpha=0.6)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        ## Display each component of the GM
        if show_components:
            ax = plt.gca()
            for k in range(self.K):
                if self.weights[k] >= 1e-5:
                    plot_confidence_ellipse(self.m[k], self.covs[k], 0.9, ax=ax,
                        ec='brown', linestyle=(0, (5, 2)),
                        alpha=max(0.3, self.weights[k] / max(self.weights)))