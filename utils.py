
#!/usr/bin/env python3
# Julien Nonin

#%%
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, label_bg='white', **kwargs):
    """Display a confidence ellipse of a bivariate normal distribution
    
    Arguments:
        mu {array-like of shape (2,)} -- mean of the distribution
        cov {array-like of shape(2,2)} -- covariance matrix
        alph {float btw 0 and 1} -- level of confidence
        ax {plt.Axes} -- axes on which to display the ellipse
        clabel {str} -- label to add to ellipse (default: {None})
        label_bg {str} -- background of clabel's textbox
        kwargs -- other arguments given to class Ellipse
    """
    c = -2 * np.log(1 - alph)  # quantile at alpha of the chi_squarred distr. with df = 2
    Lambda, Q = la.eig(cov)  # eigenvalues and eigenvectors (col. by col.)
    
    ## Compute the attributes of the ellipse
    width, heigth = 2 * np.sqrt(c * Lambda)
    # compute the value of the angle theta (in degree)
    theta = 180 * np.arctan(Q[1,0] / Q[0,0]) / np.pi if cov[1,0] else 0
        
    ## Create the ellipse
    if 'fc' not in kwargs.keys():
        kwargs['fc'] = 'None'
    level_line = Ellipse(mu, width, heigth, angle=theta, **kwargs)
    
    ## Display a label 'clabel' on the ellipse
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
        pos = Q[:,1] * np.sqrt(c * Lambda[1]) + mu  # position along the heigth
        
        ax.text(*pos, clabel, color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=1)) # white box
        
    return ax.add_patch(level_line)
  
if __name__ == '__main__':
    N = 100
    mu = np.array([0, 1])
    cov = np.array([[0.30, 0.03], [0.03, 0.15]])
    np.random.seed(0)
    x = np.random.multivariate_normal(mu, cov, N)

    plt.figure()
    ax = plt.gca()
    plot_confidence_ellipse(mu, cov, 0.99, ax, ec='crimson', clabel='99 %')
    plot_confidence_ellipse(mu, cov, 0.95, ax, ec='r')
    plot_confidence_ellipse(mu, cov, 0.90, ax, ec='orange', clabel='90 %')
    # plot_confidence_ellipse(mu, cov, 1-np.exp(-3**2/2), plt.gca(), fc='lightblue', ec='darkblue', alpha=0.3, label=r'3$\sigma$')
    plt.plot(*x.T, 'o', c='dimgrey')
    ax.plot(*mu, 'kx')
    plt.axis('equal')
    plt.show()
  