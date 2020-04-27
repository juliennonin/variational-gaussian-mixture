# Variational Bayesian Model Selection for Mixture Distributions

Two Variational Bayesian Gaussian mixture models proposed in [1] and [2] are compared. Their graphical models are respectively shown below.
![graphical models](data/img/graphical_models.png)
- The implementation for the first model [1], that treats the mixing coefficients as parameters, is still in progress. A first debugging draft is available in the Jupyter Notebook `viGMM_CB.ipynb`.
- The full Bayesian GMM model [2] is implemented in the class `VariationalGaussianMixture` in `src.viGMM_full`.
## Installation
First clone the project
```
git clone https://github.com/JulienNonin/variational-gaussian-mixture.git
cd variational-gaussian-mixture
```
`variational-gaussian-mixture` runs on Python 3.7. and only requires `matplotlib ≥ 3.2.1` and `numpy ≥ 1.18.1`.

## Usage
Import the class `VariationalGaussianMixture`
```
from src.viGMM_full import VariationalGaussianMixture
```
Load the "Old Faitful" data set and standardize the data.
```
X = np.loadtxt('data/faithful.txt')
X = (X - X.mean(axis=0)) / X.std(axis=0)
```
Apply the full Bayesian GMM
```
model = VariationalGaussianMixture(K=10, display=True, max_iter=200, plot_period=200)
model.fit(X)
```
This should produce the following output
![results](data/img/results_fullGMM_OF.png)

## References
[1] A. Corduneanu and C. Bishop, *Variational Bayesian Model Selection for Mixture Distributions*. in *Proc. AI
and Statistics Conf.*, Jan. 2001, pp. 27-34.

[2] C. Bishop, *Pattern Recognition and Machine Learning (Information Science and Statistics)*. New York:
Springer-Verlag, 2006.

----
## To do

- [x] Create this README
- [x] Fix random initialization of responsabilities
- [x] Compute the ELBO
