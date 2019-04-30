With the STA-663 Bayesian Hierarchical Clusteristing (BHC) package, generating Bayesian hierarchical clusters to a given data set is no longer difficult. Supply data, likelihood, and a alpha value, then a BHC is calculated at an instant!

How Does it Work?
Download
Import
Download
Use Pypi via pip to install BHC

$ pip install STA-663 BHC Package
Import
Import BHC_vanilla and Multinorm_post_marginal_likelihood

from STA-663 BHC Package import BHC, Multinorm_post_marginal_likelihood
BHC
There are three inputs to the function: data, likelihood, and number 1.

Data has to be a 2-dimensional array
Likelihood is the calculated value using the Multinorm_post_marginal_likelihood
Number 1 is a recommended number to use
Multinorm_post_marginal_likelihood
There are four inputs to the function: mu, kappa, nu, and psi.

Mu is a 2-dimensional array
kappa is a positive integer
nu is a positive integer
psi is a 2-dimensional array