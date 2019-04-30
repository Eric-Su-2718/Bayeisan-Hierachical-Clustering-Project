from scipy.stats import multivariate_normal

def Multinorm_post_marginal_likelihood(mu_0, kappa_0, nu_0, Psi_0):
    """
    Returns a function that computes the marginal likelihood 
    for a Normal-inverse-Wishart model with specifies data and prior parameters.
    """
        
    def marginal_likelihood(data):
        
        n = data.shape[0]
        p = data.shape[1]
        xbar = data.mean(axis = 0)
    
        #update posterior parameters
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * xbar) / kappa_n
        nu_n = nu_0 + n
    
        if (n == 1):
            C = np.zeros((p, p))
        else:
            C = (n - 1) * np.cov(data.T)
        Psi_n = Psi_0 + C + ((n * kappa_0) / (kappa_n)) * np.dot((xbar - mu_0).reshape(-1, p).T, (xbar - mu_0).reshape(-1, p))
    
        return (1 / np.pi**(n * p / 2)) * (np.exp(multigammaln(nu_n / 2, p)) / np.exp(multigammaln(nu_0 / 2, p))) * (np.linalg.det(Psi_0)**(nu_0 / 2) / np.linalg.det(Psi_n)**(nu_n / 2)) * (kappa_0 / kappa_n)**(p / 2)

    return marginal_likelihood

