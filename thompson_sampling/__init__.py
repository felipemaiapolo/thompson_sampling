import numpy as np
import scipy
from scipy import stats

def add_ones(arr):
    """
    Add a column of ones to the left side of a 2D numpy array.

    Parameters:
    - arr (numpy.ndarray): The 2D array to which the column of ones will be added.

    Returns:
    - numpy.ndarray: The updated 2D array with a column of ones on the left.
    """
    if len(arr.shape) != 2:
        raise ValueError("Input should be a 2D numpy array.")

    # Create a column of ones with the same number of rows as the input array
    ones_col = np.ones((arr.shape[0], 1))

    # Horizontally stack the ones column with the input array
    return np.hstack((ones_col, arr))

class BayesianLinearRegression(object):
    ### Adapted from https://github.com/tonyduan/conjugate-bayes
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown relationship. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(0, σ²V)
    Parameters
    ----------
    mu: prior for N(mu, v) on the model β
    v:  prior for N(mu, v) on the model β
    a:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    """
    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ \
                 (np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr)
        v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ \
                np.linalg.inv(np.eye(m) + x_tr @ self.v @ x_tr.T) @ \
                (y_tr - x_tr @ self.mu.T)

        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = np.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return stats.t(df=2 * self.a, loc=x_te @ self.mu, scale=scales)

    def get_conditional_beta(self, sigma2):
        return stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return stats.invgamma(self.a, scale=self.b)

class ContextualThompsonSampling(object):
    def __init__(self, k, x_dim = None, initial_data = None):
        self.k = k
        if initial_data == None:
            self.x_dim = x_dim
            self.model = BayesianLinearRegression(mu=np.zeros(k*(x_dim+1)), v=1*np.eye(k*(x_dim+1)), a=1, b=1)
        else:
            assert self.k==initial_data['d'].shape[1]
            self.x_dim = initial_data['x'].shape[1]
            X = (initial_data['d'].T[:,:,None]*add_ones(initial_data['x'])[None,:,:])
            X = np.hstack([x for x in X])
            self.model = BayesianLinearRegression(mu=np.zeros(X.shape[1]), v=1*np.eye(X.shape[1]), a=1, b=1)
            self.model.fit(X, initial_data['y'])

    def update(self, new_data):
        X = (new_data['d'].T[:,:,None]*add_ones(new_data['x'])[None,:,:])
        X = np.hstack([x for x in X])
        self.model = BayesianLinearRegression(mu=np.zeros(X.shape[1]), v=1*np.eye(X.shape[1]), a=1, b=1)
        self.model.fit(X, new_data['y'])

    def evaluate_strat(self, x, random_state = None):
        assert x.shape[0] == 1
        assert x.shape[1] == self.x_dim
        
        #Sampling from sigma2
        np.random.seed(random_state)
        u=np.random.uniform(0,1,1)
        sigma2 = self.model.get_marginal_sigma2().ppf(u)

        #Sampling from beta
        beta_mu = self.model.get_conditional_beta(sigma2=sigma2).mean
        beta_cov = self.model.get_conditional_beta(sigma2=sigma2).cov
        beta = np.random.multivariate_normal(beta_mu, beta_cov, 1)[0].squeeze()

        #Output
        X = np.eye(self.k)[:,:,None]*add_ones(x)[None,:,:]
        print(beta.shape)
        beta = np.vstack([beta[i*(self.x_dim+1):(i+1)*(self.x_dim+1)] for i in range(self.k)])[:,None,:]
        expect_y = np.diag((X*beta).sum(-1))
        best_strat = np.zeros(self.k)
        best_strat[np.argmax(expect_y)] = 1
        return {'best_strat': best_strat.reshape((1,-1)), 'expect_y': expect_y, 'beta': beta}