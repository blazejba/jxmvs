import jax
import time
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def initialize_gmm_params(K, D):
    """
    Initialize the parameters of the GMM.
    K: Number of clusters
    D: Dimension of the data
    
    Returns:
    pi: Initial mixing proportions
    mu: Initial means
    sigma: Initial covariance matrices
    """
    pi = jnp.ones(K) / K
    mu = random.normal(random.PRNGKey(0), (K, D))
    sigma = jnp.stack([jnp.eye(D) for _ in range(K)])
    return pi, mu, sigma


def e_step(data, pi, mu, sigma):
    """
    E-step: Compute the responsibilities.
    data: Input data, shape (N, D)
    pi: Mixing proportions, shape (K,)
    mu: Means of the Gaussians, shape (K, D)
    sigma: Covariance matrices of the Gaussians, shape (K, D, D)
    
    Returns:
    responsibilities: Posterior probabilities, shape (N, K)
    """
    N, D = data.shape
    K = len(pi)
    
    responsibilities = jnp.zeros((N, K))
    for k in range(K):
        likelihood = multivariate_normal.pdf(data, mu[k], sigma[k])
        responsibilities = responsibilities.at[:, k].add(pi[k] * likelihood)
        
    # Normalize the responsibilities so that they sum to 1 for each data point
    responsibilities = responsibilities / jnp.sum(responsibilities, axis=1, keepdims=True)
    
    return responsibilities


@jax.jit
def update_covariance(data, mu_k, responsibilities_k):
    """
    Helper function to compute the updated covariance matrix for a single cluster.
    data: Input data, shape (N, D)
    mu_k: Mean of the Gaussian for cluster k, shape (D,)
    responsibilities_k: Posterior probabilities for cluster k, shape (N,)
    
    Returns:
    sigma_k: Updated covariance matrix for cluster k, shape (D, D)
    """
    diff = data - mu_k
    sigma_k = jnp.dot(responsibilities_k * diff.T, diff) / jnp.sum(responsibilities_k)
    return sigma_k


@jax.jit
def m_step(data, responsibilities):
    """
    M-step: Update the parameters.
    data: Input data, shape (N, D)
    responsibilities: Posterior probabilities, shape (N, K)
    
    Returns:
    pi: Updated mixing proportions, shape (K,)
    mu: Updated means of the Gaussians, shape (K, D)
    sigma: Updated covariance matrices of the Gaussians, shape (K, D, D)
    """
    N, D = data.shape
    K = responsibilities.shape[1]
    
    # Update mixing proportions
    pi = jnp.sum(responsibilities, axis=0) / N
    
    # Update means
    mu = jnp.dot(responsibilities.T, data) / jnp.sum(responsibilities, axis=0)[:, jnp.newaxis]
    
    # Update covariance matrices
    sigma = jnp.stack([update_covariance(data, mu[k], responsibilities[:, k]) for k in range(K)])
    
    return pi, mu, sigma


def gmm_em(data, K, max_iters=100, tol=1e-4):
    """
    Fit a Gaussian Mixture Model using the EM algorithm.
    data: Input data, shape (N, D)
    K: Number of clusters
    max_iters: Maximum number of iterations
    tol: Convergence tolerance
    
    Returns:
    pi: Final mixing proportions, shape (K,)
    mu: Final means of the Gaussians, shape (K, D)
    sigma: Final covariance matrices of the Gaussians, shape (K, D, D)
    """
    _, D = data.shape
    pi, mu, sigma = initialize_gmm_params(K, D)
    
    for _ in range(max_iters):
        # E-step
        responsibilities = e_step(data, pi, mu, sigma)
        
        # M-step
        pi_new, mu_new, sigma_new = m_step(data, responsibilities)
        
        # Check for convergence
        if jnp.all(jnp.abs(mu_new - mu) < tol):
            break
        
        pi, mu, sigma = pi_new, mu_new, sigma_new
        
    return pi, mu, sigma


def plot_gmm(data, pi, mu, sigma, filename='gmm_clustering.png'):
    """
    Plot the GMM model along with the data and save the figure to a file.
    data: Input data, shape (N, D)
    pi: Mixing proportions, shape (K,)
    mu: Means of the Gaussians, shape (K, D)
    sigma: Covariance matrices of the Gaussians, shape (K, D, D)
    filename: Name of the file to save the plot, with file extension
    """
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=5, label='Data Points')
    
    ax = plt.gca()
    for k in range(len(pi)):
        eigvals, eigvecs = np.linalg.eigh(sigma[k])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(mu[k], 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]), angle=angle)
        ell.set_alpha(0.2)
        ell.set_facecolor('red')
        ax.add_artist(ell)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('GMM Clustering')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(filename)
    
    print(f"Figure saved to {filename}")


def main():
    # Test data generation (2D points)
    N = 300  # Number of data points
    D = 2  # Dimensionality of the data
    K = 3  # Number of clusters

    # Generate synthetic data for testing
    true_mu = jnp.array([
        [2.0, 3.0],
        [7.0, 5.0],
        [6.0, 10.0]
    ])
    true_sigma = jnp.array([
        [[1.0,  0.5], [ 0.5, 1.0]],
        [[1.5,  0.2], [ 0.2, 0.8]],
        [[1.0, -0.6], [-0.6, 1.0]]
    ])
    data = jnp.vstack(
        [random.multivariate_normal(random.PRNGKey(k), true_mu[k], true_sigma[k],(N // K,))
         for k in range(K)])

    def time_gmm(backend, jit_compile):
        jax.config.update("jax_platform_name", backend)
        
        start_time = time.time()
        
        if jit_compile:
            gmm_em(data, K)
        else:
            with jax.disable_jit():
                gmm_em(data, K)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(
            f"Backend: {backend}, "
            f"JIT: {'Yes' if jit_compile else 'No'}, "
            f"Time: {elapsed_time:.5f} seconds"
        )


    # Timing the code for different combinations
    for backend in ["cpu", "gpu"]:
        for jit_compile in [False, True]:
            time_gmm(backend, jit_compile)
            
    # Fit the GMM model
    # pi, mu, sigma = gmm_em(data, K)

    # Plot the GMM along with the data
    # plot_gmm(data, pi, mu, sigma)




if __name__ == '__main__':
    main()
