"""Statistical utilities for SAE feature analysis.

Includes vectorized Phi coefficient for co-occurrence and power-law 
fitting for manifold geometry.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def compute_phi_matrix(firing_history: mx.array) -> mx.array:
    """Compute the Phi coefficient matrix for feature co-occurrence.
    
    Args:
        firing_history: Binary matrix (N_docs, N_features) where 1 means fired.
        
    Returns:
        Phi matrix (N_features, N_features).
    """
    n_docs = firing_history.shape[0]
    
    # n11: Both fire (N_features, N_features)
    n11 = mx.matmul(firing_history.T, firing_history)
    
    # Individual counts (N_features,)
    counts = firing_history.sum(axis=0)
    
    # Contingency table components
    # n10: A fires, B doesn't
    n10 = counts[:, None] - n11
    # n01: B fires, A doesn't
    n01 = counts[None, :] - n11
    # n00: Neither fire
    n00 = n_docs - (n11 + n10 + n01)
    
    # Denominator components (marginal totals)
    r1 = (n11 + n10)  # Total A fires
    r0 = (n01 + n00)  # Total A doesn't
    c1 = (n11 + n01)  # Total B fires
    c0 = (n10 + n00)  # Total B doesn't
    
    # Compute Phi: (n11*n00 - n10*n01) / sqrt(r1 * r0 * c1 * c0)
    numerator = (n11 * n00) - (n10 * n01)
    denominator = mx.sqrt(r1 * r0 * c1 * c0)
    
    # Avoid division by zero for features that never fire
    phi = mx.where(denominator > 1e-9, numerator / denominator, 0.0)
    
    return phi


def fit_power_law(activations: mx.array) -> tuple[float, mx.array]:
    """Fit a power law to the eigenvalue spectrum of the covariance matrix.
    
    Args:
        activations: (N_samples, D_dim) matrix.
        
    Returns:
        (slope, eigenvalues)
    """
    # Center activations
    mean = activations.mean(axis=0)
    x = activations - mean
    
    # SVD for eigenvalues (using covariance Sigma = X.T @ X / N)
    # We use NumPy because MLX SVD is not yet supported on the GPU,
    # and switching streams in MLX can be version-dependent.
    # We cast to float32 because NumPy linalg doesn't support float16/bfloat16.
    import numpy as np
    x_np = np.array(x.astype(mx.float32), copy=False)
    s_np = np.linalg.svd(x_np, compute_uv=False)
    s = mx.array(s_np)
    # Fit log-log slope: log(lambda_i) = -alpha * log(i) + C
    # We take only non-zero eigenvalues and use NumPy for the log-fit
    lambdas_np = np.array(s_np**2 / x.shape[0], copy=False)
    valid_mask = lambdas_np > 1e-12
    l_valid = lambdas_np[valid_mask]
    indices = np.arange(1, len(l_valid) + 1)
    
    log_i = np.log(indices)
    log_L = np.log(l_valid)
    
    # polyfit returns [slope, intercept]
    coeffs = np.polyfit(log_i, log_L, deg=1)
    alpha = -float(coeffs[0])
    
    return alpha, mx.array(lambdas_np)


def clustering_entropy(lambdas: mx.array) -> float:
    """Calculate the 'negentropy' (clustering entropy) of the spectrum.
    
    Measures how much lower the entropy is than a Gaussian with 
    the same covariance.
    """
    # Normalize lambdas to represent variance distribution
    total_var = mx.sum(lambdas)
    p = lambdas / total_var
    
    # Shannon entropy of the spectrum
    entropy = -mx.sum(p * mx.log(p + 1e-12))
    
    # Max entropy is for uniform distribution (Gaussian white noise)
    max_entropy = mx.log(len(lambdas))
    
    return float((max_entropy - entropy).item())
