"""Native MLX implementation of Sparse Autoencoders (SAE).

Includes JumpReLU gating as described in the "Gemma Scope" release.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class JumpReLU(nn.Module):
    """JumpReLU activation function.
    
    f(x) = ReLU(x) * (x > threshold)
    """
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def __call__(self, x: mx.array) -> mx.array:
        # We use a hard gate for inference/analysis. 
        # For training, a straight-through estimator or soft approximation 
        # would be needed, but we focus on the core architecture first.
        return mx.where(x > self.threshold, mx.maximum(x, 0), 0)


class SAE(nn.Module):
    """Sparse Autoencoder with optional JumpReLU gating.
    
    Architecture:
        x_cent = x - b_pre
        h = Activation(W_enc @ x_cent + b_enc)
        x_hat = W_dec @ h + b_dec
    """
    
    def __init__(
        self, 
        input_dim: int, 
        dict_size: int, 
        jump_relu_threshold: float | None = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        
        # Parameters
        self.W_enc = mx.random.normal((input_dim, dict_size)) * 0.02
        self.b_enc = mx.zeros((dict_size,))
        
        self.W_dec = mx.random.normal((dict_size, input_dim)) * 0.02
        self.b_dec = mx.zeros((input_dim,))
        
        self.b_pre = mx.zeros((input_dim,))
        
        # Gating
        if jump_relu_threshold is not None:
            self.activation = JumpReLU(threshold=jump_relu_threshold)
        else:
            self.activation = nn.ReLU()

    def encode(self, x: mx.array) -> mx.array:
        """Project input into sparse feature space."""
        x_cent = x - self.b_pre
        h = mx.matmul(x_cent, self.W_enc) + self.b_enc
        return self.activation(h)

    def decode(self, f: mx.array) -> mx.array:
        """Reconstruct input from sparse features."""
        return mx.matmul(f, self.W_dec) + self.b_dec

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Full forward pass. Returns (reconstruction, features)."""
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @property
    def l0(self) -> mx.array:
        """Estimate average L0 sparsity (conceptually, depends on input)."""
        # This is a placeholder; real L0 is computed during a forward pass.
        return mx.array(0.0)
