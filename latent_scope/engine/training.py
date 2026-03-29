"""Training infrastructure for native MLX SAEs.

Includes high-performance activation buffering and optimization loops.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

if TYPE_CHECKING:
    from latent_scope.core import HookedModel
    from latent_scope.engine.sae import SAE


class ActivationBuffer:
    """Buffer to collect and shuffle activations from a model.
    
    Prevents correlations within a sequence from biasing the SAE weights.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.data: list[mx.array] = []
        self._current_size = 0

    def add(self, activations: mx.array):
        """Add activations (batch, seq, dim) to the buffer."""
        # Reshape to (batch * seq, dim)
        flat = activations.reshape(-1, activations.shape[-1])
        self.data.append(flat)
        self._current_size += flat.shape[0]

    def shuffle(self):
        """Shuffle the current buffer content."""
        if not self.data:
            return
        
        # Concatenate and shuffle
        full = mx.concatenate(self.data, axis=0)
        indices = mx.array(random.sample(range(full.shape[0]), full.shape[0]))
        shuffled = full[indices]
        
        # Redistribute back (optional, or just use as is)
        self.data = [shuffled]

    def get_batch(self, batch_size: int) -> Iterable[mx.array]:
        """Yield batches of activations."""
        if not self.data:
            return
            
        full = mx.concatenate(self.data, axis=0)
        n = full.shape[0]
        
        for i in range(0, n, batch_size):
            yield full[i:i + batch_size]
            
        # Clear after yielding
        self.data = []
        self._current_size = 0


class SAETrainer:
    """Training loop for a native MLX SAE."""
    
    def __init__(
        self, 
        sae: SAE, 
        learning_rate: float = 3e-4, 
        l1_alpha: float = 3e-4
    ):
        self.sae = sae
        self.optimizer = opt.Adam(learning_rate=learning_rate)
        self.l1_alpha = l1_alpha

    def loss_fn(self, model, x):
        """Reconstruction MSE + L1 Regularization."""
        x_hat, features = model(x)
        
        # MSE
        mse = mx.mean((x - x_hat)**2)
        
        # L1 (sparsity)
        l1 = mx.mean(mx.sum(mx.abs(features), axis=-1))
        
        return mse + self.l1_alpha * l1

    def train_step(self, x: mx.array):
        """Run a single optimization step."""
        loss_and_grads = nn.value_and_grad(self.sae, self.loss_fn)
        loss, grads = loss_and_grads(self.sae, x)
        self.optimizer.update(self.sae, grads)
        mx.eval(self.sae.parameters(), self.optimizer.state)
        return float(loss.item())

    def train_on_corpus(
        self, 
        hooked_model: HookedModel, 
        layer: int, 
        texts: list[str], 
        batch_size: int = 128,
        buffer_capacity: int = 50000
    ):
        """Train the SAE on a corpus of text."""
        buffer = ActivationBuffer(capacity=buffer_capacity)
        path = f"model.layers.{layer}"
        
        losses = []
        
        for text in texts:
            # Collect activations
            cache = hooked_model.run_with_cache(text, layers=[layer])
            # (1, seq, dim)
            acts = mx.array(cache[path])
            buffer.add(acts)
            
            # If buffer is full, train
            if buffer._current_size >= buffer_capacity:
                buffer.shuffle()
                for batch in buffer.get_batch(batch_size):
                    loss = self.train_step(batch)
                    losses.append(loss)
        
        # Final cleanup pass
        buffer.shuffle()
        for batch in buffer.get_batch(batch_size):
            loss = self.train_step(batch)
            losses.append(loss)
            
        return losses
