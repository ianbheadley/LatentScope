"""High-level analysis tools for Sparse Autoencoders.

Focuses on Lobe detection (functional modularity) and Manifold 
analysis (geometric bottleneck).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from latent_scope.engine.stats import compute_phi_matrix, fit_power_law, clustering_entropy

if TYPE_CHECKING:
    from latent_scope.core import HookedModel
    from latent_scope.engine.sae import SAE


@dataclass
class LobeResult:
    """Summary of a functional lobe detection run."""
    phi_matrix: mx.array
    clusters: dict[int, list[int]]  # cluster_id -> feature_indices
    cluster_labels: np.ndarray       # (N_features,)
    cluster_profiles: dict[int, list[str]] # cluster_id -> top_tokens
    feature_profiles: list[list[str]] # List of top tokens per feature (indexed by feature_id)


@dataclass
class ManifoldResult:
    """Summary of a manifold geometric analysis run."""
    slope: float
    entropy: float
    eigenvalues: mx.array


class SAEAnalyzer:
    """Analyze SAE features for modularity and geometric structure."""
    
    def __init__(self, model: HookedModel, sae: SAE):
        self.model = model
        self.sae = sae

    def run_lobe_detection(
        self, 
        texts: list[str], 
        layer: int, 
        n_clusters: int = 5
    ) -> LobeResult:
        """Detect functional modularity using co-occurrence (Phi)."""
        firing_history = []
        
        # 1. Collect firing history (N_features, N_tokens)
        for text in texts:
            cache = self.model.run_with_cache(text, layers=[layer])
            # (1, seq, dim)
            acts = mx.array(cache[f"model.layers.{layer}"])
            # Feature activations: (seq, N_features)
            features = self.sae.encode(acts[0])
            # Binary firing mask
            fired = (features > 0).astype(mx.float32)
            firing_history.append(fired)
            
        # Concatenate: (Total_tokens, N_features)
        history = mx.concatenate(firing_history, axis=0)
        
        # 2. Compute Phi matrix
        phi = compute_phi_matrix(history)
        
        # 3. Spectral Clustering (using SciPy/Sklearn for robust clustering)
        phi_np = np.asarray(phi)
        
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            random_state=42
        )
        # Use abs(phi) as similarity matrix
        labels = sc.fit_predict(np.abs(phi_np))
        
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(i)
            
        # 4. Profiling: Identify what each Lobe represents
        # Get all tokens across documents
        all_token_texts = []
        for text in texts:
            ids = self.model.tokenizer.encode(text)
            # Decode each token individually to keep alignment
            tokens = [self.model.tokenizer.decode([i]) for i in ids]
            all_token_texts.extend(tokens)
            
        cluster_profiles = {}
        for cluster_id, feature_indices in clusters.items():
            # (Tokens, Cluster_Features)
            cluster_feats = history[:, feature_indices]
            # Average feature activation for this cluster per token
            token_scores = cluster_feats.mean(axis=1)
            
            # Find top-5 tokens that activate this lobe
            top_idx = np.argsort(-np.asarray(token_scores))[:7]
            
            seen = set()
            top_tokens = []
            for idx in top_idx:
                if idx < len(all_token_texts):
                    t = all_token_texts[int(idx)].strip()
                    if t and t not in seen:
                        top_tokens.append(t)
                        seen.add(t)
            
            cluster_profiles[cluster_id] = top_tokens[:5]
            
        # 5. Feature-level Profiling: Calculate top tokens for every single dot
        feature_profiles = []
        # Get top-5 token indices for every feature (vectorized)
        f_hist_np = np.asarray(history)
        # (5, N_features)
        top_indices = np.argsort(-f_hist_np, axis=0)[:5, :]
        
        for f_idx in range(f_hist_np.shape[1]):
            f_top_idx = top_indices[:, f_idx]
            f_tokens = []
            for t_idx in f_top_idx:
                # Only include if it actually fired (activation > 0)
                if f_hist_np[t_idx, f_idx] > 1e-6 and t_idx < len(all_token_texts):
                    tok = all_token_texts[int(t_idx)].strip()
                    if tok and tok not in f_tokens:
                        f_tokens.append(tok)
            feature_profiles.append(f_tokens)
            
        return LobeResult(
            phi_matrix=phi,
            clusters=clusters,
            cluster_labels=labels,
            cluster_profiles=cluster_profiles,
            feature_profiles=feature_profiles
        )

    def analyze_manifold(self, texts: list[str], layer: int) -> ManifoldResult:
        """Analyze the 'fractal cucumber' shape of the activations."""
        all_acts = []
        
        for text in texts:
            cache = self.model.run_with_cache(text, layers=[layer])
            all_acts.append(mx.array(cache[f"model.layers.{layer}"])[0])
            
        # Combine all tokens: (Total_tokens, Dim)
        data = mx.concatenate(all_acts, axis=0)
        
        # Fit power law
        slope, eigenvalues = fit_power_law(data)
        
        # Compute negentropy
        entropy = clustering_entropy(eigenvalues)
        
        return ManifoldResult(
            slope=slope,
            entropy=entropy,
            eigenvalues=eigenvalues
        )
