"""Unified projection engine — every linear 2D/3D projection picks a point on the
Grassmannian Gr(n, d).  Different methods pick different points.
We can interpolate between any two via geodesics.

Methods:
  pca           — eigenvectors of Σ (max variance)
  lda           — generalized eigenvectors of Σ_w⁻¹ Σ_b (max class separation)
  cpca(α)       — eigenvectors of Σ_fg − α·Σ_bg (contrastive)
  probe_aligned — pin one axis to a direction vector, PCA in orthogonal complement
  null_space    — erase a concept direction, then PCA on residual
  grassmannian  — geodesic interpolation between any two bases
"""

from __future__ import annotations
import numpy as np


# ── Low-level math ────────────────────────────────────────────────────────────

def _safe_svd(X: np.ndarray, n: int = 3):
    """SVD with fallback for degenerate matrices."""
    try:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        return S, Vt
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(42)
        Vt = rng.standard_normal((n, X.shape[1]))
        Vt /= np.linalg.norm(Vt, axis=1, keepdims=True)
        return np.ones(n), Vt


def _extract_basis(M: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Top-n eigenvectors of symmetric matrix M. Returns (eigenvalues, basis rows)."""
    try:
        eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(42)
        V = rng.standard_normal((n, M.shape[0]))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        return np.ones(n), V
    idx = np.argsort(-eigvals)[:n]
    return eigvals[idx], eigvecs[:, idx].T  # (n, d)


def _orthogonalize(V: np.ndarray) -> np.ndarray:
    """QR-orthogonalize rows of V. Returns (n, d) orthonormal rows."""
    Q, _ = np.linalg.qr(V.T)
    return Q[:, :V.shape[0]].T


def _to_sample_space(X: np.ndarray):
    """When n_samples << n_dims, project to sample span for fast operations."""
    N, D = X.shape
    if N >= D:
        return X, np.eye(D)
    S, Vt = _safe_svd(X, N)
    k = min(N, len(S))
    mask = S[:k] > 1e-10
    k = max(mask.sum(), 3)
    X_reduced = X @ Vt[:k].T  # (N, k)
    return X_reduced, Vt[:k]  # (k, D)


# ── Projection methods ────────────────────────────────────────────────────────

def _project_pca(X_centered: np.ndarray, n: int = 3):
    """PCA projection. Uses randomized SVD for large matrices."""
    N, D = X_centered.shape
    if N > 50 and D > 500:
        k = min(n + 10, N, D)
        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((D, k))
        Y = X_centered @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ X_centered
        _, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
        S, Vt = S_b, Vt_b
        total = float(np.sum(X_centered ** 2))
    else:
        S, Vt = _safe_svd(X_centered, n)
        total = float((S ** 2).sum()) or 1.0
    n = min(n, X_centered.shape[0], X_centered.shape[1], len(S))
    total = max(total, 1.0)
    var_ratio = [round(float(s) ** 2 / total, 6) for s in S[:n]]
    return Vt[:n], S, var_ratio


def _project_lda(X_centered: np.ndarray, labels: list[int], n: int = 3):
    """LDA: maximize between-class / within-class scatter."""
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return _project_pca(X_centered, n)

    X_r, Vt_map = _to_sample_space(X_centered)
    labels_arr = np.array(labels)
    d = X_r.shape[1]
    overall_mean = X_r.mean(axis=0)

    S_w = np.zeros((d, d), dtype=np.float64)
    S_b = np.zeros((d, d), dtype=np.float64)
    for c in unique_labels:
        mask = labels_arr == c
        Xc = X_r[mask]
        mean_c = Xc.mean(axis=0)
        diff_c = Xc - mean_c
        S_w += diff_c.T @ diff_c
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_b += mask.sum() * (mean_diff @ mean_diff.T)

    S_w += np.eye(d) * 1e-6
    try:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(S_w, S_b))
        idx = np.argsort(-eigvals)[:n]
        basis_r = eigvecs[:, idx].T
        ev = eigvals[idx]
    except np.linalg.LinAlgError:
        return _project_pca(X_centered, n)

    basis = _orthogonalize(basis_r @ Vt_map)
    total = float(np.abs(ev).sum()) or 1.0
    info = [round(float(np.abs(e)) / total, 6) for e in ev]
    return basis, ev, info


def _project_cpca(X_centered: np.ndarray, labels: list[int],
                   alpha: float = 1.0, target_label: int = -1, n: int = 3):
    """Contrastive PCA: Σ_target − α·Σ_background."""
    unique_labels = sorted(set(labels))
    labels_arr = np.array(labels)

    if target_label == -1:
        target_label = max(unique_labels, key=lambda c: (labels_arr == c).sum())
        bg_labels = [c for c in unique_labels if c != target_label]
    else:
        bg_labels = [c for c in unique_labels if c != target_label]

    fg_mask = np.isin(labels_arr, [c for c in unique_labels if c not in bg_labels])
    bg_mask = ~fg_mask

    if fg_mask.sum() < 2 or bg_mask.sum() < 2:
        return _project_pca(X_centered, n)

    X_r, Vt_map = _to_sample_space(X_centered)
    X_fg = X_r[fg_mask]
    X_bg = X_r[bg_mask]

    d = X_r.shape[1]
    Sigma_fg = (X_fg.T @ X_fg) / max(X_fg.shape[0] - 1, 1)
    Sigma_bg = (X_bg.T @ X_bg) / max(X_bg.shape[0] - 1, 1)

    M = Sigma_fg - alpha * Sigma_bg
    eigvals, basis_r = _extract_basis(M, n)
    basis = _orthogonalize(basis_r @ Vt_map)
    total = float(np.abs(eigvals).sum()) or 1.0
    info = [round(float(np.abs(e)) / total, 6) for e in eigvals]
    return basis, eigvals, info


def _project_probe_aligned(X_centered: np.ndarray, direction: np.ndarray, n: int = 3):
    """Pin first axis to a probe direction, PCA in orthogonal complement."""
    d = X_centered.shape[1]
    w = direction / (np.linalg.norm(direction) + 1e-12)
    proj_onto_w = X_centered @ w
    X_residual = X_centered - np.outer(proj_onto_w, w)
    S, Vt = _safe_svd(X_residual, n - 1)
    basis = np.zeros((n, d), dtype=np.float64)
    basis[0] = w
    m = min(n - 1, Vt.shape[0])
    basis[1:1 + m] = Vt[:m]
    total = float((S[:m] ** 2).sum()) or 1.0
    info = [0.0] + [round(float(s) ** 2 / total, 6) for s in S[:m]]
    return basis, S, info


def _project_null_space(X_centered: np.ndarray, direction: np.ndarray, n: int = 3):
    """Erase a concept direction, PCA on residual."""
    d = X_centered.shape[1]
    w = direction / (np.linalg.norm(direction) + 1e-12)
    X_erased = X_centered - np.outer(X_centered @ w, w)
    S, Vt = _safe_svd(X_erased, n)
    n_actual = min(n, X_erased.shape[0], X_erased.shape[1])
    total = float((S ** 2).sum()) or 1.0
    info = [round(float(s) ** 2 / total, 6) for s in S[:n_actual]]
    return Vt[:n_actual], S, info


def _grassmannian_interpolate(W_A: np.ndarray, W_B: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation on Gr(n, d) between two n-dim subspaces. t ∈ [0, 1]."""
    if t <= 0.0:
        return W_A.copy()
    if t >= 1.0:
        return W_B.copy()
    M = W_A @ W_B.T
    U, sigma, VhT = np.linalg.svd(M)
    sigma = np.clip(sigma, -1.0, 1.0)
    theta = np.arccos(sigma)
    W_A_aligned = U.T @ W_A
    W_B_aligned = VhT @ W_B
    Y = W_B_aligned - np.diag(sigma) @ W_A_aligned
    norms = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    Q = Y / norms
    W_t = np.diag(np.cos(theta * t)) @ W_A_aligned + np.diag(np.sin(theta * t)) @ Q
    return _orthogonalize(W_t)


# ── Public dispatcher ─────────────────────────────────────────────────────────

def compute_projection(X_centered: np.ndarray, method: str, params: dict,
                        labels: list[int], n: int = 3):
    """Unified projection dispatcher. Returns (basis, info_dict).

    basis: (n, hidden_dim) orthonormal rows — the projection axes.
    info_dict: {method, variance_explained, label}.
    """
    if method == "pca":
        basis, S, var_ratio = _project_pca(X_centered, n)
        return basis, {"method": "pca", "variance_explained": var_ratio,
                       "label": "PCA"}

    elif method == "lda":
        basis, eigvals, info = _project_lda(X_centered, labels, n)
        return basis, {"method": "lda", "variance_explained": info,
                       "label": "LDA (class separation)"}

    elif method == "cpca":
        alpha = params.get("alpha", 1.0)
        target = params.get("target_group", -1)
        basis, eigvals, info = _project_cpca(X_centered, labels, alpha, target, n)
        return basis, {"method": "cpca", "variance_explained": info,
                       "label": f"cPCA (α={alpha:.1f})"}

    elif method == "probe_aligned":
        direction = np.array(params.get("direction", []), dtype=np.float64)
        if direction.size != X_centered.shape[1]:
            return compute_projection(X_centered, "pca", {}, labels, n)
        basis, S, info = _project_probe_aligned(X_centered, direction, n)
        return basis, {"method": "probe_aligned", "variance_explained": info,
                       "label": "Probe-Aligned"}

    elif method == "null_space":
        direction = np.array(params.get("direction", []), dtype=np.float64)
        if direction.size != X_centered.shape[1]:
            return compute_projection(X_centered, "pca", {}, labels, n)
        basis, S, info = _project_null_space(X_centered, direction, n)
        return basis, {"method": "null_space", "variance_explained": info,
                       "label": "Null-Space (concept erased)"}

    elif method == "grassmannian":
        method_a = params.get("method_a", "pca")
        method_b = params.get("method_b", "lda")
        t = params.get("t", 0.5)
        params_a = params.get("params_a", {})
        params_b = params.get("params_b", {})
        basis_a, _ = compute_projection(X_centered, method_a, params_a, labels, n)
        basis_b, _ = compute_projection(X_centered, method_b, params_b, labels, n)
        basis = _grassmannian_interpolate(basis_a, basis_b, t)
        return basis, {"method": "grassmannian",
                       "variance_explained": [0.0] * n,
                       "label": f"Grassmannian ({method_a}→{method_b}, t={t:.2f})"}

    return compute_projection(X_centered, "pca", {}, labels, n)
