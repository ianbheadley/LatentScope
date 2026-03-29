"""Flask application factory for Latent Scope."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from typing import Optional, Dict, List

import numpy as np

from flask import Flask, Response, jsonify, request, stream_with_context

from latent_scope.runtime import SharedModelRuntime
from latent_scope.engine.projections import compute_projection

# Optional: weight interpreter for vocab projections
try:
    from latent_scope.engine.weight_interpreter import WeightInterpreter
    _HAS_WEIGHT_INTERP = True
except ImportError:
    _HAS_WEIGHT_INTERP = False


# ── Data directory ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
PRESETS_FILE = os.path.join(DATA_DIR, "truth_domains.json")


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip() or "file"


# ── Per-process state (stateful activation cache) ─────────────────────────────

class WorkspaceState:
    """Thread-safe holder for the activation cache and projection state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.raw_matrix: Optional[np.ndarray] = None     # (n_items*n_layers, hidden_dim)
        self.raw_items: Optional[List[dict]] = None
        self.num_layers: int = 0
        self.act_cache: Optional[Dict[str, Dict[int, np.ndarray]]] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_vt: Optional[np.ndarray] = None
        self.current_basis: Optional[np.ndarray] = None
        self.group_labels: Optional[List[int]] = None
        self.group_names: Optional[List[str]] = None
        self.W_U: Optional[np.ndarray] = None
        self.truth_axis: Optional[np.ndarray] = None     # calibrated truth direction


_state = WorkspaceState()


# ── SSE streaming helper ───────────────────────────────────────────────────────

def _sse(q: queue.Queue):
    """Drain a queue and yield SSE-formatted strings."""
    while True:
        item = q.get()
        if item is None:
            break
        try:
            payload = json.dumps(item)
        except Exception as e:
            import traceback
            traceback.print_exc()
            payload = json.dumps({"type": "error", "message": f"Server JSON error: {str(e)}"})
        yield f"data: {payload}\n\n"


def _run_in_thread(fn, q: queue.Queue):
    def wrapper():
        try:
            fn(q)
        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)
    t = threading.Thread(target=wrapper, daemon=True)
    t.start()


# ── Application factory ────────────────────────────────────────────────────────

def create_app(runtime: SharedModelRuntime) -> Flask:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    app = Flask(__name__, static_folder=None)

    # ── Pages ──────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        from latent_scope.ui.workspace import render_workspace
        return render_workspace(runtime.current_model_name())

    # ── Model management ───────────────────────────────────────────────────

    @app.route("/api/model/load", methods=["POST"])
    def load_model():
        model_id = (request.json or {}).get("model_id", "").strip()
        if not model_id:
            return jsonify({"error": "model_id required"}), 400

        def _load(q):
            q.put({"type": "status", "message": f"Loading {model_id}…"})
            runtime.load(model_id)
            q.put({"type": "status", "message": f"Loaded: {model_id}"})
            q.put({"type": "loaded", "model": model_id,
                   "num_layers": runtime.model.num_layers,
                   "hidden_dim": runtime.model.hidden_dim})

        q: queue.Queue = queue.Queue()
        _run_in_thread(_load, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    @app.route("/api/model/unload", methods=["POST"])
    def unload_model():
        runtime.unload()
        _state.act_cache = None
        _state.raw_matrix = None
        return jsonify({"status": "unloaded"})

    @app.route("/api/models")
    def list_models():
        return jsonify({"models": runtime.refresh_models()})

    @app.route("/api/model/status")
    def model_status():
        name = runtime.current_model_name()
        return jsonify({
            "loaded": name is not None,
            "model": name,
            "num_layers": runtime.model.num_layers if runtime.model else 0,
            "hidden_dim": runtime.model.hidden_dim if runtime.model else 0,
        })

    # ── Encoding ───────────────────────────────────────────────────────────

    @app.route("/api/encode", methods=["POST"])
    def encode():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _encode(q):
            model = runtime.model
            raw_groups = params.get("groups", [])
            if not raw_groups:
                q.put({"type": "error", "message": "No groups provided."})
                return

            items: list[dict] = []
            for g in raw_groups:
                gname = (g.get("name") or "Group").strip()
                color = g.get("color") or "#1f77b4"
                for line in (g.get("items") or "").split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if "|" in line:
                        label, text = line.split("|", 1)
                        label, text = label.strip(), text.strip()
                    else:
                        label = text = line
                    if label and text:
                        items.append({"label": label, "text": text,
                                      "group": gname, "color": color})

            if len(items) < 2:
                q.put({"type": "error", "message": "Provide at least 2 items across all groups."})
                return

            num_layers = model.num_layers
            q.put({"type": "status", "message": f"Encoding {len(items)} items…"})

            all_layers = list(range(num_layers))
            all_acts: list[dict[int, np.ndarray]] = []
            for i, item in enumerate(items):
                q.put({"type": "status", "message": f"Encoding {i + 1}/{len(items)}: {item['label'][:40]}"})
                cache = model.run_with_cache(item["text"], layers=all_layers)
                layer_vecs: dict[int, np.ndarray] = {}
                for ll in all_layers:
                    path = f"model.layers.{ll}"
                    if path not in cache:
                        continue
                    acts = cache[path]
                    vec = np.asarray(acts[0, -1, :], dtype=np.float64)
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_vecs[ll] = vec
                all_acts.append(layer_vecs)

            with _state.lock:
                _state.act_cache = {items[ii]["label"]: all_acts[ii] for ii in range(len(items))}

            # Cache W_U once
            if _state.W_U is None and _HAS_WEIGHT_INTERP:
                try:
                    interp = WeightInterpreter(model)
                    if interp.W_U is not None:
                        _state.W_U = np.asarray(interp.W_U, dtype=np.float32)
                except Exception:
                    pass

            q.put({"type": "status", "message": "Computing projection…"})

            n_items = len(items)
            hidden_dim = next(iter(all_acts[0].values())).shape[0]
            matrix = np.zeros((n_items * num_layers, hidden_dim), dtype=np.float32)
            for ii in range(n_items):
                for ll in range(num_layers):
                    matrix[ii * num_layers + ll] = all_acts[ii][ll]

            seen_groups: dict[str, int] = {}
            group_labels = []
            for it in items:
                if it["group"] not in seen_groups:
                    seen_groups[it["group"]] = len(seen_groups)
                group_labels.append(seen_groups[it["group"]])

            with _state.lock:
                _state.raw_matrix = matrix
                _state.raw_items = items
                _state.num_layers = num_layers
                _state.group_labels = group_labels
                _state.group_names = list(seen_groups.keys())

            X = matrix.astype(np.float64)
            mean = X.mean(axis=0, keepdims=True)
            X_centered = X - mean
            basis, info = compute_projection(X_centered, "pca", {}, group_labels, n=3)
            coords = X_centered @ basis.T
            var_ratio = info.get("variance_explained", [0, 0, 0])

            with _state.lock:
                _state.pca_mean = mean[0]
                _state.pca_vt = basis
                _state.current_basis = basis

            frames, trajectories = _build_frames(items, coords, num_layers)

            # Analytics
            analytics = _compute_analytics(matrix, items, group_labels, list(seen_groups.keys()), num_layers)

            q.put({
                "type": "result",
                "frames": frames,
                "trajectories": trajectories,
                "variance_explained": var_ratio,
                "n_layers": num_layers,
                "n_items": n_items,
                "groups": list({it["group"]: it["color"] for it in items}.items()),
                "item_labels": [it["label"] for it in items],
                **analytics,
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_encode, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Reproject ──────────────────────────────────────────────────────────

    @app.route("/api/reproject", methods=["POST"])
    def reproject():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _reproject(q):
            with _state.lock:
                matrix = _state.raw_matrix
                items = _state.raw_items
                num_layers = _state.num_layers
                group_labels = _state.group_labels or []
                group_names = _state.group_names or []
                act_cache = _state.act_cache or {}

            if matrix is None or items is None:
                q.put({"type": "error", "message": "No cached activations. Run encoding first."})
                return

            method = params.get("method", "pca")
            proj_params = params.get("proj_params", {})
            n_items = len(items)
            hidden_dim = matrix.shape[1]

            basis_layer = proj_params.get("basis_layer", num_layers - 1)
            basis_layer = max(0, min(basis_layer, num_layers - 1))

            uses_labels = method in ("lda", "cpca", "probe_aligned", "null_space", "grassmannian")
            direction_source = ""

            if method in ("probe_aligned", "null_space"):
                ops = proj_params.get("direction_ops", [])
                if ops and act_cache:
                    direction = np.zeros(hidden_dim, dtype=np.float64)
                    parts = []
                    for op in ops:
                        text = op.get("text", "")
                        sign = 1.0 if op.get("op") == "+" else -1.0
                        if text in act_cache and basis_layer in act_cache[text]:
                            direction += sign * act_cache[text][basis_layer]
                        else:
                            cache = runtime.model.run_with_cache(text, layers=[basis_layer])
                            path = f"model.layers.{basis_layer}"
                            if path in cache:
                                direction += sign * np.asarray(cache[path][0, -1, :], dtype=np.float64)
                        parts.append(f"{op.get('op','+')} {text}")
                    proj_params["direction"] = direction.tolist()
                    direction_source = " ".join(parts)
                elif len(set(group_labels)) >= 2:
                    groups_sorted = sorted(set(group_labels))
                    g0, g1 = groups_sorted[0], groups_sorted[1]
                    vecs_g0 = [matrix[ii * num_layers + basis_layer] for ii, gl in enumerate(group_labels) if gl == g0]
                    vecs_g1 = [matrix[ii * num_layers + basis_layer] for ii, gl in enumerate(group_labels) if gl == g1]
                    if vecs_g0 and vecs_g1:
                        c0 = np.mean(vecs_g0, axis=0).astype(np.float64)
                        c1 = np.mean(vecs_g1, axis=0).astype(np.float64)
                        proj_params["direction"] = (c0 - c1).tolist()
                        g0_name = group_names[g0] if g0 < len(group_names) else f"G{g0}"
                        g1_name = group_names[g1] if g1 < len(group_names) else f"G{g1}"
                        direction_source = f"centroid({g0_name}) − centroid({g1_name})"

            if uses_labels and method != "pca":
                X_basis = np.array([matrix[ii * num_layers + basis_layer] for ii in range(n_items)], dtype=np.float64)
                X_basis_centered = X_basis - X_basis.mean(axis=0, keepdims=True)
                basis, info = compute_projection(X_basis_centered, method, proj_params, group_labels, n=3)
            else:
                X_all = matrix.astype(np.float64)
                mean_all = X_all.mean(axis=0, keepdims=True)
                X_all_centered = X_all - mean_all
                basis, info = compute_projection(X_all_centered, "pca", {}, group_labels, n=3)

            X_full = matrix.astype(np.float64)
            full_mean = X_full.mean(axis=0, keepdims=True)
            coords = (X_full - full_mean) @ basis.T

            with _state.lock:
                _state.pca_mean = full_mean[0]
                _state.pca_vt = basis
                _state.current_basis = basis

            frames, trajectories = _build_frames(items, coords, num_layers)
            proj_label = info.get("label", method)
            if uses_labels:
                proj_label += f" @ layer {basis_layer}"
            if direction_source:
                proj_label += f" | dir: {direction_source}"

            q.put({
                "type": "reproject_result",
                "frames": frames,
                "trajectories": trajectories,
                "variance_explained": info.get("variance_explained", [0, 0, 0]),
                "projection_label": proj_label,
                "method": method,
                "basis_layer": basis_layer,
                "direction_source": direction_source,
                "n_items": n_items,
                "n_layers": num_layers,
                "group_names": group_names,
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_reproject, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Lens (vocabulary projection) ───────────────────────────────────────

    @app.route("/api/lens", methods=["POST"])
    def lens():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}
        label = params.get("label", "")
        layer_idx = params.get("layer", 0)

        with _state.lock:
            act_cache = _state.act_cache
            W_U = _state.W_U

        if act_cache is None or W_U is None:
            return jsonify({"error": "Run encoding first and ensure model has W_U"}), 400

        vecs = act_cache.get(label)
        if vecs is None or layer_idx not in vecs:
            return jsonify({"error": f"No cache for '{label}' at layer {layer_idx}"}), 404

        vec = vecs[layer_idx].astype(np.float32)
        logits = vec @ W_U.T
        top_idx = np.argpartition(-logits, 5)[:5]
        top_idx = top_idx[np.argsort(-logits[top_idx])]
        tokens = [{"token": runtime.model.tokenizer.decode([int(i)]), "score": round(float(logits[int(i)]), 2)}
                  for i in top_idx]
        return jsonify({"label": label, "tokens": tokens})

    # ── Vector math ────────────────────────────────────────────────────────

    @app.route("/api/vector_math", methods=["POST"])
    def vector_math():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _math(q):
            import mlx.core as mx
            model = runtime.model
            ops = params.get("ops", [])
            layer_idx = params.get("layer", 0)

            result_vec = mx.zeros((model.hidden_dim,), dtype=mx.float32)
            for op in ops:
                text = op.get("text")
                if not text:
                    continue
                sign = 1.0 if op.get("op") == "+" else -1.0
                cache = model.run_with_cache(text, layers=[layer_idx])
                path = f"model.layers.{layer_idx}"
                if path in cache:
                    vec = mx.array(cache[path][0, -1, :]).astype(mx.float32)
                    result_vec = result_vec + sign * vec

            coord = None
            with _state.lock:
                if _state.pca_mean is not None and _state.pca_vt is not None:
                    vec_np = np.asarray(result_vec, dtype=np.float64)
                    coord = [round(float(x), 5) for x in (vec_np - _state.pca_mean) @ _state.pca_vt[:3].T]

            tokens = []
            if _state.W_U is not None:
                import mlx.core as mx
                rv_f32 = mx.array(np.asarray(result_vec, dtype=np.float32))
                W_U_mx = mx.array(_state.W_U)
                logits_mx = rv_f32 @ W_U_mx.T
                logits = np.asarray(logits_mx)
                top_idx = np.argpartition(-logits, 5)[:5]
                top_idx = top_idx[np.argsort(-logits[top_idx])]
                tokens = [{"token": model.tokenizer.decode([int(i)]), "score": float(logits[int(i)])} for i in top_idx]

            q.put({"type": "math_result", "tokens": tokens, "coord": coord})

        q: queue.Queue = queue.Queue()
        _run_in_thread(_math, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Sessions ───────────────────────────────────────────────────────────

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        sessions = [f[:-5] for f in sorted(os.listdir(SESSIONS_DIR)) if f.endswith(".json")]
        return jsonify({"sessions": sessions})

    @app.route("/api/sessions/<name>", methods=["GET"])
    def load_session(name):
        filepath = os.path.join(SESSIONS_DIR, f"{_safe_filename(name)}.json")
        if not os.path.exists(filepath):
            return jsonify({"error": "Session not found"}), 404
        with open(filepath) as f:
            return jsonify({"session": json.load(f)})

    @app.route("/api/sessions/<name>", methods=["POST"])
    def save_session(name):
        session = (request.json or {}).get("session", {})
        safe = _safe_filename(name)
        filepath = os.path.join(SESSIONS_DIR, f"{safe}.json")
        with open(filepath, "w") as f:
            json.dump(session, f, indent=2)
        return jsonify({"saved": safe})

    @app.route("/api/sessions/folder", methods=["POST"])
    def open_sessions_folder():
        if sys.platform == "darwin":
            subprocess.Popen(["open", SESSIONS_DIR])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", SESSIONS_DIR])
        else:
            subprocess.Popen(["xdg-open", SESSIONS_DIR])
        return jsonify({"status": "opened"})

    # ── Presets ────────────────────────────────────────────────────────────

    @app.route("/api/presets", methods=["GET"])
    def list_presets():
        if not os.path.exists(PRESETS_FILE):
            return jsonify({"presets": []})
        with open(PRESETS_FILE) as f:
            data = json.load(f)
        return jsonify({"presets": [{"key": p.get("key"), "name": p.get("name"),
                                     "description": p.get("description")} for p in data]})

    @app.route("/api/presets/<key>", methods=["GET"])
    def load_preset(key):
        if not os.path.exists(PRESETS_FILE):
            return jsonify({"error": "No presets file"}), 404
        with open(PRESETS_FILE) as f:
            data = json.load(f)
        preset = next((p for p in data if p.get("key") == key), None)
        if not preset:
            return jsonify({"error": "Preset not found"}), 404
        colors = {"true": "#1f77b4", "false": "#ff7f0e", "gibberish": "#2ca02c"}
        groups = []
        for cat in ["true", "false", "gibberish"]:
            if cat in preset:
                groups.append({
                    "name": f"{preset.get('name', 'Preset')}: {cat.capitalize()}",
                    "color": colors[cat],
                    "items": "\n".join(preset[cat]),
                })
        return jsonify({"groups": groups})

    # ── Operator Extraction ────────────────────────────────────────────────

    @app.route("/api/extract_operator", methods=["POST"])
    def extract_operator():
        params = request.json or {}
        name = params.get("name", "extracted_op")
        layer = params.get("layer", -1)
        
        with _state.lock:
            basis = _state.current_basis
            
        if basis is None or len(basis) == 0:
            return jsonify({"error": "No active projection in memory to extract."}), 400
            
        op_dir = os.path.join(DATA_DIR, "operators")
        os.makedirs(op_dir, exist_ok=True)
        
        from latent_scope.engine.operator_miner import Operator
        op = Operator(name=name, vector=basis[0], layer=layer)
        filepath = os.path.join(op_dir, f"{name}.npz")
        op.save_to_file(filepath)
        
        return jsonify({"status": "success", "filepath": filepath})

    @app.route("/api/operators", methods=["GET"])
    def list_operators():
        op_dir = os.path.join(DATA_DIR, "operators")
        if not os.path.exists(op_dir):
            return jsonify({"operators": []})
        ops = [f[:-4] for f in os.listdir(op_dir) if f.endswith(".npz")]
        return jsonify({"operators": sorted(ops)})

    # ── Truth axis calibration ─────────────────────────────────────────────

    @app.route("/api/truth_axis/calibrate", methods=["POST"])
    def calibrate_truth_axis():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _calibrate(q):
            from latent_scope.engine.truth_axis import TruthAxis
            ta = TruthAxis(runtime.model)
            true_stmts = params.get("true_statements", [])
            false_stmts = params.get("false_statements", [])
            layer = params.get("layer", runtime.model.num_layers // 2)

            q.put({"type": "status", "message": "Calibrating truth axis…"})
            result = ta.calibrate(true_stmts, false_stmts, layer=layer)

            with _state.lock:
                _state.truth_axis = np.asarray(result.direction, dtype=np.float64)

            q.put({
                "type": "truth_axis_result",
                "separation": round(float(result.separation), 4),
                "layer": layer,
                "direction": result.direction.tolist() if hasattr(result.direction, "tolist") else list(result.direction),
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_calibrate, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Pivot probe ────────────────────────────────────────────────────────

    @app.route("/api/pivot_probe/run", methods=["POST"])
    def run_pivot_probe():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _probe(q):
            from latent_scope.engine.pivot_probe import PivotProbe
            pp = PivotProbe(runtime.model)
            concept_a = params.get("concept_a", "")
            concept_b = params.get("concept_b", "")
            layer = params.get("layer", runtime.model.num_layers // 2)

            q.put({"type": "status", "message": f"Running pivot probe: {concept_a} vs {concept_b}…"})
            result = pp.probe(concept_a, concept_b, layer=layer)

            q.put({
                "type": "pivot_result",
                "spearman_rho": round(float(result.spearman_rho), 4),
                "null_axis_beat": result.null_axis_beat,
                "intruder_ratio": round(float(result.intruder_ratio), 4),
                "axis": result.axis.tolist() if hasattr(result.axis, "tolist") else list(result.axis),
                "layer": layer,
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_probe, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── SAE training ───────────────────────────────────────────────────────

    @app.route("/api/sae/train", methods=["POST"])
    def train_sae():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _train(q):
            from latent_scope.engine.sae import SAE
            from latent_scope.engine.training import SAETrainer, ActivationBuffer

            layer = params.get("layer", runtime.model.num_layers // 2)
            hidden_dim = runtime.model.hidden_dim
            n_features = params.get("n_features", hidden_dim * 4)
            sparsity = params.get("sparsity", 0.01)
            n_steps = params.get("n_steps", 1000)
            texts = params.get("texts", [])

            if not texts:
                q.put({"type": "error", "message": "Provide texts for SAE training."})
                return

            q.put({"type": "status", "message": f"Building activation buffer at layer {layer}…"})
            sae = SAE(hidden_dim, n_features)
            trainer = SAETrainer(sae, runtime.model, layer=layer, sparsity_coeff=sparsity)

            buf = ActivationBuffer(runtime.model, layer=layer)
            buf.fill(texts)

            q.put({"type": "status", "message": f"Training SAE ({n_steps} steps)…"})
            for step, metrics in trainer.train_iter(buf, n_steps=n_steps):
                if step % 50 == 0:
                    q.put({
                        "type": "sae_progress",
                        "step": step,
                        "loss": round(float(metrics.get("loss", 0)), 5),
                        "l0": round(float(metrics.get("l0", 0)), 3),
                        "dead_pct": round(float(metrics.get("dead_pct", 0)), 3),
                    })

            # Save
            import mlx.core as mx
            save_path = os.path.join(DATA_DIR, f"sae_layer{layer}.npz")
            mx.savez(save_path, **{k: v for k, v in sae.parameters().items()})
            q.put({"type": "sae_done", "path": save_path, "layer": layer})

        q: queue.Queue = queue.Queue()
        _run_in_thread(_train, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Activation steering ────────────────────────────────────────────────

    @app.route("/api/steer", methods=["POST"])
    def steer():
        if not runtime.model:
            return jsonify({"error": "No model loaded"}), 400
        params = request.json or {}

        def _steer(q):
            print("[Surgeon] Incoming /api/steer request")
            try:
                from latent_scope.engine.surgeon import ActivationSurgeon
                from latent_scope.engine.operator_miner import Operator
            except Exception as e:
                print(f"[Surgeon] Import error: {e}")
                q.put({"type": "error", "message": str(e)})
                return

            print(f"[Surgeon] Initializing ActivationSurgeon... Model loaded: {runtime.model is not None}")
            surgeon = ActivationSurgeon(runtime.model)
            prompt = params.get("prompt", "")
            layer = params.get("layer", runtime.model.num_layers // 2)
            scale = float(params.get("scale", 1.0))
            max_tokens = int(params.get("max_tokens", 100))
            use_current = params.get("use_current_operator", False)
            operator_name = params.get("operator_name")
            
            print(f"[Surgeon] Request params: prompt='{prompt[:20]}...', layer={layer}, scale={scale}, max_tokens={max_tokens}, op_name={operator_name}")

            if operator_name and operator_name != "current":
                filepath = os.path.join(DATA_DIR, "operators", f"{operator_name}.npz")
                print(f"[Surgeon] Loading explicit operator from {filepath}")
                if not os.path.exists(filepath):
                    print(f"[Surgeon] Operator file not found: {filepath}")
                    q.put({"type": "error", "message": f"Operator {operator_name} not found."})
                    return
                op = Operator.load_from_file(filepath)
                layer = op.layer
                q.put({"type": "status", "message": f"Applying extracted operator '{operator_name}'…"})
            elif use_current or operator_name == "current":
                print("[Surgeon] Using current UI basis vector")
                with _state.lock:
                    basis = _state.current_basis
                    
                if basis is None or len(basis) == 0:
                    print("[Surgeon] No active UI vector found in _state")
                    q.put({"type": "error", "message": "No active projection available. Please map and select a projection first."})
                    return
                op = Operator(name="ui_vector", vector=basis[0], layer=layer)
                q.put({"type": "status", "message": "Applying active projection operator…"})
            else:
                positive = params.get("positive", "")
                negative = params.get("negative", "")
                print(f"[Surgeon] Extracting contrastive operator: '{positive}' vs '{negative}'")
                q.put({"type": "status", "message": "Extracting contrastive operator…"})
                op = surgeon.contrastive_operator("op", positive, negative, layer=layer)

            print("[Surgeon] Setup complete. Generating outputs via surgeon.operate()...")
            q.put({"type": "status", "message": "Generating baseline…"})
            result = surgeon.operate(prompt, op, layer=layer, scale=scale, max_tokens=max_tokens)
            print("[Surgeon] Generation complete!")

            q.put({
                "type": "steer_result",
                "baseline": str(result.baseline),
                "intervened": str(result.intervened),
                "layer": int(layer),
                "scale": float(scale),
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_steer, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    # ── Unified /run endpoint (matches postRun() in frontend JS) ──────────

    # Actions that don't require a loaded model
    _NO_MODEL_ACTIONS = {
        "save_session", "load_session", "list_sessions", "open_sessions_folder",
        "load_presets", "load_preset",
        "save_tunnel", "load_tunnels", "delete_tunnel",
        "save_groups", "load_saved",
    }

    @app.route("/run", methods=["POST"])
    def run():
        """Single dispatch endpoint: all frontend postRun() calls land here."""
        params = request.json or {}
        action = params.get("action", "project")

        # Check model only for actions that need it
        if action not in _NO_MODEL_ACTIONS and not runtime.model:
            def _err(q):
                q.put({"type": "error", "message": "No model loaded."})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_err, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Session actions ──
        if action == "save_session":
            name = params.get("save_name", "session")
            session = params.get("session", {})
            safe = _safe_filename(name)
            filepath = os.path.join(SESSIONS_DIR, f"{safe}.json")
            with open(filepath, "w") as f:
                json.dump(session, f, indent=2)
            def _saved(q):
                q.put({"type": "status", "message": f"Session saved: {safe}"})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_saved, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "load_session":
            name = params.get("session_name", "")
            safe = _safe_filename(name)
            filepath = os.path.join(SESSIONS_DIR, f"{safe}.json")
            def _load_s(q):
                if os.path.exists(filepath):
                    with open(filepath) as f:
                        q.put({"type": "session_data", "session": json.load(f)})
                else:
                    q.put({"type": "error", "message": f"Session not found: {safe}"})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_load_s, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "list_sessions":
            def _list_s(q):
                sessions = [f[:-5] for f in sorted(os.listdir(SESSIONS_DIR)) if f.endswith(".json")]
                q.put({"type": "session_list", "sessions": sessions})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_list_s, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "open_sessions_folder":
            if sys.platform == "darwin":
                subprocess.Popen(["open", SESSIONS_DIR])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", SESSIONS_DIR])
            else:
                subprocess.Popen(["xdg-open", SESSIONS_DIR])
            def _opened(q):
                q.put({"type": "status", "message": "Opened sessions folder"})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_opened, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Preset actions ──
        if action == "load_presets":
            def _presets(q):
                if not os.path.exists(PRESETS_FILE):
                    q.put({"type": "presets_list", "presets": []})
                    return
                with open(PRESETS_FILE) as f:
                    data = json.load(f)
                q.put({"type": "presets_list", "presets": [
                    {"key": p.get("key"), "name": p.get("name"), "description": p.get("description")}
                    for p in data]})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_presets, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "load_preset":
            key = params.get("key", "")
            def _preset(q):
                if not os.path.exists(PRESETS_FILE):
                    q.put({"type": "error", "message": "No presets file"}); return
                with open(PRESETS_FILE) as f:
                    data = json.load(f)
                preset = next((p for p in data if p.get("key") == key), None)
                if not preset:
                    q.put({"type": "error", "message": f"Preset '{key}' not found"}); return
                colors = {"true": "#1f77b4", "false": "#ff7f0e", "gibberish": "#2ca02c"}
                groups = []
                for cat in ["true", "false", "gibberish"]:
                    if cat in preset:
                        groups.append({"name": f"{preset.get('name','Preset')}: {cat.capitalize()}",
                                       "color": colors[cat], "items": "\n".join(preset[cat])})
                q.put({"type": "preset_data", "groups": groups})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_preset, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Tunnel actions ──
        tunnel_file = os.path.join(DATA_DIR, "tunnels.json")

        if action == "save_tunnel":
            name = params.get("name", "Tunnel")
            color = params.get("color", "#888")
            trajectory = params.get("trajectory", [])
            def _save_t(q):
                tunnels = {}
                if os.path.exists(tunnel_file):
                    try:
                        with open(tunnel_file) as f: tunnels = json.load(f)
                    except Exception: pass
                tunnels[name] = {"color": color, "trajectory": trajectory}
                with open(tunnel_file, "w") as f: json.dump(tunnels, f, indent=2)
                q.put({"type": "status", "message": f"Saved tunnel '{name}'"})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_save_t, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "load_tunnels":
            def _load_t(q):
                if os.path.exists(tunnel_file):
                    with open(tunnel_file) as f: tunnels = json.load(f)
                else:
                    tunnels = {}
                q.put({"type": "tunnel_list", "tunnels": tunnels})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_load_t, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        if action == "delete_tunnel":
            name = params.get("name")
            def _del_t(q):
                if os.path.exists(tunnel_file):
                    with open(tunnel_file) as f: tunnels = json.load(f)
                    if name in tunnels:
                        del tunnels[name]
                        with open(tunnel_file, "w") as f: json.dump(tunnels, f, indent=2)
                q.put({"type": "status", "message": f"Deleted tunnel '{name}'"})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_del_t, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Lens action ──
        if action == "lens":
            label = params.get("label", "")
            layer_idx = params.get("layer", 0)
            def _lens(q):
                with _state.lock:
                    act_cache = _state.act_cache
                    W_U = _state.W_U
                if act_cache is None:
                    q.put({"type": "error", "message": "Run encoding first."}); return
                vecs = act_cache.get(label)
                if vecs is None or layer_idx not in vecs:
                    q.put({"type": "error", "message": f"No cache for '{label}' at layer {layer_idx}"}); return
                vec = vecs[layer_idx].astype(np.float32)
                if W_U is not None:
                    logits = vec @ W_U.T
                    top_idx = np.argpartition(-logits, 5)[:5]
                    top_idx = top_idx[np.argsort(-logits[top_idx])]
                    tokens = [{"token": runtime.model.tokenizer.decode([int(i)]),
                               "score": round(float(logits[int(i)]), 2)} for i in top_idx]
                else:
                    tokens = []
                q.put({"type": "lens_result", "label": label, "tokens": tokens})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_lens, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Direction probe ──
        if action == "direction_probe":
            def _dir_probe(q):
                with _state.lock:
                    act_cache = _state.act_cache
                    W_U = _state.W_U
                    hidden_dim = runtime.model.hidden_dim
                if act_cache is None:
                    q.put({"type": "error", "message": "Run encoding first."}); return
                ops = params.get("ops", [])
                layer_idx = params.get("layer", 0)
                direction = np.zeros(hidden_dim, dtype=np.float64)
                for op in ops:
                    text = op.get("text")
                    if not text: continue
                    sign = 1.0 if op.get("op") == "+" else -1.0
                    if text in act_cache and layer_idx in act_cache[text]:
                        direction += sign * act_cache[text][layer_idx]
                    else:
                        cache = runtime.model.run_with_cache(text, layers=[layer_idx])
                        path = f"model.layers.{layer_idx}"
                        if path in cache:
                            direction += sign * np.asarray(cache[path][0, -1, :], dtype=np.float64)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    q.put({"type": "error", "message": "Direction vector is near-zero."}); return
                direction /= norm
                projections = []
                for label, layer_vecs in act_cache.items():
                    if layer_idx in layer_vecs:
                        projections.append({"label": label, "projection": round(float(layer_vecs[layer_idx] @ direction), 4)})
                projections.sort(key=lambda p: p["projection"])
                dir_tokens = []
                if W_U is not None:
                    dir_f32 = direction.astype(np.float32)
                    logits = dir_f32 @ W_U.T
                    top_idx = np.argpartition(-logits, 5)[:5]
                    top_idx = top_idx[np.argsort(-logits[top_idx])]
                    dir_tokens = [{"token": runtime.model.tokenizer.decode([int(i)]),
                                   "score": round(float(logits[int(i)]), 2)} for i in top_idx]
                q.put({"type": "direction_result", "projections": projections,
                       "dir_tokens": dir_tokens, "layer": layer_idx})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_dir_probe, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Vector math ──
        if action == "vector_math":
            def _vmath(q):
                import mlx.core as mx
                ops = params.get("ops", [])
                layer_idx = params.get("layer", 0)
                result_vec = mx.zeros((runtime.model.hidden_dim,), dtype=mx.float32)
                for op in ops:
                    text = op.get("text")
                    if not text: continue
                    sign = 1.0 if op.get("op") == "+" else -1.0
                    cache = runtime.model.run_with_cache(text, layers=[layer_idx])
                    path = f"model.layers.{layer_idx}"
                    if path in cache:
                        result_vec = result_vec + sign * mx.array(cache[path][0, -1, :]).astype(mx.float32)
                coord = None
                with _state.lock:
                    if _state.pca_mean is not None and _state.pca_vt is not None:
                        vec_np = np.asarray(result_vec, dtype=np.float64)
                        coord = [round(float(x), 5) for x in (vec_np - _state.pca_mean) @ _state.pca_vt[:3].T]
                tokens = []
                with _state.lock:
                    W_U = _state.W_U
                if W_U is not None:
                    rv = np.asarray(result_vec, dtype=np.float32)
                    logits = rv @ W_U.T
                    top_idx = np.argpartition(-logits, 5)[:5]
                    top_idx = top_idx[np.argsort(-logits[top_idx])]
                    tokens = [{"token": runtime.model.tokenizer.decode([int(i)]),
                               "score": float(logits[int(i)])} for i in top_idx]
                q.put({"type": "math_result", "tokens": tokens, "coord": coord})
            q: queue.Queue = queue.Queue()
            _run_in_thread(_vmath, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Reproject ──
        if action == "reproject":
            def _repr(q):
                with _state.lock:
                    matrix = _state.raw_matrix
                    items = _state.raw_items
                    num_layers = _state.num_layers
                    group_labels = _state.group_labels or []
                    group_names = _state.group_names or []
                    act_cache = _state.act_cache or {}

                if matrix is None or items is None:
                    q.put({"type": "error", "message": "No cached activations. Run encoding first."}); return

                method = params.get("method", "pca")
                proj_params = params.get("proj_params", {})
                n_items = len(items)
                hidden_dim = matrix.shape[1]

                basis_layer = proj_params.get("basis_layer", num_layers - 1)
                basis_layer = max(0, min(basis_layer, num_layers - 1))
                uses_labels = method in ("lda", "cpca", "probe_aligned", "null_space", "grassmannian")
                direction_source = ""

                if method in ("probe_aligned", "null_space"):
                    ops = proj_params.get("direction_ops", [])
                    if ops and act_cache:
                        direction = np.zeros(hidden_dim, dtype=np.float64)
                        parts = []
                        for op in ops:
                            text = op.get("text", "")
                            sign = 1.0 if op.get("op") == "+" else -1.0
                            if text in act_cache and basis_layer in act_cache[text]:
                                direction += sign * act_cache[text][basis_layer]
                            else:
                                cache = runtime.model.run_with_cache(text, layers=[basis_layer])
                                path = f"model.layers.{basis_layer}"
                                if path in cache:
                                    direction += sign * np.asarray(cache[path][0, -1, :], dtype=np.float64)
                            parts.append(f"{op.get('op','+')} {text}")
                        proj_params["direction"] = direction.tolist()
                        direction_source = " ".join(parts)
                    elif len(set(group_labels)) >= 2:
                        gA_name = proj_params.get("probe_group_a")
                        gB_name = proj_params.get("probe_group_b")
                        
                        try:
                            g0 = group_names.index(gA_name) if gA_name in group_names else list(set(group_labels))[0]
                            g1 = group_names.index(gB_name) if gB_name in group_names else list(set(group_labels))[1]
                        except:
                            groups_sorted = sorted(set(group_labels))
                            g0, g1 = groups_sorted[0], groups_sorted[1]
                        
                        vecs_g0 = [matrix[ii * num_layers + basis_layer] for ii, gl in enumerate(group_labels) if gl == g0]
                        vecs_g1 = [matrix[ii * num_layers + basis_layer] for ii, gl in enumerate(group_labels) if gl == g1]
                        if vecs_g0 and vecs_g1:
                            c0 = np.mean(vecs_g0, axis=0).astype(np.float64)
                            c1 = np.mean(vecs_g1, axis=0).astype(np.float64)
                            proj_params["direction"] = (c0 - c1).tolist()
                            g0_name = group_names[g0] if g0 < len(group_names) else f"G{g0}"
                            g1_name = group_names[g1] if g1 < len(group_names) else f"G{g1}"
                            direction_source = f"centroid({g0_name}) − centroid({g1_name})"

                if uses_labels and method != "pca":
                    if method == "cpca":
                        tgn = proj_params.get("target_group_name")
                        if tgn and tgn in group_names:
                            proj_params["target_group"] = group_names.index(tgn)
                            
                    X_basis = np.array([matrix[ii * num_layers + basis_layer] for ii in range(n_items)], dtype=np.float64)
                    X_basis_centered = X_basis - X_basis.mean(axis=0, keepdims=True)
                    basis, info = compute_projection(X_basis_centered, method, proj_params, group_labels, n=3)
                else:
                    X_all = matrix.astype(np.float64)
                    mean_all = X_all.mean(axis=0, keepdims=True)
                    X_all_centered = X_all - mean_all
                    basis, info = compute_projection(X_all_centered, "pca", {}, group_labels, n=3)

                X_full = matrix.astype(np.float64)
                full_mean = X_full.mean(axis=0, keepdims=True)
                coords = (X_full - full_mean) @ basis.T

                with _state.lock:
                    _state.pca_mean = full_mean[0]
                    _state.pca_vt = basis
                    _state.current_basis = basis

                frames, trajectories = _build_frames(items, coords, num_layers)
                proj_label = info.get("label", method)
                if uses_labels:
                    proj_label += f" @ layer {basis_layer}"
                if direction_source:
                    proj_label += f" | dir: {direction_source}"

                q.put({
                    "type": "reproject_result",
                    "frames": frames,
                    "trajectories": trajectories,
                    "variance_explained": info.get("variance_explained", [0, 0, 0]),
                    "projection_label": proj_label,
                    "method": method,
                    "basis_layer": basis_layer,
                    "direction_source": direction_source,
                    "n_items": n_items,
                    "n_layers": num_layers,
                    "group_names": group_names,
                })

            q: queue.Queue = queue.Queue()
            _run_in_thread(_repr, q)
            return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

        # ── Default: encoding ──
        def _encode_run(q):
            model = runtime.model
            raw_groups = params.get("groups", [])
            if not raw_groups:
                q.put({"type": "error", "message": "No groups provided."}); return

            items: list[dict] = []
            for g in raw_groups:
                gname = (g.get("name") or "Group").strip()
                color = g.get("color") or "#1f77b4"
                for line in (g.get("items") or "").split("\n"):
                    line = line.strip()
                    if not line: continue
                    if "|" in line:
                        label, text = line.split("|", 1)
                        label, text = label.strip(), text.strip()
                    else:
                        label = text = line
                    if label and text:
                        items.append({"label": label, "text": text, "group": gname, "color": color})

            if len(items) < 2:
                q.put({"type": "error", "message": "Provide at least 2 items across all groups."}); return

            num_layers = model.num_layers
            q.put({"type": "status", "message": f"Encoding {len(items)} items…"})

            all_layers = list(range(num_layers))
            all_acts: list[dict] = []
            for i, item in enumerate(items):
                q.put({"type": "status", "message": f"Encoding {i + 1}/{len(items)}: {item['label'][:40]}"})
                cache = model.run_with_cache(item["text"], layers=all_layers)
                layer_vecs: dict = {}
                for ll in all_layers:
                    path = f"model.layers.{ll}"
                    if path not in cache: continue
                    vec = np.asarray(cache[path][0, -1, :], dtype=np.float64)
                    layer_vecs[ll] = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                all_acts.append(layer_vecs)

            with _state.lock:
                _state.act_cache = {items[ii]["label"]: all_acts[ii] for ii in range(len(items))}

            if _state.W_U is None and _HAS_WEIGHT_INTERP:
                try:
                    interp = WeightInterpreter(model)
                    if interp.W_U is not None:
                        _state.W_U = np.asarray(interp.W_U, dtype=np.float32)
                except Exception:
                    pass

            q.put({"type": "status", "message": "Computing projection…"})
            n_items = len(items)
            hidden_dim = next(iter(all_acts[0].values())).shape[0]
            matrix = np.zeros((n_items * num_layers, hidden_dim), dtype=np.float32)
            for ii in range(n_items):
                for ll in range(num_layers):
                    matrix[ii * num_layers + ll] = all_acts[ii][ll]

            seen_groups: dict[str, int] = {}
            group_labels = []
            for it in items:
                if it["group"] not in seen_groups:
                    seen_groups[it["group"]] = len(seen_groups)
                group_labels.append(seen_groups[it["group"]])

            with _state.lock:
                _state.raw_matrix = matrix
                _state.raw_items = items
                _state.num_layers = num_layers
                _state.group_labels = group_labels
                _state.group_names = list(seen_groups.keys())

            X = matrix.astype(np.float64)
            mean = X.mean(axis=0, keepdims=True)
            X_centered = X - mean
            basis, info = compute_projection(X_centered, "pca", {}, group_labels, n=3)
            coords = X_centered @ basis.T

            with _state.lock:
                _state.pca_mean = mean[0]
                _state.pca_vt = basis
                _state.current_basis = basis

            frames, trajectories = _build_frames(items, coords, num_layers)
            analytics = _compute_analytics(matrix, items, group_labels, list(seen_groups.keys()), num_layers)

            q.put({
                "type": "result",
                "frames": frames,
                "trajectories": trajectories,
                "variance_explained": info.get("variance_explained", [0, 0, 0]),
                "n_layers": num_layers,
                "n_items": n_items,
                "groups": list({it["group"]: it["color"] for it in items}.items()),
                "item_labels": [it["label"] for it in items],
                **analytics,
            })

        q: queue.Queue = queue.Queue()
        _run_in_thread(_encode_run, q)
        return Response(stream_with_context(_sse(q)), mimetype="text/event-stream")

    return app


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_frames(items, coords, num_layers):
    n_items = len(items)
    frames = []
    for ll in range(num_layers):
        pts = []
        for ii, item in enumerate(items):
            row = ii * num_layers + ll
            pt = {
                "label": item["label"], "group": item["group"], "color": item["color"],
                "x": round(float(coords[row, 0]), 5),
                "y": round(float(coords[row, 1]), 5),
            }
            if coords.shape[1] > 2:
                pt["z"] = round(float(coords[row, 2]), 5)
            pts.append(pt)
        frames.append({"layer": ll, "points": pts})

    trajectories = {}
    for ii, item in enumerate(items):
        traj = {
            "xs": [round(float(coords[ii * num_layers + ll, 0]), 5) for ll in range(num_layers)],
            "ys": [round(float(coords[ii * num_layers + ll, 1]), 5) for ll in range(num_layers)],
            "group": item["group"], "color": item["color"],
        }
        if coords.shape[1] > 2:
            traj["zs"] = [round(float(coords[ii * num_layers + ll, 2]), 5) for ll in range(num_layers)]
        trajectories[item["label"]] = traj

    return frames, trajectories


def _compute_analytics(matrix, items, group_labels, group_names, num_layers):
    n_items = len(items)
    n_groups = len(group_names)
    item_group_idx = group_labels

    within_sim, between_sim, discriminability, silhouette_scores = [], [], [], []
    sep_by_pair: dict[str, list] = {}
    similarity_matrices = []

    for ll in range(num_layers):
        layer_acts = matrix[ll::num_layers]
        norms = np.linalg.norm(layer_acts, axis=1, keepdims=True) + 1e-9
        norm_acts = layer_acts / norms
        sim = norm_acts @ norm_acts.T
        similarity_matrices.append(np.round(sim, 4).tolist())

        w_sims, b_sims = [], []
        for i in range(n_items):
            for j in range(i + 1, n_items):
                s = float(sim[i, j])
                (w_sims if item_group_idx[i] == item_group_idx[j] else b_sims).append(s)
        w = float(np.mean(w_sims)) if w_sims else 0.0
        b = float(np.mean(b_sims)) if b_sims else 0.0
        within_sim.append(round(w, 4))
        between_sim.append(round(b, 4))
        discriminability.append(round(w - b, 4))

        if n_groups >= 2:
            sil_scores_i = []
            for i in range(n_items):
                gi = item_group_idx[i]
                same = [1.0 - float(sim[i, j]) for j in range(n_items) if j != i and item_group_idx[j] == gi]
                a_i = float(np.mean(same)) if same else 0.0
                b_i = min(
                    (float(np.mean([1.0 - float(sim[i, j]) for j in range(n_items) if item_group_idx[j] == gk]))
                     for gk in range(n_groups) if gk != gi
                     and any(item_group_idx[j] == gk for j in range(n_items))),
                    default=0.0
                )
                denom = max(a_i, b_i)
                sil_scores_i.append((b_i - a_i) / denom if denom > 1e-10 else 0.0)
            silhouette_scores.append(round(float(np.mean(sil_scores_i)), 4))
        else:
            silhouette_scores.append(0.0)

        centroids = {}
        for gname in group_names:
            idxs = [i for i, g in enumerate(item_group_idx) if group_names[g] == gname]
            if idxs:
                centroids[gname] = layer_acts[idxs].mean(axis=0)
        for i, g1 in enumerate(group_names):
            for j in range(i + 1, len(group_names)):
                g2 = group_names[j]
                if g1 in centroids and g2 in centroids:
                    key = f"{g1} \u2194 {g2}"
                    sep_by_pair.setdefault(key, []).append(
                        round(float(np.linalg.norm(centroids[g1] - centroids[g2])), 4))

    sep_curves = {}
    for key, dists in sep_by_pair.items():
        max_d = max(dists) if dists else 1.0
        sep_curves[key] = {"raw": dists, "norm": [round(d / max_d, 4) for d in dists]}

    disc_velocity = [0.0]
    for ll in range(1, num_layers):
        disc_velocity.append(abs(discriminability[ll] - discriminability[ll - 1]))
    threshold = float(np.percentile(disc_velocity, 85)) if len(disc_velocity) > 1 else 0.0
    critical_layers = [i for i, v in enumerate(disc_velocity) if v >= threshold]

    return {
        "within_sim": within_sim,
        "between_sim": between_sim,
        "discriminability": discriminability,
        "sep_curves": sep_curves,
        "similarity_matrices": similarity_matrices,
        "silhouette_scores": silhouette_scores,
        "critical_layers": critical_layers,
    }
