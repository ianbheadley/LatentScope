# Latent Scope

**Latent Scope** is a LLM interpretability workspace. It bridges the gap between visual concept discovery and active model steering entirely. 

You can visual search for and isolate semantic features (like "truthfulness", "politeness", or "geographical awareness") across a model's residual stream, export the resulting vectors as an `Operator`, and then use as a steering vector into the model's forward-pass.

## Quick Start
1. Install MLX (pip install mlx)
2. Launch the workspace from your terminal:
   ```bash
   python -m latent_scope
   ```
3. Open your browser to `http://127.0.0.1:5100`.
4. **Load your model**, prepare a semantic dataset using Word Groups in the sidebar, and click **Encode & Map**.

## Workflow: From Visual Discovery to Surgery

1. **Encode & Project**: Use methods like **cPCA**, **LDA**, or **Probe-Aligned** to cleanly separate your target concepts from background noise in the 3D Viewer.
2. **Review the Timeline**: Scrub through the model layers to find where the feature separates most aggressively.
3. **Extract Operator**: Click `Extract Operator` along the top bar to save this specific geometric axis to disk as a `.npz` file.
4. **Activation Surgeon**: Test your operator immediately in the browser UI. The Surgical panel lets you sweep a scale multiplier (-10x to +10x) across live generation and generates side-by-side with the model's raw baseline.

---
