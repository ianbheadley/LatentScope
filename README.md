# Latent Scope

**Latent Scope** is a LLM interpretability workspace. It bridges the gap between visual concept discovery and active model steering entirely. 

Unlike standard dashboards that only broadly visualize embeddings, Latent Scope lets you mathematically isolate fine semantic features (like "truthfulness", "politeness", or "geographical awareness") across a model's residual stream, export the resulting vectors as an `Operator`, and directly inject them into the model's forward-pass to predictably steer live generation.

## Quick Start
1. Install MLX (pip install mlx)
2. Launch the workspace from your terminal:
   ```bash
   python -m latent_scope
   ```
3. Open your browser to `http://127.0.0.1:5100`.
4. **Load your model**, prepare a semantic dataset using Word Groups in the sidebar, and click **Encode & Map**.

## Workflow: From Visual Discovery to Surgery

Latent Scope enables a tight, iterative loop for mechanistic interpretability.

1. **Encode & Project**: Use methods like **cPCA**, **LDA**, or **Probe-Aligned** to cleanly separate your target concepts from background noise in the 3D Viewer.
2. **Review the Timeline**: Scrub through the model layers to find where the feature separates most aggressively (for Llama-3, this is often between Layers 6-15).
3. **Extract Operator**: Click `Extract Operator` along the top bar to save this specific geometric axis to disk as a `.npz` file.
4. **Activation Surgeon**: Test your operator immediately in the browser UI. The Surgical panel lets you sweep a scale multiplier (-10x to +10x) across live generation and generates side-by-side with the model's raw baseline.

---

### Case Study: Steering Geographical Factuality

To demonstrate how the Activation Surgeon can deeply manipulate structural reasoning within an LLM (`Llama-3.2-3B-Instruct`), we evaluated the model's spatial certainty by mapping a `Capitals True` dataset against varying forms of generic knowledge and falsehoods. 

At **Layer 6**, Contrastive PCA (cPCA) cleanly isolated the "Geographical Factuality" (or "Encyclopedic Truth") vector. We tested two interventions on the exact same baseline prompt:

#### 1. The Hyper-Factual Override (Scale: +4.0x)
By *boosting* this targeted vector into the residual stream at Layer 6, we supercharged the model's encyclopedic certainty. Notice how it immediately strips away its default conversational "Instruct" persona and generates a relentlessly dry, deterministic list of historical and geographical facts.

> **Prompt:** `the capital of france is`
> 
> **Raw Baseline:** *... (wait for it) ...PARIS! But did you know that the capital of France is not just a simple fact? It's a city with a rich history...*
>
> **Surgical Output:** *The capital of France is Paris. The city is located in the Île-de-France region, which is the most populous metropolitan area in France. The city is also the capital of the Île-de-France region and is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral...*

#### 2. The "Aphasia" Lobotomy (Scale: -6.5x)
By conversely *subtracting* the Capitals Vector during generation, the model's internal representation of the "truth" fractures completely. It falls into a hallucinatory repetition loop where it inherently recognizes the topic but structurally contradicts the fact in real-time.

> **Prompt:** `the capital of france is`
> 
> **Surgical Output:** *Paris. The capital of France is Paris, which is a well-known fact. However, it's also a fact that the capital of France is not Paris, but rather the city of Paris, which is a bit more nuanced. To clarify, the capital of France is not Paris, but rather the city of Paris...*
