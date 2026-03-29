"""Standalone page shell for Latent Scope.

Provides page_shell() — a function that returns a full HTML page as a string.
CSS/JS is adapted from base_explorer.py in Wittgenstein's Monster, simplified
for a single-page app (no nav links to a command center).
"""

from __future__ import annotations

_CSS = """
:root {
    color-scheme: light;
    --bg: #f8f9fa;
    --bg-accent: #e9ecef;
    --panel: #ffffff;
    --panel-strong: #f1f3f5;
    --text: #212529;
    --muted: #6c757d;
    --border: #dee2e6;
    --accent: #4dabf7;
    --accent-strong: #228be6;
    --button-bg: #0050ff;
    --button-hover: #003cc8;
    --button-text: #ffffff;
    --status: #495057;
    --error: #fa5252;
    --success: #40c057;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    --plot-paper: #f8f9fa;
    --plot-panel: #ffffff;
    --tag-bg: rgba(34, 139, 230, 0.1);
    --font-sans: 'Inter', 'SF Pro', 'Roboto', system-ui, sans-serif;
    --font-mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
:root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #121212;
    --bg-accent: #1e1e1e;
    --panel: #1e1e1e;
    --panel-strong: #252525;
    --text: #f8f9fa;
    --muted: #adb5bd;
    --border: #333333;
    --accent: #4dabf7;
    --accent-strong: #74c0fc;
    --button-bg: #0050ff;
    --button-hover: #3b71f7;
    --button-text: #ffffff;
    --status: #ced4da;
    --error: #ff6b6b;
    --success: #51cf66;
    --shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    --plot-paper: #121212;
    --plot-panel: #1e1e1e;
    --tag-bg: rgba(116, 192, 252, 0.1);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: var(--font-sans);
    background:
        radial-gradient(circle at top left, var(--bg-accent), transparent 32rem),
        linear-gradient(180deg, var(--bg), color-mix(in srgb, var(--bg) 90%, black 10%));
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
a { color: inherit; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Header */
header {
    flex-shrink: 0;
    z-index: 10;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    background: color-mix(in srgb, var(--bg) 82%, transparent);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    height: 52px;
}
.header-left { display: flex; align-items: center; gap: 10px; }
.wordmark {
    font-size: 14px;
    font-weight: 700;
    color: var(--accent-strong);
    letter-spacing: -0.02em;
}
.tag {
    font-size: 11px;
    background: var(--tag-bg);
    color: var(--accent-strong);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    max-width: 46ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.model-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #888; flex-shrink: 0;
    transition: background 0.3s;
}
.model-dot.active { background: #4a7c59; }
.model-dot.loading { background: #d4903a; animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.header-actions { display: flex; align-items: center; gap: 8px; }
.theme-toggle {
    display: inline-flex; align-items: center; justify-content: center;
    background: transparent; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 12px; font-size: 13px; cursor: pointer;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
    font-family: inherit;
}
.theme-toggle:hover { color: var(--text); border-color: var(--accent); background: var(--tag-bg); }

.model-loader {
    display: flex; align-items: center; gap: 6px;
}
.model-loader input {
    font-size: 11px; padding: 6px 10px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--panel-strong);
    color: var(--text); font-family: var(--font-mono); width: 280px; margin: 0;
    height: 28px;
}
.model-loader input:focus {
    outline: 1px solid var(--accent); border-color: var(--accent);
    box-shadow: none;
}
.model-loader button {
    font-size: 11px; padding: 0 12px; border-radius: 6px; height: 28px;
    line-height: 26px;
}

/* Fullscreen overlays */
.fullscreen-wrapper {
    position: fixed !important; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 1000; background: var(--panel);
    padding: 20px; box-sizing: border-box;
    display: flex; flex-direction: column;
}

/* Panels */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    box-shadow: var(--shadow);
    transition: border-color 0.2s;
}
.panel:hover { border-color: color-mix(in srgb, var(--accent) 20%, var(--border)); }
.panel h2 {
    font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Buttons */
button {
    background: var(--button-bg); color: var(--button-text);
    border: 1px solid transparent; font-family: inherit; font-size: 12px;
    padding: 0 14px; border-radius: 6px; cursor: pointer; height: 28px;
    transition: background 0.15s, box-shadow 0.15s; font-weight: 500;
}
button:hover { background: var(--button-hover); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.ghost-button {
    background: transparent; color: var(--muted);
    border: 1px solid var(--border); padding: 0 10px; height: 28px;
}
.ghost-button:hover { color: var(--text); border-color: var(--accent); background: var(--tag-bg); }

/* Forms */
input, textarea, select {
    width: 100%;
    background: var(--panel-strong);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: inherit;
    font-size: 11px;
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 8px; height: 28px;
    transition: border-color 0.15s;
}
input:focus, textarea:focus, select:focus {
    outline: 1px solid var(--accent); border-color: var(--accent); box-shadow: none;
}
textarea { height: auto; padding: 8px 10px; }

/* Range slider */
input[type="range"] {
    -webkit-appearance: none; appearance: none;
    height: 5px; border-radius: 3px; background: var(--border);
    padding: 0; margin-bottom: 8px; cursor: pointer;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: var(--button-bg); border: 2px solid var(--panel);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2); cursor: pointer;
    transition: background 0.15s, transform 0.15s;
}
input[type="range"]::-webkit-slider-thumb:hover { background: var(--button-hover); transform: scale(1.1); }
input[type="range"]:focus { outline: none; box-shadow: none; border-color: transparent; }

/* Status */
.status { font-size: 11px; color: var(--status); min-height: 16px; }
.error { color: var(--error); }
#status { padding: 4px 0; }

/* Spinner */
.spinner {
    display: inline-block; width: 12px; height: 12px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Tabs */
.tab-bar { display: flex; gap: 4px; margin-bottom: 12px; border-bottom: 1px solid var(--border); }
.tab-btn {
    background: transparent; color: var(--muted); border: none;
    border-bottom: 2px solid transparent; border-radius: 0;
    padding: 6px 12px; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; cursor: pointer; transition: color 0.15s, border-color 0.15s;
    transform: none; font-family: inherit;
}
.tab-btn:hover { color: var(--text); background: transparent; transform: none; }
.tab-btn.active { color: var(--accent-strong); border-bottom-color: var(--accent); background: transparent; }

/* Animations */
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
"""

_JS_UTILS = """
// Theme
(function() {
    const saved = localStorage.getItem('ls-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    else if (window.matchMedia('(prefers-color-scheme: dark)').matches)
        document.documentElement.setAttribute('data-theme', 'dark');
})();
function toggleTheme() {
    const root = document.documentElement;
    const current = root.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('ls-theme', next);
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = next === 'dark' ? '☀' : '☾';
}

// postRun — SSE fetch, dispatches to callback on each event
function postRun(params, onData, onDone) {
    fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
    }).then(async r => {
        if (!r.ok || !r.body) throw new Error('Request failed ' + r.status);
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            const parts = buffer.split('\\n\\n');
            buffer = parts.pop();
            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith('data:')) continue;
                try {
                    const data = JSON.parse(line.slice(5).trim());
                    if (onData) onData(data);
                } catch(e) {}
            }
        }
        if (onDone) onDone();
    }).catch(err => {
        if (onData) onData({type: 'error', message: err.message});
        if (onDone) onDone();
    });
}

// Model loading (uses /api/model/load REST endpoint)
function loadModel() {
    const inp = document.getElementById('model-input');
    const modelId = inp ? inp.value.trim() : '';
    if (!modelId) return;
    setModelLoading(true);
    fetch('/api/model/load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model_id: modelId}),
    }).then(async r => {
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            const parts = buffer.split('\\n\\n');
            buffer = parts.pop();
            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith('data:')) continue;
                try {
                    const data = JSON.parse(line.slice(5).trim());
                    if (data.type === 'loaded') setModelLoaded(data.model, data.num_layers);
                    else if (data.type === 'error') { setModelLoading(false); alert('Load failed: ' + data.message); }
                } catch(e) {}
            }
        }
    }).catch(err => { setModelLoading(false); alert('Load error: ' + err.message); });
}

function setModelLoading(on) {
    const dot = document.getElementById('model-dot');
    if (dot) { dot.className = 'model-dot ' + (on ? 'loading' : ''); }
}

function setModelLoaded(name, numLayers) {
    const dot = document.getElementById('model-dot');
    if (dot) dot.className = 'model-dot active';
    const tag = document.getElementById('model-tag');
    if (tag) tag.textContent = name;
    if (typeof onModelLoaded === 'function') onModelLoaded(numLayers);
}

// getPlotlyConfig — unified config for all plots
function getPlotlyConfig() {
    return {
        responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['sendDataToCloud', 'editInChartStudio'],
        toImageButtonOptions: {format: 'png', filename: 'latent-scope', height: 800, width: 1200, scale: 2},
    };
}

function getPlotlyLayout(overrides) {
    return Object.assign({
        plot_bgcolor: 'var(--plot-panel)',
        paper_bgcolor: 'transparent',
        font: {family: 'inherit', color: 'var(--text)'},
    }, overrides || {});
}

// Populate model datalist from /api/models
function populateModelList() {
    fetch('/api/models').then(r => r.json()).then(data => {
        const dl = document.getElementById('model-list');
        if (!dl) return;
        dl.innerHTML = '';
        (data.models || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            dl.appendChild(opt);
        });
        // If input is empty and there are models, hint the first one
        const inp = document.getElementById('model-input');
        if (inp && !inp.value && data.models && data.models.length) {
            inp.placeholder = data.models[0];
        }
    }).catch(() => {});
}

// Check model status on load, then populate list
fetch('/api/model/status').then(r => r.json()).then(data => {
    if (data.loaded) {
        setModelLoaded(data.model, data.num_layers);
        const inp = document.getElementById('model-input');
        if (inp) inp.value = data.model;
    }
    populateModelList();
}).catch(() => { populateModelList(); });

function toggleSidebarLocal() {
    const sb = document.querySelector('.sidebar');
    if (!sb) return;
    if (sb.style.display === 'none') {
        sb.style.display = 'flex';
    } else {
        sb.style.display = 'none';
    }
}

"""


def page_shell(title: str, body: str, extra_js: str = "", model_label: str = "") -> str:
    """Return a full standalone HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Latent Scope</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body>

<header>
  <div class="header-left">
    <button class="ghost-button" onclick="toggleSidebarLocal()" style="font-size:14px; padding:0 8px; border:none; color:var(--text)" title="Toggle Sidebar">≡</button>
    <span class="wordmark">Latent Scope</span>
    <span style="color:var(--border); margin:0 4px;">/</span>
    <div class="model-loader">
      <input id="model-input" list="model-list" placeholder="Search or type model ID…"
             onkeydown="if(event.key==='Enter')loadModel()" autocomplete="off" spellcheck="false" style="width:300px;">
      <datalist id="model-list"></datalist>
    </div>
  </div>
  <div class="header-actions">
    <div style="display:flex; align-items:center; gap:6px; margin-right: 12px; font-family:var(--font-mono)">
        <div class="model-dot" id="model-dot"></div>
        <span class="tag" id="model-tag">{model_label or "No model loaded"}</span>
    </div>
    <button onclick="loadModel()">Load</button>
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()" style="border-radius:999px; height:28px; width:28px; padding:0">☾</button>
  </div>
</header>

{body}

<script>
{_JS_UTILS}
{extra_js}
</script>
</body>
</html>"""
