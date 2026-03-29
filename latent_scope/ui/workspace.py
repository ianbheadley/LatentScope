"""Main workspace page for Latent Scope."""

from __future__ import annotations
from latent_scope.ui.shell import page_shell

_BODY = """
<div id="app">

<!-- ── LEFT SIDEBAR ─────────────────────────────────────────────────────── -->
<aside class="panel sidebar">

  <div class="sidebar-top" style="padding-bottom: 12px;">
    <!-- Presets & Sessions — top -->
    <details class="sidebar-section" style="margin-bottom: 12px; margin-top: 4px;">
      <summary>Presets &amp; Sessions</summary>
      <div style="display:flex;flex-direction:column;gap:8px">
        <div>
          <label class="sidebar-label">Presets</label>
          <div style="display:flex;gap:6px;align-items:center">
            <select id="preset-sel" class="gname" style="flex:1;margin:0">
              <option value="">Choose preset&hellip;</option>
            </select>
            <button type="button" class="ghost-button small-btn" onclick="applyPreset(false)">Load</button>
            <button type="button" class="ghost-button small-btn" title="Append to existing groups" onclick="applyPreset(true)">Add</button>
          </div>
        </div>
        <div>
          <label class="sidebar-label">Save session</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input id="session-name" class="gname" placeholder="Name&hellip;" style="flex:1;margin:0">
            <button type="button" class="ghost-button small-btn" onclick="saveSession()">Save</button>
          </div>
        </div>
        <div>
          <label class="sidebar-label">Restore session</label>
          <div style="display:flex;gap:6px;align-items:center">
            <select id="session-sel" class="gname" style="flex:1;margin:0" onchange="onSelectSession(this.value)">
              <option value="">Choose&hellip;</option>
            </select>
            <button type="button" class="ghost-button small-btn" onclick="openSessionsFolder()" title="Open sessions folder">&#x1F4C2;</button>
          </div>
        </div>
      </div>
    </details>

    <!-- Encode right below -->
    <button type="button" id="run-btn" onclick="runEncoding()" style="width:100%; margin-bottom:8px;">Encode &amp; Map &rarr;</button>
    <div id="status-area" class="status" style="min-height:14px;font-size:11px;"></div>
  </div>

  <div class="sidebar-scroll">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
      <span style="font-size:10px;font-weight:600;color:var(--text);letter-spacing:0.04em;text-transform:uppercase;">Word Groups</span>
      <button type="button" class="ghost-button small-btn" onclick="addGroup()">+ Group</button>
    </div>

    <div id="groups-container"></div>

    <!-- Variance bar — shown after encoding -->
    <div id="pca-info" style="display:none; margin-top:12px;" class="info-box compact">
      <div id="pca-bar"></div>
      <div id="pca-text" style="font-size:10px;color:var(--muted);margin-top:3px"></div>
    </div>

    <div id="legend-controls" style="display:none; justify-content:space-between; align-items:center; margin-top:12px;">
        <span style="font-size:9px; color:var(--muted); text-transform:uppercase;">Legend</span>
        <div style="display:flex; gap:6px;">
            <button class="ghost-button" type="button" style="font-size:9px; padding:0 4px; height:18px;" onclick="toggleAllGroups(true)">All</button>
            <button class="ghost-button" type="button" style="font-size:9px; padding:0 4px; height:18px;" onclick="toggleAllGroups(false)">None</button>
        </div>
    </div>
    <div id="legend-area" style="display:none;flex-wrap:wrap;gap:6px;margin-top:2px"></div>
  </div>

</aside>

<!-- ── MAIN VIZ AREA ─────────────────────────────────────────────────────── -->
<main class="main-area">

  <div id="empty-state" class="empty-state">
    <div style="font-size:48px;opacity:0.15">&#x2B61;</div>
    <div style="font-size:17px;font-weight:600;opacity:0.4;margin-top:8px">Activation Space</div>
    <div style="font-size:12px;opacity:0.4;margin-top:6px;max-width:260px;text-align:center;line-height:1.6">
      Add word groups on the left,<br>then click <em>Encode &amp; Map</em>.
    </div>
  </div>

  <div id="plot-area" style="display:none;flex-direction:column;height:100%">

    <!-- ── VIEW CONTROLS ──────────────────────────────────────── -->
    <div class="layer-bar" style="border-bottom:none; padding-bottom:0;">
      <div class="view-toggles">
        <label class="toggle-label" title="Toggle 3D visualization"><input type="checkbox" id="is-3d-chk" onchange="renderLayer(currentLayer)" checked> 3D</label>
        <label class="toggle-label" title="Lock axes bounds across frames"><input type="checkbox" id="lock-axis-chk" onchange="renderLayer(currentLayer)"> Lock</label>
        <label class="toggle-label" title="Show trails of past moves"><input type="checkbox" id="trails-chk" checked onchange="renderLayer(currentLayer)"> Trails</label>
        <label class="toggle-label" title="Show point labels"><input type="checkbox" id="labels-chk" onchange="renderLayer(currentLayer)"> Labels</label>
        <label class="toggle-label" title="Draw hull areas around clusters"><input type="checkbox" id="hulls-chk" onchange="renderLayer(currentLayer)"> Hulls</label>
      </div>
    </div>

    <!-- ── PROJECTION CONTROLS ────────────────────────────────────────── -->
    <div class="proj-bar" id="proj-bar">
      <span class="proj-bar-label">Projection</span>
      <select id="proj-method" class="proj-select" onchange="onProjectionMethodChange()">
        <option value="pca">PCA</option>
        <option value="lda">LDA</option>
        <option value="cpca">cPCA</option>
        <option value="probe_aligned">Probe-Aligned</option>
        <option value="null_space">Null-Space</option>
        <option value="grassmannian">Grassmannian</option>
      </select>

      <!-- cPCA controls -->
      <div id="cpca-controls" style="display:none;align-items:center;gap:6px">
        <select id="cpca-target-group" class="proj-select" onchange="runReproject()">
          <option value="-1">Target: Auto</option>
        </select>
        <span class="proj-bar-label">&alpha; <span id="cpca-alpha-val">1.0</span></span>
        <input type="range" id="cpca-alpha" min="0" max="20" step="0.1" value="1.0"
               style="width:100px" oninput="onCpcaAlpha(this.value)">
      </div>

      <!-- Basis layer -->
      <div id="basis-layer-controls" style="display:none;align-items:center;gap:6px">
        <span class="proj-bar-label"
              title="Which layer's activations are used to compute the projection axes. The resulting basis is then applied to all layers. Fix it to the final layer for a stable coordinate frame while you scrub.">Basis L<span id="basis-layer-val">27</span></span>
        <input type="range" id="basis-layer-slider" min="0" max="27" value="27"
               style="width:90px" oninput="onBasisLayer(this.value)">
      </div>

      <!-- Grassmannian -->
      <div id="grassmannian-controls" style="display:none;align-items:center;gap:6px">
        <select id="grass-method-a" class="proj-select" onchange="runReproject()">
          <option value="pca">PCA</option>
          <option value="lda">LDA</option>
          <option value="cpca">cPCA</option>
        </select>
        <span class="proj-bar-label" style="opacity:0.5">&rarr;</span>
        <select id="grass-method-b" class="proj-select" onchange="runReproject()">
          <option value="pca">PCA</option>
          <option value="lda" selected>LDA</option>
          <option value="cpca">cPCA</option>
        </select>
        <span class="proj-bar-label">t <span id="grass-t-val">0.50</span></span>
        <input type="range" id="grass-t" min="0" max="1" step="0.01" value="0.5"
               style="width:80px" oninput="onGrassT(this.value)">
      </div>

      <!-- Direction source note -->
      <div id="direction-controls" style="display:none;align-items:center;gap:4px">
        <select id="probe-group-a" class="proj-select" onchange="runReproject()"></select>
        <span class="proj-bar-label" style="opacity:0.6">&minus;</span>
        <select id="probe-group-b" class="proj-select" onchange="runReproject()"></select>
        <span class="proj-bar-label" style="opacity:0.6;margin-left:6px" id="direction-source-label"></span>
      </div>

      <!-- RIGHT SIDE: variance status + extract + saved views -->
      <div style="display:flex;align-items:center;gap:6px;margin-left:auto;flex-shrink:0">
        <span id="proj-status" style="font-size:10px;color:var(--muted);white-space:nowrap"></span>
        <button type="button" class="ghost-button small-btn" onclick="extractOperator()" title="Save this projection vector for Activation Surgeon" style="display:flex;align-items:center;gap:4px;">
          <span style="font-size:11px">&DownArrowBar;</span> Extract Operator
        </button>  
        <span id="variance-label" style="font-size:11px;color:var(--muted);width:80px;text-align:right"></span>
        <div style="width:1px;height:14px;background:var(--border);flex-shrink:0"></div>
        <select id="views-sel" class="proj-select" style="font-size:10px;max-width:120px"
                onchange="onSelectView(this.value)" title="Saved views">
          <option value="">Views&hellip;</option>
        </select>
        <button type="button" class="ghost-button small-btn" onclick="saveCurrentView()"
                title="Save current projection + layer as a named view" style="padding:2px 7px;font-size:11px">+ View</button>
      </div>
    </div>

    <!-- ── PLOTS ──────────────────────────────────────────────────────── -->
    <div id="plotly-div" style="flex:2;min-height:0;position:relative;">
      <!-- Mini Map inset overlay -->
      <div id="mini-map-div" style="position:absolute; bottom:10px; right:10px; width:160px; height:160px; background:var(--panel); border:1px solid var(--border); border-radius:6px; z-index:10; pointer-events:none; box-shadow:var(--shadow);"></div>
      <!-- Context Menu -->
      <div id="context-menu" style="display:none; position:absolute; background:var(--panel); border:1px solid var(--border); box-shadow:var(--shadow); z-index:100; border-radius:6px; overflow:hidden;">
        <div class="ctx-item" onclick="ctxViewText()">View Raw Text</div>
        <div class="ctx-item" onclick="ctxIsolate()">Isolate Cluster</div>
      </div>
    </div>

    <!-- ── LAYER SLIDER (TIMELINE) ────────────────────────────────────── -->
    <div class="layer-bar" style="border-top:1px solid var(--border); border-bottom:none; background:color-mix(in srgb, var(--panel-strong) 40%, transparent);">
      <button type="button" id="play-btn" onclick="togglePlay()" style="min-width:30px">&#x25B6;</button>
      <span id="layer-label" style="white-space:nowrap;min-width:60px;font-size:11px;color:var(--muted);text-align:right;">L 0</span>
      <input type="range" id="layer-slider" min="0" max="27" value="0"
             style="flex:1" oninput="onSlider(this.value)">
    </div>

    <div style="display:flex;flex:1.2;min-height:0;border-top:1px solid var(--border)">
      <div style="display:flex;flex-direction:column;flex:1;min-height:0; position:relative;" id="charts-wrapper-1">
        <button class="ghost-button" style="position:absolute;top:5px;right:5px;z-index:10;padding:2px 6px;height:22px;font-size:10px;" onclick="toggleFullscreen(this, 'charts-wrapper-1')" title="Expand Chart">⛶</button>
        <div id="sep-div" style="flex:1;min-height:0"></div>
      </div>
      <div style="flex:1;min-height:0;border-left:1px solid var(--border); position:relative;" id="charts-wrapper-2">
        <button class="ghost-button" style="position:absolute;top:5px;right:5px;z-index:10;padding:2px 6px;height:22px;font-size:10px;" onclick="toggleFullscreen(this, 'charts-wrapper-2')" title="Expand Heatmap">⛶</button>
        <div id="heatmap-div" style="height:100%"></div>
      </div>
    </div>

    <!-- ── ACTIVATION SURGEON ────────────────────────────────────────── -->
    <div class="panel surgeon-panel" style="margin-top:8px; display:flex; flex-direction:column; gap:8px; padding:12px; border-top:1px solid var(--border); overflow:visible;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:11px;font-weight:600;color:var(--text);letter-spacing:0.04em;text-transform:uppercase;">Activation Surgeon</span>
        <span style="font-size:10px;color:var(--muted)">Injects current projection vector at current layer</span>
      </div>
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <div style="flex:1;">
           <div style="display:flex; justify-content:space-between; align-items:center;">
             <span class="sidebar-label" style="margin:0;">Prompt</span>
             <select id="surgeon-op-sel" class="proj-select" style="max-width:140px; margin-bottom:4px;" title="Select which operator to inject">
               <option value="current">Current UI Vector</option>
             </select>
           </div>
           <textarea id="surgeon-prompt" class="gitems" rows="2" style="width:100%; margin:0; resize:vertical; font-size:12px; padding:6px; min-height:45px;" placeholder="Prompt text to steer..."></textarea>
        </div>
        <div style="width:180px; display:flex; flex-direction:column; gap:6px;">
           <div style="display:flex; justify-content:space-between; align-items:center;">
             <span class="sidebar-label" style="margin:0;">Scale</span>
             <span id="surgeon-scale-val" style="font-size:11px;font-family:monospace;color:var(--primary)">3.0&times;</span>
           </div>
           <input type="range" id="surgeon-scale" min="-10" max="10" step="0.5" value="3.0" style="width:100%" oninput="document.getElementById('surgeon-scale-val').innerHTML=parseFloat(this.value).toFixed(1)+'&times;'">
           <button type="button" class="ghost-button" style="background:var(--bg-highlight); color:var(--text); justify-content:center; margin-top:4px;" onclick="runSurgery()" id="surgeon-btn">&#x2694; Operate</button>
        </div>
      </div>
      <div id="surgeon-results" style="display:none; gap:12px; margin-top:4px; min-height:100px;">
         <div style="flex:1; background:var(--bg-dark); padding:8px; border-radius:4px; border:1px solid var(--border); overflow-y:auto; max-height:220px;">
           <div style="font-size:10px; color:var(--muted); text-transform:uppercase; margin-bottom:4px; font-weight:600">Raw Baseline</div>
           <div id="surgeon-baseline" style="font-size:12px; color:var(--text); white-space:pre-wrap; font-family:var(--font-mono)"></div>
         </div>
         <div style="flex:1; background:var(--bg-dark); padding:8px; border-radius:4px; border:1px solid var(--border); border-top:2px solid var(--primary); overflow-y:auto; max-height:220px;">
           <div style="font-size:10px; color:var(--primary); text-transform:uppercase; margin-bottom:4px; font-weight:600">Surgical Output (Steered)</div>
           <div id="surgeon-intervened" style="font-size:12px; color:var(--text); white-space:pre-wrap; font-family:var(--font-mono)"></div>
         </div>
      </div>
    </div>
    
  </div>
</main>

</div>

<style>
#app { display:flex; height:calc(100vh - 52px); overflow:hidden; }

/* ── Sidebar ── */
.sidebar {
  width:280px; min-width:220px; flex-shrink:0;
  border-right:1px solid var(--border); border-radius:0;
  display:flex; flex-direction:column; overflow:hidden; box-shadow:none;
}
.sidebar-top {
  padding:10px 12px 8px; border-bottom:1px solid var(--border); flex-shrink:0;
}
.sidebar-scroll {
  flex:1; overflow-y:auto; padding:10px 12px;
  display:flex; flex-direction:column; gap:8px;
}
.sidebar-label {
  display:block; font-size:9px; text-transform:uppercase;
  letter-spacing:0.08em; color:var(--muted); margin-bottom:4px;
}
.sidebar-section {
  border:1px solid var(--border); border-radius:6px; font-size:11px;
}
.sidebar-section summary {
  padding:7px 10px; cursor:pointer; font-weight:600; font-size:11px;
  user-select:none; list-style:none; display:flex; align-items:center; gap:6px;
}
.sidebar-section summary::before { content:'\25B8'; font-size:9px; color:var(--muted); }
.sidebar-section[open] summary::before { content:'\25BE'; }
.sidebar-section[open] > :not(summary) { padding:0 10px 10px; }

/* ── Group blocks ── */
.group-block {
  border:1px solid var(--border); border-radius:6px;
  padding:8px 10px; display:flex; flex-direction:column; gap:5px;
}
.group-hdr { display:flex; align-items:center; gap:5px; cursor:pointer; }
.expand-arrow { font-size:10px; color:var(--muted); width:10px; flex-shrink:0; }
.item-count {
  font-size:10px; color:var(--muted); background:var(--tag-bg);
  padding:1px 5px; border-radius:8px; flex-shrink:0;
}
.color-dot {
  width:11px; height:11px; border-radius:50%;
  border:1.5px solid rgba(0,0,0,0.15); cursor:pointer; flex-shrink:0;
}
.gname {
  flex:1; font-size:11px; padding:3px 7px; border-radius:4px;
  border:1px solid var(--border); background:var(--panel-strong); color:var(--text);
}
.gname:focus { outline:1px solid var(--accent); }
.remove-btn {
  background:none; border:none; color:var(--muted); cursor:pointer;
  font-size:12px; padding:0; line-height:1;
}
.remove-btn:hover { color:var(--text); }
.gitems {
  font-size:11px; font-family:monospace; min-height:40px; resize:vertical;
  border-radius:4px; border:1px solid var(--border);
  padding:3px 5px; background:var(--panel-strong); color:var(--text); line-height:1.5;
}
.small-btn { font-size:11px; padding:4px 9px; border-radius:5px; }
.info-box.compact { padding:5px 7px; border:1px solid var(--border); border-radius:6px; background:var(--panel-strong); }

/* ── Legend ── */
#legend-area { display:flex; }
.leg-item { display:flex; align-items:center; gap:5px; font-size:11px; }
.leg-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }

/* ── Main area ── */
.main-area { flex:1; overflow:hidden; display:flex; flex-direction:column; }
.empty-state {
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:100%; color:var(--muted);
}

/* ── Layer bar ── */
.layer-bar {
  display:flex; align-items:center; gap:8px;
  padding:6px 12px; border-bottom:1px solid var(--border); flex-shrink:0;
}
#play-btn { padding:3px 8px; font-size:11px; border-radius:5px; min-width:30px; }
#layer-slider { accent-color:var(--accent); }
.view-toggles { display:flex; align-items:center; gap:10px; flex-shrink:0; }
.toggle-label { font-size:11px; display:flex; align-items:center; gap:3px; cursor:pointer; white-space:nowrap; }

/* ── Projection bar ── */
.proj-bar {
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:5px 12px; border-bottom:1px solid var(--border);
  background:color-mix(in srgb, var(--panel-strong) 60%, transparent);
  flex-shrink:0; min-height:32px;
}
.proj-bar-label { font-size:10px; color:var(--muted); white-space:nowrap; flex-shrink:0; }
.proj-select {
  font-size:11px; padding:3px 6px; border-radius:5px;
  border:1px solid var(--border); background:var(--panel-strong);
  color:var(--text); font-family:inherit; margin:0;
}
.proj-select:focus { outline:1px solid var(--accent); }
.proj-bar input[type="range"] {
  height:4px; margin:0; padding:0;
  accent-color:var(--accent);
}
/* Context menu */
.ctx-item {
  padding: 8px 12px; font-size: 11px; cursor: pointer; color: var(--text);
  border-bottom: 1px solid var(--border); transition: background 0.15s;
}
.ctx-item:hover { background: var(--tag-bg); }
.ctx-item:last-child { border-bottom: none; }

.proj-bar input[type="range"]::-webkit-slider-thumb { width:13px; height:13px; }
</style>
"""

_JS = r"""
// ── State ─────────────────────────────────────────────────────────────────────
const PALETTE = ['#0050ff','#ff5e00','#00c853','#d500f9','#ffd600','#00e5ff','#ff1744']; // High contrast
let groupCount   = 0;
let allFrames    = [];
let allTrajs     = {};
let currentLayer = 0;
let numLayers    = 28;
let playTimer    = null;
let DATA         = null;
let groupsMeta   = [];
let globalBounds = null;
let hoverTimer   = null;
let savedViews   = [];  // [{name, projection, view}]
let hiddenGroups = new Set();
let ctxPointLabel= null;

function onModelLoaded(nLayers) { if (nLayers) numLayers = nLayers; }

// ── Group management ──────────────────────────────────────────────────────────
function addGroup(name, color, items) {
  const idx = groupCount++;
  const col = color || PALETTE[idx % PALETTE.length];
  const itemCount = items ? items.split('\n').filter(l => l.trim()).length : 0;
  const div = document.createElement('div');
  div.className = 'group-block';
  div.id = 'gb-' + idx;
  div.innerHTML = `
    <div class="group-hdr" onclick="toggleGroupExpand(${idx}, event)">
      <span class="expand-arrow" id="arrow-${idx}">&#x25B8;</span>
      <span class="color-dot" id="dot-${idx}" style="background:${col}"
            onclick="event.stopPropagation();pickColor(${idx})" title="Click to change colour"></span>
      <input class="gname" id="gn-${idx}" value="${name || 'Group ' + (idx + 1)}" placeholder="Group name"
             onclick="event.stopPropagation()">
      <span class="item-count">${itemCount}</span>
      <button class="remove-btn" onclick="event.stopPropagation();removeGroup(${idx})">&#x2715;</button>
    </div>
    <textarea class="gitems" id="gi-${idx}" rows="5" style="display:none"
      placeholder="One item per line&#10;Or: label | full text to encode"
      oninput="updateItemCount(${idx})">${items || ''}</textarea>
  `;
  document.getElementById('groups-container').appendChild(div);
}

function toggleGroupExpand(idx, event) {
  const ta = document.getElementById('gi-' + idx);
  const arrow = document.getElementById('arrow-' + idx);
  if (!ta) return;
  const hidden = ta.style.display === 'none';
  ta.style.display = hidden ? '' : 'none';
  arrow.textContent = hidden ? '\u25BE' : '\u25B8';
}

function updateItemCount(idx) {
  const ta = document.getElementById('gi-' + idx);
  const block = document.getElementById('gb-' + idx);
  if (!ta || !block) return;
  block.querySelector('.item-count').textContent =
    ta.value.split('\n').filter(l => l.trim()).length;
}

function removeGroup(idx) {
  const el = document.getElementById('gb-' + idx);
  if (el) el.remove();
}

function pickColor(idx) {
  const inp = document.createElement('input');
  inp.type = 'color';
  inp.value = rgbToHex(document.getElementById('dot-' + idx).style.background);
  inp.addEventListener('input', e => {
    const newColor = e.target.value;
    document.getElementById('dot-' + idx).style.background = newColor;
    const groupName = document.getElementById('gn-' + idx)?.value?.trim();
    if (groupName && allFrames.length) {
      allFrames.forEach(f => f.points.forEach(p => { if (p.group === groupName) p.color = newColor; }));
      Object.values(allTrajs).forEach(t => { if (t.group === groupName) t.color = newColor; });
      renderLayer(currentLayer);
    }
  });
  inp.click();
}

function rgbToHex(str) {
  if (!str) return '#888888';
  if (str.startsWith('#')) return str;
  const m = str.match(/\d+/g);
  if (!m) return '#888888';
  return '#' + m.slice(0,3).map(x => parseInt(x).toString(16).padStart(2,'0')).join('');
}

function collectGroups() {
  const groups = [];
  document.querySelectorAll('.group-block').forEach(bl => {
    const idx = bl.id.split('-')[1];
    const name  = document.getElementById('gn-' + idx)?.value?.trim() || 'Group';
    const color = rgbToHex(document.getElementById('dot-' + idx)?.style?.background || '#888');
    const items = document.getElementById('gi-' + idx)?.value || '';
    if (items.trim()) groups.push({name, color, items});
  });
  return groups;
}

function toggleAllGroups(show) {
  hiddenGroups.clear();
  if (!show) {
    const groups = collectGroups();
    groups.forEach(g => hiddenGroups.add(g.name));
  }
  _updateLegendUI();
  renderLayer(currentLayer);
}

function _updateLegendUI() {
  document.querySelectorAll('.leg-item').forEach(el => {
    const gName = el.getAttribute('data-group');
    el.style.opacity = hiddenGroups.has(gName) ? '0.4' : '1';
  });
}

function toggleGroupVis(gName) {
  if (hiddenGroups.has(gName)) hiddenGroups.delete(gName);
  else hiddenGroups.add(gName);
  _updateLegendUI();
  renderLayer(currentLayer);
}

// Fullscreen
function toggleFullscreen(btn, wrapperId) {
  const wr = document.getElementById(wrapperId);
  if (wr.classList.contains('fullscreen-wrapper')) {
    wr.classList.remove('fullscreen-wrapper');
    btn.textContent = '⛶';
  } else {
    wr.classList.add('fullscreen-wrapper');
    btn.textContent = '✖';
  }
  // Trigger plotly resize
  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}

// ── Projection Engine ─────────────────────────────────────────────────────────
let projDebounce = null;
let projBusy = false;
let pendingReproject = false;

function onProjectionMethodChange() {
  const method = document.getElementById('proj-method').value;
  const needsLabels = method !== 'pca';
  document.getElementById('basis-layer-controls').style.display = needsLabels ? 'flex' : 'none';
  document.getElementById('cpca-controls').style.display        = method === 'cpca' ? 'flex' : 'none';
  document.getElementById('direction-controls').style.display   = (method === 'probe_aligned' || method === 'null_space') ? 'flex' : 'none';
  document.getElementById('grassmannian-controls').style.display = method === 'grassmannian' ? 'flex' : 'none';
  runReproject();
}

function onBasisLayer(val) {
  document.getElementById('basis-layer-val').textContent = val;
  clearTimeout(projDebounce); projDebounce = setTimeout(runReproject, 16);
}

function onCpcaAlpha(val) {
  document.getElementById('cpca-alpha-val').textContent = parseFloat(val).toFixed(1);
  clearTimeout(projDebounce); projDebounce = setTimeout(runReproject, 16);
}

function onGrassT(val) {
  document.getElementById('grass-t-val').textContent = parseFloat(val).toFixed(2);
  clearTimeout(projDebounce); projDebounce = setTimeout(runReproject, 16);
}

function collectReprojParams() {
  const method = document.getElementById('proj-method').value;
  const projParams = {};
  const basisEl = document.getElementById('basis-layer-slider');
  if (basisEl) projParams.basis_layer = parseInt(basisEl.value);
  if (method === 'cpca') {
    projParams.alpha = parseFloat(document.getElementById('cpca-alpha').value);
    const tgt = document.getElementById('cpca-target-group');
    if (tgt && tgt.value !== "-1") {
      projParams.target_group_name = tgt.value;
    }
  }
  if (method === 'grassmannian') {
    projParams.method_a = document.getElementById('grass-method-a').value;
    projParams.method_b = document.getElementById('grass-method-b').value;
    projParams.t = parseFloat(document.getElementById('grass-t').value);
  }
  if (method === 'probe_aligned' || method === 'null_space') {
    projParams.probe_group_a = document.getElementById('probe-group-a')?.value;
    projParams.probe_group_b = document.getElementById('probe-group-b')?.value;
  }
  return {method, projParams};
}

function runReproject() {
  if (!allFrames.length) return;
  if (projBusy) { pendingReproject = true; return; }
  pendingReproject = false;
  const {method, projParams} = collectReprojParams();
  const statusEl = document.getElementById('proj-status');
  statusEl.textContent = 'Reprojecting\u2026';
  projBusy = true;

  postRun({action: 'reproject', method, proj_params: projParams}, (data) => {
    if (data.type === 'reproject_result') {
      projBusy = false;
      allFrames = data.frames;
      allTrajs  = data.trajectories;
      _recomputeBounds();
      _updateVarianceLabel(data.variance_explained, data.projection_label);
      renderLayer(currentLayer);
      if (pendingReproject) runReproject();
    } else if (data.type === 'error') {
      projBusy = false;
      statusEl.textContent = data.message || 'Error';
      if (pendingReproject) runReproject();
    }
  }, () => { projBusy = false; if (pendingReproject) runReproject(); });
}

function _recomputeBounds() {
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity, minZ=Infinity, maxZ=-Infinity;
  allFrames.forEach(f => f.points.forEach(p => {
    if (p.x < minX) minX=p.x; if (p.x > maxX) maxX=p.x;
    if (p.y < minY) minY=p.y; if (p.y > maxY) maxY=p.y;
    if (p.z !== undefined) { if (p.z < minZ) minZ=p.z; if (p.z > maxZ) maxZ=p.z; }
  }));
  globalBounds = {x:[minX,maxX], y:[minY,maxY], z:[minZ,maxZ]};
}

function _updateVarianceLabel(v, projLabel) {
  const statusEl = document.getElementById('proj-status');
  if (v && v[0] > 0) {
    statusEl.textContent =
      `PC1 ${(v[0]*100).toFixed(1)}%  PC2 ${(v[1]*100).toFixed(1)}%` +
      (v[2] ? `  PC3 ${(v[2]*100).toFixed(1)}%` : '');
  } else {
    statusEl.textContent = projLabel || '';
  }
  if (v && (v[0]+v[1]) > 0) {
    const pct1 = Math.round(v[0]/(v[0]+v[1]+(v[2]||0)+1e-9)*100);
    document.getElementById('pca-bar').innerHTML =
      `<div style="height:4px;border-radius:2px;overflow:hidden;background:var(--border)">
         <div style="height:100%;width:${pct1}%;background:#1f77b4;float:left"></div>
         <div style="height:100%;width:${100-pct1}%;background:#ff7f0e;float:left"></div>
       </div>`;
    document.getElementById('pca-text').textContent =
      `PC1 ${(v[0]*100).toFixed(1)}% · PC2 ${(v[1]*100).toFixed(1)}%` +
      (v[2] ? ` · PC3 ${(v[2]*100).toFixed(1)}%` : '');
  }
}

// ── Sessions ──────────────────────────────────────────────────────────────────
function collectSessionState() {
  const p = {};
  ['proj-method','grass-method-a','grass-method-b'].forEach(id => {
    const el = document.getElementById(id); if (el) p[id] = el.value;
  });
  ['basis-layer-slider','cpca-alpha','grass-t'].forEach(id => {
    const el = document.getElementById(id); if (el) p[id] = parseFloat(el.value);
  });
  return {
    groups: collectGroups(),
    projection: p,
    view: {
      layer: currentLayer,
      is3d:    document.getElementById('is-3d-chk')?.checked || false,
      lockAxis:document.getElementById('lock-axis-chk')?.checked !== false,
      trails:  document.getElementById('trails-chk')?.checked !== false,
      labels:  document.getElementById('labels-chk')?.checked !== false,
    },
    savedViews: savedViews.slice(),
  };
}

function restoreSessionState(session) {
  document.getElementById('groups-container').innerHTML = '';
  groupCount = 0;
  (session.groups || []).forEach(g => addGroup(g.name, g.color, g.items));

  if (session.projection) {
    const p = session.projection;
    ['proj-method','grass-method-a','grass-method-b'].forEach(id => {
      const el = document.getElementById(id); if (el && p[id]) el.value = p[id];
    });
    const basisEl = document.getElementById('basis-layer-slider');
    if (basisEl && p['basis-layer-slider'] != null) {
      basisEl.value = p['basis-layer-slider'];
      document.getElementById('basis-layer-val').textContent = basisEl.value;
    }
    const alphaEl = document.getElementById('cpca-alpha');
    if (alphaEl && p['cpca-alpha'] != null) {
      alphaEl.value = p['cpca-alpha'];
      document.getElementById('cpca-alpha-val').textContent = parseFloat(alphaEl.value).toFixed(1);
    }
    const gtEl = document.getElementById('grass-t');
    if (gtEl && p['grass-t'] != null) {
      gtEl.value = p['grass-t'];
      document.getElementById('grass-t-val').textContent = parseFloat(gtEl.value).toFixed(2);
    }
    onProjectionMethodChange();
  }

  if (session.view) {
    const v = session.view;
    if (document.getElementById('is-3d-chk'))    document.getElementById('is-3d-chk').checked    = !!v.is3d;
    if (document.getElementById('lock-axis-chk')) document.getElementById('lock-axis-chk').checked = v.lockAxis !== false;
    if (document.getElementById('trails-chk'))   document.getElementById('trails-chk').checked   = v.trails !== false;
    if (document.getElementById('labels-chk'))   document.getElementById('labels-chk').checked   = v.labels !== false;
    if (v.layer != null) currentLayer = v.layer;
  }

  // Restore saved views
  savedViews = Array.isArray(session.savedViews) ? session.savedViews : [];
  _renderViewsList();
}

function saveSession() {
  const name = document.getElementById('session-name').value.trim();
  if (!name) { setStatus('Enter a session name.', true); return; }
  postRun({action:'save_session', save_name:name, session:collectSessionState()}, (data) => {
    if (data.type === 'status') {
      setStatus(data.message);
      document.getElementById('session-name').value = '';
      loadSessionList();
    }
  });
}

function openSessionsFolder() {
  postRun({action:'open_sessions_folder'}, () => {});
}

function loadSessionList() {
  postRun({action:'list_sessions'}, (data) => {
    const sel = document.getElementById('session-sel');
    if (!sel) return;
    sel.innerHTML = '<option value="">Choose\u2026</option>';
    (data.sessions || []).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      sel.appendChild(opt);
    });
  });
}

function onSelectSession(name) {
  if (!name) return;
  document.getElementById('session-sel').value = '';
  postRun({action:'load_session', session_name:name}, (data) => {
    if (data.type === 'session_data' && data.session) {
      restoreSessionState(data.session);
      setStatus('Restored: ' + name);
    } else if (data.type === 'error') {
      setStatus(data.message, true);
    }
  });
}

// ── Saved Views ───────────────────────────────────────────────────────────────
function collectViewSnapshot() {
  const {method, projParams} = collectReprojParams();
  return {
    projection: {method, projParams},
    view: {
      layer:    currentLayer,
      is3d:     document.getElementById('is-3d-chk')?.checked || false,
      lockAxis: document.getElementById('lock-axis-chk')?.checked !== false,
      trails:   document.getElementById('trails-chk')?.checked !== false,
      labels:   document.getElementById('labels-chk')?.checked !== false,
    }
  };
}

function saveCurrentView() {
  const defaultName = 'View ' + (savedViews.length + 1);
  const name = (window.prompt('Name this view:', defaultName) || '').trim();
  if (!name) return;
  // Replace if same name exists, otherwise append
  const existingIdx = savedViews.findIndex(v => v.name === name);
  const snapshot = { name, ...collectViewSnapshot() };
  if (existingIdx >= 0) {
    savedViews[existingIdx] = snapshot;
    setStatus('Updated view: ' + name);
  } else {
    savedViews.push(snapshot);
    setStatus('Saved view: ' + name);
  }
  _renderViewsList();
}

function deleteCurrentView() {
  const sel = document.getElementById('views-sel');
  const idx = parseInt(sel.value);
  if (isNaN(idx) || idx < 0 || idx >= savedViews.length) return;
  const name = savedViews[idx].name;
  savedViews.splice(idx, 1);
  _renderViewsList();
  setStatus('Deleted view: ' + name);
}

function onSelectView(val) {
  if (val === '') return;
  const idx = parseInt(val);
  if (isNaN(idx) || idx < 0 || idx >= savedViews.length) return;
  _applyView(savedViews[idx]);
  // Reset dropdown back to placeholder so it's re-selectable
  document.getElementById('views-sel').value = '';
}

function _applyView(v) {
  if (!v) return;
  // Restore projection controls
  const p = v.projection;
  if (p) {
    const methodEl = document.getElementById('proj-method');
    if (methodEl && p.method) methodEl.value = p.method;
    const pp = p.projParams || {};
    if (pp.basis_layer != null) {
      const el = document.getElementById('basis-layer-slider');
      if (el) { el.value = pp.basis_layer; document.getElementById('basis-layer-val').textContent = pp.basis_layer; }
    }
    if (pp.alpha != null) {
      const el = document.getElementById('cpca-alpha');
      if (el) { el.value = pp.alpha; document.getElementById('cpca-alpha-val').textContent = parseFloat(pp.alpha).toFixed(1); }
    }
    if (pp.t != null) {
      const el = document.getElementById('grass-t');
      if (el) { el.value = pp.t; document.getElementById('grass-t-val').textContent = parseFloat(pp.t).toFixed(2); }
    }
    if (pp.method_a) { const el = document.getElementById('grass-method-a'); if (el) el.value = pp.method_a; }
    if (pp.method_b) { const el = document.getElementById('grass-method-b'); if (el) el.value = pp.method_b; }
    onProjectionMethodChange();  // shows/hides sub-controls + triggers reproject
  }
  // Restore view toggles + layer
  const vw = v.view;
  if (vw) {
    if (document.getElementById('is-3d-chk'))    document.getElementById('is-3d-chk').checked    = !!vw.is3d;
    if (document.getElementById('lock-axis-chk')) document.getElementById('lock-axis-chk').checked = vw.lockAxis !== false;
    if (document.getElementById('trails-chk'))   document.getElementById('trails-chk').checked   = vw.trails !== false;
    if (document.getElementById('labels-chk'))   document.getElementById('labels-chk').checked   = vw.labels !== false;
    if (vw.layer != null) {
      currentLayer = vw.layer;
      const sl = document.getElementById('layer-slider'); if (sl) sl.value = currentLayer;
      const lb = document.getElementById('layer-label');  if (lb) lb.textContent = 'Layer ' + currentLayer;
    }
  }
  setStatus('Applied view: ' + v.name);
}

function _renderViewsList() {
  const sel = document.getElementById('views-sel');
  if (!sel) return;
  sel.innerHTML = '<option value="">Views\u2026</option>';
  savedViews.forEach((v, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    // Build a compact description of the view
    const method = v.projection?.method || '?';
    const layer  = v.view?.layer ?? '?';
    opt.textContent = v.name + ' (' + method + ' L' + layer + ')';
    sel.appendChild(opt);
  });
}

// ── Presets ───────────────────────────────────────────────────────────────────
function loadPresets() {
  postRun({action:'load_presets'}, (data) => {
    const sel = document.getElementById('preset-sel');
    sel.innerHTML = '<option value="">Select preset\u2026</option>';
    (data.presets || []).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.key; opt.textContent = p.name;
      sel.appendChild(opt);
    });
  });
}

function applyPreset(append) {
  const key = document.getElementById('preset-sel').value;
  if (!key) return;
  postRun({action:'load_preset', key}, (data) => {
    if (data.groups) {
      if (!append) {
        document.getElementById('groups-container').innerHTML = '';
        groupCount = 0;
      }
      data.groups.forEach(g => {
        const colorToUse = append ? undefined : g.color;
        addGroup(g.name, colorToUse, g.items);
      });
    } else if (data.type === 'error') {
      setStatus(data.message, true);
    }
  });
}

function extractOperator() {
  const name = prompt("Name this Operator (Steering Vector):", "my_operator");
  if (!name) return;
  
  const basisEl = document.getElementById('basis-layer-slider');
  const layer = basisEl ? parseInt(basisEl.value) : -1;
  const method = document.getElementById('proj-method').value;
  
  setStatus('Extracting ' + method + ' vector\u2026');
  fetch('/api/extract_operator', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, layer })
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      setStatus("Error: " + data.error, true);
    } else {
      setStatus("Saved " + name);
      loadOperators(); // Refresh dropdown
    }
  })
  .catch(err => setStatus("Failed to extract: " + err, true));
}

function loadOperators() {
  fetch('/api/operators').then(r=>r.json()).then(data => {
    const sel = document.getElementById('surgeon-op-sel');
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="current">Current UI Vector</option>';
    if (data.operators && data.operators.length) {
      data.operators.forEach(op => {
        sel.innerHTML += `<option value="${op}">Extracted: ${op}</option>`;
      });
    }
    sel.value = current;
  }).catch(()=>{});
}
// Load automatically on script start
loadOperators();

function runSurgery() {
  const prompt = document.getElementById('surgeon-prompt').value.trim();
  if (!prompt) { setStatus("Enter a prompt for the surgeon.", true); return; }
  
  const scale = parseFloat(document.getElementById('surgeon-scale').value);
  const layer = currentLayer;  // Use timeline's current layer
  const operatorName = document.getElementById('surgeon-op-sel').value;
  
  const btn = document.getElementById('surgeon-btn');
  btn.disabled = true;
  document.getElementById('surgeon-results').style.display = 'flex';
  document.getElementById('surgeon-baseline').textContent = 'Generating...';
  document.getElementById('surgeon-intervened').textContent = 'Generating...';
  
  setStatus('Operating on prompt (may take 10-20s)\\u2026');
  
  fetch('/api/steer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      prompt: prompt, 
      scale: scale, 
      layer: layer,
      operator_name: operatorName,
      max_tokens: 100
    })
  })
  .then(resp => {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    function read() {
      reader.read().then(({done, value}) => {
        if (done) { 
          btn.disabled = false;
          return; 
        }
        buffer += decoder.decode(value, {stream:true});
        let lines = buffer.split('\n\n');
        buffer = lines.pop(); // keep partial chunk
        lines.forEach(block => {
          if (block.startsWith('data: ')) {
            try {
              const data = JSON.parse(block.substring(6));
              if (data.type === 'status') {
                setStatus(data.message);
              } else if (data.type === 'steer_result') {
                document.getElementById('surgeon-baseline').textContent = data.baseline;
                document.getElementById('surgeon-intervened').textContent = data.intervened;
                setStatus("Surgery complete.", false);
              } else if (data.type === 'error') {
                setStatus(data.message, true);
                document.getElementById('surgeon-intervened').textContent = "Failed.";
                document.getElementById('surgeon-baseline').textContent = "Failed.";
              }
            } catch(e) {
                document.getElementById('surgeon-baseline').textContent = "JSON parse error: " + e.toString();
                document.getElementById('surgeon-intervened').textContent = "Block content:\\n" + block.substring(0, 500) + "...";
                setStatus("UI parsing failed.", true);
            }
          }
        });
        read();
      }).catch(err => {
         setStatus("Stream error: " + err, true);
         btn.disabled = false;
      });
    }
    read();
  }).catch(err => {
    setStatus("Surgeon failed: " + err, true);
    btn.disabled = false;
  });
}

// ── Encoding ──────────────────────────────────────────────────────────────────
function runEncoding() {
  const groups = collectGroups();
  if (!groups.length) { setStatus('Add at least one group with items.', true); return; }
  stopPlay();
  document.getElementById('run-btn').disabled = true;
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('plot-area').style.display = 'none';
  setStatus('Connecting\u2026');

  postRun({groups}, (data) => {
    if (data.type === 'status') {
      setStatus(data.message || '');
    } else if (data.type === 'error') {
      setStatus(data.message || 'Error', true);
      document.getElementById('run-btn').disabled = false;
    } else if (data.type === 'result') {
      document.getElementById('run-btn').disabled = false;
      onResult(data);
    }
  }, () => { document.getElementById('run-btn').disabled = false; });
}

function setStatus(msg, isErr) {
  const el = document.getElementById('status-area');
  el.textContent = msg;
  el.style.color = isErr ? 'var(--error)' : 'var(--muted)';
}

// ── Result ────────────────────────────────────────────────────────────────────
function onResult(data) {
  DATA = data;
  allFrames  = data.frames;
  allTrajs   = data.trajectories;
  numLayers  = data.n_layers;
  groupsMeta = data.groups;
  currentLayer = 0;

  _recomputeBounds();

  const slider = document.getElementById('layer-slider');
  slider.max = numLayers - 1;
  slider.value = 0;

  // Reset projection bar
  document.getElementById('proj-method').value = 'pca';
  document.getElementById('basis-layer-controls').style.display = 'none';
  document.getElementById('cpca-controls').style.display = 'none';
  document.getElementById('direction-controls').style.display = 'none';
  document.getElementById('grassmannian-controls').style.display = 'none';
  const basisSlider = document.getElementById('basis-layer-slider');
  basisSlider.max = numLayers - 1;
  basisSlider.value = numLayers - 1;
  document.getElementById('basis-layer-val').textContent = numLayers - 1;

  // Variance bar + legend
  document.getElementById('pca-info').style.display = 'block';
  _updateVarianceLabel(data.variance_explained, 'PCA');

  const pA = document.getElementById('probe-group-a');
  const pB = document.getElementById('probe-group-b');
  const cTgt = document.getElementById('cpca-target-group');
  if (data.groups) {
    const opts = data.groups.map(([name, color]) => `<option value="${name}">${name}</option>`).join('');
    if (pA && pB) {
      pA.innerHTML = opts;
      pB.innerHTML = opts;
      if (data.groups.length >= 2) {
         pA.value = data.groups[0][0];
         pB.value = data.groups[1][0];
      }
    }
    if (cTgt) {
      cTgt.innerHTML = `<option value="-1">Target: Auto</option>` + data.groups.map(([name, color]) => `<option value="${name}">Target: ${name}</option>`).join('');
    }
  }

  const legEl = document.getElementById('legend-area');
  legEl.style.display = 'flex';
  document.getElementById('legend-controls').style.display = 'flex';
  legEl.innerHTML = data.groups.map(([name, color]) =>
    `<div class="leg-item" data-group="${name}" style="cursor:pointer" onclick="toggleGroupVis('${name}')"><div class="leg-dot" style="background:${color}"></div>${name}</div>`
  ).join('');

  setStatus(`${data.n_items} items · ${numLayers} layers`);
  document.getElementById('plot-area').style.display = 'flex';
  drawMetricsCurve(data, 0);
  renderHeatmap(0);
  renderLayer(0);
}

// ── Metrics Curve (Separation & Sim) ──────────────────────────────────────────
function drawMetricsCurve(data, initLayer) {
  const colorMap = Object.fromEntries(data.groups);
  const layers = Array.from({length: numLayers}, (_, i) => i);
  const traces = [];
  
  Object.entries(data.sep_curves || {}).forEach(([key, curve]) => {
    const gA = key.split(' \u2194 ')[0];
    traces.push({
      type:'scatter', mode:'lines', name:'Sep: '+key,
      x:layers, y:curve.norm,
      line:{color:colorMap[gA]||'#888', width:2},
      hovertemplate:'Sep '+key+'<br>Layer %{x}: %{y:.2f}<extra></extra>'
    });
  });

  if (data.within_sim) {
    traces.push({type:'scatter', mode:'lines', x:layers, y:data.within_sim,
       line:{color:'#2ca02c',width:2, dash:'dot'}, name:'Within Sim', yaxis:'y2', hovertemplate:'Within: %{y:.3f}<extra></extra>'});
  }
  if (data.between_sim) {
    traces.push({type:'scatter', mode:'lines', x:layers, y:data.between_sim,
       line:{color:'#d62728',width:2, dash:'dot'}, name:'Between Sim', yaxis:'y2', hovertemplate:'Between: %{y:.3f}<extra></extra>'});
  }
  if (data.discriminability) {
     traces.push({type:'scatter', mode:'lines+markers', x:layers, y:data.discriminability,
       line:{color:'#1f77b4',width:2.5}, marker:{size:3}, name:'Disc.', yaxis:'y2', hovertemplate:'Disc: %{y:.3f}<extra></extra>'});
  }

  const shapes = [{type:'line', x0:initLayer, x1:initLayer, yref:'paper', y0:0, y1:1,
                    line:{color:'rgba(0,0,0,0.25)', width:1.5, dash:'dash'}, layer:'below'}];
  (data.critical_layers||[]).forEach(l => {
    shapes.push({type:'rect', x0:l-0.4, x1:l+0.4, yref:'paper', y0:0, y1:1,
                  fillcolor:'rgba(255,0,0,0.05)', line:{width:0}, layer:'below'});
  });

  Plotly.newPlot('sep-div', traces, {
    margin:{t:6,r:36,b:28,l:36},
    xaxis:{title:{text:'Layer',font:{size:10}}, range:[-0.5,numLayers-0.5]},
    yaxis:{title:{text:'Separation',font:{size:10}}, range:[-0.05,1.1]},
    yaxis2:{overlaying:'y', side:'right', showgrid:false, title:{text:'Sim / Score',font:{size:10,color:'#1f77b4'}}, range:[-0.1,1.05]},
    plot_bgcolor:'var(--plot-panel)', paper_bgcolor:'transparent',
    font:{family:'inherit', color:'var(--text)', size:10},
    showlegend:true, legend:{orientation:'h', x:0.5, xanchor:'center', y:-0.3, font:{size:9}},
    shapes,
  }, {responsive:true, displayModeBar:false});
  
  document.getElementById('sep-div').on('plotly_click', d => {
    if (d.points?.length) renderLayer(Math.round(d.points[0].x));
  });
}

// ── Heatmap ───────────────────────────────────────────────────────────────────
function renderHeatmap(layer) {
  if (!DATA?.similarity_matrices) return;
  const sim = DATA.similarity_matrices[layer];
  const labels = DATA.item_labels;
  if (!sim || !labels) return;
  Plotly.react('heatmap-div', [{
    z:sim, x:labels, y:labels, type:'heatmap',
    colorscale:[[0,'#2166ac'],[0.25,'#67a9cf'],[0.5,'#f7f7f7'],[0.75,'#d6936b'],[1,'#b2182b']],
    zmin:-0.2, zmax:1.0, zmid:0.4,
    colorbar:{title:{text:'Sim',side:'right',font:{size:9}}, len:0.85, thickness:8, tickfont:{size:8}},
    hovertemplate:'<b>%{y}</b> vs <b>%{x}</b><br>%{z:.3f}<extra></extra>',
  }], {
    margin:{t:20,r:55,b:55,l:55},
    xaxis:{tickangle:-45, tickfont:{size:8,family:'monospace'}},
    yaxis:{tickfont:{size:8,family:'monospace'}, autorange:'reversed'},
    plot_bgcolor:'var(--plot-panel)', paper_bgcolor:'transparent',
    font:{family:'inherit', color:'var(--text)'},
    annotations:[{x:0.5, y:1.04, xref:'paper', yref:'paper',
      text:'<b>Similarity</b> \u2014 L' + layer,
      showarrow:false, font:{size:10}}],
    uirevision:'heat',
  }, {responsive:true, displayModeBar:false});
}

// ── Scatter render ────────────────────────────────────────────────────────────
function renderLayer(layer) {
  currentLayer = layer;
  document.getElementById('layer-label').textContent = `Layer ${layer} / ${numLayers-1}`;
  document.getElementById('layer-slider').value = layer;
  if (!allFrames.length) return;

  const show3D     = document.getElementById('is-3d-chk').checked;
  const showTrails = document.getElementById('trails-chk').checked;
  const showLabels = document.getElementById('labels-chk').checked;
  const lockAxis   = document.getElementById('lock-axis-chk').checked;
  const showHulls  = document.getElementById('hulls-chk') ? document.getElementById('hulls-chk').checked : false;
  const frame      = allFrames[layer];
  const colorMap   = {};
  frame.points.forEach(p => { colorMap[p.group] = p.color; });
  const traces     = [];

  if (showTrails) {
    Object.values(allTrajs).forEach(t => {
      const tr = {
        type:show3D?'scatter3d':'scatter', mode:'lines',
        x:t.xs.slice(0,layer+1), y:t.ys.slice(0,layer+1),
        line:{color:t.color, width:2}, opacity:0.15, showlegend:false, hoverinfo:'skip'
      };
      if (show3D && t.zs) tr.z = t.zs.slice(0,layer+1);
      traces.push(tr);
    });
  }

  Object.keys(colorMap).forEach(g => {
    if (hiddenGroups.has(g)) return;
    const pts = frame.points.filter(p => p.group === g);
    if (!pts.length) return;
    const tr = {
      type:show3D?'scatter3d':'scatter',
      mode:showLabels?'markers+text':'markers',
      x:pts.map(p=>p.x), y:pts.map(p=>p.y),
      text:pts.map(p=>p.label), textposition:'top center',
      textfont:{size:10},
      marker:{size:show3D?6:10, color:colorMap[g], line:{color:'rgba(255,255,255,0.7)',width:1}},
      name:g, hoverinfo:'text'
    };
    if (show3D) tr.z = pts.map(p=>p.z||0);
    traces.push(tr);
    
    if (showHulls && show3D && pts.length >= 4) {
      traces.push({
        type:'mesh3d', x:pts.map(p=>p.x), y:pts.map(p=>p.y), z:pts.map(p=>p.z||0),
        alphahull:0, opacity:0.15, color:colorMap[g], hoverinfo:'skip', showlegend:false
      });
    }
  });

  const projMethod = document.getElementById('proj-method')?.value || 'pca';
  const axisNames = {
    pca:['PC1','PC2','PC3'], lda:['LD1','LD2','LD3'], cpca:['cPC1','cPC2','cPC3'],
    probe_aligned:['Probe','Orth1','Orth2'], null_space:['Res1','Res2','Res3'],
    grassmannian:['Gr1','Gr2','Gr3'],
  }[projMethod] || ['Axis1','Axis2','Axis3'];

  const layout = {
    margin:{t:4,r:4,b:4,l:4},
    plot_bgcolor:'var(--plot-panel)', paper_bgcolor:'transparent',
    font:{family:'inherit', color:'var(--text)'},
    legend:{orientation:'h', y:-0.08, font:{size:10}},
    uirevision:'true',
  };
  if (show3D) {
    layout.scene = {
      xaxis:{title:axisNames[0], showgrid:true, gridcolor:'rgba(128,128,128,0.2)'},
      yaxis:{title:axisNames[1], showgrid:true, gridcolor:'rgba(128,128,128,0.2)'},
      zaxis:{title:axisNames[2], showgrid:true, gridcolor:'rgba(128,128,128,0.2)'},
      camera:{eye:{x:1.5,y:1.5,z:1.5}}
    };
    if (lockAxis && globalBounds) {
      layout.scene.xaxis.range = globalBounds.x;
      layout.scene.yaxis.range = globalBounds.y;
      layout.scene.zaxis.range = globalBounds.z;
    }
  } else {
    layout.xaxis = {title:axisNames[0], scaleanchor:'y', scaleratio:1, showgrid:true, gridcolor:'rgba(128,128,128,0.2)'};
    layout.yaxis = {title:axisNames[1], showgrid:true, gridcolor:'rgba(128,128,128,0.2)'};
    if (lockAxis && globalBounds) {
      layout.xaxis.range = globalBounds.x;
      layout.yaxis.range = globalBounds.y;
    }
  }

  Plotly.react('plotly-div', traces, layout, {responsive:true});
  
  // Render Mini-Map
  const miniTraces = [];
  Object.keys(colorMap).forEach(g => {
    if (hiddenGroups.has(g)) return;
    const pts = frame.points.filter(p => p.group === g);
    if (!pts.length) return;
    miniTraces.push({
      type:'scatter', mode:'markers', x:pts.map(p=>p.x), y:pts.map(p=>p.y),
      marker:{size:3, color:colorMap[g]}, hoverinfo:'skip'
    });
  });
  Plotly.react('mini-map-div', miniTraces, {
    margin:{t:0,r:0,b:0,l:0}, showlegend:false, plot_bgcolor:'transparent', paper_bgcolor:'transparent',
    xaxis:{visible:false}, yaxis:{visible:false, scaleanchor:'x', scaleratio:1}
  }, {staticPlot:true});

  Plotly.relayout('sep-div',   {'shapes[0].x0':layer, 'shapes[0].x1':layer});
  renderHeatmap(layer);

  // Lens on hover & Context Menu & Sync Heatmap Highlight
  const gd = document.getElementById('plotly-div');
  gd.on('plotly_hover', (ev) => {
    if (hoverTimer) clearTimeout(hoverTimer);
    const pt = ev.points[0];
    if (!pt?.text) return;
    hoverTimer = setTimeout(() => {
      postRun({action:'lens', label:pt.text, layer:currentLayer}, (data) => {
        if (data.type === 'lens_result' && data.tokens) {
          setStatus('Lens \u2192 ' + data.tokens.slice(0,5).map(t=>t.token).join(' · '));
        }
      });
    }, 160);
  });
  
  // Right click Context Menu
  gd.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });
  gd.on('plotly_click', (ev) => {
    if (ev.event.button === 2 || ev.event.ctrlKey) {
      ev.event.preventDefault();
      const pt = ev.points[0];
      if (!pt?.text) return;
      ctxPointLabel = pt.text;
      const menu = document.getElementById('context-menu');
      menu.style.display = 'block';
      let rect = gd.getBoundingClientRect();
      menu.style.left = (ev.event.clientX - rect.left) + 'px';
      menu.style.top = (ev.event.clientY - rect.top) + 'px';
    } else {
        document.getElementById('context-menu').style.display = 'none';
    }
  });
}

function ctxViewText() {
  document.getElementById('context-menu').style.display = 'none';
  if (!ctxPointLabel) return;
  alert('Raw text for: ' + ctxPointLabel + '\n\n' + 'This feature typically loads the exact text slice from the backend if encoded.');
}

function ctxIsolate() {
  document.getElementById('context-menu').style.display = 'none';
  if (!ctxPointLabel) return;
  // find group this label belongs to
  let grp = null;
  allFrames[currentLayer].points.forEach(p => { if (p.label === ctxPointLabel) grp = p.group; });
  if (!grp) return;
  // Hide all but this group
  hiddenGroups.clear();
  const groups = collectGroups();
  groups.forEach(g => { if (g.name !== grp) hiddenGroups.add(g.name); });
  _updateLegendUI();
  renderLayer(currentLayer);
}

// Global click dismiss ctx menu
document.addEventListener('click', (e) => {
  if (!e.target.closest('#context-menu') && document.getElementById('context-menu')) {
    document.getElementById('context-menu').style.display = 'none';
  }
});

function onSlider(val) { renderLayer(parseInt(val)); }

function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  document.getElementById('play-btn').textContent = '\u23F8';
  playTimer = setInterval(() => {
    const next = (currentLayer+1) % numLayers;
    renderLayer(next);
    if (next === numLayers-1) stopPlay();
  }, 240);
}

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  document.getElementById('play-btn').textContent = '\u25B6';
}

// Init
loadPresets();
loadSessionList();
addGroup('Group 1', '#1f77b4', '');
addGroup('Group 2', '#ff7f0e', '');
"""


def render_workspace(model_name: str = "") -> str:
    return page_shell("Activation Space", _BODY, _JS, model_label=model_name)
