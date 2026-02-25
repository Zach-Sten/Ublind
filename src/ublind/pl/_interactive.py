"""Interactive scatter plot with cluster sonification."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ublind.pl._utils import get_ublind_uns


def interactive(
    adata,
    *,
    color_by: str = "leiden",
    embedding: Optional[str] = None,
    dims: tuple[int, int] = (0, 1),
    scale: str = "pentatonic",
    root: int = 0,
    n_chord_notes: int = 12,
    point_size: int = 4,
    width: int = 700,
    height: int = 700,
):
    """
    Interactive scatter plot — click a cluster to hear its chord.

    Renders an HTML widget in Jupyter with Web Audio API.
    Each cluster's points are mapped to pitches; clicking a cluster
    plays them as an arpeggiated chord.

    Parameters
    ----------
    adata : AnnData
        Must have been preprocessed with ``ub.pp.preprocess()``.
    color_by : str
        Categorical column in ``adata.obs`` for cluster identity.
    embedding : str, optional
        Override embedding key. Defaults to whatever was preprocessed.
    dims : tuple
        Which 2 dims to plot.
    scale : str
        Scale for pitch mapping.
    root : int
        Root note (0=C).
    n_chord_notes : int
        Max notes to play per cluster chord.
    point_size : int
        Scatter point radius in pixels.
    width, height : int
        Widget size.

    Returns
    -------
    IPython.display.HTML
    """
    from IPython.display import HTML
    from ublind._core.notes import map_to_pitches, SCALES

    ub = get_ublind_uns(adata)
    coords = ub["coords"]
    subsample_idx = ub.get("subsample_idx")

    emb_key = embedding or ub["embedding"]
    d0, d1 = dims
    x = coords[:, d0]
    y = coords[:, d1]

    # Get cluster labels
    col = adata.obs[color_by]
    if subsample_idx is not None:
        col = col.iloc[subsample_idx]
    else:
        col = col.iloc[:len(coords)]

    cat = col.astype("category")
    codes = cat.cat.codes.values
    names = list(cat.cat.categories)

    # Colors from scanpy or defaults
    color_key = f"{color_by}_colors"
    if color_key in adata.uns:
        palette = list(adata.uns[color_key])
    else:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ]

    # Compute pitches for both dims
    pitches_d0 = map_to_pitches(x, scale=scale, root=root)
    pitches_d1 = map_to_pitches(y, scale=scale, root=root)

    # Build per-cluster chord data
    cluster_data = []
    for i, name in enumerate(names):
        mask = codes == i
        if not mask.any():
            continue
        # Sample notes for chord
        idx = np.where(mask)[0]
        rng = np.random.default_rng(i)
        if len(idx) > n_chord_notes:
            sampled = rng.choice(idx, n_chord_notes, replace=False)
        else:
            sampled = idx
        # Use pitches from both dims for richer chords
        notes = sorted(set(
            [int(x) for x in pitches_d0[sampled]] +
            [int(x) for x in pitches_d1[sampled]]
        ))
        # Deduplicate and limit
        if len(notes) > n_chord_notes:
            notes = sorted([int(x) for x in rng.choice(notes, n_chord_notes, replace=False)])
        notes = sorted(notes)

        cluster_data.append({
            "name": name,
            "color": palette[i % len(palette)],
            "cx": float(x[mask].mean()),
            "cy": float(y[mask].mean()),
            "notes": notes,
            "n_cells": int(mask.sum()),
        })

    # Build point arrays for JS
    import json
    points_js = json.dumps([
        {"x": float(x[i]), "y": float(y[i]),
         "c": int(codes[i]), "p0": int(pitches_d0[i]), "p1": int(pitches_d1[i])}
        for i in range(len(x))
    ])
    clusters_js = json.dumps(cluster_data)
    palette_js = json.dumps(palette)
    names_js = json.dumps(names)

    html = _build_html(
        points_js, clusters_js, palette_js, names_js,
        emb_key, d0, d1, point_size, width, height,
    )

    return HTML(html)


def _build_html(points_js, clusters_js, palette_js, names_js,
                emb_key, d0, d1, point_size, width, height):
    return f"""
<div id="ublind-interactive" style="display:inline-block; position:relative;">
  <canvas id="ublind-canvas" width="{width}" height="{height}"
    style="border:1px solid #ccc; cursor:crosshair;"></canvas>
  <div id="ublind-info" style="
    position:absolute; top:10px; right:10px; background:rgba(255,255,255,0.92);
    padding:8px 12px; border-radius:6px; font:13px monospace; color:#000;
    display:none; box-shadow:0 2px 8px rgba(0,0,0,0.15);
  "></div>
  <div style="text-align:center; margin-top:6px; font:12px sans-serif; color:#888;">
    Click cluster to hear chord · Hover for notes · Scroll to zoom · Drag to pan
  </div>
</div>

<script>
(function() {{
  const points = {points_js};
  const clusters = {clusters_js};
  const palette = {palette_js};
  const names = {names_js};

  const canvas = document.getElementById('ublind-canvas');
  const ctx = canvas.getContext('2d');
  const info = document.getElementById('ublind-info');
  const W = canvas.width, H = canvas.height;
  const pad = 50;

  // Audio
  let audioCtx = null;
  function getAudio() {{
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }}

  // Data bounds
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const xMin0 = Math.min(...xs), xMax0 = Math.max(...xs);
  const yMin0 = Math.min(...ys), yMax0 = Math.max(...ys);

  // Zoom/pan state
  let zoom = 1.0;
  let panX = 0, panY = 0;  // in data coords
  let isDragging = false;
  let dragStartX, dragStartY, panStartX, panStartY;

  function toCanvas(px, py) {{
    const xSpan = (xMax0 - xMin0) / zoom || 1;
    const ySpan = (yMax0 - yMin0) / zoom || 1;
    const cx = (xMin0 + xMax0) / 2 + panX;
    const cy = (yMin0 + yMax0) / 2 + panY;
    return [
      pad + (px - (cx - xSpan/2)) / xSpan * (W - 2*pad),
      (H - pad) - (py - (cy - ySpan/2)) / ySpan * (H - 2*pad)
    ];
  }}

  function toData(canvasX, canvasY) {{
    const xSpan = (xMax0 - xMin0) / zoom || 1;
    const ySpan = (yMax0 - yMin0) / zoom || 1;
    const cx = (xMin0 + xMax0) / 2 + panX;
    const cy = (yMin0 + yMax0) / 2 + panY;
    return [
      (canvasX - pad) / (W - 2*pad) * xSpan + (cx - xSpan/2),
      ((H - pad) - canvasY) / (H - 2*pad) * ySpan + (cy - ySpan/2)
    ];
  }}

  // Draw
  let highlightCluster = undefined;
  function draw() {{
    ctx.clearRect(0, 0, W, H);
    const ps = Math.max(1, {point_size} * Math.sqrt(zoom));

    for (const p of points) {{
      const [cx, cy] = toCanvas(p.x, p.y);
      if (cx < -10 || cx > W+10 || cy < -10 || cy > H+10) continue;
      const alpha = (highlightCluster !== undefined && p.c !== highlightCluster) ? 0.1 : 0.6;
      ctx.beginPath();
      ctx.arc(cx, cy, ps, 0, Math.PI * 2);
      ctx.fillStyle = palette[p.c % palette.length] + (alpha < 0.5 ? '1a' : '99');
      ctx.fill();
    }}

    // Cluster labels
    const fontSize = Math.max(9, Math.min(14, 11 * Math.sqrt(zoom)));
    ctx.font = 'bold ' + fontSize + 'px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const cl of clusters) {{
      const [cx, cy] = toCanvas(cl.cx, cl.cy);
      if (cx < -50 || cx > W+50 || cy < -20 || cy > H+20) continue;
      const m = ctx.measureText(cl.name);
      const bh = fontSize + 4;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(cx - m.width/2 - 4, cy - bh/2, m.width + 8, bh);
      ctx.strokeStyle = cl.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - m.width/2 - 4, cy - bh/2, m.width + 8, bh);
      ctx.fillStyle = '#333';
      ctx.fillText(cl.name, cx, cy);
    }}

    // Axes
    ctx.fillStyle = '#888';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('{emb_key} ' + ({d0}+1), W/2, H - 8);
    ctx.save();
    ctx.translate(12, H/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('{emb_key} ' + ({d1}+1), 0, 0);
    ctx.restore();

    // Zoom indicator
    if (zoom !== 1.0) {{
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(zoom.toFixed(1) + 'x', 8, 16);
    }}
  }}

  draw();

  // ── Zoom (scroll wheel) ──
  canvas.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Zoom toward mouse position
    const [dataX, dataY] = toData(mx, my);
    const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
    const newZoom = Math.max(0.5, Math.min(50, zoom * factor));

    // Adjust pan so the point under cursor stays put
    panX += dataX - ((xMin0 + xMax0)/2 + panX);
    panX *= newZoom / zoom;
    panX -= dataX - ((xMin0 + xMax0)/2);
    panX = (panX * zoom + (dataX - (xMin0+xMax0)/2) * (zoom - newZoom)) / newZoom;

    // Simpler: just zoom toward center for stability
    zoom = newZoom;
    // Recalc pan to keep mouse point stable
    const [newMx, newMy] = toCanvas(dataX, dataY);
    const xSpan = (xMax0 - xMin0) / zoom || 1;
    const ySpan = (yMax0 - yMin0) / zoom || 1;
    panX += (mx - newMx) / (W - 2*pad) * xSpan;
    panY -= (my - newMy) / (H - 2*pad) * ySpan;

    draw();
  }}, {{passive: false}});

  // ── Pan (drag) ──
  canvas.addEventListener('mousedown', function(e) {{
    if (e.button === 0) {{
      isDragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      panStartX = panX;
      panStartY = panY;
      canvas.style.cursor = 'grabbing';
    }}
  }});

  canvas.addEventListener('mousemove', function(e) {{
    if (isDragging) {{
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      const xSpan = (xMax0 - xMin0) / zoom || 1;
      const ySpan = (yMax0 - yMin0) / zoom || 1;
      panX = panStartX - dx / (W - 2*pad) * xSpan;
      panY = panStartY + dy / (H - 2*pad) * ySpan;
      draw();
      return;
    }}

    // Hover: play note
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const pi = findPoint(mx, my);
    if (pi !== null && pi !== lastHover) {{
      lastHover = pi;
      const p = points[pi];
      playNote(p.p0, 0, 0.3, 0.08);
      canvas.style.cursor = 'pointer';
    }} else if (pi === null) {{
      lastHover = -1;
      if (!isDragging) canvas.style.cursor = 'crosshair';
    }}
  }});

  canvas.addEventListener('mouseup', function(e) {{
    if (isDragging) {{
      isDragging = false;
      canvas.style.cursor = 'crosshair';
    }}
  }});

  canvas.addEventListener('mouseleave', function() {{
    isDragging = false;
  }});

  // Double-click to reset
  canvas.addEventListener('dblclick', function(e) {{
    e.preventDefault();
    zoom = 1.0; panX = 0; panY = 0;
    highlightCluster = undefined;
    info.style.display = 'none';
    draw();
  }});

  // ── Click → play cluster chord ──
  canvas.addEventListener('click', function(e) {{
    if (Math.abs(e.clientX - dragStartX) > 5 || Math.abs(e.clientY - dragStartY) > 5) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const ci = findCluster(mx, my);
    if (ci !== null) {{
      const cl = clusters[ci];
      highlightCluster = names.indexOf(cl.name);
      draw();
      info.style.display = 'block';
      info.innerHTML = '<b style="color:#000">' + cl.name + '</b><br>' +
        '<span style="color:#333">' + cl.n_cells + ' cells · ' + cl.notes.length + ' notes</span>';
      playChord(cl.notes);

      setTimeout(() => {{ highlightCluster = undefined; draw(); info.style.display = 'none'; }},
        cl.notes.length * 80 + 1800);
    }}
  }});

  // ── Audio helpers ──
  function midiToFreq(note) {{
    return 440 * Math.pow(2, (note - 69) / 12);
  }}

  function playNote(midi, delay, duration, gain) {{
    const ac = getAudio();
    const osc = ac.createOscillator();
    const env = ac.createGain();
    osc.type = 'sine';
    osc.frequency.value = midiToFreq(midi);
    const osc2 = ac.createOscillator();
    const env2 = ac.createGain();
    osc2.type = 'sine';
    osc2.frequency.value = midiToFreq(midi) * 2.01;

    const t = ac.currentTime + delay;
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(gain, t + 0.02);
    env.gain.linearRampToValueAtTime(gain * 0.6, t + 0.1);
    env.gain.linearRampToValueAtTime(0, t + duration);
    env2.gain.setValueAtTime(0, t);
    env2.gain.linearRampToValueAtTime(gain * 0.2, t + 0.02);
    env2.gain.linearRampToValueAtTime(0, t + duration);

    osc.connect(env).connect(ac.destination);
    osc2.connect(env2).connect(ac.destination);
    osc.start(t); osc.stop(t + duration + 0.1);
    osc2.start(t); osc2.stop(t + duration + 0.1);
  }}

  function playChord(notes) {{
    const spacing = 0.08;
    const duration = 1.5;
    const gain = 0.15;
    notes.forEach((n, i) => playNote(n, i * spacing, duration, gain));
  }}

  function findCluster(mx, my) {{
    let best = null, bestDist = Infinity;
    for (let i = 0; i < clusters.length; i++) {{
      const [cx, cy] = toCanvas(clusters[i].cx, clusters[i].cy);
      const d = Math.sqrt((mx-cx)**2 + (my-cy)**2);
      if (d < bestDist) {{ bestDist = d; best = i; }}
    }}
    return best;
  }}

  let lastHover = -1;
  function findPoint(mx, my) {{
    let best = null, bestDist = Infinity;
    const threshold = Math.max(10, 20 / Math.sqrt(zoom));
    for (let i = 0; i < points.length; i++) {{
      const [cx, cy] = toCanvas(points[i].x, points[i].y);
      const d = Math.sqrt((mx-cx)**2 + (my-cy)**2);
      if (d < bestDist && d < threshold) {{ bestDist = d; best = i; }}
    }}
    return best;
  }}
}})();
</script>
"""
