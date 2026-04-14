"""Bundled static HTML for the ragrag dashboard.

This is a single self-contained page. No build step, no framework, no
external assets. The JS polls ``/status`` every 2 s and re-renders the
cards in-place. Kept small enough that the whole thing fits in a
readable Python string literal.
"""
from __future__ import annotations


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ragrag dashboard</title>
  <style>
    :root {
      color-scheme: light dark;
      --accent: #3b82f6;
      --bg: #0f172a;
      --card: #1e293b;
      --ink: #e2e8f0;
      --muted: #94a3b8;
      --ok: #10b981;
      --warn: #f59e0b;
      --err: #ef4444;
    }
    body {
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
      margin: 0;
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 600;
    }
    header .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: #0b3b68;
      color: #93c5fd;
      margin-left: 8px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .card {
      background: var(--card);
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    dl {
      margin: 0;
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 6px 12px;
    }
    dt { color: var(--muted); }
    dd { margin: 0; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 4px 8px;
      border-bottom: 1px solid #334155;
    }
    th { color: var(--muted); font-weight: 500; }
    .bar {
      display: block;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      overflow: hidden;
      margin-top: 2px;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.4s ease;
    }
    .pill {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 11px;
      background: #334155;
      color: var(--muted);
    }
    .pill.ok { background: rgba(16,185,129,0.18); color: var(--ok); }
    .pill.warn { background: rgba(245,158,11,0.18); color: var(--warn); }
    .pill.err { background: rgba(239,68,68,0.18); color: var(--err); }
    button {
      background: var(--err);
      color: white;
      border: 0;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.1); }
    .muted { color: var(--muted); }
    code { font-family: ui-monospace, "JetBrains Mono", monospace; font-size: 12px; }
    footer { margin-top: 24px; color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>ragrag dashboard <span class="badge" id="device">?</span></h1>
    <div class="muted" id="version">Loading…</div>
  </header>

  <div class="grid">
    <section class="card" id="daemon-card">
      <h2>Daemon</h2>
      <dl>
        <dt>Uptime</dt>     <dd id="uptime">—</dd>
        <dt>Idle</dt>       <dd id="idle">—</dd>
        <dt>Models</dt>     <dd id="models">—</dd>
      </dl>
      <p style="margin-top:12px;">
        <button id="shutdown-btn">Shut down daemon</button>
      </p>
    </section>

    <section class="card" id="indexing-card">
      <h2>Indexing</h2>
      <dl>
        <dt>Status</dt>  <dd id="idx-status">idle</dd>
        <dt>File</dt>    <dd id="idx-file" class="muted">—</dd>
        <dt>Pages</dt>   <dd id="idx-pages">—</dd>
        <dt>Queue</dt>   <dd id="idx-queue">—</dd>
      </dl>
      <div class="bar" style="margin-top:12px;"><span id="idx-bar"></span></div>
    </section>

    <section class="card" id="resources-card">
      <h2>Resources</h2>
      <dl>
        <dt>CPU</dt>   <dd><span id="cpu">—</span></dd>
        <dt>GPU</dt>   <dd><span id="gpu">—</span></dd>
        <dt>RAM</dt>   <dd><span id="ram">—</span></dd>
        <dt>VRAM</dt>  <dd><span id="vram">—</span></dd>
      </dl>
    </section>

    <section class="card" id="indexes-card" style="grid-column: 1 / -1;">
      <h2>Recent queries</h2>
      <table id="recent-table">
        <thead><tr><th>Query</th><th>Top hit</th><th>Status</th><th>Wall (ms)</th></tr></thead>
        <tbody id="recent-body"></tbody>
      </table>
    </section>
  </div>

  <footer id="footer">Polling every 2 s.</footer>

  <script>
    async function fetchStatus() {
      try {
        const resp = await fetch('/status', { cache: 'no-store' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return await resp.json();
      } catch (err) {
        document.getElementById('footer').textContent = 'Disconnected: ' + err.message;
        return null;
      }
    }

    function fmtSeconds(s) {
      if (s == null) return '—';
      if (s < 60) return s.toFixed(0) + ' s';
      if (s < 3600) return (s / 60).toFixed(1) + ' m';
      return (s / 3600).toFixed(1) + ' h';
    }

    function setBar(id, pct) {
      const el = document.getElementById(id);
      if (!el) return;
      el.style.width = Math.max(0, Math.min(100, pct)) + '%';
    }

    function render(status) {
      if (!status) return;
      document.getElementById('version').textContent = 'ragrag ' + (status.version || '?');
      document.getElementById('device').textContent = status.device_mode || 'cpu';
      document.getElementById('uptime').textContent = fmtSeconds(status.uptime_s);
      document.getElementById('idle').textContent = fmtSeconds(status.idle_s);
      const models = status.models_loaded || [];
      document.getElementById('models').textContent = models.length ? models.join(', ') : '(none)';

      const idx = status.indexing || {};
      document.getElementById('idx-status').textContent = idx.active_file ? 'running' : 'idle';
      document.getElementById('idx-file').textContent = idx.active_file || '—';
      if (idx.pages_total) {
        document.getElementById('idx-pages').textContent = `${idx.pages_done || 0} / ${idx.pages_total}`;
        setBar('idx-bar', ((idx.pages_done || 0) / idx.pages_total) * 100);
      } else {
        document.getElementById('idx-pages').textContent = '—';
        setBar('idx-bar', 0);
      }
      document.getElementById('idx-queue').textContent = `${idx.queued_files || 0} queued · ${idx.completed_files || 0} done`;

      const res = status.resources || {};
      document.getElementById('cpu').textContent = res.cpu_pct != null ? res.cpu_pct + ' %' : '—';
      document.getElementById('gpu').textContent = res.gpu_pct != null ? res.gpu_pct + ' %' : '—';
      if (res.mem_rss_mib != null) {
        document.getElementById('ram').textContent = `${res.mem_rss_mib} MiB (process)`;
      } else {
        document.getElementById('ram').textContent = '—';
      }
      if (res.vram_free_mib != null) {
        document.getElementById('vram').textContent = `${res.vram_used_mib} / ${res.vram_total_mib} MiB used`;
      } else {
        document.getElementById('vram').textContent = '— (CPU mode)';
      }

      const tbody = document.getElementById('recent-body');
      tbody.innerHTML = '';
      for (const row of status.recent_queries || []) {
        const tr = document.createElement('tr');
        const q = document.createElement('td'); q.textContent = row.query;
        const p = document.createElement('td'); p.innerHTML = row.top1_path ? '<code>' + row.top1_path + '</code>' : '<span class="muted">—</span>';
        const s = document.createElement('td');
        const pill = document.createElement('span');
        pill.className = 'pill ' + (row.status === 'complete' ? 'ok' : (row.status === 'partial' ? 'warn' : 'err'));
        pill.textContent = row.status || '?';
        s.appendChild(pill);
        const w = document.createElement('td'); w.textContent = row.wall_ms;
        tr.append(q, p, s, w);
        tbody.appendChild(tr);
      }
    }

    document.getElementById('shutdown-btn').addEventListener('click', async () => {
      if (!confirm('Shut down the ragrag daemon?')) return;
      try {
        const resp = await fetch('/shutdown', {
          method: 'POST',
          headers: { 'X-Ragrag-Confirm': 'yes' },
        });
        const body = await resp.json();
        document.getElementById('footer').textContent = body.ack ? 'Daemon shutting down.' : 'Shutdown failed.';
      } catch (err) {
        document.getElementById('footer').textContent = 'Shutdown error: ' + err.message;
      }
    });

    async function tick() {
      const status = await fetchStatus();
      if (status) render(status);
    }
    tick();
    setInterval(tick, 2000);
  </script>
</body>
</html>
"""
