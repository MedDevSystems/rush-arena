// region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): LevelOfDetailRenderer; TECH(8): Canvas 2D]
// @purpose Draw a big battle (up to thousands of agents, ~100 CPs, a map hundreds of tiles wide) at any zoom
//   with bounded cost, in the neon theme (design/theme.js tokens and hooks).
// @invariants
// - Level of detail is a pure function of zoom (CSS px per world px): FAR < 0.26 <= MID < 0.72 <= NEAR
// - FAR: team dots + territory heat + CP discs + kill/capture pings; no per-agent hooks, no bullets
//   MID: batched simplified bodies (circle / hex; exp12 replays: class silhouettes per team×class), facing ticks,
//        batched tracers, theme CPs, kill/capture effects
//   NEAR: theme.drawAgent (exp12: class silhouettes in the theme's look) / drawBullets / drawEffect on culled sets,
//         neon static blocks built lazily
// - exp12 class identity: team = colour, class = silhouette (CLASS_SHAPES keyed by fork name, same on both sides)
// - Everything drawn in world space is culled to the viewport; effects are capped per frame (FX_CAP)
// - Static layers: FAR/MID are one canvas each (<= 2048 / 4096 px); NEAR is 32x32-tile blocks, LRU-cached,
//   at most BLOCKS_PER_FRAME built per frame (the MID layer shows through until a block exists)
// @links LINKS_TO: replay2.js (state source), director_big.js (camera), app_big.js (loop), design/theme.js
// endregion MODULE_CONTRACT
(function () {
  "use strict";
  const LOD_FAR = 0.26, LOD_NEAR = 0.72;
  const BLOCK_TILES = 32, BLOCK_CACHE = 24, BLOCKS_PER_FRAME = 2;
  const FX_CAP = 260, PING_CAP = 160;
  // Far-zoom fire (redesign 25.09.2026): screen-constant sizes, so shots read with the whole 500v500 map on screen.
  const FAR_STREAK_PX = 12, FAR_STREAK_W = 1.6, FAR_MUZZLE_PX = 2.2, FAR_SPARK_PX = 1.8;
  const FAR_BULLET_ALPHA = 0.14, FAR_MUZZLE_ALPHA = 0.12, FAR_SPARK_ALPHA = 0.18;   // ~7 overlaps saturate
  // 6000 per-bullet strokes: measured 3.6 ms max draw with 4186 bullets on screen (Warfront500, 8640 px rules)
  const FAR_BULLET_CAP = 6000, FAR_SPARK_CAP = 900;
  const MID_MIN_SIL_PX = 7;                          // min on-screen radius of a class silhouette at MID zoom
  const TAU = Math.PI * 2;
  const EVK = { kill: 1, hit: 2, capture: 3, respawn: 4, dash: 5, end: 6, refill: 7 };
  // exp15: ammo bar under the body, «empty» marker, attention lines (mid/near only, budgeted), refill glow on CP rings
  const AMMO_COL = "#e9eefc", AMMO_LOW_COL = "#ffb347", AMMO_LOW = 0.3, EMPTY_COL = "#ff9f1c";
  const ATTN_LINE_CAP = 600, ATTN_ALPHA = { enemy: 0.42, ally: 0.3, cp: 0.34 };
  const REFILL_COL = "#ffd166", REFILL_WIN = 1.5;       // decisions of refill events that light a CP ring

  function rgba(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }
  function rgb(hex) { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function canvas(w, h) { const c = document.createElement("canvas"); c.width = Math.max(1, Math.ceil(w)); c.height = Math.max(1, Math.ceil(h)); return c; }
  function lodOf(z) { return z < LOD_FAR ? "far" : z < LOD_NEAR ? "mid" : "near"; }

  // region CLASS_SILHOUETTES
  // exp12 classes (forks12 order). Team = colour, class = silhouette. Shapes are unit polygons facing +x (radius 1 =
  // the agent's body radius), rotated with the agent's facing; "circle" is the control fork. Chosen to differ in
  // outline at 10–20 px on screen: round / notched arrow / square / needle / plus / dart / star / shield.
  const CLASS_ORDER = ["base", "hunter", "heavy", "sniper", "tank", "scout", "assault", "guardian"];
  function star(n, ro, ri) { const p = []; for (let i = 0; i < 2 * n; i++) { const q = (i * Math.PI) / n, rr = i % 2 ? ri : ro; p.push([Math.cos(q) * rr, Math.sin(q) * rr]); } return p; }
  const CLASS_SHAPES = [
    "circle",                                                                   // base: Контроль
    [[1.3, 0], [-0.85, 0.95], [-0.25, 0], [-0.85, -0.95]],                      // hunter: arrowhead with a notch
    [[0.92, 0.92], [-0.92, 0.92], [-0.92, -0.92], [0.92, -0.92]],               // heavy: square
    [[1.55, 0], [0, 0.5], [-1.05, 0], [0, -0.5]],                               // sniper: long thin diamond
    // tank: thick plus (an octagon read as a circle = Контроль at 16 px in the legend)
    [[1.1, 0.42], [0.42, 0.42], [0.42, 1.1], [-0.42, 1.1], [-0.42, 0.42], [-1.1, 0.42],
     [-1.1, -0.42], [-0.42, -0.42], [-0.42, -1.1], [0.42, -1.1], [0.42, -0.42], [1.1, -0.42]],
    [[1.25, 0], [-0.8, 0.72], [-0.8, -0.72]],                                   // scout: dart
    star(5, 1.22, 0.55),                                                        // assault: five-point star
    [[1.15, 0], [0.35, 0.95], [-0.9, 0.95], [-0.9, -0.95], [0.35, -0.95]],      // guardian: shield (flat back)
  ];
  // add class c's silhouette at (x, y), radius R, facing a, to a Path2D or a 2D context (both have moveTo/lineTo)
  function classPath(p, c, x, y, R, a) {
    const sh = CLASS_SHAPES[c] || "circle";
    if (sh === "circle") { p.moveTo(x + R, y); p.arc(x, y, R, 0, TAU); return; }
    const ca = Math.cos(a), sa = Math.sin(a);
    for (let i = 0; i < sh.length; i++) {
      const u = sh[i][0] * R, v = sh[i][1] * R, px = x + u * ca - v * sa, py = y + u * sa + v * ca;
      if (i === 0) p.moveTo(px, py); else p.lineTo(px, py);
    }
    p.closePath();
  }
  // inline SVG of a class silhouette (legend chips, class panel), facing up-right like a moving soldier
  function classSvg(c, fill, stroke, px) {
    const sh = CLASS_SHAPES[c] || "circle", R = 0.36 * px, cx = px / 2, a = -Math.PI / 2;
    let d;
    if (sh === "circle") d = `<circle cx="${cx}" cy="${cx}" r="${R}"/>`;
    else d = `<path d="${sh.map(([u, v], i) => `${i ? "L" : "M"}${(cx + (u * Math.cos(a) - v * Math.sin(a)) * R).toFixed(1)} ${(cx + (u * Math.sin(a) + v * Math.cos(a)) * R).toFixed(1)}`).join(" ")} Z"/>`;
    return `<svg class="cls" width="${px}" height="${px}" viewBox="0 0 ${px} ${px}" aria-hidden="true"><g fill="${fill}" stroke="${stroke}" stroke-width="1.6" stroke-linejoin="round">${d}</g></svg>`;
  }
  // endregion CLASS_SILHOUETTES

  class BigRenderer {
    constructor(cv, replay, theme) {
      this.cv = cv; this.ctx = cv.getContext("2d");
      this.r = replay; this.T = theme;
      const N = replay.N;
      this.s = { x: new Float32Array(N), y: new Float32Array(N), a: new Float32Array(N), hp: new Float32Array(N),
                 sh: new Float32Array(N), alive: new Uint8Array(N), dash: new Uint8Array(N),
                 am: replay.hasAmmo && replay.hasAmmo() ? new Uint8Array(N) : null,
                 cap: replay.hasAmmo && replay.hasAmmo() ? new Uint8Array(N) : null,
                 att: replay.hasAttn && replay.hasAttn() ? new Uint16Array(N) : null };
      this.showAttn = true;                          // attention lines at mid/near zoom (key L toggles)
      this.refillGlow = new Float32Array(replay.C);
      const m = replay.map;
      this.W = m.w; this.H = m.h; this.tile = m.tile; this.cols = m.cols; this.rows = m.rows;
      this.wall = new Uint8Array(this.cols * this.rows);
      for (const [row, c0, c1] of m.walls) for (let c = c0; c <= c1; c++) this.wall[row * this.cols + c] = 1;
      this.blocks = new Map();
      this.stats = { lod: "", agents: 0, bullets: 0, fx: 0, cps: 0, blocksBuilt: 0, blocksCached: 0, drawMs: 0 };
      // territory grid: ~128 cells along the long side
      this.cell = Math.max(m.w, m.h) / 128;
      this.tw = Math.ceil(m.w / this.cell); this.th = Math.ceil(m.h / this.cell);
      this.terrCv = canvas(this.tw, this.th); this.terrCtx = this.terrCv.getContext("2d");
      this.terrImg = this.terrCtx.createImageData(this.tw, this.th);
      this.cntB = new Float32Array(this.tw * this.th); this.cntR = new Float32Array(this.tw * this.th);
      this.teamRGB = [rgb(theme.team[0].core), rgb(theme.team[1].core)];
      this.cpContested = new Uint8Array(replay.C);
      this.hover = -1;
      this.farLayer = this._staticLayer(Math.min(2048 / Math.max(m.w, m.h), 0.25), "far");
      this.midLayer = null;
    }

    isWall(c, r) { return c >= 0 && r >= 0 && c < this.cols && r < this.rows && this.wall[r * this.cols + c] === 1; }

    resize(w, h) {
      const dpr = window.devicePixelRatio || 1;
      this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
      this.cv.style.width = w + "px"; this.cv.style.height = h + "px";
      this.w = w; this.h = h; this.dpr = dpr;
    }

    // region static layers
    // One canvas for the whole map. "far": walls as bright solid shapes (structure must read at 1-4 px per
    // tile) plus a faint 25-tile grid for scale. "mid": dark walls with lit edges, 5-tile grid.
    _staticLayer(scale, kind) {
      const C = this.T.color, t = this.tile;
      const cv = canvas(this.W * scale, this.H * scale), g = cv.getContext("2d");
      g.scale(scale, scale);
      g.fillStyle = C.floor; g.fillRect(0, 0, this.W, this.H);
      const step = kind === "far" ? 25 : 5;
      g.strokeStyle = kind === "far" ? rgba(C.gridMajor, 0.8) : C.grid; g.lineWidth = 1 / scale;
      g.beginPath();
      for (let c = step; c < this.cols; c += step) { g.moveTo(c * t, 0); g.lineTo(c * t, this.H); }
      for (let r = step; r < this.rows; r += step) { g.moveTo(0, r * t); g.lineTo(this.W, r * t); }
      g.stroke();
      g.fillStyle = kind === "far" ? rgba(C.wallEdge, 0.46) : C.wallFill;
      for (const [row, c0, c1] of this.r.map.walls) g.fillRect(c0 * t, row * t, (c1 - c0 + 1) * t, t);
      g.beginPath();
      for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
        if (!this.wall[r * this.cols + c]) continue;
        const x = c * t, y = r * t;
        if (!this.isWall(c, r - 1)) { g.moveTo(x, y); g.lineTo(x + t, y); }
        if (!this.isWall(c, r + 1)) { g.moveTo(x, y + t); g.lineTo(x + t, y + t); }
        if (!this.isWall(c - 1, r)) { g.moveTo(x, y); g.lineTo(x, y + t); }
        if (!this.isWall(c + 1, r)) { g.moveTo(x + t, y); g.lineTo(x + t, y + t); }
      }
      g.strokeStyle = kind === "far" ? rgba(C.wallEdgeHot, 0.55) : rgba(C.wallEdge, 0.9);
      g.lineWidth = kind === "far" ? 1.2 / scale : Math.max(1 / scale, 2.5);
      g.stroke();
      g.strokeStyle = rgba(C.wallEdge, 0.6); g.lineWidth = 2 / scale; g.strokeRect(0, 0, this.W, this.H);
      cv.scale = scale;
      return cv;
    }

    // Neon block for NEAR zoom: same look as theme.buildStaticLayer (floor grid, hatched wall bodies,
    // triple-stroked edges) but for a 32x32-tile window and without the arena border, so blocks tile seamlessly.
    _block(bx, by, scale) {
      const C = this.T.color, t = this.tile, B = BLOCK_TILES;
      const c0 = bx * B, r0 = by * B, size = B * t;
      const cv = canvas(size * scale, size * scale), g = cv.getContext("2d");
      g.scale(scale, scale); g.translate(-c0 * t, -r0 * t);
      g.fillStyle = C.floor; g.fillRect(c0 * t, r0 * t, size, size);
      g.lineWidth = 1 / scale; g.strokeStyle = C.grid; g.beginPath();
      for (let c = c0; c <= c0 + B; c++) if (c % 5) { g.moveTo(c * t, r0 * t); g.lineTo(c * t, (r0 + B) * t); }
      for (let r = r0; r <= r0 + B; r++) if (r % 5) { g.moveTo(c0 * t, r * t); g.lineTo((c0 + B) * t, r * t); }
      g.stroke();
      g.lineWidth = 1.2 / scale + 0.4; g.strokeStyle = C.gridMajor; g.beginPath();
      for (let c = Math.ceil(c0 / 5) * 5; c <= c0 + B; c += 5) { g.moveTo(c * t, r0 * t); g.lineTo(c * t, (r0 + B) * t); }
      for (let r = Math.ceil(r0 / 5) * 5; r <= r0 + B; r += 5) { g.moveTo(c0 * t, r * t); g.lineTo((c0 + B) * t, r * t); }
      g.stroke();
      const tiles = [];
      for (let r = r0 - 1; r <= r0 + B; r++) for (let c = c0 - 1; c <= c0 + B; c++) if (this.isWall(c, r)) tiles.push(c, r);
      if (tiles.length) {
        g.save(); g.beginPath();
        for (let i = 0; i < tiles.length; i += 2) g.rect(tiles[i] * t, tiles[i + 1] * t, t, t);
        g.fillStyle = C.wallFill; g.fill(); g.clip();
        g.strokeStyle = C.wallHatch; g.lineWidth = 1.5; g.beginPath();
        const X0 = c0 * t, Y0 = r0 * t;
        for (let d = -size; d < 2 * size; d += 10) { g.moveTo(X0 + d, Y0); g.lineTo(X0 + d + size, Y0 + size); }
        g.stroke(); g.restore();
        g.beginPath();
        for (let i = 0; i < tiles.length; i += 2) {
          const c = tiles[i], r = tiles[i + 1], x = c * t, y = r * t;
          if (!this.isWall(c, r - 1)) { g.moveTo(x, y); g.lineTo(x + t, y); }
          if (!this.isWall(c, r + 1)) { g.moveTo(x, y + t); g.lineTo(x + t, y + t); }
          if (!this.isWall(c - 1, r)) { g.moveTo(x, y); g.lineTo(x, y + t); }
          if (!this.isWall(c + 1, r)) { g.moveTo(x + t, y); g.lineTo(x + t, y + t); }
        }
        g.lineCap = "round";
        g.strokeStyle = rgba(C.wallEdge, 0.13); g.lineWidth = 12; g.stroke();
        g.strokeStyle = rgba(C.wallEdge, 0.38); g.lineWidth = 4.5; g.stroke();
        g.strokeStyle = rgba(C.wallEdgeHot, 0.95); g.lineWidth = 1.3; g.stroke();
      }
      return cv;
    }

    _drawStatic(ctx, z, x0, y0, x1, y1) {
      const dev = z * this.dpr;
      let layer = this.farLayer;
      if (dev > this.farLayer.scale * 1.25) {
        if (!this.midLayer) this.midLayer = this._staticLayer(Math.min(4096 / Math.max(this.W, this.H), 0.8), "mid");
        layer = this.midLayer;
      }
      const s = layer.scale;
      const sx = Math.max(0, x0), sy = Math.max(0, y0), ex = Math.min(this.W, x1), ey = Math.min(this.H, y1);
      if (ex > sx && ey > sy) ctx.drawImage(layer, sx * s, sy * s, (ex - sx) * s, (ey - sy) * s, sx, sy, ex - sx, ey - sy);
      if (z < LOD_NEAR) return;
      const size = BLOCK_TILES * this.tile, scale = Math.min(1.5, Math.max(1, this.dpr));
      let built = 0;
      for (let by = Math.max(0, Math.floor(sy / size)); by * size < ey; by++) {
        for (let bx = Math.max(0, Math.floor(sx / size)); bx * size < ex; bx++) {
          const key = bx + "," + by;
          let b = this.blocks.get(key);
          if (!b) {
            if (built >= BLOCKS_PER_FRAME) continue;
            b = this._block(bx, by, scale); built++;
            this.blocks.set(key, b);
            if (this.blocks.size > BLOCK_CACHE) this.blocks.delete(this.blocks.keys().next().value);
          } else { this.blocks.delete(key); this.blocks.set(key, b); }            // LRU touch
          ctx.drawImage(b, bx * size, by * size, size, size);
        }
      }
      this.stats.blocksBuilt += built; this.stats.blocksCached = this.blocks.size;
    }
    // endregion

    // region territory
    _territory(ctx, alpha) {
      const s = this.s, N = this.r.N, T = this.r.T, cs = this.cell, tw = this.tw, th = this.th;
      const B = this.cntB, R = this.cntR;
      B.fill(0); R.fill(0);
      for (let k = 0; k < N; k++) {
        if (!s.alive[k]) continue;
        const cx = Math.min(tw - 1, Math.max(0, (s.x[k] / cs) | 0)), cy = Math.min(th - 1, Math.max(0, (s.y[k] / cs) | 0));
        const arr = k < T ? B : R;
        arr[cy * tw + cx] += 1;
        if (cx > 0) arr[cy * tw + cx - 1] += 0.5; if (cx < tw - 1) arr[cy * tw + cx + 1] += 0.5;
        if (cy > 0) arr[(cy - 1) * tw + cx] += 0.5; if (cy < th - 1) arr[(cy + 1) * tw + cx] += 0.5;
      }
      const d = this.terrImg.data, [b0, b1] = this.teamRGB;
      for (let i = 0; i < tw * th; i++) {
        const b = B[i], r = R[i], sum = b + r, o = i * 4;
        if (sum <= 0) { d[o + 3] = 0; continue; }
        const dom = (b - r) / sum, inten = Math.min(1, sum / 4);
        if (b > 0.9 && r > 0.9 && Math.abs(dom) < 0.6) { d[o] = 255; d[o + 1] = 240; d[o + 2] = 220; d[o + 3] = 200 * inten; }   // front
        else { const c = dom > 0 ? b0 : b1; d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 150 * inten * Math.abs(dom); }
      }
      this.terrCtx.putImageData(this.terrImg, 0, 0);
      ctx.save();
      ctx.globalAlpha = alpha; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(this.terrCv, 0, 0, tw * cs, th * cs);
      ctx.restore();
    }
    // endregion

    // region control points
    _cps(ctx, t, z, lod, now, x0, y0, x1, y1) {
      const r = this.r, T = this.T, R = r.cpR, s = this.s, N = r.N, TS = r.T;
      let n = 0;
      for (let c = 0; c < r.C; c++) {
        const cx = r.cpXY[2 * c], cy = r.cpXY[2 * c + 1];
        const pad = Math.max(R, 8 / z);
        if (cx + pad < x0 || cx - pad > x1 || cy + pad < y0 || cy - pad > y1) continue;
        const st = r.cpState(t, c);
        let b = 0, d = 0;
        for (let k = 0; k < N; k++) if (s.alive[k]) { const dx = s.x[k] - cx, dy = s.y[k] - cy; if (dx * dx + dy * dy <= R * R) { if (k < TS) b++; else d++; } }
        const contested = b > 0 && d > 0;
        n++;
        if (lod === "far") {
          const col = st.owner === 1 ? T.team[0].core : st.owner === 2 ? T.team[1].core : T.cp.neutral;
          const rad = Math.max(R, 4.5 / z);
          ctx.fillStyle = rgba(col, st.owner ? 0.30 : 0.14);
          ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU); ctx.fill();
          ctx.lineWidth = 1.6 / z;
          ctx.strokeStyle = contested && ((now / 180) | 0) % 2 === 0 ? "#ffffff" : rgba(col, 0.95);
          ctx.stroke();
          if (st.prog > 0 && st.cap) {
            const cc = T.team[st.cap - 1].core;
            ctx.strokeStyle = cc; ctx.lineWidth = 2.6 / z;
            ctx.beginPath(); ctx.arc(cx, cy, rad + 2.5 / z, -Math.PI / 2, -Math.PI / 2 + TAU * st.prog); ctx.stroke();
          }
        } else {
          T.drawControlPoint(ctx, { x: cx, y: cy, r: R, owner: st.owner, progress: st.prog, cap_team: st.cap, contested }, { t: now, zoom: z }, c);
        }
        const g = this.refillGlow[c];
        if (g > 0.01) {                      // exp15: agents refilling here — a warm pulsing ring outside the CP ring
          const rad = (lod === "far" ? Math.max(R, 4.5 / z) : R) + 5 / z, pulse = 0.65 + 0.35 * Math.sin(now / 140);
          ctx.globalCompositeOperation = "lighter";
          ctx.strokeStyle = rgba(REFILL_COL, Math.min(0.85, 0.25 + 0.12 * g) * pulse);
          ctx.lineWidth = Math.min(7, 1.8 + 0.6 * g) / z;
          ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU); ctx.stroke();
          ctx.globalCompositeOperation = "source-over";
        }
      }
      return n;
    }

    // exp15: refill intensity per CP from refill events (a = cp, b = agents) over the last REFILL_WIN decisions
    _refills(t) {
      const r = this.r, g = this.refillGlow;
      g.fill(0);
      if (!r.hasAmmo || !r.hasAmmo()) return 0;
      const [e0, e1] = r.eventRange(t - REFILL_WIN - r.k, t + 0.5);
      let n = 0;
      for (let e = e0; e < e1; e++) {
        if (r.eK[e] !== EVK.refill) continue;
        const cp = r.eA[e]; if (cp >= r.C) continue;
        const age = Math.max(0, t - r.eT[e]);
        g[cp] = Math.max(g[cp], r.eB[e] * Math.max(0, 1 - age / (REFILL_WIN + r.k))); n++;
      }
      return n;
    }

    // exp15: ammo bars (mid, batched) — a dim track, a fill (white, amber when low), an amber dot for «empty».
    _ammoMid(ctx, z, x0, y0, x1, y1) {
      const s = this.s, r = this.r, R0 = this.T.size.agent, rad = r.agentRadius;
      if (!s.am) return 0;
      const track = new Path2D(), fill = new Path2D(), low = new Path2D(), empty = new Path2D();
      const h = 2.2 / z, dot = 2.6 / z;
      let n = 0;
      for (let k = 0; k < r.N; k++) {
        if (!s.alive[k]) continue;
        const x = s.x[k], y = s.y[k];
        if (x < x0 - 30 || x > x1 + 30 || y < y0 - 30 || y > y1 + 30) continue;
        const R = Math.max(rad ? R0 * rad[k] / 12 : R0, (MID_MIN_SIL_PX / z) * (rad ? rad[k] / 12 : 1));
        const w = 2 * R, bx = x - R, by = y + R + 3 / z, cap = s.cap[k] || 1, f = Math.min(1, s.am[k] / cap);
        track.rect(bx, by, w, h);
        if (s.am[k] === 0) { empty.moveTo(x + dot, y - R - 4 / z); empty.arc(x, y - R - 4 / z, dot, 0, TAU); }
        else (f < AMMO_LOW ? low : fill).rect(bx, by, w * f, h);
        n++;
      }
      ctx.fillStyle = "rgba(255,255,255,0.14)"; ctx.fill(track);
      ctx.fillStyle = rgba(AMMO_COL, 0.85); ctx.fill(fill);
      ctx.fillStyle = AMMO_LOW_COL; ctx.fill(low);
      ctx.fillStyle = EMPTY_COL; ctx.fill(empty);
      return n;
    }

    // exp15: near — the same bar as a crisp segmented magazine under the HP arc; empty -> blinking amber outline
    _ammoNear(ctx, k, x, y, below, now) {           // below = the outer radius of the HP/shield rings
      const s = this.s, cap = s.cap[k] || 1, am = s.am[k], f = Math.min(1, am / cap);
      const w = Math.max(22, 1.6 * below), h = 3.2, bx = x - w / 2, by = y + below + 5;
      ctx.fillStyle = "rgba(255,255,255,0.12)"; ctx.fillRect(bx, by, w, h);
      if (am === 0) {
        const on = ((now / 260) | 0) % 2 === 0;
        ctx.strokeStyle = rgba(EMPTY_COL, on ? 0.95 : 0.45); ctx.lineWidth = 1.2; ctx.strokeRect(bx - 0.5, by - 0.5, w + 1, h + 1);
        return;
      }
      ctx.fillStyle = f < AMMO_LOW ? AMMO_LOW_COL : rgba(AMMO_COL, 0.9);
      ctx.fillRect(bx, by, w * f, h);
      if (cap <= 40) {                       // magazine ticks: one per round up to 40 rounds
        ctx.strokeStyle = "rgba(6,9,26,0.7)"; ctx.lineWidth = 0.7; ctx.beginPath();
        for (let i = 1; i < cap; i++) { const xx = bx + (w * i) / cap; ctx.moveTo(xx, by); ctx.lineTo(xx, by + h); }
        ctx.stroke();
      }
    }

    // exp15: attention lines — agent -> attended enemy (team colour), ally (white) or CP (warm, dashed); mid/near only
    _attnLines(ctx, z, x0, y0, x1, y1) {
      const s = this.s, r = this.r;
      if (!s.att || !this.showAttn) return 0;
      const base = r.attnCpBase, N = r.N, TS = r.T;
      const enemy = [new Path2D(), new Path2D()], ally = new Path2D(), cp = new Path2D();
      let n = 0;
      for (let k = 0; k < N && n < ATTN_LINE_CAP; k++) {
        if (!s.alive[k]) continue;
        const x = s.x[k], y = s.y[k];
        if (x < x0 || x > x1 || y < y0 || y > y1) continue;
        const v = s.att[k];
        if (v === 65535) continue;
        let tx, ty, p;
        if (v < base) {
          if (v >= N || !s.alive[v]) continue;
          tx = s.x[v]; ty = s.y[v];
          const same = (k < TS) === (v < TS);
          p = same ? ally : enemy[k < TS ? 0 : 1];
        } else {
          const c = v - base; if (c >= r.C) continue;
          tx = r.cpXY[2 * c]; ty = r.cpXY[2 * c + 1]; p = cp;
        }
        p.moveTo(x, y); p.lineTo(tx, ty); n++;
      }
      ctx.lineWidth = 1.1 / z; ctx.lineCap = "round";
      ctx.globalCompositeOperation = "lighter";
      for (const team of [0, 1]) { ctx.strokeStyle = rgba(this.T.team[team].core, ATTN_ALPHA.enemy); ctx.stroke(enemy[team]); }
      ctx.strokeStyle = rgba("#ffffff", ATTN_ALPHA.ally); ctx.stroke(ally);
      ctx.globalCompositeOperation = "source-over";
      ctx.setLineDash([6 / z, 5 / z]); ctx.strokeStyle = rgba(REFILL_COL, ATTN_ALPHA.cp); ctx.stroke(cp); ctx.setLineDash([]);
      return n;
    }
    // endregion

    // region agents
    _agentsFar(ctx, z, x0, y0, x1, y1) {
      const s = this.s, N = this.r.N, TS = this.r.T, T = this.T;
      const dot = Math.min(3.4, Math.max(1.7, 12 * z * 2.2)) / z, glow = dot * 2.6;
      let n = 0;
      for (let team = 0; team < 2; team++) {
        const k0 = team === 0 ? 0 : TS, k1 = team === 0 ? TS : N;
        ctx.beginPath();
        for (let k = k0; k < k1; k++) {
          if (!s.alive[k]) continue;
          const x = s.x[k], y = s.y[k];
          if (x < x0 || x > x1 || y < y0 || y > y1) continue;
          ctx.rect(x - glow / 2, y - glow / 2, glow, glow);
        }
        ctx.globalCompositeOperation = "lighter";
        ctx.fillStyle = rgba(T.team[team].glow, 0.22); ctx.fill();
        ctx.globalCompositeOperation = "source-over";
        ctx.beginPath();
        for (let k = k0; k < k1; k++) {
          if (!s.alive[k]) continue;
          const x = s.x[k], y = s.y[k];
          if (x < x0 || x > x1 || y < y0 || y > y1) continue;
          ctx.rect(x - dot / 2, y - dot / 2, dot, dot); n++;
        }
        ctx.fillStyle = T.team[team].core; ctx.fill();
      }
      return n;
    }

    // MID with classes (exp12): one Path2D per (team, class) — 16 batched silhouettes, each rotated with facing and
    // sized by the agent's body-radius trait; glow halo + deep fill + team-colour outline + white facing tick.
    _agentsMidClasses(ctx, z, x0, y0, x1, y1) {
      const s = this.s, r = this.r, N = r.N, TS = r.T, T = this.T, R0 = T.size.agent, rad = r.agentRadius;
      const bodies = [], tick = [new Path2D(), new Path2D()];
      for (let i = 0; i < 16; i++) bodies.push(null);
      let n = 0;
      for (let k = 0; k < N; k++) {
        if (!s.alive[k]) continue;
        const x = s.x[k], y = s.y[k];
        if (x < x0 - 30 || x > x1 + 30 || y < y0 - 30 || y > y1 + 30) continue;
        const team = k < TS ? 0 : 1, c = r.agentClass[k] & 7, key = team * 8 + c, a = s.a[k];
        // at least MID_MIN_SIL_PX screen radius so the outline reads at 0.26–0.72 zoom (true size wins when larger)
        const R = Math.max(rad ? R0 * rad[k] / 12 : R0, (MID_MIN_SIL_PX / z) * (rad ? rad[k] / 12 : 1));
        const p = bodies[key] || (bodies[key] = new Path2D());
        classPath(p, c, x, y, R, a);
        tick[team].moveTo(x + Math.cos(a) * (R - 2), y + Math.sin(a) * (R - 2)); tick[team].lineTo(x + Math.cos(a) * (R + 11), y + Math.sin(a) * (R + 11));
        n++;
      }
      ctx.lineJoin = "round";
      for (let team = 0; team < 2; team++) {
        const TT = T.team[team];
        ctx.globalCompositeOperation = "lighter";
        ctx.strokeStyle = rgba(TT.glow, 0.28); ctx.lineWidth = Math.max(6, 3 / z);
        for (let c = 0; c < 8; c++) if (bodies[team * 8 + c]) ctx.stroke(bodies[team * 8 + c]);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = rgba(TT.core, 0.34); ctx.strokeStyle = TT.core; ctx.lineWidth = Math.max(2.4, 1.3 / z);
        for (let c = 0; c < 8; c++) { const p = bodies[team * 8 + c]; if (p) { ctx.fill(p); ctx.stroke(p); } }
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = Math.max(2, 1.1 / z); ctx.stroke(tick[team]);
      }
      ctx.lineJoin = "miter";
      return n;
    }

    // NEAR with classes: the theme's agent look (bloom, deep body, team outline, white facing wedge, HP arc over a dim
    // track, 8-segment shield ring) but with the class silhouette as the body and bars normalised by the agent's
    // own maximum (tank HP ×1.4 reads full at full). Drawn here because theme.drawAgent only knows team shapes.
    _classAgentNear(ctx, k, x, y, view, flash) {
      const r = this.r, T = this.T, S = T.size, s = this.s, team = r.teamOf(k), TT = T.team[team];
      const R = r.agentRadius ? S.agent * r.agentRadius[k] / 12 : S.agent, a = s.a[k];
      const maxHp = r.agentMaxHp ? r.agentMaxHp[k] : (r.rules.hp_max || 100);
      const hpF = Math.max(0, Math.min(1, s.hp[k] / maxHp));
      const shF = Math.max(0, Math.min(1, s.sh[k] / ((r.rules.shield_max || 40) * maxHp / (r.rules.hp_max || 100))));
      const low = hpF < 0.3, beat = low ? 0.5 + 0.5 * Math.sin(view.t / 90) : 0;
      ctx.globalCompositeOperation = "lighter";
      T.util.drawGlow(ctx, TT.glow, x, y, R * 3.4, 0.55 + 0.4 * flash);
      ctx.globalCompositeOperation = "source-over";
      ctx.beginPath(); classPath(ctx, r.agentClass[k] & 7, x, y, R, a);
      ctx.fillStyle = flash > 0.05 ? rgba("#ffffff", 0.25 + 0.6 * flash) : TT.deep; ctx.fill();
      ctx.lineWidth = 2.4; ctx.lineJoin = "round"; ctx.strokeStyle = TT.core; ctx.stroke(); ctx.lineJoin = "miter";
      const ca = Math.cos(a), sa = Math.sin(a), tipX = x + ca * (R + S.wedge), tipY = y + sa * (R + S.wedge);
      const bx = x + ca * (R - 3), by = y + sa * (R - 3);
      ctx.beginPath(); ctx.moveTo(tipX, tipY); ctx.lineTo(bx - sa * 5, by + ca * 5); ctx.lineTo(bx + sa * 5, by - ca * 5); ctx.closePath();
      ctx.fillStyle = "#ffffff"; ctx.fill();
      const ext = r.agentClass[k] === 3 ? 1.55 : r.agentClass[k] === 1 ? 1.3 : 1.25;       // needle / arrow reach further
      const hr = R * Math.max(1, ext * 0.85) + S.hpArcGap, a0 = -Math.PI / 2;
      ctx.lineWidth = 2.4;
      ctx.strokeStyle = rgba(TT.core, 0.16); ctx.beginPath(); ctx.arc(x, y, hr, 0, TAU); ctx.stroke();
      ctx.strokeStyle = low ? rgba("#ffffff", 0.55 + 0.45 * beat) : TT.core;
      ctx.beginPath(); ctx.arc(x, y, hr, a0, a0 + TAU * hpF); ctx.stroke();
      const seg = S.shieldSegments, lit = shF * seg;
      if (lit > 0.01) {
        const sr = hr + (S.shieldGap - S.hpArcGap), span = TAU / seg;
        ctx.lineWidth = 1.8;
        for (let i = 0; i < seg; i++) {
          const f = Math.max(0, Math.min(1, lit - i)); if (f <= 0) break;
          const s0 = a0 + i * span + 0.07;
          ctx.strokeStyle = rgba(T.color.shield, 0.35 + 0.55 * f);
          ctx.beginPath(); ctx.arc(x, y, sr, s0, s0 + (span - 0.14) * f); ctx.stroke();
        }
      }
      const ring = hr + (S.shieldGap - S.hpArcGap);
      if (s.am) this._ammoNear(ctx, k, x, y, ring, view.t);
      if (view.zoom >= S.labelMinZoom) {
        ctx.font = "600 11px " + T.fonts.ui; ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillStyle = rgba(TT.text || TT.core, 0.9); ctx.fillText(r.classTitle(k), x, y + hr + (s.am ? 16 : 9));
      }
    }

    _agentsMid(ctx, z, x0, y0, x1, y1) {
      if (this.r.hasClasses()) return this._agentsMidClasses(ctx, z, x0, y0, x1, y1);
      const s = this.s, N = this.r.N, TS = this.r.T, T = this.T, R0 = T.size.agent, rad = this.r.agentRadius;
      let n = 0;
      for (let team = 0; team < 2; team++) {
        const k0 = team === 0 ? 0 : TS, k1 = team === 0 ? TS : N;
        const body = new Path2D(), tick = new Path2D();
        for (let k = k0; k < k1; k++) {
          if (!s.alive[k]) continue;
          const x = s.x[k], y = s.y[k];
          if (x < x0 - 30 || x > x1 + 30 || y < y0 - 30 || y > y1 + 30) continue;
          const a = s.a[k], R = rad ? R0 * rad[k] / 12 : R0;      // exp12: body radius per agent (trait)
          if (team === 1) {
            for (let i = 0; i < 6; i++) { const q = a + Math.PI / 6 + (i * TAU) / 6, px = x + Math.cos(q) * R * 1.08, py = y + Math.sin(q) * R * 1.08; if (i === 0) body.moveTo(px, py); else body.lineTo(px, py); }
            body.closePath();
          } else { body.moveTo(x + R, y); body.arc(x, y, R, 0, TAU); }
          tick.moveTo(x + Math.cos(a) * (R - 2), y + Math.sin(a) * (R - 2)); tick.lineTo(x + Math.cos(a) * (R + 11), y + Math.sin(a) * (R + 11));
          n++;
        }
        ctx.globalCompositeOperation = "lighter";
        ctx.strokeStyle = rgba(T.team[team].glow, 0.28); ctx.lineWidth = Math.max(6, 3 / z); ctx.stroke(body);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = T.team[team].deep; ctx.fill(body);
        ctx.strokeStyle = T.team[team].core; ctx.lineWidth = Math.max(2.4, 1.3 / z); ctx.stroke(body);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = Math.max(2, 1.1 / z); ctx.stroke(tick);
      }
      return n;
    }

    _agentsNear(ctx, z, now, t, x0, y0, x1, y1) {
      const s = this.s, r = this.r, T = this.T;
      // body flash for 90 ms after a hit
      const flash = new Map();
      const [e0, e1] = r.eventRange(t - 2, t + 0.5);
      for (let e = e0; e < e1; e++) {
        if (r.eK[e] !== EVK.hit) continue;
        const age = now - (r.msAt(r.eT[e]) - r.msAt(0.5));
        if (age >= 0 && age < 90) flash.set(r.eB[e], Math.max(flash.get(r.eB[e]) || 0, 1 - age / 90));
      }
      const view = { t: now, zoom: z }, classes = r.hasClasses();
      let n = 0;
      for (let k = 0; k < r.N; k++) {
        if (!s.alive[k]) continue;
        const x = s.x[k], y = s.y[k];
        if (x < x0 - 40 || x > x1 + 40 || y < y0 - 40 || y > y1 + 40) continue;
        if (classes) { this._classAgentNear(ctx, k, x, y, view, flash.get(k) || 0); n++; continue; }
        // exp12: HP / shield normalised by the agent's own maximum (traits), so a tank's bar is not "over full"
        const hk = r.agentMaxHp ? (r.rules.hp_max || 100) / r.agentMaxHp[k] : 1;
        T.drawAgent(ctx, { x, y, angle: s.a[k], hp: s.hp[k] * hk, shield: s.sh[k] * hk, alive: true, team: r.teamOf(k), label: r.agentName(k) }, view, { flash: flash.get(k) || 0 });
        if (s.am) this._ammoNear(ctx, k, x, y, T.size.agent + T.size.shieldGap, now);
        n++;
      }
      return n;
    }
    // endregion

    // region bullets
    _bullets(ctx, t, z, lod, x0, y0, x1, y1) {
      const r = this.r, T = this.T, k = T.size.tracerFrames;
      if (lod === "near") {
        const list = [];
        r.forBullets(t, x0 - 50, y0 - 50, x1 + 50, y1 + 50, (x, y, vx, vy, team) => list.push({ x, y, vx, vy, team }));
        T.drawBullets(ctx, list);
        return list.length;
      }
      // MID: short tracer + bright head per bullet, batched per team (a long bare line reads as noise)
      const p = [new Path2D(), new Path2D()], h = [new Path2D(), new Path2D()], hr = 1.6 / z;
      const n = r.forBullets(t, x0, y0, x1, y1, (x, y, vx, vy, team) => {
        p[team].moveTo(x - vx * k, y - vy * k); p[team].lineTo(x, y);
        h[team].moveTo(x + hr, y); h[team].arc(x, y, hr, 0, TAU);
      });
      ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round";
      for (let team = 0; team < 2; team++) {
        ctx.strokeStyle = rgba(T.team[team].glow, 0.55); ctx.lineWidth = Math.max(3, 1.6 / z); ctx.stroke(p[team]);
        ctx.fillStyle = T.team[team].bullet; ctx.fill(h[team]);
      }
      ctx.globalCompositeOperation = "source-over";
      return n;
    }

    // FAR: the whole battle on screen. Every bullet is a faint, nearly transparent screen-constant streak drawn
    // with its OWN stroke in additive mode: overlapping marks inside one path are rasterised once and do not add
    // up, separate strokes do — so a lone shot is barely visible and sustained fire across a front sums into a
    // bright band. Muzzles follow the same rule (faint alone, bright where many fire).
    _bulletsFar(ctx, t, z, x0, y0, x1, y1) {
      const r = this.r, T = this.T, L = FAR_STREAK_PX / z, w = FAR_STREAK_W / z, mdot = FAR_MUZZLE_PX / z;
      const fresh = 1.5 * r.fpd;                   // frames: the bullet left the barrel within ~1.5 decisions
      const col = [rgba(T.team[0].bullet, FAR_BULLET_ALPHA), rgba(T.team[1].bullet, FAR_BULLET_ALPHA)];
      const mcol = `rgba(255,250,230,${FAR_MUZZLE_ALPHA})`;
      let n = 0, m = 0;
      ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round"; ctx.lineWidth = w;
      r.forBullets(t, x0, y0, x1, y1, (x, y, vx, vy, team, owner) => {
        if (n >= FAR_BULLET_CAP) return;
        const sp = Math.hypot(vx, vy) || 1;
        ctx.strokeStyle = col[team];
        ctx.beginPath(); ctx.moveTo(x - (vx / sp) * L, y - (vy / sp) * L); ctx.lineTo(x, y); ctx.stroke();
        n++;
        if (owner !== 65535 && this.s.alive[owner]) {
          const ox = this.s.x[owner], oy = this.s.y[owner];
          if (Math.hypot(x - ox, y - oy) <= sp * fresh + 30) {
            ctx.fillStyle = mcol; ctx.beginPath(); ctx.arc(ox, oy, mdot, 0, TAU); ctx.fill(); m++;
          }
        }
      });
      ctx.globalCompositeOperation = "source-over";
      this.stats.muzzles = m;
      return n;
    }
    // endregion

    // region effects
    // FAR/MID: pings for kills and captures (screen-constant rings). NEAR: the theme's effects and decals.
    _effects(ctx, t, z, lod, now, x0, y0, x1, y1) {
      const r = this.r, T = this.T, hz = r.hz;
      let n = 0;
      if (lod !== "near") {
        const winK = 0.9 * hz, winC = 1.6 * hz, winH = 0.3 * hz;
        const [e0, e1] = r.eventRange(t - winC, t + 0.5);
        const sparks = [], sr = FAR_SPARK_PX / z;
        let nh = 0;
        for (let e = e1 - 1; e >= e0 && n < PING_CAP; e--) {
          const kind = r.eK[e], age = t - r.eT[e];
          if (age < -0.5) continue;
          if (kind === EVK.hit && age < winH && nh < FAR_SPARK_CAP) {
            const k = r.eB[e], x = this.s.x[k], y = this.s.y[k];
            if (x < x0 || x > x1 || y < y0 || y > y1) continue;
            const g = sr * (1.4 - Math.max(0, age) / winH);
            sparks.push(x, y, g);
            nh++;
          } else if (kind === EVK.kill && age < winK) {
            const x = r.eX[e], y = r.eY[e];
            if (x < x0 || x > x1 || y < y0 || y > y1) continue;
            const k = Math.max(0, age / winK), vt = r.teamOf(r.eB[e]);
            ctx.strokeStyle = rgba(T.team[vt].core, 0.9 * (1 - k)); ctx.lineWidth = 1.6 / z;
            ctx.beginPath(); ctx.arc(x, y, (3 + 11 * k) / z, 0, TAU); ctx.stroke();
            n++;
          } else if (kind === EVK.capture) {
            const c = r.eA[e], x = r.cpXY[2 * c], y = r.cpXY[2 * c + 1];
            if (x < x0 - 400 || x > x1 + 400 || y < y0 - 400 || y > y1 + 400) continue;
            const k = Math.max(0, age / winC), col = T.team[r.eB[e]].core;
            ctx.strokeStyle = rgba(col, 0.9 * (1 - k)); ctx.lineWidth = 2.2 / z;
            ctx.beginPath(); ctx.arc(x, y, r.cpR + (8 + 40 * k) / z, 0, TAU); ctx.stroke();
            n++;
          }
        }
        if (nh) {                                   // hit sparks: faint, one fill each so they add up where fire is dense
          ctx.globalCompositeOperation = "lighter"; ctx.fillStyle = `rgba(255,250,230,${FAR_SPARK_ALPHA})`;
          for (let i = 0; i < sparks.length; i += 3) { ctx.beginPath(); ctx.arc(sparks[i], sparks[i + 1], sparks[i + 2], 0, TAU); ctx.fill(); }
          ctx.globalCompositeOperation = "source-over";
        }
        this.stats.sparks = nh;
        return n + nh;
      }
      const fx = T.fx, view = { t: now, zoom: z };
      const maxDec = ((fx.decal || 4000) / 1000) * hz;
      const [e0, e1] = r.eventRange(t - maxDec, t + 0.5);
      const half = r.msAt(0.5);
      const inView = (x, y, m) => x > x0 - m && x < x1 + m && y > y0 - m && y < y1 + m;
      for (let e = e1 - 1; e >= e0 && n < FX_CAP; e--) {
        const kind = r.eK[e], ms = r.msAt(r.eT[e]) - half, age = now - ms;
        if (age < 0) continue;
        if (kind === EVK.kill) {
          const x = r.eX[e], y = r.eY[e];
          if (!inView(x, y, 60)) continue;
          const team = r.teamOf(r.eB[e]);
          if (age < (fx.decal || 4000)) { T.drawDecal(ctx, { x, y, t0: ms, team }, view); n++; }
          if (age < (fx.death || 800)) { T.drawEffect(ctx, { type: "death", x, y, t0: ms, team, seed: e }, view); n++; }
        } else if (kind === EVK.hit && age < (fx.shieldHit || 260)) {
          const k = r.eB[e], x = this.s.x[k], y = this.s.y[k];
          if (!inView(x, y, 40) || !this.s.alive[k]) continue;
          const src = r.eA[e], sx = src === 65535 ? x - 1 : this.s.x[src], sy = src === 65535 ? y : this.s.y[src];
          T.drawEffect(ctx, { type: "hitSpark", x, y, t0: ms, team: src === 65535 ? 1 - r.teamOf(k) : r.teamOf(src), angle: Math.atan2(y - sy, x - sx), seed: e }, view);
          T.drawEffect(ctx, { type: (r.eV[e] & 128) ? "shieldHit" : "hitMarker", x, y, t0: ms, team: r.teamOf(k), seed: e }, view);
          n += 2;
        } else if (kind === EVK.respawn && age < (fx.respawn || 700)) {
          const x = r.eX[e], y = r.eY[e];
          if (!inView(x, y, 60)) continue;
          T.drawEffect(ctx, { type: "respawn", x, y, t0: ms, team: r.teamOf(r.eA[e]), seed: e }, view); n++;
        } else if (kind === EVK.capture && age < (fx.capture || 1300)) {
          const c = r.eA[e], x = r.cpXY[2 * c], y = r.cpXY[2 * c + 1];
          if (!inView(x, y, r.cpR + 80)) continue;
          T.drawEffect(ctx, { type: "capture", x, y, t0: ms, team: r.eB[e], r: r.cpR, seed: e }, view); n++;
        } else if (kind === EVK.dash && age < (fx.dash || 320)) {
          const k = r.eA[e], x = this.s.x[k], y = this.s.y[k];
          if (!inView(x, y, 60) || !this.s.alive[k]) continue;
          T.drawEffect(ctx, { type: "dash", x, y, t0: ms, team: r.teamOf(k), points: [{ x: x - Math.cos(this.s.a[k]) * 40, y: y - Math.sin(this.s.a[k]) * 40 }, { x, y }], seed: e }, view); n++;
        }
      }
      // muzzle flashes: bullets born in the last muzzle window
      const mw = ((fx.muzzle || 70) / 1000) * hz;
      r.forBullets(t, x0, y0, x1, y1, (x, y, vx, vy, team, owner) => {
        if (n >= FX_CAP || owner === 65535) return;
        // forBullets reports current position; flash only while the bullet is fresh (within the muzzle window)
        const sp = Math.hypot(vx, vy) * r.fpd;
        const ox = this.s.x[owner], oy = this.s.y[owner];
        const dist = Math.hypot(x - ox, y - oy);
        if (dist > sp * mw + 30) return;
        const ang = Math.atan2(vy, vx), R = T.size.agent + 8;
        T.drawEffect(ctx, { type: "muzzle", x: ox + Math.cos(ang) * R, y: oy + Math.sin(ang) * R, t0: now - 10, team, seed: owner }, view);
        n++;
      });
      return n;
    }
    // endregion

    draw(t, cam) {
      const t0 = performance.now();
      const ctx = this.ctx, dpr = this.dpr, z = cam.zoom, T = this.T, r = this.r;
      const lod = lodOf(z);
      const now = r.msAt(t);
      r.agentsInto(t, this.s);
      const hw = this.w / 2 / z, hh = this.h / 2 / z;
      const x0 = cam.x - hw, x1 = cam.x + hw, y0 = cam.y - hh, y1 = cam.y + hh;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = T.color.bg; ctx.fillRect(0, 0, this.w, this.h);
      ctx.setTransform(dpr * z, 0, 0, dpr * z, dpr * (this.w / 2 - cam.x * z), dpr * (this.h / 2 - cam.y * z));
      this._drawStatic(ctx, z, x0, y0, x1, y1);
      const terrA = lod === "far" ? 0.62 : lod === "mid" ? 0.62 * Math.max(0, (LOD_NEAR - z) / (LOD_NEAR - LOD_FAR)) * 0.6 : 0;
      if (terrA > 0.01) this._territory(ctx, terrA);
      const S = this.stats;
      S.lod = lod; S.zoom = +z.toFixed(3);
      S.refills = this._refills(t);
      S.cps = this._cps(ctx, t, z, lod, now, x0, y0, x1, y1);
      S.lines = lod === "far" ? 0 : this._attnLines(ctx, z, x0, y0, x1, y1);    // under the bodies: bodies stay readable
      S.bullets = lod === "far" ? 0 : this._bullets(ctx, t, z, lod, x0, y0, x1, y1);
      S.agents = lod === "far" ? this._agentsFar(ctx, z, x0, y0, x1, y1) : lod === "mid" ? this._agentsMid(ctx, z, x0, y0, x1, y1) : this._agentsNear(ctx, z, now, t, x0, y0, x1, y1);
      S.ammo = lod === "mid" ? this._ammoMid(ctx, z, x0, y0, x1, y1) : 0;
      S.marks = lod === "far" || !r.hasClasses() ? 0 : S.agents;        // silhouettes drawn (classes replay)
      if (lod === "far") S.bullets = this._bulletsFar(ctx, t, z, x0, y0, x1, y1);   // over the dots: fire must read
      S.fx = this._effects(ctx, t, z, lod, now, x0, y0, x1, y1);
      const mark = cam.mode === "follow" ? cam.follow : this.hover;
      if (mark >= 0 && this.s.alive[mark]) {
        ctx.save(); ctx.strokeStyle = T.color.wallEdgeHot; ctx.lineWidth = 1.6 / z; ctx.setLineDash([5 / z, 5 / z]);
        ctx.beginPath(); ctx.arc(this.s.x[mark], this.s.y[mark], T.size.agent + 10 / z + 8, 0, TAU); ctx.stroke(); ctx.restore();
      }
      S.drawMs = performance.now() - t0;
      return this.s;
    }

    // region minimap
    buildMini(wCss) {
      const dpr = window.devicePixelRatio || 1;
      const k = wCss / this.W;
      this.miniK = k; this.miniW = wCss; this.miniH = this.H * k;
      const cv = canvas(this.miniW * dpr, this.miniH * dpr), g = cv.getContext("2d");
      g.drawImage(this.farLayer, 0, 0, cv.width, cv.height);
      this.miniBase = cv;
    }
    drawMini(mc, cam) {
      const dpr = window.devicePixelRatio || 1, g = mc.getContext("2d"), r = this.r, s = this.s, T = this.T, k = this.miniK;
      if (mc.width !== Math.round(this.miniW * dpr)) { mc.width = Math.round(this.miniW * dpr); mc.height = Math.round(this.miniH * dpr); mc.style.width = this.miniW + "px"; mc.style.height = this.miniH + "px"; }
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.drawImage(this.miniBase, 0, 0);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.save(); g.globalAlpha = 0.55; g.imageSmoothingEnabled = true; g.globalCompositeOperation = "lighter";
      g.drawImage(this.terrCv, 0, 0, this.tw * this.cell * k, this.th * this.cell * k); g.restore();
      for (let c = 0; c < r.C; c++) {
        const st = r.cpState(cam.t, c), col = st.owner === 1 ? T.team[0].core : st.owner === 2 ? T.team[1].core : T.cp.neutral;
        g.fillStyle = rgba(col, st.owner ? 0.85 : 0.5);
        g.beginPath(); g.arc(r.cpXY[2 * c] * k, r.cpXY[2 * c + 1] * k, Math.max(1.8, r.cpR * k), 0, TAU); g.fill();
      }
      for (let team = 0; team < 2; team++) {
        g.fillStyle = T.team[team].core;
        const k0 = team ? r.T : 0, k1 = team ? r.N : r.T;
        for (let a = k0; a < k1; a++) if (s.alive[a]) g.fillRect(s.x[a] * k - 0.6, s.y[a] * k - 0.6, 1.2, 1.2);
      }
      const hw = this.w / 2 / cam.zoom, hh = this.h / 2 / cam.zoom;
      g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 1;
      g.strokeRect((cam.x - hw) * k, (cam.y - hh) * k, hw * 2 * k, hh * 2 * k);
    }
    // endregion
  }

  window.Arena = window.Arena || {};
  window.Arena.BigRenderer = BigRenderer;
  window.Arena.lodOf = lodOf;
  window.Arena.LOD = { FAR: LOD_FAR, NEAR: LOD_NEAR };
  window.Arena.CLASS_ORDER = CLASS_ORDER;
  window.Arena.classSvg = classSvg;
})();
