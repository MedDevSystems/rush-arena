/* region MODULE_CONTRACT [DOMAIN(7): Spectator; CONCEPT(8): NeonTheme; TECH(6): Canvas2D]
 * @purpose Neon/Tron visual theme for the 2D arena spectator: palette tokens, sizes, effect timings,
 *          and small draw hooks that paint one thing each on a Canvas 2D context.
 * @scope   Look only. No simulation, no replay parsing, no camera maths — the player owns those and
 *          calls the hooks with plain state objects (field names follow World11.snapshot, see
 *          design/THEME_SPEC.md).
 * @input   Canvas 2D context already transformed to world (logical px) or screen (CSS px) space.
 * @output  Pixels. Hooks never keep references to state between calls, except the sprite cache.
 * @invariants
 *  - No ctx.shadowBlur in per-entity hooks: glow is pre-rendered sprites + 'lighter' compositing.
 *  - Every hook restores globalAlpha / globalCompositeOperation / lineDash it touched.
 *  - Team 0 = BLUE (round body), team 1 = RED (hex body): hue AND shape carry the team.
 * @rationale
 *  Q: Why orange-red instead of pure red for team 1?
 *  A: Blue vs orange is the pair that survives all three common colour-vision deficiencies; pure red
 *  A: next to the violet walls collapses for protanopes. The shape cue is the second, hue-free channel.
 * @changes LAST_CHANGE: [v0.1.0] Initial neon theme for experiment 11 (workstream D).
 * endregion MODULE_CONTRACT */
(function (root) {
  'use strict';

  const TAU = Math.PI * 2;
  const spriteCache = new Map();

  // region BLOCK_HELPERS
  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  function makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.ceil(w));
    c.height = Math.max(1, Math.ceil(h));
    return c;
  }

  // Deterministic 0..1 noise from an integer seed: effects look random but do not flicker per frame.
  function hash01(seed) {
    let t = (seed + 0x6d2b79f5) | 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function easeOut(k) { return 1 - (1 - k) * (1 - k); }
  // endregion BLOCK_HELPERS

  // region BLOCK_TOKENS
  const THEME = {
    name: 'neon',
    version: '0.1.0',

    fonts: {
      display: "'Orbitron', system-ui, sans-serif",   // scores, CP letters, banners
      ui: "'Rajdhani', system-ui, sans-serif",        // kill feed, labels, HUD small print
      googleFontsHref: 'https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap',
    },

    color: {
      bg: '#03050c',
      floor: '#060915',
      grid: '#0c1229',
      gridMajor: '#152049',
      wallFill: '#0b0c20',
      wallHatch: '#1d1a48',
      wallEdge: '#8a7bff',
      wallEdgeHot: '#d6d0ff',
      text: '#e8ecff',
      textDim: '#7d86b0',
      panel: 'rgba(5,7,16,0.74)',
      panelEdge: 'rgba(138,123,255,0.35)',
      shield: '#eafcff',
      white: '#ffffff',
    },

    team: [
      { name: 'BLUE', core: '#35c8ff', glow: '#168dff', deep: '#051f33', bullet: '#a6ecff', text: '#72d9ff', shape: 'circle' },
      { name: 'RED',  core: '#ff8433', glow: '#ff4f12', deep: '#331105', bullet: '#ffcaa0', text: '#ffa066', shape: 'hex' },
    ],

    cp: {
      neutral: '#b9b3e3',
      contested: '#ffffff',
      labels: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    },

    size: {
      agent: 12,            // PLAYER_RADIUS
      bullet: 3,            // BULLET_RADIUS
      hpArcGap: 4.5,        // HP arc radius = agent + gap
      shieldGap: 8.5,       // shield ring radius = agent + gap
      shieldSegments: 8,    // 40 shield / 8 = 5 HP per lit segment
      tracerFrames: 1.7,    // tracer tail length = |v| * frames  (24 px/frame -> ~41 px)
      wedge: 9,             // facing wedge reaches agent + wedge
      labelMinZoom: 1.05,   // agent labels only when zoomed in enough to read them
    },

    // Effect lifetimes in ms of GAME time (they slow down with slow-mo).
    fx: {
      muzzle: 70,
      hitSpark: 220,
      shieldHit: 260,
      hitMarker: 180,
      death: 800,
      decal: 4000,
      respawn: 700,
      dash: 320,
      capture: 1300,
      killfeed: 5000,
      banner: 2200,
    },

    camera: {
      directorPadding: 170,     // logical px around the action box
      maxZoomOverFit: 2.6,      // director never zooms tighter than 2.6x the whole-arena fit
      absMaxZoom: 1.6,          // ... nor tighter than 1.6 CSS px per logical px
      portraitMinOverFit: 1.7,  // portrait screens: director never zooms OUT past 1.7x fit, else half the phone is empty
      follow: 3.2,              // 1/s exponential approach for centre
      zoomFollow: 1.6,          // 1/s for zoom — slower than pan so framing never "breathes"
      combatMemoryFrames: 90,   // an agent counts as "in action" this long after shooting/being hit
    },

    shake: {
      kill: 5,                  // CSS px amplitude added on a kill inside the view
      max: 7,                   // hard budget: amplitude never exceeds this
      decayPerSec: 9,           // exponential decay
      minIntervalMs: 350,       // at most one new kick per interval
    },
  };
  // endregion BLOCK_TOKENS

  // region FUNC_glowSprite
  // @purpose Radial glow sprite cached per (colour, radius). Drawn with 'lighter' it reads as bloom.
  function glowSprite(hex, radius) {
    const r = Math.max(4, Math.round(radius || 64));
    const key = hex + '|' + r;
    let s = spriteCache.get(key);
    if (s) return s;
    s = makeCanvas(r * 2, r * 2);
    const g = s.getContext('2d');
    const grad = g.createRadialGradient(r, r, 0, r, r, r);
    grad.addColorStop(0, rgba(hex, 1));
    grad.addColorStop(0.22, rgba(hex, 0.5));
    grad.addColorStop(0.55, rgba(hex, 0.13));
    grad.addColorStop(1, rgba(hex, 0));
    g.fillStyle = grad;
    g.fillRect(0, 0, r * 2, r * 2);
    spriteCache.set(key, s);
    return s;
  }

  function drawGlow(ctx, hex, x, y, radius, alpha) {
    if (alpha <= 0 || radius <= 0) return;
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.drawImage(glowSprite(hex, 64), x - radius, y - radius, radius * 2, radius * 2);
    ctx.globalAlpha = 1;
  }
  // endregion FUNC_glowSprite

  // region FUNC_buildStaticLayer
  // @purpose Pre-render floor grid + walls once per map. The player blits it every frame.
  // @io geometry {w, h, tile, walls: [[col,row],...]}, scale (device px per logical px) -> canvas
  function buildStaticLayer(geom, scale) {
    const C = THEME.color;
    const cv = makeCanvas(geom.w * scale, geom.h * scale);
    const g = cv.getContext('2d');
    g.scale(scale, scale);
    const t = geom.tile;
    const cols = Math.round(geom.w / t);
    const rows = Math.round(geom.h / t);
    const wall = new Uint8Array(cols * rows);
    for (const [c, r] of geom.walls) if (c >= 0 && c < cols && r >= 0 && r < rows) wall[r * cols + c] = 1;
    const isWall = (c, r) => c >= 0 && c < cols && r >= 0 && r < rows && wall[r * cols + c] === 1;

    g.fillStyle = C.floor;
    g.fillRect(0, 0, geom.w, geom.h);

    // Floor grid: thin every tile, brighter every 5 tiles — gives motion a sense of speed.
    g.lineWidth = 1 / scale;
    g.strokeStyle = C.grid;
    g.beginPath();
    for (let c = 1; c < cols; c++) if (c % 5) { g.moveTo(c * t, 0); g.lineTo(c * t, geom.h); }
    for (let r = 1; r < rows; r++) if (r % 5) { g.moveTo(0, r * t); g.lineTo(geom.w, r * t); }
    g.stroke();
    g.lineWidth = 1.2 / scale + 0.4;
    g.strokeStyle = C.gridMajor;
    g.beginPath();
    for (let c = 5; c < cols; c += 5) { g.moveTo(c * t, 0); g.lineTo(c * t, geom.h); }
    for (let r = 5; r < rows; r += 5) { g.moveTo(0, r * t); g.lineTo(geom.w, r * t); }
    g.stroke();

    // Wall bodies: dark fill + faint diagonal hatch so they read as solid, not as holes.
    g.save();
    g.beginPath();
    for (const [c, r] of geom.walls) g.rect(c * t, r * t, t, t);
    g.fillStyle = C.wallFill;
    g.fill();
    g.clip();
    g.strokeStyle = C.wallHatch;
    g.lineWidth = 1.5;
    g.beginPath();
    for (let d = -geom.h; d < geom.w; d += 10) { g.moveTo(d, 0); g.lineTo(d + geom.h, geom.h); }
    g.stroke();
    g.restore();

    // Wall outline: only edges facing open floor, stroked three times (wide dim -> thin hot) = neon tube.
    g.beginPath();
    for (const [c, r] of geom.walls) {
      const x = c * t, y = r * t;
      if (!isWall(c, r - 1)) { g.moveTo(x, y); g.lineTo(x + t, y); }
      if (!isWall(c, r + 1)) { g.moveTo(x, y + t); g.lineTo(x + t, y + t); }
      if (!isWall(c - 1, r)) { g.moveTo(x, y); g.lineTo(x, y + t); }
      if (!isWall(c + 1, r)) { g.moveTo(x + t, y); g.lineTo(x + t, y + t); }
    }
    g.lineCap = 'round';
    g.strokeStyle = rgba(C.wallEdge, 0.13); g.lineWidth = 12; g.stroke();
    g.strokeStyle = rgba(C.wallEdge, 0.38); g.lineWidth = 4.5; g.stroke();
    g.strokeStyle = rgba(C.wallEdgeHot, 0.95); g.lineWidth = 1.3; g.stroke();

    // Arena border.
    g.strokeStyle = rgba(C.wallEdge, 0.5);
    g.lineWidth = 3;
    g.strokeRect(1.5, 1.5, geom.w - 3, geom.h - 3);
    return cv;
  }

  // @purpose Minimap background: walls only, one device pixel per cell or better.
  function buildMinimapLayer(geom, widthPx, dpr) {
    const k = (widthPx * dpr) / geom.w;
    const cv = makeCanvas(geom.w * k, geom.h * k);
    const g = cv.getContext('2d');
    g.fillStyle = 'rgba(6,9,21,0.9)';
    g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = rgba(THEME.color.wallEdge, 0.55);
    for (const [c, r] of geom.walls) g.fillRect(c * geom.tile * k, r * geom.tile * k, Math.ceil(geom.tile * k), Math.ceil(geom.tile * k));
    return cv;
  }
  // endregion FUNC_buildStaticLayer

  // region FUNC_drawControlPoint
  // @purpose CP: owner-coloured dashed ring slowly rotating, capture sweep in the capturer's colour,
  //          white flicker when contested, big letter in the middle.
  // @io cp {x, y, r, owner 0|1|2, progress 0..1, cap_team 0|1|2, contested?}, view {t, zoom}, index
  function cpColor(owner) { return owner === 1 ? THEME.team[0].core : owner === 2 ? THEME.team[1].core : THEME.cp.neutral; }

  function drawControlPoint(ctx, cp, view, index) {
    const t = view.t;
    const col = cpColor(cp.owner);
    const pulse = 0.5 + 0.5 * Math.sin(t / 420 + index);

    ctx.fillStyle = rgba(col, cp.owner ? 0.07 + 0.03 * pulse : 0.035);
    ctx.beginPath(); ctx.arc(cp.x, cp.y, cp.r, 0, TAU); ctx.fill();

    ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = rgba(col, 0.10);
    ctx.lineWidth = 12;
    ctx.beginPath(); ctx.arc(cp.x, cp.y, cp.r, 0, TAU); ctx.stroke();

    const contested = !!cp.contested;
    ctx.setLineDash([22, 12]);
    ctx.lineDashOffset = -t * 0.018;
    ctx.lineWidth = 3;
    ctx.strokeStyle = contested && ((t / 90) | 0) % 2 === 0 ? rgba(THEME.cp.contested, 0.95) : rgba(col, 0.85);
    ctx.beginPath(); ctx.arc(cp.x, cp.y, cp.r, 0, TAU); ctx.stroke();
    ctx.setLineDash([]);

    if (cp.progress > 0 && cp.cap_team) {
      const cc = THEME.team[cp.cap_team - 1].core;
      const a0 = -Math.PI / 2;
      ctx.strokeStyle = rgba(cc, 0.22);
      ctx.lineWidth = 16;
      ctx.beginPath(); ctx.arc(cp.x, cp.y, cp.r - 9, a0, a0 + TAU * cp.progress); ctx.stroke();
      ctx.strokeStyle = rgba(cc, 0.95);
      ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(cp.x, cp.y, cp.r - 9, a0, a0 + TAU * cp.progress); ctx.stroke();
      const ex = cp.x + Math.cos(a0 + TAU * cp.progress) * (cp.r - 9);
      const ey = cp.y + Math.sin(a0 + TAU * cp.progress) * (cp.r - 9);
      drawGlow(ctx, cc, ex, ey, 26, 0.9);
    }
    ctx.globalCompositeOperation = 'source-over';

    ctx.font = '700 34px ' + THEME.fonts.display;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = rgba(col, 0.55 + 0.25 * pulse);
    ctx.fillText(THEME.cp.labels[index] || String(index + 1), cp.x, cp.y + 1);
  }
  // endregion FUNC_drawControlPoint

  // region FUNC_drawAgent
  // @purpose One agent: bloom, team-shaped body, facing wedge, HP arc, segmented shield ring, label.
  // @io a {x, y, angle, hp, shield, alive, team, label?}, view {t, zoom}, st {flash 0..1, dashing?}
  function bodyPath(ctx, shape, x, y, r, angle) {
    ctx.beginPath();
    if (shape === 'hex') {
      for (let i = 0; i < 6; i++) {
        const a = angle + Math.PI / 6 + (i * TAU) / 6;
        const px = x + Math.cos(a) * r * 1.08, py = y + Math.sin(a) * r * 1.08;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
    } else {
      ctx.arc(x, y, r, 0, TAU);
    }
  }

  function drawAgent(ctx, a, view, st) {
    if (!a.alive) return;
    const T = THEME.team[a.team];
    const S = THEME.size;
    const r = S.agent;
    const flash = (st && st.flash) || 0;
    const lowHp = a.hp < 30;
    const beat = lowHp ? 0.5 + 0.5 * Math.sin(view.t / 90) : 0;

    ctx.globalCompositeOperation = 'lighter';
    drawGlow(ctx, T.glow, a.x, a.y, r * 3.4, 0.55 + 0.4 * flash);
    ctx.globalCompositeOperation = 'source-over';

    bodyPath(ctx, T.shape, a.x, a.y, r, a.angle);
    ctx.fillStyle = flash > 0.05 ? rgba('#ffffff', 0.25 + 0.6 * flash) : T.deep;
    ctx.fill();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = T.core;
    ctx.stroke();

    // Facing wedge: where the barrel points is the single most important thing to read.
    const ca = Math.cos(a.angle), sa = Math.sin(a.angle);
    const tipX = a.x + ca * (r + S.wedge), tipY = a.y + sa * (r + S.wedge);
    const bx = a.x + ca * (r - 3), by = a.y + sa * (r - 3);
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(bx - sa * 5, by + ca * 5);
    ctx.lineTo(bx + sa * 5, by - ca * 5);
    ctx.closePath();
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // HP arc (clockwise from 12 o'clock) over a dim track.
    const hr = r + S.hpArcGap;
    const a0 = -Math.PI / 2;
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = rgba(T.core, 0.16);
    ctx.beginPath(); ctx.arc(a.x, a.y, hr, 0, TAU); ctx.stroke();
    ctx.strokeStyle = lowHp ? rgba('#ffffff', 0.55 + 0.45 * beat) : T.core;
    ctx.beginPath(); ctx.arc(a.x, a.y, hr, a0, a0 + TAU * Math.max(0, Math.min(1, a.hp / 100))); ctx.stroke();

    // Shield: 8 segments, lit ones bright white-cyan. A glance tells "shield up / broken".
    const seg = S.shieldSegments;
    const lit = (a.shield / 40) * seg;
    if (lit > 0.01) {
      const sr = r + S.shieldGap;
      const span = TAU / seg;
      ctx.lineWidth = 1.8;
      for (let i = 0; i < seg; i++) {
        const fill = Math.max(0, Math.min(1, lit - i));
        if (fill <= 0) break;
        const s0 = a0 + i * span + 0.07;
        ctx.strokeStyle = rgba(THEME.color.shield, 0.35 + 0.55 * fill);
        ctx.beginPath(); ctx.arc(a.x, a.y, sr, s0, s0 + (span - 0.14) * fill); ctx.stroke();
      }
    }

    if (a.label && view.zoom >= S.labelMinZoom) {
      ctx.font = '600 11px ' + THEME.fonts.ui;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = rgba(T.text, 0.8);
      ctx.fillText(a.label, a.x, a.y + r + 12);
    }
  }
  // endregion FUNC_drawAgent

  // region FUNC_drawBullets
  // @purpose Tracers batched per team (one path per team), bright heads with glow.
  // @io bullets [{x, y, vx, vy, team}]
  function drawBullets(ctx, bullets) {
    if (!bullets.length) return;
    const k = THEME.size.tracerFrames;
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (let team = 0; team < 2; team++) {
      const T = THEME.team[team];
      ctx.beginPath();
      let any = false;
      for (const b of bullets) {
        if (b.team !== team) continue;
        any = true;
        ctx.moveTo(b.x - b.vx * k, b.y - b.vy * k);
        ctx.lineTo(b.x, b.y);
      }
      if (!any) continue;
      ctx.strokeStyle = rgba(T.glow, 0.35); ctx.lineWidth = 6; ctx.stroke();
      ctx.strokeStyle = rgba(T.bullet, 0.95); ctx.lineWidth = 2; ctx.stroke();
      for (const b of bullets) if (b.team === team) drawGlow(ctx, T.glow, b.x, b.y, 11, 0.8);
    }
    ctx.fillStyle = '#ffffff';
    for (const b of bullets) { ctx.beginPath(); ctx.arc(b.x, b.y, 1.6, 0, TAU); ctx.fill(); }
    ctx.globalCompositeOperation = 'source-over';
  }
  // endregion FUNC_drawBullets

  // region FUNC_drawEffect
  // @purpose Transient effects. Returns false once the effect has expired, so the player can drop it.
  // @io fx {type, x, y, t0, team, angle?, seed?, r?, points?}, view {t}
  function drawEffect(ctx, fx, view) {
    const dur = THEME.fx[fx.type] || 300;
    const k = (view.t - fx.t0) / dur;
    if (k >= 1) return false;
    if (k < 0) return true;
    const T = THEME.team[fx.team || 0];
    const inv = 1 - k;
    ctx.globalCompositeOperation = 'lighter';

    switch (fx.type) {
      case 'muzzle': {
        drawGlow(ctx, T.bullet, fx.x, fx.y, 18 * inv + 4, inv);
        drawGlow(ctx, '#ffffff', fx.x, fx.y, 6 * inv + 2, inv);
        break;
      }
      case 'hitSpark': {
        ctx.strokeStyle = rgba(T.bullet, inv);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let i = 0; i < 7; i++) {
          const ang = (fx.angle || 0) + Math.PI + (i - 3) * 0.38 + (hash01((fx.seed || 0) + i) - 0.5) * 0.4;
          const d0 = 3 + 10 * easeOut(k), d1 = d0 + 5 + 9 * inv * hash01((fx.seed || 0) + 31 * i);
          ctx.moveTo(fx.x + Math.cos(ang) * d0, fx.y + Math.sin(ang) * d0);
          ctx.lineTo(fx.x + Math.cos(ang) * d1, fx.y + Math.sin(ang) * d1);
        }
        ctx.stroke();
        drawGlow(ctx, T.glow, fx.x, fx.y, 14 * inv, 0.8 * inv);
        break;
      }
      case 'shieldHit': {
        const rr = THEME.size.agent + THEME.size.shieldGap + 3 * k;
        ctx.strokeStyle = rgba(THEME.color.shield, 0.9 * inv);
        ctx.lineWidth = 3 * inv + 0.5;
        ctx.beginPath(); ctx.arc(fx.x, fx.y, rr, 0, TAU); ctx.stroke();
        break;
      }
      case 'hitMarker': {
        const s = 5 + 3 * k;
        ctx.strokeStyle = rgba('#ffffff', inv);
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
          ctx.moveTo(fx.x + dx * s, fx.y + dy * s);
          ctx.lineTo(fx.x + dx * (s + 5), fx.y + dy * (s + 5));
        }
        ctx.stroke();
        break;
      }
      case 'death': {
        const e = easeOut(k);
        drawGlow(ctx, '#ffffff', fx.x, fx.y, 60 * inv, 0.7 * inv);
        drawGlow(ctx, T.glow, fx.x, fx.y, 110 * inv + 20, 0.9 * inv);
        ctx.strokeStyle = rgba(T.core, inv);
        ctx.lineWidth = 4 * inv + 0.5;
        ctx.beginPath(); ctx.arc(fx.x, fx.y, 12 + 70 * e, 0, TAU); ctx.stroke();
        ctx.fillStyle = rgba(T.bullet, inv);
        for (let i = 0; i < 14; i++) {
          const ang = (i / 14) * TAU + hash01((fx.seed || 0) + i) * 0.5;
          const d = (40 + 70 * hash01((fx.seed || 0) + 97 * i)) * e;
          ctx.beginPath(); ctx.arc(fx.x + Math.cos(ang) * d, fx.y + Math.sin(ang) * d, 2.6 * inv + 0.3, 0, TAU); ctx.fill();
        }
        break;
      }
      case 'respawn': {
        const rr = 14 + 80 * (1 - easeOut(k));
        ctx.strokeStyle = rgba(T.core, 0.9 * Math.min(1, k * 3) * inv + 0.1);
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(fx.x, fx.y, rr, 0, TAU); ctx.stroke();
        ctx.beginPath(); ctx.arc(fx.x, fx.y, rr * 0.6, 0, TAU); ctx.stroke();
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(fx.x - rr * 1.3, fx.y); ctx.lineTo(fx.x + rr * 1.3, fx.y);
        ctx.moveTo(fx.x, fx.y - rr * 1.3); ctx.lineTo(fx.x, fx.y + rr * 1.3);
        ctx.stroke();
        drawGlow(ctx, T.glow, fx.x, fx.y, 50 * Math.sin(Math.PI * k) + 10, 0.9);
        break;
      }
      case 'dash': {
        const pts = fx.points || [];
        const n = pts.length;
        if (n > 1) {
          ctx.strokeStyle = rgba(T.core, 0.5 * inv);
          ctx.lineWidth = 10 * inv + 1;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < n; i++) ctx.lineTo(pts[i].x, pts[i].y);
          ctx.stroke();
        }
        ctx.lineWidth = 1.5;
        for (let i = 0; i < n; i++) {
          const w = (i + 1) / n;
          ctx.strokeStyle = rgba(T.core, 0.6 * inv * w);
          ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, THEME.size.agent * (0.55 + 0.45 * w), 0, TAU); ctx.stroke();
        }
        break;
      }
      case 'capture': {
        const e = easeOut(k);
        const base = fx.r || 150;
        ctx.strokeStyle = rgba(T.core, 0.9 * inv);
        ctx.lineWidth = 7 * inv + 0.5;
        ctx.beginPath(); ctx.arc(fx.x, fx.y, base + 90 * e, 0, TAU); ctx.stroke();
        if (k > 0.15) {
          const k2 = (k - 0.15) / 0.85;
          ctx.lineWidth = 3 * (1 - k2) + 0.5;
          ctx.beginPath(); ctx.arc(fx.x, fx.y, base + 45 * easeOut(k2), 0, TAU); ctx.stroke();
        }
        drawGlow(ctx, T.glow, fx.x, fx.y, base * 1.3, 0.35 * inv);
        break;
      }
      default: break;
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    return true;
  }

  // @purpose Floor decal left by a death: fades over THEME.fx.decal ms. Drawn under agents.
  function drawDecal(ctx, d, view) {
    const k = (view.t - d.t0) / THEME.fx.decal;
    if (k >= 1) return false;
    const T = THEME.team[d.team || 0];
    const a = 0.3 * (1 - k);
    ctx.strokeStyle = rgba(T.core, a);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(d.x, d.y, 16, 0, TAU); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(d.x - 7, d.y - 7); ctx.lineTo(d.x + 7, d.y + 7);
    ctx.moveTo(d.x + 7, d.y - 7); ctx.lineTo(d.x - 7, d.y + 7);
    ctx.stroke();
    return true;
  }
  // endregion FUNC_drawEffect

  // region FUNC_screenSpace
  // All screen-space hooks take `ui` = {w, h, s (ui scale), compact (bool)} in CSS px.
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawVignette(ctx, ui) {
    const g = ctx.createRadialGradient(ui.w / 2, ui.h / 2, Math.min(ui.w, ui.h) * 0.35, ui.w / 2, ui.h / 2, Math.hypot(ui.w, ui.h) * 0.62);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, ui.w, ui.h);
  }

  // @purpose Top-centre score panel. Returns its bottom y so other widgets can stack under it.
  // @io hud {score:[b,r], scoreToWin, timeLeftSec, lives:[b,r], livesMax, alive:[b,r], teamSize}
  function drawHud(ctx, hud, ui) {
    const s = ui.s;
    const pw = Math.min(640 * s, ui.w - 32);
    const ph = (ui.compact ? 50 : 62) * s;
    const x0 = (ui.w - pw) / 2, y0 = 12;
    const mid = ui.w / 2;

    ctx.fillStyle = THEME.color.panel;
    roundRect(ctx, x0, y0, pw, ph, 10 * s); ctx.fill();
    ctx.strokeStyle = THEME.color.panelEdge; ctx.lineWidth = 1; ctx.stroke();

    // Tug-of-war score bar: each team fills outward from the centre.
    const barY = y0 + ph - 13 * s, barH = 5 * s, half = pw / 2 - 18 * s;
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fillRect(mid - half, barY, half * 2, barH);
    for (let team = 0; team < 2; team++) {
      const T = THEME.team[team];
      const frac = Math.max(0, Math.min(1, hud.score[team] / hud.scoreToWin));
      const len = half * frac;
      const x = team === 0 ? mid - len : mid;
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = rgba(T.glow, 0.35);
      ctx.fillRect(x, barY - 3 * s, len, barH + 6 * s);
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = T.core;
      ctx.fillRect(x, barY, len, barH);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillRect(mid - 0.5, barY - 4 * s, 1, barH + 8 * s);

    // Scores. Rajdhani, not Orbitron: Orbitron's slashed square zero reads as an icon at HUD size.
    const fs = (ui.compact ? 26 : 34) * s;
    ctx.font = '700 ' + fs + 'px ' + THEME.fonts.ui;
    ctx.textBaseline = 'alphabetic';
    const numY = y0 + (ui.compact ? 27 : 33) * s;
    ctx.textAlign = 'left';
    ctx.fillStyle = THEME.team[0].core;
    ctx.fillText(String(Math.floor(hud.score[0])), x0 + 16 * s, numY);
    ctx.textAlign = 'right';
    ctx.fillStyle = THEME.team[1].core;
    ctx.fillText(String(Math.floor(hud.score[1])), x0 + pw - 16 * s, numY);

    // Timer.
    const tl = Math.max(0, Math.ceil(hud.timeLeftSec));
    const mm = String(Math.floor(tl / 60)).padStart(2, '0'), ss = String(tl % 60).padStart(2, '0');
    ctx.textAlign = 'center';
    ctx.font = '700 ' + (ui.compact ? 15 : 18) * s + 'px ' + THEME.fonts.display;
    ctx.fillStyle = THEME.color.text;
    ctx.fillText(mm + ':' + ss, mid, numY - 2 * s);

    // Alive pips + lives next to the scores.
    const pipR = 3.2 * s, pipGap = 9 * s;
    const pipY = numY - 6 * s;
    for (let team = 0; team < 2; team++) {
      const T = THEME.team[team];
      for (let i = 0; i < hud.teamSize; i++) {
        const px = team === 0 ? mid - 44 * s - i * pipGap : mid + 44 * s + i * pipGap;
        ctx.beginPath(); ctx.arc(px, pipY, pipR, 0, TAU);
        if (i < hud.alive[team]) { ctx.fillStyle = T.core; ctx.fill(); }
        else { ctx.strokeStyle = rgba(T.core, 0.45); ctx.lineWidth = 1; ctx.stroke(); }
      }
      if (!ui.compact) {
        ctx.font = '600 ' + 12 * s + 'px ' + THEME.fonts.ui;
        ctx.fillStyle = rgba(T.text, 0.85);
        ctx.textAlign = team === 0 ? 'right' : 'left';
        const lx = team === 0 ? mid - 44 * s - hud.teamSize * pipGap - 4 * s : mid + 44 * s + hud.teamSize * pipGap + 4 * s;
        ctx.fillText('LIVES ' + hud.lives[team], lx, pipY + 4 * s);
      }
    }
    return y0 + ph;
  }

  // @io feed [{t, killer:{label, team}, victim:{label, team}, how?:'dash'|'shot'}], view {t}, top y
  function drawKillfeed(ctx, feed, view, ui, top) {
    const s = ui.s;
    const rowH = 24 * s, pad = 8 * s;
    const maxRows = ui.compact ? 3 : 5;
    const live = feed.filter((e) => view.t - e.t < THEME.fx.killfeed).slice(-maxRows);
    ctx.font = '700 ' + 15 * s + 'px ' + THEME.fonts.ui;
    ctx.textBaseline = 'middle';
    let y = top + 10 * s;
    for (const e of live) {
      const age = view.t - e.t;
      const a = Math.min(1, age / 150) * Math.min(1, (THEME.fx.killfeed - age) / 600);
      const kT = THEME.team[e.killer.team], vT = THEME.team[e.victim.team];
      const icon = e.how === 'dash' ? '  »━  ' : '  ━╸  ';
      const wK = ctx.measureText(e.killer.label).width;
      const wI = ctx.measureText(icon).width;
      const wV = ctx.measureText(e.victim.label).width;
      const w = wK + wI + wV + pad * 2;
      const x = ui.w - 16 - w;
      ctx.globalAlpha = a;
      ctx.fillStyle = THEME.color.panel;
      roundRect(ctx, x, y, w, rowH, 6 * s); ctx.fill();
      ctx.fillStyle = kT.core; ctx.fillRect(x, y + 4 * s, 2 * s, rowH - 8 * s);
      ctx.textAlign = 'left';
      ctx.fillStyle = kT.text; ctx.fillText(e.killer.label, x + pad, y + rowH / 2 + 1);
      ctx.fillStyle = rgba('#ffffff', 0.75); ctx.fillText(icon, x + pad + wK, y + rowH / 2 + 1);
      ctx.fillStyle = vT.text; ctx.fillText(e.victim.label, x + pad + wK + wI, y + rowH / 2 + 1);
      ctx.globalAlpha = 1;
      y += rowH + 5 * s;
    }
  }

  // @io mm {layer (canvas from buildMinimapLayer), arenaW, arenaH, agents, cps, viewRect {x,y,w,h}}
  function drawMinimap(ctx, mm, ui) {
    const w = Math.min(ui.compact ? 120 : 200, ui.w * 0.34);
    const h = w * (mm.arenaH / mm.arenaW);
    const x0 = 16, y0 = ui.h - h - 16;
    const k = w / mm.arenaW;
    ctx.fillStyle = THEME.color.panel;
    roundRect(ctx, x0 - 4, y0 - 4, w + 8, h + 8, 6); ctx.fill();
    ctx.drawImage(mm.layer, x0, y0, w, h);
    for (let i = 0; i < mm.cps.length; i++) {
      const cp = mm.cps[i];
      ctx.strokeStyle = cpColor(cp.owner);
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(x0 + cp.x * k, y0 + cp.y * k, Math.max(3, cp.r * k), 0, TAU); ctx.stroke();
    }
    for (const a of mm.agents) {
      if (!a.alive) continue;
      ctx.fillStyle = THEME.team[a.team].core;
      ctx.beginPath(); ctx.arc(x0 + a.x * k, y0 + a.y * k, 2.2, 0, TAU); ctx.fill();
    }
    if (mm.viewRect) {
      const v = mm.viewRect;
      const vx = Math.max(0, v.x), vy = Math.max(0, v.y);
      const vw = Math.min(mm.arenaW, v.x + v.w) - vx, vh = Math.min(mm.arenaH, v.y + v.h) - vy;
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + vx * k, y0 + vy * k, vw * k, vh * k);
    }
    ctx.strokeStyle = THEME.color.panelEdge;
    ctx.strokeRect(x0 - 0.5, y0 - 0.5, w + 1, h + 1);
  }

  // @io banner {text, sub?, team? (0|1|null), t0, dur?}
  function drawBanner(ctx, banner, view, ui) {
    const dur = banner.dur || THEME.fx.banner;
    const k = (view.t - banner.t0) / dur;
    if (k >= 1 || k < 0) return k < 0;
    const a = Math.min(1, k / 0.08) * Math.min(1, (1 - k) / 0.2);
    const col = banner.team == null ? THEME.color.text : THEME.team[banner.team].core;
    const glow = banner.team == null ? THEME.color.wallEdge : THEME.team[banner.team].glow;
    const y = ui.h * 0.24;
    const fs = (ui.compact ? 20 : 32) * ui.s;
    ctx.globalAlpha = a;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '900 ' + fs + 'px ' + THEME.fonts.display;
    ctx.shadowColor = glow;       // one text item per frame: shadowBlur is affordable here
    ctx.shadowBlur = 22;
    ctx.fillStyle = col;
    ctx.fillText(banner.text, ui.w / 2, y + (1 - Math.min(1, k / 0.08)) * 8);
    ctx.shadowBlur = 0;
    if (banner.sub) {
      ctx.font = '600 ' + fs * 0.5 + 'px ' + THEME.fonts.ui;
      ctx.fillStyle = rgba(THEME.color.text, 0.85);
      ctx.fillText(banner.sub, ui.w / 2, y + fs * 0.95);
    }
    ctx.globalAlpha = 1;
    return true;
  }
  // endregion FUNC_screenSpace

  THEME.util = { rgba, hash01, easeOut, glowSprite, drawGlow, roundRect };
  THEME.buildStaticLayer = buildStaticLayer;
  THEME.buildMinimapLayer = buildMinimapLayer;
  THEME.drawControlPoint = drawControlPoint;
  THEME.drawAgent = drawAgent;
  THEME.drawBullets = drawBullets;
  THEME.drawEffect = drawEffect;
  THEME.drawDecal = drawDecal;
  THEME.drawVignette = drawVignette;
  THEME.drawHud = drawHud;
  THEME.drawKillfeed = drawKillfeed;
  THEME.drawMinimap = drawMinimap;
  THEME.drawBanner = drawBanner;

  root.ARENA_THEME = THEME;
  if (typeof module !== 'undefined' && module.exports) module.exports = THEME;
})(typeof window !== 'undefined' ? window : globalThis);
