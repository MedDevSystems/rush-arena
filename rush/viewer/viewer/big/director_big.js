// region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): MultiHotspotDirector; TECH(7): browser JS]
// @purpose Camera for a big battle. Auto mode keeps an event-density grid (kills, hits, captures, contact
//   between armies), holds on the hottest hotspot with hysteresis, flies to a new one by zooming out on the
//   way, and cuts to an establishing overview every so often. Other modes: overview, free (pan/zoom), follow.
// @invariants
// - Camera state {x, y, zoom} is world px / CSS px per world px; zoom is clamped to [fit*0.9, maxZoom]
// - A hotspot is held at least HOLD_MIN seconds of wall time; a switch needs SWITCH_RATIO x its heat
// - Heat is recomputed at most every HEAT_EVERY seconds of wall time (grid cost is bounded, not per frame)
// @links LINKS_TO: replay2.js (events, agent state), render_big.js, app_big.js
// endregion MODULE_CONTRACT
(function () {
  "use strict";
  const HEAT_EVERY = 0.25, HOLD_MIN = 3.5, HOLD_MAX = 14, SWITCH_RATIO = 1.45;
  const OVERVIEW_EVERY = 55, OVERVIEW_FOR = 5;          // seconds of wall time in auto mode
  const EVK = { kill: 1, hit: 2, capture: 3 };

  class BigDirector {
    constructor(replay) {
      this.r = replay;
      const m = replay.map;
      this.cellW = Math.max(600, Math.max(m.w, m.h) / 22);
      this.gw = Math.ceil(m.w / this.cellW); this.gh = Math.ceil(m.h / this.cellW);
      this.heat = new Float32Array(this.gw * this.gh);
      this.wx = new Float32Array(this.gw * this.gh); this.wy = new Float32Array(this.gw * this.gh);
      this.x = m.w / 2; this.y = m.h / 2; this.zoom = 0.1;
      this.tx = this.x; this.ty = this.y; this.tz = this.zoom;
      this.mode = "auto"; this.follow = -1;
      this.cur = -1; this.curHeat = 0; this.heldFor = 0; this.sinceHeat = 1e9;
      this.sinceOverview = 0; this.overviewLeft = 0;
      this.hotspots = [];
      this.t = 0;
    }

    // pad = CSS px covered by the HUD at the top and the control bar at the bottom: the overview fits the map
    // into what is left and shifts the camera so the map sits in the visible band, not under the chrome.
    setView(w, h, pad = { top: 0, bottom: 0 }) {
      this.vw = w; this.vh = h; this.pad = pad;
      const usable = Math.max(120, h - pad.top - pad.bottom);
      this.fitZoom = Math.min(w / this.r.map.w, usable / this.r.map.h) * 0.97;
      this.maxZoom = 2.2;
    }
    clampZoom(z) { return Math.max(this.fitZoom * 0.9, Math.min(this.maxZoom, z)); }
    overviewCenter() { const p = this.pad || { top: 0, bottom: 0 }; return [this.r.map.w / 2, this.r.map.h / 2 - (p.top - p.bottom) / 2 / this.fitZoom]; }
    snapToOverview() { [this.x, this.y] = this.overviewCenter(); this.tx = this.x; this.ty = this.y; this.zoom = this.tz = this.fitZoom; }
    screenToWorld(sx, sy) { return [this.x + (sx - this.vw / 2) / this.zoom, this.y + (sy - this.vh / 2) / this.zoom]; }
    panBy(dx, dy) { this.x -= dx / this.zoom; this.y -= dy / this.zoom; this.tx = this.x; this.ty = this.y; }
    zoomAt(f, sx, sy) {
      const [wx, wy] = this.screenToWorld(sx, sy);
      this.zoom = this.clampZoom(this.zoom * f); this.tz = this.zoom;
      this.x = wx - (sx - this.vw / 2) / this.zoom; this.y = wy - (sy - this.vh / 2) / this.zoom; this.tx = this.x; this.ty = this.y;
    }
    flyTo(x, y, z) { this.tx = x; this.ty = y; if (z) this.tz = this.clampZoom(z); this.flying = true; }
    centerOn(x, y) { this.flyTo(x, y, Math.max(this.zoom, 0.45)); }

    // region heat
    _computeHeat(t, s) {
      const r = this.r, H = this.heat, WX = this.wx, WY = this.wy, cw = this.cellW, gw = this.gw, gh = this.gh;
      H.fill(0); WX.fill(0); WY.fill(0);
      const tau = 2.2 * r.hz, win = 5 * r.hz;
      const add = (x, y, w) => {
        const cx = Math.min(gw - 1, Math.max(0, (x / cw) | 0)), cy = Math.min(gh - 1, Math.max(0, (y / cw) | 0)), i = cy * gw + cx;
        H[i] += w; WX[i] += w * x; WY[i] += w * y;
      };
      const [e0, e1] = r.eventRange(t - win, t + 0.5);
      for (let e = e0; e < e1; e++) {
        const k = r.eK[e], dec = Math.exp(-Math.max(0, t - r.eT[e]) / tau);
        if (k === EVK.kill) add(r.eX[e], r.eY[e], 4 * dec);
        else if (k === EVK.hit) { const v = r.eB[e]; if (s.alive[v]) add(s.x[v], s.y[v], 0.7 * dec); }
        else if (k === EVK.capture) { const c = r.eA[e]; add(r.cpXY[2 * c], r.cpXY[2 * c + 1], 6 * dec); }
      }
      // contact: cells where both armies stand (fronts are worth watching even between kills)
      const B = new Uint16Array(gw * gh), R = new Uint16Array(gw * gh);
      for (let k = 0; k < r.N; k++) {
        if (!s.alive[k]) continue;
        const i = Math.min(gh - 1, (s.y[k] / cw) | 0) * gw + Math.min(gw - 1, (s.x[k] / cw) | 0);
        if (k < r.T) B[i]++; else R[i]++;
      }
      for (let i = 0; i < gw * gh; i++) if (B[i] && R[i]) { const w = 0.35 * Math.min(B[i], R[i]); H[i] += w; WX[i] += w * ((i % gw) + 0.5) * cw; WY[i] += w * (((i / gw) | 0) + 0.5) * cw; }
      // 3x3 box sum = hotspot score; keep the top few for the UI
      const scores = [];
      for (let cy = 0; cy < gh; cy++) for (let cx = 0; cx < gw; cx++) {
        let h = 0, x = 0, y = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const X = cx + dx, Y = cy + dy; if (X < 0 || Y < 0 || X >= gw || Y >= gh) continue;
          const i = Y * gw + X; h += H[i]; x += WX[i]; y += WY[i];
        }
        if (h > 0.5) scores.push({ i: cy * gw + cx, heat: h, x: x / h, y: y / h });
      }
      scores.sort((a, b) => b.heat - a.heat);
      // non-maximum suppression: hotspots at least 2 cells apart
      const picked = [];
      for (const s0 of scores) {
        if (picked.every((p) => Math.hypot(p.x - s0.x, p.y - s0.y) > 2 * cw)) picked.push(s0);
        if (picked.length >= 5) break;
      }
      this.hotspots = picked;
    }
    // endregion

    update(t, s, dt) {
      this.t = t;
      const r = this.r;
      if (this.mode === "overview") { [this.tx, this.ty] = this.overviewCenter(); this.tz = this.fitZoom; }
      else if (this.mode === "follow" && this.follow >= 0 && s.alive[this.follow]) { this.tx = s.x[this.follow]; this.ty = s.y[this.follow]; this.tz = Math.max(this.tz, 0.9); }
      else if (this.mode === "auto") {
        this.sinceHeat += dt; this.heldFor += dt; this.sinceOverview += dt;
        if (this.sinceHeat >= HEAT_EVERY) { this.sinceHeat = 0; this._computeHeat(t, s); }
        if (this.overviewLeft > 0) {
          this.overviewLeft -= dt;
          [this.tx, this.ty] = this.overviewCenter(); this.tz = this.fitZoom;
        } else if (this.sinceOverview > OVERVIEW_EVERY) {
          this.sinceOverview = 0; this.overviewLeft = OVERVIEW_FOR; this.cur = -1;
        } else {
          const best = this.hotspots[0];
          const curSpot = this.hotspots.find((h) => h.i === this.cur) || (this.cur >= 0 ? { heat: 0 } : null);
          const cand = best && (!curSpot || (this.heldFor > HOLD_MIN && best.heat > curSpot.heat * SWITCH_RATIO) || (this.heldFor > HOLD_MAX && best.i !== this.cur));
          if (cand) { this.cur = best.i; this.heldFor = 0; }
          const spot = this.hotspots.find((h) => h.i === this.cur);
          if (spot) {
            this.tx = spot.x; this.ty = spot.y;
            // frame ~2.4 cells: close enough to see agents (MID/NEAR), wide enough to see the fight's shape
            this.tz = this.clampZoom(Math.min(this.vw, this.vh * 1.4) / (this.cellW * 2.4));
          } else if (!best) { [this.tx, this.ty] = this.overviewCenter(); this.tz = this.fitZoom; }
        }
      }
      // motion: exponential approach; a long jump zooms out on the way ("fly"), so cuts read as travel
      const dist = Math.hypot(this.tx - this.x, this.ty - this.y);
      const span = Math.max(this.vw, this.vh) / this.zoom;
      let zt = this.tz;
      if (dist > span * 0.6) zt = Math.min(zt, this.clampZoom(Math.max(this.vw, this.vh) / (dist * 1.8)));
      const kp = 1 - Math.exp(-2.6 * dt), kz = 1 - Math.exp(-1.9 * dt);
      this.x += (this.tx - this.x) * kp; this.y += (this.ty - this.y) * kp;
      this.zoom = Math.exp(Math.log(this.zoom) + (Math.log(zt) - Math.log(this.zoom)) * kz);
      if (this.mode === "free" && dist < 1) this.flying = false;
    }
  }

  window.Arena = window.Arena || {};
  window.Arena.BigDirector = BigDirector;
})();
