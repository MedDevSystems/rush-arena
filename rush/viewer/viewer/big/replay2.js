// region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): ReplayDecodeV2; TECH(8): streams + typed arrays]
// @purpose Stream-decode a schema-2 replay (replay11.BigReplayWriter): gunzip as bytes arrive, parse the
//   JSON header, drop each typed-array chunk into preallocated arrays, and answer "where is everything at
//   decision t" with interpolation between stored (decimated) frames. Playback may start after the first chunk.
// @invariants
// - Mirrors replay11.load_replay_v2 byte for byte: section = u8 type | 3 pad | u32 len | payload; every array
//   inside a payload starts on a 4-byte boundary; frame rows are delta-coded within a chunk (row 0 absolute)
// - Time is the decision index everywhere; loadedDec is the last decision whose frame, bullets and events are in
// - Arrays are allocated once from header counts; nothing is reallocated while streaming
// @links LINKS_TO: replay11.py (format owner), render_big.js, director_big.js, app_big.js
// endregion MODULE_CONTRACT
(function () {
  "use strict";
  const SEC_FRAMES = 1, SEC_BULLETS = 2, SEC_EVENTS = 3;
  const SEC_AMMO = 4, SEC_ATTN = 5;              // exp15: optional per chunk, written before the chunk's events
  const EV = { kill: 1, hit: 2, capture: 3, respawn: 4, dash: 5, end: 6, refill: 7 };
  const NONE16 = 65535;
  const TELEPORT_PX = 220;
  const al4 = (o) => (o + 3) & ~3;

  function lowerBound(arr, n, v) { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } return lo; }

  class Replay2 {
    constructor(header, totalBytes) {
      const h = header;
      this.header = h; this.map = h.map; this.rules = h.rules; this.teams = h.teams; this.timing = h.timing;
      this.result = h.result; this.q = h.quant; this.k = h.decimate;
      this.N = h.n_agents; this.T = h.team_size; this.C = h.n_cps; this.F = h.n_frames; this.nDec = h.n_decisions;
      this.fpd = h.timing.frames_per_decision; this.hz = h.timing.decision_hz; this.fps = h.timing.fps;
      this.totalBytes = totalBytes;
      const F = this.F, N = this.N, C = this.C;
      this.dec = new Uint32Array(F); this.phys = new Uint32Array(F);
      this.score = new Float32Array(F * 2); this.lives = new Uint32Array(F * 2);
      this.X = new Int16Array(F * N); this.Y = new Int16Array(F * N);
      this.ANG = new Uint8Array(F * N); this.HP = new Uint8Array(F * N); this.SH = new Uint8Array(F * N); this.ST = new Uint8Array(F * N);
      this.CPS = new Uint8Array(F * C); this.CPP = new Uint8Array(F * C);
      const M = h.n_bullets, E = h.n_events;
      this.bBirth = new Uint16Array(M); this.bLife = new Uint16Array(M); this.bOwner = new Uint16Array(M);
      this.bX = new Uint16Array(M); this.bY = new Uint16Array(M); this.bVX = new Int8Array(M); this.bVY = new Int8Array(M);
      this.eT = new Uint16Array(E); this.eA = new Uint16Array(E); this.eB = new Uint16Array(E);
      this.eX = new Uint16Array(E); this.eY = new Uint16Array(E); this.eK = new Uint8Array(E); this.eV = new Uint8Array(E);
      this.framesLoaded = 0; this.bLoaded = 0; this.eLoaded = 0; this.loadedDec = -1; this.complete = false;
      this.maxLife = h.max_bullet_life || 60;
      this.cpXY = new Float32Array(C * 2);
      for (let c = 0; c < C; c++) { this.cpXY[2 * c] = this.map.cps[c][0]; this.cpXY[2 * c + 1] = this.map.cps[c][1]; }
      this.cpR = this.map.cp_radius;
      this.listeners = [];
      this.durationSec = Math.max(0, this.nDec - 1) / this.hz;
      // exp12 mixed armies (record_big12): which fork every agent is. Absent in older files -> no classes.
      this.forks = Array.isArray(h.forks) && Array.isArray(h.agent_fork) && h.agent_fork.length === this.N ? h.forks : null;
      this.agentFork = this.forks ? Uint8Array.from(h.agent_fork) : null;
      this.classStats = h.class_stats || null;
      this.agentRadius = Array.isArray(h.agent_radius) ? Float32Array.from(h.agent_radius) : null;
      this.agentMaxHp = Array.isArray(h.agent_max_hp) ? Float32Array.from(h.agent_max_hp) : null;
      // class = the fork's NAME (same silhouette for the same fork on both sides); render_big.CLASS_ORDER maps a
      // name to a silhouette, unknown names get one by their order among the replay's distinct names
      const order = (window.Arena && window.Arena.CLASS_ORDER) || [];
      const names = this.forks ? [...new Set(this.forks.map((f) => f.name))] : [];
      this.forkClass = this.forks ? Uint8Array.from(this.forks.map((f) => {
        const i = order.indexOf(f.name); return i >= 0 ? i : names.indexOf(f.name) % 8;
      })) : null;
      this.agentClass = this.forks ? Uint8Array.from(this.agentFork, (e) => this.forkClass[e]) : null;
      // exp15 streams (header.ext): ammo + magazine per frame per agent, attended target per frame per agent
      const ext = h.ext || {};
      this.AM = ext.ammo ? new Uint8Array(F * N) : null; this.CAP = ext.ammo ? new Uint8Array(F * N) : null;
      this.ATT = ext.attn ? new Uint16Array(F * N) : null;
      this.attnCpBase = ext.attn_cp_base || 0xF000;
      this.agentStats = h.agent_stats || null;
    }

    hasAmmo() { return !!this.AM; }
    hasAttn() { return !!this.ATT; }

    hasClasses() { return !!this.forks; }
    forkOf(k) { return this.agentFork ? this.agentFork[k] : -1; }
    classTitle(k) { return this.forks ? this.forks[this.agentFork[k]].title_ru : ""; }
    classOf(k) { return this.agentClass ? this.agentClass[k] : -1; }
    teamOf(k) { return k < this.T ? 0 : 1; }
    agentName(k) { return (k < this.T ? "B" : "R") + ((k % this.T) + 1); }
    clampT(t) { return Math.max(0, Math.min(Math.max(0, this.loadedDec), t)); }
    msAt(t) { return (t * this.fpd * 1000) / this.fps; }

    // region section parsers
    _frames(u8, off, len) {
      const buf = u8.buffer, base = u8.byteOffset + off;
      let o = 0;
      const dv = new DataView(buf, base, len);
      const f0 = dv.getUint32(0, true), n = dv.getUint32(4, true); o = 8;
      const N = this.N, C = this.C;
      const u32 = (cnt) => { const a = new Uint32Array(buf, base + o, cnt); o = al4(o + cnt * 4); return a; };
      const i16 = (cnt) => { const a = new Int16Array(buf, base + o, cnt); o = al4(o + cnt * 2); return a; };
      const u8a = (cnt) => { const a = new Uint8Array(buf, base + o, cnt); o = al4(o + cnt); return a; };
      this.dec.set(u32(n), f0); this.phys.set(u32(n), f0);
      const sc = u32(2 * n); for (let i = 0; i < 2 * n; i++) this.score[f0 * 2 + i] = sc[i] / this.q.score;
      this.lives.set(u32(2 * n), f0 * 2);
      const xd = i16(n * N), yd = i16(n * N);
      const X = this.X, Y = this.Y;
      for (let i = 0; i < n; i++) {
        const dst = (f0 + i) * N, src = i * N;
        if (i === 0) for (let a = 0; a < N; a++) { X[dst + a] = xd[src + a]; Y[dst + a] = yd[src + a]; }
        else for (let a = 0; a < N; a++) { X[dst + a] = X[dst - N + a] + xd[src + a]; Y[dst + a] = Y[dst - N + a] + yd[src + a]; }
      }
      this.ANG.set(u8a(n * N), f0 * N); this.HP.set(u8a(n * N), f0 * N); this.SH.set(u8a(n * N), f0 * N); this.ST.set(u8a(n * N), f0 * N);
      this.CPS.set(u8a(n * C), f0 * C); this.CPP.set(u8a(n * C), f0 * C);
      this.framesLoaded = f0 + n;
    }
    _bullets(u8, off, len) {
      const buf = u8.buffer, base = u8.byteOffset + off;
      const m = new DataView(buf, base, len).getUint32(0, true);
      let o = 4;
      const u16 = () => { const a = new Uint16Array(buf, base + o, m); o = al4(o + m * 2); return a; };
      const i8 = () => { const a = new Int8Array(buf, base + o, m); o = al4(o + m); return a; };
      const at = this.bLoaded;
      this.bBirth.set(u16(), at); this.bLife.set(u16(), at); this.bOwner.set(u16(), at); this.bX.set(u16(), at); this.bY.set(u16(), at);
      this.bVX.set(i8(), at); this.bVY.set(i8(), at);
      this.bLoaded += m;
    }
    _ammo(u8, off, len) {
      const buf = u8.buffer, base = u8.byteOffset + off, dv = new DataView(buf, base, len);
      const f0 = dv.getUint32(0, true), n = dv.getUint32(4, true), N = this.N;
      this.AM.set(new Uint8Array(buf, base + 8, n * N), f0 * N);
      this.CAP.set(new Uint8Array(buf, base + al4(8 + n * N), n * N), f0 * N);
    }
    _attn(u8, off, len) {
      const buf = u8.buffer, base = u8.byteOffset + off, dv = new DataView(buf, base, len);
      const f0 = dv.getUint32(0, true), n = dv.getUint32(4, true), N = this.N;
      // copy through a byte view: the u16 payload may sit on an odd offset of the shared body buffer
      const bytes = new Uint8Array(buf, base + 8, n * N * 2), dst = new Uint8Array(this.ATT.buffer, f0 * N * 2, n * N * 2);
      dst.set(bytes);
    }
    _events(u8, off, len) {
      const buf = u8.buffer, base = u8.byteOffset + off;
      const m = new DataView(buf, base, len).getUint32(0, true);
      let o = 4;
      const u16 = () => { const a = new Uint16Array(buf, base + o, m); o = al4(o + m * 2); return a; };
      const u8a = () => { const a = new Uint8Array(buf, base + o, m); o = al4(o + m); return a; };
      const at = this.eLoaded;
      this.eT.set(u16(), at); this.eA.set(u16(), at); this.eB.set(u16(), at); this.eX.set(u16(), at); this.eY.set(u16(), at);
      this.eK.set(u8a(), at); this.eV.set(u8a(), at);
      this.eLoaded += m;
      // a chunk is complete once its events are in (sections come frames -> bullets -> events)
      const lastFrame = this.framesLoaded - 1;
      this.loadedDec = this.framesLoaded >= this.F ? this.nDec - 1 : this.dec[lastFrame];
      for (const fn of this.listeners) fn(this);
    }
    // endregion

    frameIndex(t) {
      const F = this.framesLoaded;
      if (F <= 0) return 0;
      let i = Math.min(F - 1, Math.max(0, Math.floor(t / this.k)));
      while (i + 1 < F && this.dec[i + 1] <= t) i++;
      while (i > 0 && this.dec[i] > t) i--;
      return i;
    }

    // Fill preallocated state arrays for fractional decision t. out = {x,y,a,hp,sh (Float32), alive, dash (Uint8)}
    agentsInto(t, out) {
      t = this.clampT(t);
      const i = this.frameIndex(t), j = Math.min(this.framesLoaded - 1, i + 1);
      const span = this.dec[j] - this.dec[i];
      const f = span > 0 ? Math.min(1, Math.max(0, (t - this.dec[i]) / span)) : 0;
      const N = this.N, q = this.q.pos, bi = i * N, bj = j * N;
      const X = this.X, Y = this.Y, A = this.ANG, ST = this.ST, HP = this.HP, SH = this.SH;
      const tp = TELEPORT_PX * q * Math.max(1, span / 2);
      for (let k = 0; k < N; k++) {
        const pa = ST[bi + k] & 1, qa = ST[bj + k] & 1;
        let x = X[bi + k], y = Y[bi + k], ang = A[bi + k] * (Math.PI / 128);
        let alive = pa;
        const dx = X[bj + k] - x, dy = Y[bj + k] - y;
        if (pa && qa && dx * dx + dy * dy < tp * tp) {
          x += dx * f; y += dy * f;
          let da = A[bj + k] * (Math.PI / 128) - ang;
          if (da > Math.PI) da -= 2 * Math.PI; else if (da < -Math.PI) da += 2 * Math.PI;
          ang += da * f;
        } else if (!pa && qa && f > 0.5) { alive = 1; x = X[bj + k]; y = Y[bj + k]; ang = A[bj + k] * (Math.PI / 128); }
        out.x[k] = x / q; out.y[k] = y / q; out.a[k] = ang; out.alive[k] = alive;
        out.hp[k] = HP[bi + k] + (HP[bj + k] - HP[bi + k]) * (pa && qa ? f : 0);
        out.sh[k] = SH[bi + k] + (SH[bj + k] - SH[bi + k]) * (pa && qa ? f : 0);
        out.dash[k] = (ST[bi + k] & 4) ? 1 : 0;
      }
      if (this.AM && out.am) { out.am.set(this.AM.subarray(bi, bi + N)); out.cap.set(this.CAP.subarray(bi, bi + N)); }
      if (this.ATT && out.att) out.att.set(this.ATT.subarray(bi, bi + N));
      return out;
    }

    // Visit bullets alive at t (visible for birth - 0.5 <= t < birth + life - 0.5) inside the world rect.
    forBullets(t, x0, y0, x1, y1, fn) {
      const lo = lowerBound(this.bBirth, this.bLoaded, t - this.maxLife - 1);
      const hi = lowerBound(this.bBirth, this.bLoaded, t + 0.5 + 1e-6);
      const q = this.q.pos, vs = this.fpd / this.q.vel;
      let n = 0;
      for (let b = lo; b < hi; b++) {
        const age = t - this.bBirth[b];
        if (age < -0.5 || age >= this.bLife[b] - 0.5) continue;
        const vx = this.bVX[b] / this.q.vel, vy = this.bVY[b] / this.q.vel;
        const x = this.bX[b] / q + this.bVX[b] * vs * age, y = this.bY[b] / q + this.bVY[b] * vs * age;
        if (x < x0 || x > x1 || y < y0 || y > y1) continue;
        const o = this.bOwner[b];
        fn(x, y, vx, vy, o === NONE16 ? 0 : this.teamOf(o), o);
        n++;
      }
      return n;
    }

    cpState(t, c) {
      const i = this.frameIndex(this.clampT(t)), v = this.CPS[i * this.C + c];
      return { owner: v & 3, cap: (v >> 2) & 3, prog: this.CPP[i * this.C + c] / 100 };
    }

    scoreAt(t) {
      t = this.clampT(t);
      const i = this.frameIndex(t), j = Math.min(this.framesLoaded - 1, i + 1);
      const span = this.dec[j] - this.dec[i], f = span > 0 ? (t - this.dec[i]) / span : 0;
      return [this.score[2 * i] + (this.score[2 * j] - this.score[2 * i]) * f, this.score[2 * i + 1] + (this.score[2 * j + 1] - this.score[2 * i + 1]) * f];
    }
    livesAt(t) { const i = this.frameIndex(this.clampT(t)); return [this.lives[2 * i], this.lives[2 * i + 1]]; }

    eventRange(t0, t1) { return [lowerBound(this.eT, this.eLoaded, t0), lowerBound(this.eT, this.eLoaded, t1)]; }

    sectorAt(x, y) {
      const s = this.map.sectors || [];
      for (const z of s) if (x >= z.x0 && x < z.x1 && y >= z.y0 && y < z.y1) return z.name;
      return "";
    }

    summary() {
      const counts = {};
      const names = ["", "kill", "hit", "capture", "respawn", "dash", "end", "refill"];
      for (let e = 0; e < this.eLoaded; e++) { const k = names[this.eK[e]]; counts[k] = (counts[k] || 0) + 1; }
      return { agents: this.N, teamSize: this.T, cps: this.C, frames: this.framesLoaded, decisions: this.nDec, decimate: this.k,
               bullets: this.bLoaded, events: this.eLoaded, counts, map: `${this.map.name} ${this.map.w}x${this.map.h}`,
               wallRuns: this.map.walls.length, bytes: this.totalBytes, result: this.result, teams: this.teams.map((x) => x.label) };
    }
  }

  // region loader
  // Resolves with the Replay2 as soon as the first chunk is decoded; the rest keeps streaming in.
  // opts.onProgress(bytesDecoded, totalBytes), opts.onDone(replay, ms)
  async function loadReplay2(source, opts = {}) {
    const t0 = performance.now();
    let stream, compressedBytes = 0;
    if (source instanceof Blob) { stream = source.stream(); compressedBytes = source.size; }
    else {
      const resp = await fetch(source);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${source}`);
      compressedBytes = +resp.headers.get("content-length") || 0;
      stream = resp.body;
    }
    const reader = stream.pipeThrough(new DecompressionStream("gzip")).getReader();
    let head = new Uint8Array(0);      // bytes before the body buffer exists
    let body = null, filled = 0, parsed = 0, replay = null, total = 0, bodyStart = 0;
    let resolveFirst, rejectFirst;
    const first = new Promise((res, rej) => { resolveFirst = res; rejectFirst = rej; });
    let firstDone = false;

    function append(chunk) {
      if (!body) {
        const merged = new Uint8Array(head.length + chunk.length); merged.set(head); merged.set(chunk, head.length); head = merged;
        if (head.length < 8) return;
        const magic = String.fromCharCode(head[0], head[1], head[2], head[3]);
        if (magic !== "ARB2") throw new Error("not an arena-replay schema-2 file (magic " + JSON.stringify(magic) + ")");
        const hlen = new DataView(head.buffer, head.byteOffset + 4, 4).getUint32(0, true);
        if (head.length < 8 + hlen) return;
        const header = JSON.parse(new TextDecoder().decode(head.subarray(8, 8 + hlen)).replace(/\0+$/, ""));
        if (header.format !== "arena-replay" || header.version !== 2) throw new Error("unsupported header " + header.format + " v" + header.version);
        bodyStart = 8 + hlen;
        total = bodyStart + header.sections_bytes;
        body = new Uint8Array(new ArrayBuffer(total));
        body.set(head.subarray(0, Math.min(head.length, total)));
        filled = Math.min(head.length, total);
        parsed = bodyStart;
        replay = new Replay2(header, total);
        replay.compressedBytes = compressedBytes;
        console.log("[IMP:9][Replay2.header][INIT]", JSON.stringify({ agents: header.n_agents, cps: header.n_cps, frames: header.n_frames,
          decisions: header.n_decisions, decimate: header.decimate, chunks: header.n_chunks, bullets: header.n_bullets,
          events: header.n_events, rawBytes: total, gzBytes: compressedBytes, map: `${header.map.name} ${header.map.cols}x${header.map.rows}` }));
        head = null;
      } else {
        const take = Math.min(chunk.length, total - filled);
        body.set(chunk.subarray(0, take), filled); filled += take;
      }
      // parse every complete section
      while (parsed + 8 <= filled) {
        const dv = new DataView(body.buffer, parsed, 8);
        const typ = dv.getUint8(0), len = dv.getUint32(4, true);
        if (parsed + 8 + len > filled) break;
        if (typ === SEC_FRAMES) replay._frames(body, parsed + 8, len);
        else if (typ === SEC_BULLETS) replay._bullets(body, parsed + 8, len);
        else if (typ === SEC_AMMO && replay.AM) replay._ammo(body, parsed + 8, len);
        else if (typ === SEC_ATTN && replay.ATT) replay._attn(body, parsed + 8, len);
        else if (typ === SEC_EVENTS) {
          replay._events(body, parsed + 8, len);
          if (!firstDone) { firstDone = true; replay.firstChunkMs = performance.now() - t0; resolveFirst(replay); }
        }
        parsed += 8 + len;
      }
      if (opts.onProgress) opts.onProgress(filled, total);
    }

    (async () => {
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          append(value);
        }
        if (!replay) throw new Error("empty or truncated replay");
        replay.complete = true;
        replay.loadedDec = replay.nDec - 1;
        replay.loadMs = performance.now() - t0;
        console.log("[IMP:9][Replay2.load][RESULT]", JSON.stringify({ ...replay.summary(), firstChunkMs: Math.round(replay.firstChunkMs), loadMs: Math.round(replay.loadMs) }));
        for (const fn of replay.listeners) fn(replay);
        if (opts.onDone) opts.onDone(replay);
        if (!firstDone) { firstDone = true; resolveFirst(replay); }
      } catch (e) {
        console.error("[IMP:9][Replay2.load][ERROR]", e);
        if (!firstDone) { firstDone = true; rejectFirst(e); }
      }
    })();
    return first;
  }
  // endregion

  // Sniff the first bytes of a (possibly gzipped) file: 2 for schema 2, 1 otherwise.
  async function sniffVersion(source) {
    try {
      let blob;
      if (source instanceof Blob) blob = source.slice(0, 4096);
      else { const r = await fetch(source, { headers: { Range: "bytes=0-4095" } }); blob = await r.blob(); }
      const u8 = new Uint8Array(await blob.arrayBuffer());
      if (u8[0] === 0x41 && u8[1] === 0x52 && u8[2] === 0x42 && u8[3] === 0x32) return 2;
      if (u8[0] !== 0x1f || u8[1] !== 0x8b) return 1;
      const rd = new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
      const { value } = await rd.read().catch(() => ({ value: null }));
      rd.cancel().catch(() => {});
      return value && value[0] === 0x41 && value[1] === 0x52 && value[2] === 0x42 && value[3] === 0x32 ? 2 : 1;
    } catch (_) { return 1; }
  }

  window.Arena = window.Arena || {};
  window.Arena.loadReplay2 = loadReplay2;
  window.Arena.Replay2 = Replay2;
  window.Arena.sniffVersion = sniffVersion;
  window.Arena.EV = EV;
})();
