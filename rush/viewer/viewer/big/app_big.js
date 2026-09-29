// region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): BigBattlePlayer; TECH(7): browser JS]
// @purpose Player shell for schema-2 big-battle replays: streaming load with progress, playback clock bounded
//   by what has loaded, army HUD (alive, CPs held, lives, score race), a kill feed filtered for scale
//   (multi-kills, streaks, streak ends, clashes, CP flips), timeline with notable markers, camera controls,
//   minimap navigation, and a once-per-second stats line for automated checks.
// @invariants
// - Playhead t is a fractional DECISION index in [0, loadedDec]; the clock never runs past loaded data
// - Notables are derived incrementally as event chunks arrive (events arrive in time order)
// - Every load logs [IMP:9] lines whose counts must equal the writer's log (Replay2.header / Replay2.load)
// @links LINKS_TO: replay2.js, render_big.js, director_big.js, design/theme.js
// endregion MODULE_CONTRACT
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const T = window.ARENA_THEME;
  const SPEEDS = [0.5, 1, 2, 4, 8, 16];
  const EVK = { kill: 1, hit: 2, capture: 3, respawn: 4, dash: 5, end: 6 };
  const S = { r: null, t: 0, playing: false, speed: 2, dir: null, ren: null, last: 0, frames: 0, fpsT: 0, fps: 0,
              notes: [], noteProc: 0, killer: null, streak: null, clash: new Map(), feedKey: "", statsOn: false, lastStats: null };

  function fmt(sec) { sec = Math.max(0, sec); return `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`; }
  const TEAM_RU = ["СИНИЕ", "КРАСНЫЕ"];             // the theme's names are English; the show is Russian
  function teamName(k) { return TEAM_RU[k] || T.team[k].name; }
  const REASON_RU = { score: "по очкам", elimination: "уничтожение", timeout: "время вышло" };
  function cpName(c) { const n = S.r && S.r.map.cp_names; return n && n[c] ? n[c] : String(c + 1); }   // "Ц3" when the replay carries names
  const LOD_RU = { far: "ДАЛЕКО", mid: "СРЕДНЕ", near: "БЛИЗКО" };
  function killsWord(n) { const a = n % 10, b = n % 100; return a === 1 && b !== 11 ? "убийство" : a >= 2 && a <= 4 && (b < 12 || b > 14) ? "убийства" : "убийств"; }
  function teamCol(k) { return T.team[k].core; }
  const TEAM_GEN = ["синих", "красных"];
  // class silhouette chip in the team's colours (render_big.CLASS_SHAPES — the same shape as on the field)
  function clsIcon(entry, px) { const f = S.r.forks[entry], TT = T.team[f.team]; return window.Arena.classSvg(S.r.forkClass[entry], TT.deep, TT.core, px || 16); }
  // exp12 mixed armies: «Снайпер синих»; older replays: the agent id («B12»)
  function nm(k) {
    const r = S.r, tm = r.teamOf(k);
    if (r.hasClasses()) return `<b style="color:${teamCol(tm)}">${r.classTitle(k)}</b> ${TEAM_GEN[tm]}`;
    return `<b style="color:${teamCol(tm)}">${r.agentName(k)}</b>`;
  }

  // region notables
  function processNotables() {
    const r = S.r, hz = r.hz;
    if (!S.killer) { S.killer = new Map(); S.streak = new Uint16Array(r.N); S.cpHeld = [0, 0]; S.cpOwner = new Uint8Array(r.C); }
    for (let e = S.noteProc; e < r.eLoaded; e++) {
      const k = r.eK[e], t = r.eT[e];
      if (k === EVK.kill) {
        const killer = r.eA[e], victim = r.eB[e], x = r.eX[e], y = r.eY[e];
        if (S.streak[victim] >= 6 && killer !== 65535) S.notes.push({ t, kind: "end", team: r.teamOf(killer), x, y, html: `${nm(killer)} прерывает серию: ${nm(victim)} · ${S.streak[victim]}` });
        S.streak[victim] = 0;
        if (killer !== 65535) {
          const list = (S.killer.get(killer) || []).filter((tt) => t - tt <= 3 * hz); list.push(t); S.killer.set(killer, list);
          if (list.length === 3) S.notes.push({ t, kind: "multi", team: r.teamOf(killer), x, y, html: `${nm(killer)} <span class="tag">ТРОЙНОЕ</span>` });
          if (list.length === 5) S.notes.push({ t, kind: "multi", team: r.teamOf(killer), x, y, html: `${nm(killer)} <span class="tag hot">НЕУДЕРЖИМ</span>` });
          const s = ++S.streak[killer];
          if (s % 5 === 0) S.notes.push({ t, kind: "streak", team: r.teamOf(killer), x, y, html: `${nm(killer)} · ${s} ${killsWord(s)} без смерти` });
        }
        // clash: 8 kills within 3 s inside one 1500 px cell
        const key = ((x / 1500) | 0) + "," + ((y / 1500) | 0);
        const c = S.clash.get(key) || { ts: [], lastNote: -1e9, sides: [0, 0] };
        c.ts = c.ts.filter((tt) => t - tt <= 3 * hz); c.ts.push(t);
        c.sides[r.teamOf(victim)]++;
        if (c.ts.length >= 8 && t - c.lastNote > 10 * hz) {
          c.lastNote = t;
          const where = r.sectorAt(x, y);
          S.notes.push({ t, kind: "clash", team: -1, x, y, html: `<span class="tag clash">СХВАТКА</span> ${where ? where + " · " : ""}${c.ts.length} погибших за 3 с` });
        }
        S.clash.set(key, c);
      } else if (k === EVK.capture) {
        const cp = r.eA[e], team = r.eB[e], prev = r.eV[e];
        const x = r.cpXY[2 * cp], y = r.cpXY[2 * cp + 1];
        if (S.cpOwner[cp]) S.cpHeld[S.cpOwner[cp] - 1]--;
        S.cpOwner[cp] = team + 1; S.cpHeld[team]++;
        if (prev) S.notes.push({ t, kind: "flip", team, x, y, html: `<b style="color:${teamCol(team)}">${teamName(team)}</b> отбивают точку ${cpName(cp)}${r.sectorAt(x, y) ? " · " + r.sectorAt(x, y) : ""}` });
        const held = S.cpHeld[team];
        if (held > 0 && held % 10 === 0 && held > (S.cpMilestone?.[team] || 0)) {
          S.cpMilestone = S.cpMilestone || [0, 0]; S.cpMilestone[team] = held;
          S.notes.push({ t, kind: "hold", team, x, y, html: `<b style="color:${teamCol(team)}">${teamName(team)}</b> держат ${held} из ${r.C} точек` });
        }
      }
    }
    S.noteProc = r.eLoaded;
    S.notes.sort((a, b) => a.t - b.t);
  }
  // endregion

  // region HUD
  function updateHud() {
    const r = S.r, t = S.t, s = S.ren.s;
    const alive = [0, 0];
    for (let k = 0; k < r.N; k++) if (s.alive[k]) alive[k < r.T ? 0 : 1]++;
    const held = [0, 0];
    for (let c = 0; c < r.C; c++) { const o = r.cpState(t, c).owner; if (o) held[o - 1]++; }
    const sc = r.scoreAt(t), lv = r.livesAt(t), stw = r.rules.score_to_win;
    for (const k of [0, 1]) {
      const el = $("army" + k);
      el.querySelector(".alive").textContent = `${alive[k]}/${r.T}`;
      el.querySelector(".cps").textContent = `${held[k]}/${r.C}`;
      el.querySelector(".lives").textContent = r.rules.team_lives ? `${lv[k]}` : "—";
      el.querySelector(".score").textContent = Math.floor(sc[k]);
      $("race" + k).style.width = `${Math.min(100, (sc[k] / stw) * 100)}%`;
    }
    if (r.hasClasses()) {                                   // alive per class, in the army legend
      const F = r.forks.length, al = S.clsAlive || (S.clsAlive = new Uint16Array(F));
      al.fill(0);
      for (let k = 0; k < r.N; k++) if (s.alive[k]) al[r.agentFork[k]]++;
      for (let f = 0; f < F; f++) { const el = S.clsEls && S.clsEls[f]; if (el) el.textContent = al[f]; }
    }
    $("clock").textContent = fmt(r.msAt(t) / 1000);
    $("time").textContent = `${fmt(t / r.hz)} / ${fmt(r.durationSec)}`;
    $("lod").textContent = LOD_RU[S.ren.stats.lod] || "";
    const where = r.sectorAt(S.dir.x, S.dir.y);
    const hs = S.dir.hotspots.length;
    $("where").textContent = S.dir.mode === "auto" ? `${where || "поле боя"}${hs ? ` · очаг: ${hs}` : ""}` : (where || "");
    // feed: notables of the last 8 s of match time
    const win = 8 * r.hz;
    const items = [], maxRows = window.innerWidth < 560 ? 3 : 6;   // phone: the feed must not bury the map
    for (let i = S.notes.length - 1; i >= 0 && items.length < maxRows; i--) { const n = S.notes[i]; if (n.t <= t + 0.5 && n.t > t - win) items.push(n); if (n.t < t - win) break; }
    const key = items.map((n) => n.t + n.kind + n.html.length).join("|");
    if (key !== S.feedKey) {
      S.feedKey = key;
      $("feed").innerHTML = items.map((n) => `<div class="row ${n.kind}">${n.html}</div>`).join("");
    }
    $("play").textContent = S.playing ? "❚❚" : "▶";
  }

  function showBanner() {
    const res = S.r.result || {};
    const b = $("banner"), big = b.querySelector(".big"), small = b.querySelector(".small");
    if (res.winner === 1 || res.winner === 2) { const k = res.winner - 1; big.textContent = `ПОБЕДА: ${teamName(k)}`; big.style.color = teamCol(k); big.style.textShadow = `0 0 24px ${teamCol(k)}`; }
    else { big.textContent = "НИЧЬЯ"; big.style.color = T.color.text; }
    const s = res.score || [0, 0];
    const la = S.r.teams[0].label, lb = S.r.teams[1].label;           // «Все классы против Все классы» says nothing
    small.textContent = `${Math.floor(s[0])} : ${Math.floor(s[1])} · ${REASON_RU[res.reason] || res.reason || ""}${la !== lb ? ` · ${la} против ${lb}` : ""}`;
    b.classList.add("show");
    document.body.classList.add("ended");                  // banner moves up, the class table below it
    if (S.r.classStats || (S.r.forks && S.r.agentStats)) { renderClassPanel(); toggleClasses(true); }
  }

  // Army legend: one chip per class (mark, name, alive now). Built once per replay.
  function buildLegend() {
    const r = S.r;
    document.body.classList.toggle("has-classes", r.hasClasses());
    S.clsEls = [];
    for (const k of [0, 1]) $("army" + k).querySelector(".classes").innerHTML = "";
    if (!r.hasClasses()) return;
    r.forks.forEach((f, i) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.title = `${f.title_ru}${f.blurb_ru ? " — " + f.blurb_ru : ""}`;
      chip.innerHTML = `${clsIcon(i, 16)}<span class="t">${f.title_ru}</span><b>0</b>`;
      $("army" + f.team).querySelector(".classes").appendChild(chip);
      S.clsEls[i] = chip.querySelector("b");
    });
    renderClassPanel();
  }

  // exp15: per-fork stats derived in the player when the recorder wrote agent_stats but no class_stats (synthetic,
  // tourney without the hook): kills/deaths/hits/damage from events (a team kill is the victim's death, no credit),
  // fired/aimed/empty/refills from header.agent_stats. Needs the whole file: built once the load completes.
  function derivedClassStats() {
    const r = S.r;
    if (!r.forks || !r.agentStats || !r.complete) return null;
    const F = r.forks.length, z = () => new Float64Array(F);
    const a = { agents: z(), kills: z(), deaths: z(), hits: z(), damage: z() };
    for (let k = 0; k < r.N; k++) a.agents[r.agentFork[k]]++;
    for (let e = 0; e < r.eLoaded; e++) {
      const kd = r.eK[e], A = r.eA[e], B = r.eB[e];
      if (kd === EVK.kill) {
        if (B < r.N) a.deaths[r.agentFork[B]]++;
        if (A < r.N && B < r.N && r.teamOf(A) !== r.teamOf(B)) a.kills[r.agentFork[A]]++;
      } else if (kd === EVK.hit && A < r.N && B < r.N && r.teamOf(A) !== r.teamOf(B)) {
        a.hits[r.agentFork[A]]++; a.damage[r.agentFork[A]] += r.eV[e] & 127;
      }
    }
    return r.forks.map((f, i) => ({ fork: f.name, title_ru: f.title_ru, team: f.team, agents: a.agents[i],
      kills: a.kills[i], deaths: a.deaths[i], kd: a.kills[i] / Math.max(1, a.deaths[i]),
      damage_per_agent: a.damage[i] / Math.max(1, a.agents[i]), hits: a.hits[i], engage_dist: null }));
  }

  // exp15 columns from header.agent_stats, summed per fork entry
  function ammoStatsByEntry() {
    const r = S.r, st = r.agentStats;
    if (!r.forks || !st) return null;
    const F = r.forks.length, sum = (key) => { const o = new Float64Array(F); if (st[key]) st[key].forEach((v, k) => { o[r.agentFork[k]] += v; }); return o; };
    return { fired: sum("fired"), aimed: sum("aimed"), alive: sum("alive_dec"), empty: sum("empty_dec"), refills: sum("refills"),
             hasAmmo: r.hasAmmo(), hasAimed: (st.aimed || []).some((v) => v > 0) };
  }

  function renderClassPanel() {
    const r = S.r;
    const cs = r.classStats || derivedClassStats();
    if (!cs) { $("classes").innerHTML = ""; return false; }
    const ext = ammoStatsByEntry();
    // both armies side by side: one row per class (fork name), blue columns | red columns
    const pct = (v) => (v == null || !isFinite(v) ? "—" : `${Math.round(v * 100)}%`);
    const num = (v, d = 0) => (v == null || !isFinite(v) ? "—" : (+v).toFixed(d));
    const entryOf = (c) => r.forks ? r.forks.findIndex((f) => f.name === c.fork && f.team === c.team) : -1;
    const byName = new Map();
    cs.forEach((c, i) => { const row = byName.get(c.fork) || [null, null]; row[c.team] = { c, i: entryOf(c) >= 0 ? entryOf(c) : i }; byName.set(c.fork, row); });
    const cols = ext
      ? [["убийств", (e) => num(e.c.kills)], ["смертей", (e) => num(e.c.deaths)], ["У/С", (e) => num(e.c.kd, 2)],
         ["выстрелов на бойца", (e) => num(ext.fired[e.i] / Math.max(1, e.c.agents))],
         ...(ext.hasAimed ? [["прицельных", (e) => pct(ext.aimed[e.i] / Math.max(1, ext.fired[e.i]))]] : []),
         ["точность", (e) => pct(e.c.hits != null ? e.c.hits / Math.max(1, ext.fired[e.i]) : e.c.accuracy)],
         ...(ext.hasAmmo ? [["без патронов", (e) => pct(ext.empty[e.i] / Math.max(1, ext.alive[e.i]))],
                            ["пополнений на бойца", (e) => num(ext.refills[e.i] / Math.max(1, e.c.agents), 1)]] : []),
         ["дистанция, px", (e) => num(e.c.engage_dist)]]
      : [["убийств", (e) => e.c.kills], ["смертей", (e) => e.c.deaths], ["У/С", (e) => e.c.kd.toFixed(2)],
         ["урон на бойца", (e) => Math.round(e.c.damage_per_agent)], ["точность", (e) => pct(e.c.accuracy)],
         ["дистанция, px", (e) => Math.round(e.c.engage_dist)]];
    const W = cols.length;
    const cell = (e, first) => e ? cols.map(([, f], j) => `<td${first && j === 0 ? ' class="sep"' : ""}>${f(e)}</td>`).join("")
                                 : `<td class="na${first ? " sep" : ""}" colspan="${W}">нет в армии</td>`;
    const rows = [...byName.values()].map(([b, rd]) => {
      const e = b || rd, title = e.c.title_ru;
      const icons = (b ? clsIcon(b.i, 18) : "") + (rd ? clsIcon(rd.i, 18) : "");
      return `<tr><td class="cl">${icons} <b>${title}</b></td>${cell(b, false)}${cell(rd, true)}</tr>`;
    }).join("");
    const sub = (first) => cols.map(([h], j) => `<th${first && j === 0 ? ' class="sep"' : ""}>${h}</th>`).join("");
    const notes = ["Дистанция — среднее расстояние от стрелка до цели при попадании."];
    if (ext && ext.hasAimed) notes.push("Прицельный — выстрел, когда пуля первой задела бы врага.");
    if (ext && ext.hasAmmo) notes.push("Без патронов — доля времени жизни с пустым магазином.");
    $("classes").innerHTML = `<h2>КЛАССЫ · ИТОГ МАТЧА</h2><div class="tw"><table><thead>` +
      `<tr><th></th><th colspan="${W}" style="color:${teamCol(0)}">${teamName(0)} · ${r.teams[0].label}</th><th class="sep" colspan="${W}" style="color:${teamCol(1)}">${teamName(1)} · ${r.teams[1].label}</th></tr>` +
      `<tr><th>класс</th>${sub(false)}${sub(true)}</tr></thead><tbody>${rows}</tbody></table></div>` +
      `<div class="note">По ${Math.min(...cs.map((c) => c.agents))}–${Math.max(...cs.map((c) => c.agents))} бойцов в классе. ${notes.join(" ")} Клавиша C — скрыть.</div>`;
    S.panelRows = cs.length;
    return true;
  }
  function toggleAttn(on) {
    if (!S.r || !S.r.hasAttn()) return;
    S.ren.showAttn = on === undefined ? !S.ren.showAttn : on;
    $("attnbtn").classList.toggle("on", S.ren.showAttn);
  }
  function toggleClasses(on) {
    if (!S.r) return;
    if (!S.r.classStats && !$("classes").innerHTML && !renderClassPanel()) return;
    const el = $("classes"), show = on === undefined ? !el.classList.contains("show") : on;
    el.classList.toggle("show", show); $("classbtn").classList.toggle("on", show);
  }
  // endregion

  // region timeline
  function drawTimeline() {
    const c = $("timeline"), r = S.r, dpr = devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); S.tlKey = ""; }
    const key = `${r.loadedDec}|${S.notes.length}|${w}`;
    if (key !== S.tlKey) {
      S.tlKey = key;
      const off = document.createElement("canvas"); off.width = c.width; off.height = c.height;
      const g = off.getContext("2d"); g.scale(dpr, dpr);
      const X = (t) => (t / Math.max(1, r.nDec - 1)) * w;
      g.fillStyle = "rgba(255,255,255,0.04)"; g.fillRect(0, 0, w, h);
      g.fillStyle = "rgba(255,255,255,0.07)"; g.fillRect(0, 0, X(r.loadedDec), h);
      g.beginPath();
      for (let i = 0; i < r.framesLoaded; i += 2) {
        const d = (r.score[2 * i] - r.score[2 * i + 1]) / Math.max(1, r.rules.score_to_win) * 4;
        const y = h / 2 - Math.max(-1, Math.min(1, d)) * (h / 2 - 3);
        if (i === 0) g.moveTo(X(r.dec[i]), y); else g.lineTo(X(r.dec[i]), y);
      }
      g.strokeStyle = "rgba(255,255,255,0.3)"; g.lineWidth = 1; g.stroke();
      for (const n of S.notes) {
        const x = X(n.t);
        if (n.kind === "clash") { g.fillStyle = "rgba(255,255,255,0.8)"; g.fillRect(x - 1, 2, 2, h - 4); }
        else if (n.kind === "flip" || n.kind === "hold") { g.fillStyle = teamCol(n.team); g.fillRect(x - 1, n.team ? h - 7 : 2, 2, 5); }
        else { g.fillStyle = teamCol(n.team); g.fillRect(x - 0.5, n.team ? h / 2 + 2 : h / 2 - 8, 1.5, 6); }
      }
      S.tlBase = off;
    }
    const g = c.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height); g.drawImage(S.tlBase, 0, 0);
    g.scale(dpr, dpr);
    const x = (S.t / Math.max(1, r.nDec - 1)) * w;
    g.fillStyle = T.color.wallEdgeHot; g.fillRect(x - 1, 0, 2, h);
  }
  function seekFrom(ev) {
    const rect = $("timeline").getBoundingClientRect();
    seek((Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width))) * (S.r.nDec - 1));
  }
  function seek(t) { S.t = S.r.clampT(t); $("banner").classList.remove("show"); document.body.classList.remove("ended"); }
  // endregion

  // region loading
  async function open(source, label) {
    $("err").textContent = ""; $("prog").style.width = "0%"; $("loader").classList.add("loading");
    try {
      // URLs: the extension decides (.arena.bin.gz = schema 2, .arena.json.gz = schema 1) — a sniff would fetch
      // the file twice on servers without Range support. Files: sniff the first bytes (cheap slice).
      const ver = typeof source === "string" ? (/\.json(\.gz)?$/.test(source) ? 1 : 2) : await Arena.sniffVersion(source);
      if (ver === 1) {
        if (typeof source === "string") { location.href = "index.html?theme=../design/theme.js&src=" + encodeURIComponent(source.replace(/^\.\.\//, "")); return; }
        throw new Error("schema-1 replay: open it in index.html");
      }
      const r = await Arena.loadReplay2(source, { onProgress: (a, b) => { $("prog").style.width = `${((a / b) * 100).toFixed(1)}%`; } });
      start(r, label);
      r.listeners.push(() => { processNotables(); });
    } catch (e) {
      console.error("[IMP:9][open][ERROR]", e);
      $("err").textContent = String(e.message || e);
      $("loader").classList.remove("hidden", "loading");
    }
  }

  function start(r, label) {
    S.r = r; S.t = 0; S.playing = true;
    S.notes = []; S.noteProc = 0; S.killer = null; S.clash = new Map(); S.tlKey = "";
    S.ren = new Arena.BigRenderer($("view"), r, T);
    S.dir = new Arena.BigDirector(r);
    resize();
    S.dir.snapToOverview();
    processNotables();
    $("loader").classList.add("hidden");
    for (const k of [0, 1]) {
      $("army" + k).querySelector(".label").textContent = r.teams[k].label;
      $("army" + k).querySelector(".name").textContent = teamName(k);
    }
    buildLegend();
    $("classes").classList.remove("show");
    // exp15: attention lines button only when the replay has them; derived class panel once the file is complete
    document.body.classList.toggle("has-attn", r.hasAttn());
    $("attnbtn").classList.toggle("on", S.ren.showAttn);
    S.panelDone = !!r.classStats;
    r.listeners.push(() => { if (r.complete && !S.panelDone && r.forks && r.agentStats) { S.panelDone = true; renderClassPanel(); } });
    document.title = `${r.teams[0].label} vs ${r.teams[1].label} · ${r.map.name} · ${r.T}v${r.T}`;
    setCam("overview");                  // no automatic zoom-in: the show starts on the whole map
    console.log("[IMP:9][BigPlayer.start][RESULT]", JSON.stringify({ source: label, firstChunkMs: Math.round(r.firstChunkMs || 0), loadedDec: r.loadedDec, ...r.summary() }));
  }

  async function listSamples() {
    try {
      const resp = await fetch("samples/big/index.json");
      if (!resp.ok) return;
      for (const it of await resp.json()) {
        const b = document.createElement("button");
        b.textContent = `▶ ${it.title}`;
        b.onclick = () => open(`samples/big/${it.file}`, it.file);
        $("samples").appendChild(b);
      }
    } catch (_) { /* picker only */ }
  }
  // endregion

  function resize() {
    if (!S.r) return;
    const w = window.innerWidth, h = window.innerHeight;
    S.ren.resize(w, h);
    const hud = $("hud").getBoundingClientRect(), ctl = $("controls").getBoundingClientRect();
    S.dir.setView(w, h, { top: Math.ceil(hud.bottom + 6), bottom: Math.ceil(h - ctl.top) });
    const mw = w < 560 ? 116 : Math.min(230, w * 0.22);
    S.ren.buildMini(mw);
  }

  function setCam(mode) {
    S.dir.mode = mode; if (mode !== "follow") S.dir.follow = -1;
    for (const b of document.querySelectorAll("#cam button")) b.classList.toggle("on", b.dataset.cam === mode);
  }
  function setSpeed(v) { S.speed = v; for (const b of document.querySelectorAll("#speed button")) b.classList.toggle("on", parseFloat(b.dataset.v) === v); }

  // region loop
  function frame(now) {
    requestAnimationFrame(frame);
    if (!S.r) return;
    const dt = Math.min(0.1, S.last ? (now - S.last) / 1000 : 0);
    S.last = now;
    const r = S.r;
    if (S.playing) {
      S.t = Math.min(r.loadedDec, S.t + dt * r.hz * S.speed);
      if (r.complete && S.t >= r.nDec - 1) { S.playing = false; showBanner(); }
    }
    S.dir.update(S.t, S.ren.s, dt);
    S.dir.t = S.t;
    S.ren.draw(S.t, S.dir);
    S.ren.drawMini($("minimap"), S.dir);
    drawTimeline();
    updateHud();
    S.frames++;
    if (now - S.fpsT >= 1000) {
      S.fps = (S.frames * 1000) / (now - S.fpsT); S.frames = 0; S.fpsT = now;
      const st = S.ren.stats;
      S.lastStats = { fps: +S.fps.toFixed(1), lod: st.lod, zoom: st.zoom, agents: st.agents, marks: st.marks || 0, bullets: st.bullets, fx: st.fx, cps: st.cps,
                      muzzles: st.lod === "far" ? st.muzzles || 0 : null, sparks: st.lod === "far" ? st.sparks || 0 : null,
                      drawMs: +st.drawMs.toFixed(2), blocksCached: st.blocksCached, t: +S.t.toFixed(1), loadedDec: r.loadedDec, mode: S.dir.mode };
      if (r.hasAmmo() || r.hasAttn()) Object.assign(S.lastStats, { lines: st.lines || 0, ammoBars: st.ammo || 0, refills: st.refills || 0, attn: S.ren.showAttn });
      console.log("[IMP:8][BigPlayer.stats][VALUE]", JSON.stringify(S.lastStats));
      if (S.statsOn) $("stats").textContent = JSON.stringify(S.lastStats);
    }
  }
  // endregion

  // region wiring
  function wire() {
    const st = document.documentElement.style, C = T.color;
    st.setProperty("--bg", C.bg); st.setProperty("--text", C.text); st.setProperty("--dim", C.textDim);
    st.setProperty("--panel", C.panel); st.setProperty("--edge", C.panelEdge); st.setProperty("--t0", T.team[0].core); st.setProperty("--t1", T.team[1].core);
    st.setProperty("--hot", C.wallEdgeHot);
    if (T.fonts && T.fonts.googleFontsHref) { const l = document.createElement("link"); l.rel = "stylesheet"; l.href = T.fonts.googleFontsHref; document.head.appendChild(l); }
    $("pick").onclick = () => $("file").click();
    $("file").onchange = (e) => { const f = e.target.files[0]; if (f) open(f, f.name); e.target.value = ""; };
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) open(f, f.name); });
    $("play").onclick = () => { if (!S.r) return; if (!S.playing && S.t >= S.r.nDec - 1) seek(0); S.playing = !S.playing; };
    for (const v of SPEEDS) { const b = document.createElement("button"); b.textContent = `${v}x`; b.dataset.v = v; b.onclick = () => setSpeed(v); $("speed").appendChild(b); }
    setSpeed(2);
    for (const b of document.querySelectorAll("#cam button")) b.onclick = () => setCam(b.dataset.cam);
    $("classbtn").onclick = () => toggleClasses();
    $("attnbtn").onclick = () => toggleAttn();
    const tl = $("timeline"); let tlDrag = false;
    tl.addEventListener("pointerdown", (e) => { if (!S.r) return; tlDrag = true; tl.setPointerCapture(e.pointerId); seekFrom(e); });
    tl.addEventListener("pointermove", (e) => { if (tlDrag) seekFrom(e); });
    tl.addEventListener("pointerup", () => { tlDrag = false; });
    const view = $("view"); let drag = null;
    view.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; view.setPointerCapture(e.pointerId); });
    view.addEventListener("pointermove", (e) => {
      if (!S.r || !drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (drag.moved || Math.hypot(dx, dy) > 4) { drag.moved = true; S.dir.panBy(dx, dy); setCam("free"); drag.x = e.clientX; drag.y = e.clientY; }
    });
    view.addEventListener("pointerup", (e) => {
      if (drag && !drag.moved && S.r && S.dir.zoom > 0.5) {
        const [wx, wy] = S.dir.screenToWorld(e.clientX, e.clientY), s = S.ren.s;
        let best = -1, bd = 30 / S.dir.zoom;
        for (let k = 0; k < S.r.N; k++) if (s.alive[k]) { const d = Math.hypot(s.x[k] - wx, s.y[k] - wy); if (d < bd) { bd = d; best = k; } }
        if (best >= 0) { S.dir.follow = best; setCam("follow"); }
      }
      drag = null;
    });
    view.addEventListener("dblclick", () => S.r && setCam("overview"));
    view.addEventListener("wheel", (e) => { if (!S.r) return; e.preventDefault(); S.dir.zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY); if (S.dir.mode !== "follow") setCam("free"); }, { passive: false });
    $("minimap").addEventListener("pointerdown", (e) => {
      if (!S.r) return;
      const rect = e.target.getBoundingClientRect(), k = S.ren.miniK;
      setCam("free"); S.dir.centerOn((e.clientX - rect.left) / k, (e.clientY - rect.top) / k);
    });
    window.addEventListener("resize", resize);
    window.addEventListener("keydown", (e) => {
      if (!S.r) return;
      if (e.code === "Space") { e.preventDefault(); $("play").click(); }
      else if (e.key === "ArrowRight") seek(S.t + 10 * S.r.hz);
      else if (e.key === "ArrowLeft") seek(S.t - 10 * S.r.hz);
      else if (e.key === "a" || e.key === "A") setCam("auto");
      else if (e.key === "o" || e.key === "O") setCam("overview");
      else if (e.key === "f" || e.key === "F") setCam("free");
      else if (e.key === "s" || e.key === "S") { S.statsOn = !S.statsOn; $("stats").classList.toggle("show", S.statsOn); }
      else if (e.key === "c" || e.key === "C") toggleClasses();
      else if (e.key === "l" || e.key === "L" || e.code === "KeyL") toggleAttn();
      else if (e.key === "]") setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(S.speed) + 1)]);
      else if (e.key === "[") setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(S.speed) - 1)]);
    });
    const params = new URLSearchParams(location.search);
    if (params.get("speed")) setSpeed(parseFloat(params.get("speed")));
    const src = params.get("src");
    if (src) open(src, src); else listSamples();
    requestAnimationFrame(frame);
  }

  // test hook: deterministic control for automated checks
  window.BigPlayer = {
    state: S,
    seek: (t) => seek(t), pause: () => { S.playing = false; }, play: () => { S.playing = true; },
    camera: (m) => setCam(m),
    view: (x, y, z) => { setCam("free"); const d = S.dir; d.x = d.tx = x; d.y = d.ty = y; d.zoom = d.tz = d.clampZoom(z); },
    stats: () => S.lastStats,
    ready: () => !!(S.r && S.r.complete),
  };
  // endregion

  wire();
})();
