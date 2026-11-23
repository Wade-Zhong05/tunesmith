// tunesmith web UI (no build step).
// AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble mode 2026-11-22;
// redesigned 2026-11-23. See ATTRIBUTION.md.

const EXAMPLES = {
  melody: [
    "a gentle piano waltz in A minor, slow and a little sad",
    "an epic brass theme at 140 bpm",
    "a playful flute tune in D major",
    "a dreamy synth melody, ambient, very slow",
  ],
  ensemble: [
    "an upbeat pop song with piano, electric guitar and bass, with drums",
    "a calm piece for strings, harp and synth pad, no drums",
    "a jazz groove with electric piano, sax and bass, with drums",
    "a gentle folk tune for acoustic guitar and flute",
  ],
};
const OVERRIDES = {
  melody: ["key", "meter", "tempo_bpm", "instrument", "mood", "genre", "bars"],
  ensemble: ["key", "meter", "tempo_bpm", "mood", "genre"],
};
const UNDERSTOOD = {
  melody: ["key", "meter", "tempo_bpm", "instrument", "mood", "genre", "bars"],
  ensemble: ["key", "meter", "tempo_bpm", "instruments", "drums", "mood", "genre"],
};
const LABELS = { key: "Key", meter: "Meter", tempo_bpm: "Tempo", instrument: "Instrument", mood: "Mood", genre: "Style",
                 bars: "Length", instruments: "Instruments", drums: "Drums" };
// part colours on white: orange first, then ink and calm companions
const PALETTE = ["#ff6b1a", "#1d1d1b", "#3f7cc0", "#f2a65a", "#5b8c5a", "#9b6ab8"];
const DRUM_COLOR = "#a3a29b";
const ROLL = { grid: "#f1f0ec", bar: "#e2e1db", shade: "#fafaf8", white: "#ffffff", whiteEdge: "#e6e5e0",
               black: "#2b2b29", head: "#ff6b1a" };
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const pretty = (v) => String(v).replace(/_/g, " ");
const mode = () => document.querySelector("input[name=mode]:checked").value;

const state = {
  maxInstruments: 4, instruments: [],
  pieces: [], current: 0, mode: "melody", hidden: new Set(),
  preview: false, previewStart: 0, raf: null,
};
const audio = $("#audio");
const canvas = $("#roll");

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `the server answered ${res.status}`);
  return data;
}

// ---------------------------------------------------------------- animated title

function buildTitle() {
  const box = $(".title .letters");
  const word = "tunesmith";
  [...word].forEach((ch) => box.appendChild(el("span", null, ch)));
  requestAnimationFrame(() => {
    const title = $(".title");
    const width = box.getBoundingClientRect().width;
    title.style.setProperty("--w", `${width + 12}px`);
    const spans = [...box.children];
    const left = box.getBoundingClientRect().left;
    const sweep = 1700;
    spans.forEach((s) => {
      const r = s.getBoundingClientRect();
      const frac = (r.left + r.width * 0.45 - left) / width;
      s.style.setProperty("--d", `${Math.round(frac * sweep * 0.94)}ms`);
    });
    title.style.setProperty("--sweep", `${sweep}ms`);
    if (!reduceMotion) setTimeout(() => title.classList.add("run"), 250);
  });
}

function titleLoop(on) {
  const t = $(".title");
  t.classList.remove("run");
  t.classList.toggle("loop", on && !reduceMotion);
}

// ---------------------------------------------------------------- controls

function buildControls(opts) {
  const box = $("#overrides");
  const lists = { ...opts, tempo_bpm: opts.tempo };
  for (const f of OVERRIDES.melody) {
    const sel = document.createElement("select");
    sel.id = `ov-${f}`;
    sel.innerHTML = `<option value="">Auto</option>` + (lists[f] || []).map((o) =>
      typeof o === "object" ? `<option value="${o.bpm}">${pretty(o.name)} (${o.bpm} bpm)</option>` : `<option value="${o}">${pretty(o)}</option>`
    ).join("");
    const label = el("label", `ov ov-${f}`, LABELS[f]);
    label.appendChild(sel);
    box.appendChild(label);
  }
  state.maxInstruments = opts.max_instruments || 4;
  const chips = $("#instrument-chips");
  for (const inst of opts.instrument || []) {
    const b = el("button", "inst", pretty(inst));
    b.type = "button"; b.dataset.value = inst; b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => toggleInstrument(inst));
    chips.appendChild(b);
  }
}

function toggleInstrument(inst) {
  const i = state.instruments.indexOf(inst);
  if (i >= 0) state.instruments.splice(i, 1);
  else if (state.instruments.length < state.maxInstruments) state.instruments.push(inst);
  for (const b of document.querySelectorAll(".inst")) {
    const k = state.instruments.indexOf(b.dataset.value);
    b.classList.toggle("on", k >= 0);
    b.setAttribute("aria-pressed", String(k >= 0));
    b.dataset.order = k >= 0 ? String(k + 1) : "";
    b.style.setProperty("--part", k >= 0 ? PALETTE[k % PALETTE.length] : "");
  }
}

function readOverrides() {
  const out = {};
  for (const f of OVERRIDES[mode()]) {
    const v = $(`#ov-${f}`).value;
    if (v !== "") out[f] = (f === "tempo_bpm" || f === "bars") ? Number(v) : v;
  }
  if (mode() === "ensemble") {
    if (state.instruments.length) out.instruments = [...state.instruments];
    const d = document.querySelector("input[name=drums]:checked").value;
    if (d !== "") out.drums = d === "yes";
  }
  return out;
}

function showTries() {
  const p = $("#tries");
  p.textContent = "Try ";
  EXAMPLES[mode()].forEach((t, i, all) => {
    const b = el("a", "try", t);
    b.href = "#text";
    b.addEventListener("click", (e) => { e.preventDefault(); $("#text").value = t; $("#text").focus(); });
    p.appendChild(b);
    p.append(i < all.length - 2 ? ", " : i === all.length - 2 ? " or " : ".");
  });
}

function moveModeLine() {
  const label = document.querySelector("input[name=mode]:checked").closest(".mode");
  const line = $(".mode-line");
  line.style.width = `${label.offsetWidth}px`;
  line.style.transform = `translateX(${label.offsetLeft}px)`;
}

function applyMode() {
  const ens = mode() === "ensemble";
  $("#ensemble-controls").hidden = !ens;
  for (const f of ["instrument", "bars"]) {
    const label = document.querySelector(`.ov-${f}`);
    if (label) label.hidden = ens;
  }
  $("#text").placeholder = EXAMPLES[mode()][0];
  showTries();
  moveModeLine();
}

// ---------------------------------------------------------------- colours and text

function assignColors(roll, requested) {
  const order = [...(requested || [])];
  for (const t of roll.tracks || []) if (!t.drum && !order.includes(t.name)) order.push(t.name);
  roll.colors = (roll.tracks || []).map((t) => t.drum ? DRUM_COLOR : PALETTE[Math.max(0, order.indexOf(t.name)) % PALETTE.length]);
}

function describe(piece, m) {
  const mt = piece.metrics;
  let s = `${mt.n_notes} notes over ${mt.bars} bars in ${mt.meter || "free time"}`;
  if (mt.detected_key) s += `, sounding like ${mt.detected_key}`;
  if (m === "ensemble" && mt.polyphony != null) s += `, with about ${mt.polyphony.toFixed(1)} notes sounding at once`;
  else if (mt.in_scale != null) s += `; ${Math.round(mt.in_scale * 100)}% of the notes are in the requested key`;
  return s + ".";
}

function showUnderstood(attrs, overrides, m) {
  const box = $("#understood");
  box.innerHTML = "";
  for (const f of UNDERSTOOD[m]) {
    let v = attrs ? attrs[f] : null;
    if (f === "instruments" && Array.isArray(v)) v = v.map(pretty).join(", ");
    if (f === "drums" && v != null) v = v ? "yes" : "no";
    if (f === "tempo_bpm" && v != null) v = `${v} bpm`;
    if (f === "bars" && v != null) v = `${v} bars`;
    const row = el("div");
    row.appendChild(el("dt", null, LABELS[f]));
    const dd = el("dd", v == null ? "auto" : (f in overrides ? "yours" : null), v == null ? "any" : pretty(v));
    if (f in overrides) dd.title = "Set by you";
    row.appendChild(dd);
    box.appendChild(row);
  }
}

// ---------------------------------------------------------------- piano roll

const BLACK = new Set([1, 3, 6, 8, 10]);
function drumRow(p) { if (p === 35 || p === 36) return 2; if ([37, 38, 39, 40, 41, 43, 45, 47, 48, 50].includes(p)) return 1; return 0; }

function draw(beat = null, reveal = 1) {
  const piece = state.pieces[state.current];
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!piece) return;
  const roll = piece.notes;
  const pitched = roll.notes.filter((n) => !n.drum), drums = roll.notes.filter((n) => n.drum);
  const lane = drums.length ? 34 : 0;
  const ps = pitched.length ? pitched.map((n) => n.pitch) : [60, 72];
  // fit the rows to the notes, but show at least an octave and a half
  let lo = Math.min(...ps) - 2, hi = Math.max(...ps) + 2;
  const pad = Math.max(0, 18 - (hi - lo));
  lo -= Math.floor(pad / 2); hi += Math.ceil(pad / 2);
  const total = Math.max(roll.total_beats, roll.beats_per_bar * Math.max(1, roll.bars));
  const kb = 30, left = kb + 6, right = w - 2;
  const x = (b) => left + (b / total) * (right - left);
  const top = 2, bottom = h - 4 - lane;
  const rowH = (bottom - top) / (hi - lo + 1);
  const y = (p) => top + (hi - p) * rowH;
  // keyboard and pitch rows
  for (let p = lo; p <= hi; p++) {
    const black = BLACK.has(((p % 12) + 12) % 12);
    if (black) { ctx.fillStyle = ROLL.shade; ctx.fillRect(left, y(p), right - left, rowH); }
    ctx.fillStyle = black ? ROLL.black : ROLL.white;
    ctx.fillRect(0, y(p), black ? kb * 0.6 : kb, Math.max(1, rowH - 0.5));
    if (!black) { ctx.fillStyle = ROLL.whiteEdge; ctx.fillRect(0, y(p) + rowH - 0.5, kb, 0.5); }
  }
  ctx.strokeStyle = ROLL.whiteEdge; ctx.lineWidth = 1;
  ctx.strokeRect(0.5, top + 0.5, kb - 1, bottom - top - 1);
  // beat and bar lines
  for (let b = 0; b <= total + 1e-6; b += 1) {
    const isBar = Math.abs(b / roll.beats_per_bar - Math.round(b / roll.beats_per_bar)) < 1e-6;
    ctx.fillStyle = isBar ? ROLL.bar : ROLL.grid;
    ctx.fillRect(Math.round(x(b)), top, 1, h - top - 2);
  }
  if (lane) { ctx.fillStyle = ROLL.shade; ctx.fillRect(left, h - lane, right - left, lane - 2); }
  const limit = total * reveal;
  // notes: a note lights up while the playhead is on it and fades out afterwards
  for (const n of pitched) {
    if (state.hidden.has(n.track) || n.start > limit) continue;
    const nx = x(n.start), nw = Math.max(3, x(Math.min(n.start + n.dur, limit)) - nx - 1);
    let glow = 0;
    if (beat != null) {
      if (beat >= n.start && beat < n.start + n.dur) glow = 1;
      else if (beat >= n.start + n.dur && beat < n.start + n.dur + 0.6) glow = 1 - (beat - n.start - n.dur) / 0.6;
    }
    ctx.globalAlpha = beat != null && glow === 0 ? 0.72 : 1;
    ctx.fillStyle = roll.colors ? roll.colors[n.track] : PALETTE[0];
    const grow = glow * 1.5;
    roundRect(ctx, nx, y(n.pitch) + 0.5 - grow, nw, Math.max(3, rowH - 1) + grow * 2, 2.5);
    if (glow > 0) {
      ctx.globalAlpha = glow * 0.35; ctx.fillStyle = ROLL.head;
      roundRect(ctx, nx - 2, y(n.pitch) - 2 - grow, nw + 4, Math.max(3, rowH - 1) + 4 + grow * 2, 4);
    }
  }
  for (const n of drums) {
    if (state.hidden.has(n.track) || n.start > limit) continue;
    const on = beat != null && beat >= n.start && beat < n.start + 0.25;
    ctx.globalAlpha = on ? 1 : 0.8;
    ctx.fillStyle = on ? ROLL.head : DRUM_COLOR;
    roundRect(ctx, x(n.start), h - lane + 3 + drumRow(n.pitch) * 10, on ? 6 : 4, 6, 2);
  }
  ctx.globalAlpha = 1;
  if (beat != null) {
    const px = x(Math.min(beat, total));
    ctx.fillStyle = ROLL.head;
    ctx.fillRect(px - 1, 0, 2, h);
    ctx.beginPath(); ctx.moveTo(px - 6, 0); ctx.lineTo(px + 6, 0); ctx.lineTo(px, 8); ctx.closePath(); ctx.fill();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); ctx.fill();
}

// ---------------------------------------------------------------- player

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function pieceSeconds() {
  const p = state.pieces[state.current];
  if (!p) return 0;
  const r = p.notes;
  return Math.max(r.total_beats, r.beats_per_bar * Math.max(1, r.bars)) * 60 / r.qpm;
}

function setProgress(sec) {
  const total = pieceSeconds() || 1;
  const frac = Math.min(1, Math.max(0, sec / total));
  $(".scrub-fill").style.width = `${frac * 100}%`;
  $(".scrub-knob").style.left = `${frac * 100}%`;
  $("#scrub").setAttribute("aria-valuenow", String(Math.round(frac * 100)));
  $("#time").textContent = fmt(sec);
}

function frame() {
  const p = state.pieces[state.current];
  if (!p) return;
  const qpm = p.notes.qpm;
  let sec = null;
  if (!audio.paused) sec = audio.currentTime;
  else if (state.preview) sec = ((performance.now() - state.previewStart) / 1000) % (pieceSeconds() + 1.5);
  if (sec != null) {
    draw(sec * qpm / 60);
    setProgress(sec);
    state.raf = requestAnimationFrame(frame);
  } else {
    state.raf = null;
  }
}
function startLoop() { if (!state.raf) state.raf = requestAnimationFrame(frame); }

function revealAnimation() {
  if (reduceMotion) { draw(); return; }
  const t0 = performance.now();
  const step = () => {
    const k = Math.min(1, (performance.now() - t0) / 700);
    draw(null, 1 - Math.pow(1 - k, 3));
    if (k < 1) requestAnimationFrame(step); else startLoop();
  };
  requestAnimationFrame(step);
}

function selectPiece(i) {
  audio.pause();
  state.current = i;
  state.hidden.clear();
  const piece = state.pieces[i];
  for (const [k, t] of [...document.querySelectorAll(".tab")].entries()) t.setAttribute("aria-selected", String(k === i));
  audio.src = piece.wav_url || "";
  $("#play").disabled = !piece.wav_url;
  const dl = $("#download");
  dl.href = piece.midi_url;
  dl.setAttribute("download", `tunesmith-${state.mode}-${i + 1}.mid`);
  $("#dur").textContent = fmt(pieceSeconds());
  $("#summary").textContent = describe(piece, state.mode);
  const legend = $("#legend");
  legend.innerHTML = "";
  const tracks = piece.notes.tracks || [];
  if (tracks.length > 1) {
    tracks.forEach((t, k) => {
      const b = el("button");
      b.type = "button"; b.setAttribute("aria-pressed", "true");
      b.title = `Show or hide the ${pretty(t.name)} part`;
      const sw = el("i"); sw.style.background = piece.notes.colors[k];
      b.append(sw, pretty(t.name));
      b.addEventListener("click", () => {
        if (state.hidden.has(k)) state.hidden.delete(k); else state.hidden.add(k);
        b.classList.toggle("off", state.hidden.has(k));
        b.setAttribute("aria-pressed", String(!state.hidden.has(k)));
        if (!state.raf) draw();
      });
      legend.appendChild(b);
    });
  }
  setProgress(0);
  revealAnimation();
}

function showResult(res, overrides, { demo = false } = {}) {
  state.mode = res.mode;
  state.pieces = res.pieces;
  state.preview = demo && !reduceMotion;
  state.previewStart = performance.now();
  for (const p of state.pieces) assignColors(p.notes, res.mode === "ensemble" ? res.attributes.instruments : null);
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  if (demo) {
    tabs.appendChild(Object.assign(el("button", "tab", "Example"), { type: "button" }));
  } else {
    res.pieces.forEach((_, i) => {
      const t = el("button", "tab", res.pieces.length > 1 ? `Version ${i + 1}` : "Your piece");
      t.type = "button"; t.setAttribute("role", "tab");
      t.addEventListener("click", () => selectPiece(i));
      tabs.appendChild(t);
    });
  }
  $("#caption").textContent = demo
    ? `An example: what the band model wrote for "${res.text}". Press play to hear it.`
    : `${res.mode === "ensemble" ? "Band" : "Melody"} for "${res.text || "your settings"}", composed in ${res.timings.total_s.toFixed(1)} s (seed ${res.seed}).`;
  showUnderstood(res.attributes, overrides, res.mode);
  selectPiece(0);
}

function togglePlay() {
  if (!audio.src) return;
  state.preview = false;
  if (audio.paused) audio.play(); else audio.pause();
}

function seekTo(clientX) {
  const r = $("#scrub").getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
  state.preview = false;
  const sec = frac * pieceSeconds();
  if (audio.src) audio.currentTime = Math.min(sec, audio.duration || sec);
  setProgress(sec);
  const p = state.pieces[state.current];
  if (audio.paused && p) draw(sec * p.notes.qpm / 60);
}

// ---------------------------------------------------------------- compose

async function compose() {
  const text = $("#text").value.trim();
  const overrides = readOverrides();
  const m = mode();
  if (!text && !Object.keys(overrides).length) {
    $("#status").textContent = "Describe the music first, or choose settings under More control.";
    $("#text").focus();
    return;
  }
  const body = {
    text, overrides, mode: m, n: Number($("#n").value) || 1,
    temperature: Number($("#temperature").value), top_p: Number($("#top_p").value),
    cfg_scale: Number($("#cfg_scale").value), key_bias: Number($("#key_bias").value),
  };
  if ($("#seed").value !== "") body.seed = Number($("#seed").value);
  $("#go").disabled = true;
  $("#go").textContent = "Composing";
  $("#status").textContent = m === "ensemble" ? "Writing the parts" : "Writing the melody";
  titleLoop(true);
  audio.pause();
  try {
    const res = await api("/api/generate", body);
    showResult(res, overrides);
    $("#status").textContent = "";
    $(".player").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  } catch (e) {
    $("#status").textContent = `Nothing was composed: ${e.message}.`;
  } finally {
    titleLoop(false);
    $("#go").disabled = false;
    $("#go").textContent = "Compose";
  }
}

function showModelStatus(h) {
  const ens = h.composers && h.composers.ensemble;
  const melody = h.trained ? "trained" : "not trained yet (a baseline answers instead)";
  const band = ens ? (ens.trained ? "trained" : "not trained yet (a baseline answers instead)") : "unavailable";
  const reader = h.parser === "rules" ? "keyword rules" : "a fine-tuned language model";
  $("#model-status").textContent = `Melody model ${melody}. Band model ${band}. Descriptions are read with ${reader}. Running on ${h.device}.`;
}

// ---------------------------------------------------------------- start

async function init() {
  buildTitle();
  for (const id of ["temperature", "top_p", "cfg_scale", "key_bias"]) {
    const input = $(`#${id}`), out = $(`#${id}-v`);
    input.addEventListener("input", () => { out.textContent = input.value; });
  }
  for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", applyMode);
  $("#go").addEventListener("click", compose);
  $("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) compose(); });
  $("#play").addEventListener("click", togglePlay);
  audio.addEventListener("play", () => { $("#play").classList.add("on"); $("#play").setAttribute("aria-label", "Pause"); startLoop(); });
  audio.addEventListener("pause", () => { $("#play").classList.remove("on"); $("#play").setAttribute("aria-label", "Play"); });
  audio.addEventListener("ended", () => { setProgress(0); draw(); });
  const scrub = $("#scrub");
  scrub.addEventListener("pointerdown", (e) => {
    scrub.setPointerCapture(e.pointerId); seekTo(e.clientX);
    const move = (ev) => seekTo(ev.clientX);
    scrub.addEventListener("pointermove", move);
    scrub.addEventListener("pointerup", () => scrub.removeEventListener("pointermove", move), { once: true });
  });
  scrub.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    state.preview = false;
    audio.currentTime = Math.max(0, audio.currentTime + (e.key === "ArrowRight" ? 2 : -2));
    setProgress(audio.currentTime);
    if (audio.paused) draw(audio.currentTime * state.pieces[state.current].notes.qpm / 60);
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
      e.preventDefault(); togglePlay();
    }
  });
  window.addEventListener("resize", () => { moveModeLine(); if (!state.raf) draw(); });
  try { buildControls(await api("/api/options")); } catch (e) { /* the menus stay empty */ }
  applyMode();
  try { showResult(await api("demo/demo.json"), {}, { demo: true }); } catch (e) { /* no demo piece */ }
  try { showModelStatus(await api("/api/health")); } catch (e) {
    $("#model-status").textContent = "The tunesmith server is not reachable. Start it with pixi run app.";
  }
}

init();
