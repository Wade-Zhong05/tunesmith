// tunesmith web UI (no build step).
// AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.

const EXAMPLES = [
  "a gentle piano waltz in A minor, slow and a little sad",
  "an epic brass theme at 140 bpm",
  "playful flute melody in D major, 2/4, short",
  "dreamy synth pad, ambient, very slow",
  "a cheerful celtic tune on the violin, fast",
  "romantic cello piece in Eb major, 16 bars",
];
const FIELDS = ["key", "meter", "tempo_bpm", "instrument", "mood", "genre", "bars"];
const LABELS = { key: "Key", meter: "Meter", tempo_bpm: "Tempo", instrument: "Instrument", mood: "Mood", genre: "Genre", bars: "Bars" };
const $ = (sel, el = document) => el.querySelector(sel);

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  return data;
}

function pretty(v) { return String(v).replace(/_/g, " "); }

function buildOverrides(opts) {
  const grid = $("#overrides");
  const lists = { ...opts, tempo_bpm: opts.tempo };
  for (const f of FIELDS) {
    const sel = document.createElement("select");
    sel.id = `ov-${f}`;
    sel.innerHTML = `<option value="">Auto</option>` + (lists[f === "tempo_bpm" ? "tempo_bpm" : f] || []).map((o) =>
      typeof o === "object" ? `<option value="${o.bpm}">${pretty(o.name)} (${o.bpm})</option>` : `<option value="${o}">${pretty(o)}</option>`
    ).join("");
    const label = document.createElement("label");
    label.textContent = LABELS[f];
    label.appendChild(sel);
    grid.appendChild(label);
  }
}

function readOverrides() {
  const out = {};
  for (const f of FIELDS) {
    const v = $(`#ov-${f}`).value;
    if (v !== "") out[f] = (f === "tempo_bpm" || f === "bars") ? Number(v) : v;
  }
  return out;
}

function showAttrs(attrs, overrides) {
  const box = $("#attrs");
  box.innerHTML = "";
  for (const f of FIELDS) {
    const v = attrs[f];
    const chip = document.createElement("span");
    chip.className = "chip" + (v == null ? " empty" : "") + (f in overrides ? " override" : "");
    chip.innerHTML = `<b>${LABELS[f]}</b> ${v == null ? "model decides" : pretty(f === "tempo_bpm" ? v + " bpm" : v)}`;
    box.appendChild(chip);
  }
}

// ---------------------------------------------------------------- piano roll

function drawRoll(canvas, roll, playBeat = null) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const notes = roll.notes;
  const pitches = notes.map((n) => n.pitch);
  const lo = Math.min(...pitches, 60) - 2, hi = Math.max(...pitches, 72) + 2;
  const total = Math.max(roll.total_beats, roll.beats_per_bar * Math.max(1, roll.bars));
  const x = (b) => 8 + (b / total) * (w - 16);
  const rowH = (h - 12) / (hi - lo + 1);
  const y = (p) => 6 + (hi - p) * rowH;
  // pitch rows: shade black keys
  for (let p = lo; p <= hi; p++) {
    if ([1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12)) { ctx.fillStyle = "rgba(40,30,20,0.045)"; ctx.fillRect(8, y(p), w - 16, rowH); }
  }
  // bar lines
  ctx.strokeStyle = "rgba(40,30,20,0.18)";
  ctx.lineWidth = 1;
  for (let b = 0; b <= total + 1e-6; b += roll.beats_per_bar) {
    ctx.beginPath(); ctx.moveTo(x(b), 4); ctx.lineTo(x(b), h - 4); ctx.stroke();
  }
  // notes
  for (const n of notes) {
    const active = playBeat != null && playBeat >= n.start && playBeat < n.start + n.dur;
    const alpha = 0.45 + 0.55 * (n.vel / 127);
    ctx.fillStyle = active ? "#c2410c" : `rgba(23, 70, 72, ${alpha})`;
    const nx = x(n.start), nw = Math.max(3, x(n.start + n.dur) - nx - 1.5);
    roundRect(ctx, nx, y(n.pitch) + 1, nw, Math.max(3, rowH - 2), 3);
  }
  if (playBeat != null) {
    ctx.strokeStyle = "#c2410c"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x(playBeat), 2); ctx.lineTo(x(playBeat), h - 2); ctx.stroke();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); ctx.fill();
}

function fmt(s) { const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

function renderPiece(piece, idx) {
  const node = $("#piece-tpl").content.firstElementChild.cloneNode(true);
  const canvas = $(".roll", node), audio = $("audio", node), btn = $(".play", node);
  const roll = piece.notes;
  $(".download", node).href = piece.midi_url;
  $(".download", node).setAttribute("download", `tunesmith-${idx + 1}.mid`);
  const m = piece.metrics;
  const bits = [`version ${idx + 1}`, `${m.n_notes} notes`, `${m.bars} bars`, m.meter];
  if (m.detected_key) bits.push(`sounds like ${m.detected_key}`);
  if (m.in_scale != null) bits.push(`${Math.round(m.in_scale * 100)}% in key`);
  $(".stats", node).textContent = bits.join(" · ");
  if (piece.wav_url) audio.src = piece.wav_url; else btn.disabled = true;
  let raf = null;
  const tick = () => {
    drawRoll(canvas, roll, audio.currentTime * roll.qpm / 60);
    $(".time", node).textContent = fmt(audio.currentTime);
    if (!audio.paused) raf = requestAnimationFrame(tick);
  };
  btn.addEventListener("click", () => {
    document.querySelectorAll("audio").forEach((a) => { if (a !== audio) a.pause(); });
    if (audio.paused) audio.play(); else audio.pause();
  });
  audio.addEventListener("play", () => { btn.textContent = "❚❚"; raf = requestAnimationFrame(tick); });
  audio.addEventListener("pause", () => { btn.textContent = "▶"; cancelAnimationFrame(raf); });
  audio.addEventListener("ended", () => { drawRoll(canvas, roll); });
  $("#pieces").appendChild(node);
  requestAnimationFrame(() => drawRoll(canvas, roll));
  window.addEventListener("resize", () => drawRoll(canvas, roll));
}

// ---------------------------------------------------------------- main

async function compose() {
  const text = $("#text").value.trim();
  const overrides = readOverrides();
  if (!text && !Object.keys(overrides).length) { $("#status").textContent = "Type a description first."; return; }
  const body = {
    text, overrides, mode: document.querySelector("input[name=mode]:checked").value,
    n: Number($("#n").value) || 1,
    temperature: Number($("#temperature").value), top_p: Number($("#top_p").value),
    cfg_scale: Number($("#cfg_scale").value), key_bias: Number($("#key_bias").value),
  };
  if ($("#seed").value !== "") body.seed = Number($("#seed").value);
  $("#go").disabled = true;
  $("#status").textContent = "Composing…";
  try {
    const res = await api("/api/generate", body);
    $("#result").hidden = false;
    showAttrs(res.attributes, overrides);
    $("#pieces").innerHTML = "";
    res.pieces.forEach(renderPiece);
    $("#meta").textContent = `understood by ${res.parser} · composed by ${res.composer} · ${res.timings.total_s.toFixed(1)} s · seed ${res.seed}`;
    $("#status").textContent = "";
  } catch (e) {
    $("#status").textContent = `Could not compose: ${e.message}`;
  } finally {
    $("#go").disabled = false;
  }
}

async function init() {
  const ex = $("#examples");
  for (const t of EXAMPLES) {
    const b = document.createElement("button");
    b.className = "example"; b.type = "button"; b.textContent = t;
    b.addEventListener("click", () => { $("#text").value = t; $("#text").focus(); });
    ex.appendChild(b);
  }
  for (const id of ["temperature", "top_p", "cfg_scale", "key_bias"]) {
    const input = $(`#${id}`), out = $(`#${id}-v`);
    input.addEventListener("input", () => { out.textContent = input.value; });
  }
  $("#go").addEventListener("click", compose);
  $("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) compose(); });
  try { buildOverrides(await api("/api/options")); } catch (e) { /* menus stay empty */ }
  try {
    const h = await api("/api/health");
    $("#model-status").textContent = h.trained
      ? `Model: ${h.composer} · parser: ${h.parser} · ${h.device}`
      : `Preview mode: ${h.composer}. Train a composer to get real melodies.`;
    if (!h.trained) $("#model-status").classList.add("warn");
  } catch (e) {
    $("#model-status").textContent = "Server not reachable.";
  }
}

init();
