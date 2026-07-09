"use strict";

const $ = (s) => document.querySelector(s);
const POLL_MS = 450;

let stageDefs = { index: [], web: [], ask: [] };
let mode = "pdf"; // "pdf" | "url"
let selectedFile = null;

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("err", isError);
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3400);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// --------------------------------------------------------------------------- //
// Minimal, XSS-safe Markdown renderer (everything is escaped before formatting)
// --------------------------------------------------------------------------- //
const CB_OPEN = ""; // private-use sentinels: never in real text, survive escapeHtml,
const CB_CLOSE = ""; // and can't collide with ordinary digits during restore.

function renderMarkdown(src) {
  const blocks = [];
  // Pull out fenced code blocks first, escape their contents, stash a placeholder.
  src = String(src).replace(/```[\w]*\n?([\s\S]*?)```/g, (_m, code) => {
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `${CB_OPEN}${blocks.length - 1}${CB_CLOSE}`;
  });

  let out = escapeHtml(src);
  out = out.replace(/`([^`\n]+)`/g, (_m, c) => `<code>${c}</code>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_m, t, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`);

  const lines = out.split("\n");
  let html = "";
  let listType = null;
  const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };
  const blockRe = new RegExp(`^${CB_OPEN}\\d+${CB_CLOSE}$`);

  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (blockRe.test(line.trim())) { closeList(); html += line.trim(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      closeList();
      const lvl = Math.min(6, m[1].length + 2);
      html += `<h${lvl}>${m[2]}</h${lvl}>`;
      continue;
    }
    if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      if (listType !== "ul") { closeList(); listType = "ul"; html += "<ul>"; }
      html += `<li>${m[1]}</li>`;
      continue;
    }
    if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {
      if (listType !== "ol") { closeList(); listType = "ol"; html += "<ol>"; }
      html += `<li>${m[1]}</li>`;
      continue;
    }
    if ((m = line.match(/^&gt;\s?(.*)$/))) { closeList(); html += `<blockquote>${m[1]}</blockquote>`; continue; }
    if (line.trim() === "") { closeList(); continue; }
    closeList();
    html += `<p>${line}</p>`;
  }
  closeList();

  const restoreRe = new RegExp(`${CB_OPEN}(\\d+)${CB_CLOSE}`, "g");
  return html.replace(restoreRe, (_m, i) => blocks[+i] || "");
}

// --------------------------------------------------------------------------- //
// Dithering canvas — a lightweight 2D Bayer-dither approximation of the
// @paper-design "warp" shader (React/WebGL only). Speeds up on hover.
// --------------------------------------------------------------------------- //
function startDithering(canvas) {
  const ctx = canvas.getContext("2d");
  const CELL = 5; // on-screen size of one dither cell (canvas is upscaled via CSS)
  const R = 0xa8, G = 0x55, B = 0xf7; // #a855f7 — matches the purple accent
  const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  let t = 0;
  let speed = 0.02;

  function resize() {
    const w = Math.max(1, Math.floor(canvas.clientWidth / CELL));
    const h = Math.max(1, Math.floor(canvas.clientHeight / CELL));
    canvas.width = w;
    canvas.height = h;
  }

  function frame() {
    const w = canvas.width, h = canvas.height;
    if (w > 1 && h > 1) {
      const img = ctx.createImageData(w, h);
      const d = img.data;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const fx = x * 0.09, fy = y * 0.11;
          let v = Math.sin(fx + t) + Math.sin(fy * 1.3 - t * 0.8) + Math.sin((fx + fy) * 0.7 + t * 1.2);
          v = v / 3 * 0.5 + 0.5;                 // -> 0..1
          v *= 0.35 + 0.65 * (1 - y / h);         // fade toward the bottom
          const thr = (bayer[(y & 3) * 4 + (x & 3)] + 0.5) / 16;
          const idx = (y * w + x) * 4;
          d[idx] = R; d[idx + 1] = G; d[idx + 2] = B;
          d[idx + 3] = v > thr ? 255 : 0;
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    t += speed;
    requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener("resize", resize);
  const host = canvas.closest(".hero");
  if (host) {
    host.addEventListener("mouseenter", () => { speed = 0.055; });
    host.addEventListener("mouseleave", () => { speed = 0.02; });
  }
  document.addEventListener("visibilitychange", () => { speed = document.hidden ? 0 : (speed || 0.02); });
  requestAnimationFrame(frame);
}

// --------------------------------------------------------------------------- //
// Stepper rendering (compact)
// --------------------------------------------------------------------------- //
const GLYPH = { pending: "", running: "", done: "✓", error: "✕" };

function renderStepper(container, defs) {
  container.innerHTML = defs.map((s) => `
    <div class="step" data-key="${s.key}">
      <div class="dot"></div>
      <div class="lbl">${escapeHtml(s.label)}</div>
    </div>`).join("");
}

function applyProgress(container, progress) {
  (progress || []).forEach((p) => {
    const step = container.querySelector(`.step[data-key="${p.key}"]`);
    if (!step) return;
    step.classList.remove("running", "done", "error");
    if (p.status !== "pending") step.classList.add(p.status);
    step.querySelector(".dot").textContent = GLYPH[p.status] || "";
  });
}

function markAllDone(container) {
  container.querySelectorAll(".step").forEach((s) => {
    if (s.classList.contains("error")) return;
    s.classList.remove("running");
    s.classList.add("done");
    s.querySelector(".dot").textContent = "✓";
  });
}

async function drive(statusUrl, container) {
  const MAX_POLLS = 900;
  const MAX_ERRORS = 5;
  let errors = 0;
  for (let i = 0; i < MAX_POLLS; i++) {
    let s;
    try { s = await api(statusUrl); errors = 0; }
    catch (err) { if (++errors >= MAX_ERRORS) throw err; await sleep(POLL_MS); continue; }
    applyProgress(container, s.progress);
    if (s.state === "completed") { markAllDone(container); return s.result; }
    if (s.state === "failed") throw new Error(s.error || "Workflow failed");
    await sleep(POLL_MS);
  }
  throw new Error("Timed out waiting for the workflow to finish");
}

// --------------------------------------------------------------------------- //
// Health
// --------------------------------------------------------------------------- //
async function refreshHealth() {
  const badge = $("#statusBadge");
  const text = $("#statusText");
  try {
    const h = await api("/api/health");
    const up = h.temporal && h.qdrant && h.ollama;
    badge.classList.toggle("down", !up);
    text.textContent = up ? "All systems online" : "Some services offline";
    if (h.vectors != null) {
      $("#vectorsPill").textContent = `${h.vectors} chunk${h.vectors === 1 ? "" : "s"} indexed`;
    }
  } catch (_) {
    badge.classList.add("down");
    text.textContent = "API unreachable";
  }
}

// --------------------------------------------------------------------------- //
// Indexing (PDF or URL)
// --------------------------------------------------------------------------- //
function setMode(next) {
  mode = next;
  document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === next));
  document.querySelectorAll(".mode-panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== next));
  $("#indexBtn").textContent = next === "pdf" ? "Index PDF" : "Scrape & index URL";
}

function setFile(file) {
  selectedFile = file;
  $("#fileName").textContent = file ? file.name : "";
}

async function runIndex(useSample) {
  const btn = $("#indexBtn");
  const stepper = $("#indexStepper");
  btn.disabled = true;
  $("#indexResult").classList.add("hidden");

  try {
    let startUrl, statusBase, opts, sourceLabel;

    if (mode === "url" && !useSample) {
      const url = $("#urlInput").value.trim();
      if (!url) { toast("Enter a URL first.", true); return; }
      renderStepper(stepper, stageDefs.web);
      stepper.classList.remove("hidden");
      startUrl = "/api/web/start";
      statusBase = "/api/web/status/";
      opts = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) };
      sourceLabel = url;
    } else {
      renderStepper(stepper, stageDefs.index);
      stepper.classList.remove("hidden");
      const form = new FormData();
      if (useSample) form.append("use_sample", "true");
      else {
        if (!selectedFile) { toast("Choose a PDF first.", true); return; }
        form.append("file", selectedFile);
      }
      startUrl = "/api/index/start";
      statusBase = "/api/index/status/";
      opts = { method: "POST", body: form };
      sourceLabel = useSample ? "sample.pdf" : selectedFile.name;
    }

    const started = await api(startUrl, opts);
    const result = await drive(statusBase + started.workflow_id, stepper);

    const name = escapeHtml(started.file || started.url || sourceLabel);
    $("#indexResult").innerHTML =
      `Indexed <strong>${result.chunks}</strong> chunk${result.chunks === 1 ? "" : "s"} from <strong>${name}</strong>.`;
    $("#indexResult").classList.remove("hidden");
    toast(`Indexed ${result.inserted} vector${result.inserted === 1 ? "" : "s"}`);
    refreshHealth();
  } catch (err) {
    toast(`Indexing failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// Asking
// --------------------------------------------------------------------------- //
async function runAsk() {
  const question = $("#question").value.trim();
  if (!question) { toast("Type a question first.", true); return; }

  const btn = $("#askBtn");
  const stepper = $("#askStepper");
  btn.disabled = true;
  $("#answerBlock").classList.add("hidden");
  renderStepper(stepper, stageDefs.ask);
  stepper.classList.remove("hidden");

  try {
    const started = await api("/api/ask/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await drive(`/api/ask/status/${started.workflow_id}`, stepper);

    $("#answerText").innerHTML = renderMarkdown(result.answer || "_No answer returned._");

    const chunks = $("#chunks");
    const ctx = result.retrieved_context || [];
    const scores = result.scores || [];
    const sources = result.sources || [];
    chunks.innerHTML = ctx.map((text, i) => {
      const meta = sources[i] || {};
      const src = meta.document_name || "unknown";
      const page = meta.page_number != null ? `p.${meta.page_number}` : "";
      return `
        <div class="source">
          <div class="src-head">
            <span class="src-name">${escapeHtml(src)} ${escapeHtml(page)}</span>
            <span class="score">${escapeHtml(scores[i] != null ? scores[i] : "–")}</span>
          </div>
          <div>${escapeHtml(text)}</div>
        </div>`;
    }).join("");
    $("#sourcesLabel").textContent = `Sources (${ctx.length})`;
    $("#answerBlock").classList.remove("hidden");
  } catch (err) {
    toast(`Question failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// Init / wiring
// --------------------------------------------------------------------------- //
async function init() {
  try { stageDefs = await api("/api/pipeline"); }
  catch (_) {
    stageDefs = {
      index: [{ key: "read_pdf", label: "Read PDF" }, { key: "split", label: "Split" }, { key: "embed", label: "Embed" }, { key: "store", label: "Store" }],
      web: [{ key: "fetch_url", label: "Fetch URL" }, { key: "split", label: "Split" }, { key: "embed", label: "Embed" }, { key: "store", label: "Store" }],
      ask: [{ key: "embed_query", label: "Embed" }, { key: "retrieve", label: "Retrieve" }, { key: "generate", label: "Generate" }],
    };
  }

  startDithering($("#ditherCanvas"));

  $("#ctaBtn").addEventListener("click", () => $("#workspace").scrollIntoView({ behavior: "smooth" }));

  document.querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode))
  );

  const dz = $("#dropzone");
  const fileInput = $("#fileInput");
  fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith(".pdf")) setFile(f);
    else toast("Please drop a .pdf file.", true);
  });

  $("#sampleBtn").addEventListener("click", () => runIndex(true));
  $("#indexBtn").addEventListener("click", () => runIndex(false));
  $("#askBtn").addEventListener("click", runAsk);
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAsk(); });
  $("#urlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") runIndex(false); });

  const toggle = $("#sourcesToggle");
  toggle.addEventListener("click", () => {
    const open = $("#chunks").classList.toggle("hidden");
    toggle.setAttribute("aria-expanded", String(!open));
  });

  refreshHealth();
  setInterval(refreshHealth, 8000);
}

init();
