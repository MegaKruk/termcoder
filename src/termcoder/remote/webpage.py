"""The single-page client served to phones and other remote browsers.

The page is deliberately one self-contained HTML string: no build step, no
static file packaging, no framework. It connects back over a WebSocket with the
token taken from its own URL, renders the session's event stream, and offers
the three interactions a remote needs: send a message, answer an approval, and
flip the shared thinking toggle.

Kept in its own module so the server logic stays readable and the page can be
tested for basic invariants (ASCII only, expected hooks present).
"""

from __future__ import annotations

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>termcoder remote</title>
<style>
  :root {
    --bg: #10131a; --panel: #181c26; --line: #2a3040;
    --text: #d7dce6; --dim: #7d8595; --accent: #5ac8fa;
    --good: #4cd97b; --bad: #ff6b6b; --warn: #f5c451;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px; display: flex; flex-direction: column; height: 100dvh;
  }
  header {
    padding: 8px 12px; background: var(--panel);
    border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  #dot { width: 10px; height: 10px; border-radius: 50%; background: var(--bad); }
  #dot.on { background: var(--good); }
  #meta { color: var(--dim); font-size: 12px; overflow: hidden;
          text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
  #busy { display: none; color: var(--accent); font-size: 12px; }
  body.busy #busy { display: inline; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  label.toggle { color: var(--dim); font-size: 12px; display: flex;
                 align-items: center; gap: 5px; user-select: none; }
  #log { flex: 1; overflow-y: auto; padding: 10px 12px 16px; }
  .block { margin: 6px 0; white-space: pre-wrap; word-break: break-word; }
  .who { color: var(--dim); font-size: 11px; margin-bottom: 2px; }
  .user .who { color: var(--accent); }
  .assistant .who { color: var(--good); }
  .thinking { display: none; color: var(--dim); font-style: italic;
              border-left: 2px solid var(--line); padding-left: 8px; }
  body.show-thinking .thinking { display: block; }
  .status { color: var(--dim); }
  .status.info { color: var(--text); }
  .status.warning { color: var(--warn); }
  .status.error { color: var(--bad); }
  .tool { color: var(--dim); }
  .tool.done-ok { color: var(--good); }
  .tool.done-bad { color: var(--bad); }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 10px; margin: 8px 0; }
  .card.destructive { border-color: var(--bad); }
  .card .title { font-weight: bold; margin-bottom: 6px; }
  .card pre { background: #0b0e14; border: 1px solid var(--line);
              border-radius: 6px; padding: 8px; overflow-x: auto; margin: 6px 0;
              font-size: 12px; max-height: 40vh; }
  .card .note { color: var(--warn); font-size: 12px; margin: 4px 0; }
  .card .buttons { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .card button { flex: 1; min-width: 90px; padding: 10px 8px; border-radius: 6px;
                 border: 1px solid var(--line); background: #222836;
                 color: var(--text); font: inherit; }
  .card button.yes { border-color: var(--good); }
  .card button.always { border-color: var(--accent); }
  .card button.no { border-color: var(--bad); }
  .card button:disabled { opacity: 0.45; }
  .card input { width: 100%; margin-top: 8px; padding: 8px; border-radius: 6px;
                border: 1px solid var(--line); background: #0b0e14;
                color: var(--text); font: inherit; }
  .card .verdict { margin-top: 8px; color: var(--dim); font-size: 12px; }
  footer { display: flex; gap: 8px; padding: 10px 12px;
           background: var(--panel); border-top: 1px solid var(--line); }
  #box { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid var(--line);
         background: #0b0e14; color: var(--text); font: inherit; }
  #send { padding: 10px 16px; border-radius: 8px; border: 1px solid var(--accent);
          background: #1c2a38; color: var(--text); font: inherit; }
  #send:disabled { opacity: 0.45; }
  .diff-add { color: var(--good); }
  .diff-del { color: var(--bad); }
</style>
</head>
<body>
<header>
  <div id="dot"></div>
  <div id="meta">connecting...</div>
  <span id="busy">working...</span>
  <label class="toggle"><input type="checkbox" id="thinking"> thinking</label>
</header>
<div id="log"></div>
<footer>
  <input id="box" type="text" autocomplete="off"
         placeholder="message or /command" disabled>
  <button id="send" disabled>send</button>
</footer>
<script>
"use strict";
var log = document.getElementById("log");
var dot = document.getElementById("dot");
var meta = document.getElementById("meta");
var box = document.getElementById("box");
var send = document.getElementById("send");
var thinkingToggle = document.getElementById("thinking");
var socket = null;
var retryMs = 1000;
var currentAssistant = null;
var currentThinking = null;
var pendingEcho = null;
var cards = {};

function token() {
  return new URLSearchParams(location.search).get("token") || "";
}

function scrolledDown() {
  return log.scrollHeight - log.scrollTop - log.clientHeight < 60;
}

function append(el) {
  var stick = scrolledDown();
  log.appendChild(el);
  if (stick) log.scrollTop = log.scrollHeight;
}

function breakStreams() {
  currentAssistant = null;
  currentThinking = null;
}

function block(cls, who) {
  var wrap = document.createElement("div");
  wrap.className = "block " + cls;
  if (who) {
    var label = document.createElement("div");
    label.className = "who";
    label.textContent = who;
    wrap.appendChild(label);
  }
  var body = document.createElement("div");
  body.className = "body";
  wrap.appendChild(body);
  append(wrap);
  return body;
}

function statusLine(text, level) {
  breakStreams();
  var el = document.createElement("div");
  el.className = "block status " + level;
  el.textContent = text;
  append(el);
}

function setBusy(value) {
  document.body.classList.toggle("busy", value);
}

function setThinkingVisible(value) {
  document.body.classList.toggle("show-thinking", value);
  thinkingToggle.checked = value;
  var stick = scrolledDown();
  if (stick) log.scrollTop = log.scrollHeight;
}

function renderDetail(pre, text, kind) {
  pre.textContent = "";
  var lines = String(text).split("\\n");
  for (var i = 0; i < lines.length; i++) {
    var line = document.createElement("div");
    line.textContent = lines[i];
    if (kind === "diff" && lines[i].charAt(0) === "+") line.className = "diff-add";
    if (kind === "diff" && lines[i].charAt(0) === "-") line.className = "diff-del";
    pre.appendChild(line);
  }
}

function decide(id, decision) {
  var card = cards[id];
  var feedback = "";
  if (card && decision === "reject") feedback = card.feedback.value;
  post({type: "approval", request_id: id, decision: decision, feedback: feedback});
  if (card) card.buttons.forEach(function (b) { b.disabled = true; });
}

function approvalCard(ev) {
  breakStreams();
  var wrap = document.createElement("div");
  wrap.className = "card" + (ev.destructive ? " destructive" : "");
  var title = document.createElement("div");
  title.className = "title";
  title.textContent = "Approval needed: " + ev.summary;
  wrap.appendChild(title);
  if (ev.detail) {
    var pre = document.createElement("pre");
    renderDetail(pre, ev.detail, ev.detail_kind);
    wrap.appendChild(pre);
  }
  if (ev.note) {
    var note = document.createElement("div");
    note.className = "note";
    note.textContent = ev.note;
    wrap.appendChild(note);
  }
  var row = document.createElement("div");
  row.className = "buttons";
  var buttons = [];
  [["approve", "approve", "yes"],
   ["allow for session", "approve_for_session", "always"],
   ["reject", "reject", "no"]].forEach(function (spec) {
    var b = document.createElement("button");
    b.textContent = spec[0];
    b.className = spec[2];
    b.onclick = function () { decide(ev.request_id, spec[1]); };
    row.appendChild(b);
    buttons.push(b);
  });
  wrap.appendChild(row);
  var feedback = document.createElement("input");
  feedback.placeholder = "optional feedback if rejecting";
  wrap.appendChild(feedback);
  var verdict = document.createElement("div");
  verdict.className = "verdict";
  wrap.appendChild(verdict);
  append(wrap);
  cards[ev.request_id] = {buttons: buttons, feedback: feedback, verdict: verdict};
}

function resolveCard(ev) {
  var card = cards[ev.request_id];
  if (!card) return;
  card.buttons.forEach(function (b) { b.disabled = true; });
  card.verdict.textContent = "resolved: " + ev.decision + " (" + ev.resolved_by + ")";
  delete cards[ev.request_id];
}

function handle(ev) {
  if (ev.kind === "session_state") {
    meta.textContent = ev.workspace + "  |  " + ev.model;
    setBusy(ev.busy);
    setThinkingVisible(ev.show_thinking);
  } else if (ev.kind === "assistant_text") {
    currentThinking = null;
    if (!currentAssistant) currentAssistant = block("assistant", "assistant");
    currentAssistant.textContent += ev.text;
  } else if (ev.kind === "thinking") {
    if (!currentThinking) currentThinking = block("thinking", "thinking");
    currentThinking.textContent += ev.text;
    if (scrolledDown()) log.scrollTop = log.scrollHeight;
  } else if (ev.kind === "user_message") {
    breakStreams();
    if (pendingEcho) { pendingEcho.remove(); pendingEcho = null; }
    var who = ev.source === "remote" ? "you (phone)" : "you (terminal)";
    block("user", who).textContent = ev.text;
    setBusy(true);
  } else if (ev.kind === "turn_ended") {
    breakStreams();
    setBusy(false);
  } else if (ev.kind === "status") {
    statusLine(ev.text, ev.level);
  } else if (ev.kind === "tool_started") {
    breakStreams();
    statusFor("[tool] " + ev.tool_name + " " + ev.arguments_preview, "tool");
  } else if (ev.kind === "tool_finished") {
    breakStreams();
    statusFor("[tool] " + ev.tool_name + ": " + ev.summary,
              ev.ok ? "tool done-ok" : "tool done-bad");
  } else if (ev.kind === "approval_requested") {
    approvalCard(ev);
  } else if (ev.kind === "approval_resolved") {
    resolveCard(ev);
  } else if (ev.kind === "thinking_visibility") {
    setThinkingVisible(ev.enabled);
  }
}

function statusFor(text, cls) {
  var el = document.createElement("div");
  el.className = "block " + cls;
  el.textContent = text;
  append(el);
}

function post(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  }
}

function submit() {
  var text = box.value.trim();
  if (!text) return;
  post({type: "user_input", text: text});
  box.value = "";
  breakStreams();
  pendingEcho = document.createElement("div");
  pendingEcho.className = "block user";
  pendingEcho.textContent = text;
  pendingEcho.style.opacity = "0.55";
  append(pendingEcho);
}

function connect() {
  var scheme = location.protocol === "https:" ? "wss://" : "ws://";
  socket = new WebSocket(scheme + location.host + "/ws?token=" +
                         encodeURIComponent(token()));
  socket.onopen = function () {
    dot.classList.add("on");
    box.disabled = false;
    send.disabled = false;
    retryMs = 1000;
    log.textContent = "";
    cards = {};
    breakStreams();
  };
  socket.onmessage = function (raw) {
    try { handle(JSON.parse(raw.data)); } catch (err) { /* skip bad frame */ }
  };
  socket.onclose = function () {
    dot.classList.remove("on");
    box.disabled = true;
    send.disabled = true;
    meta.textContent = "disconnected, retrying...";
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 10000);
  };
  socket.onerror = function () { socket.close(); };
}

send.onclick = submit;
box.addEventListener("keydown", function (event) {
  if (event.key === "Enter") { event.preventDefault(); submit(); }
});
thinkingToggle.addEventListener("change", function () {
  post({type: "set_thinking", enabled: thinkingToggle.checked});
});
connect();
</script>
</body>
</html>
"""

__all__ = ["PAGE_HTML"]
