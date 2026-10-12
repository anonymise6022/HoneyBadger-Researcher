/* The Debate tab: scenarios, the agents' debate, and an opt-in plan.
 *
 * Kept apart from app.js, which it does not need to reach into: the tab
 * switching there already shows #view-debate generically. This file only
 * calls the bridge and prints what comes back. The reports are the exact
 * strings the validator passed, shown as-is in a monospace block, so nothing
 * here formats, rounds or computes a figure.
 *
 * `window.Debate.run(query)` lets the Ask tab hand over a question about the
 * future.
 */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = () => window.pywebview.api;
  const HEADINGS = new Set([
    "QUESTION", "THE SIMPLE PICTURE", "THE STRONGEST CASE", "SEVERAL STRONG VIEWS",
    "WHAT IF: SCENARIOS FROM HISTORY", "WHAT TO WATCH FOR", "YOU DECIDE",
    "WHAT YOU ARE LOOKING AT", "THE NUMBERS", "COMPARED WITH SIMILAR COMPANIES",
    "PRICE HISTORY", "WHAT THIS DOES NOT TELL YOU", "RAW DATA", "YOUR DECISION",
    "WHAT HISTORY SAYS ABOUT YOUR TIME FRAME", "PRACTICAL POINTS",
  ]);
  const OUTCOMES = {
    clear: ["The strongest case", "good"],
    several: ["Several strong views", "warn"],
    standard: ["Standard report", ""],
  };
  let busy = false;
  let lastTicker = null;

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  /** The validated text, escaped, with section headings picked out. */
  function reportBlock(text) {
    const lines = String(text).split("\n").map((line) => {
      const bare = line.trim();
      const heading = HEADINGS.has(bare) || bare.startsWith("OPTION ") ||
        bare.startsWith("THE DEBATE --") || bare.startsWith("OUTLOOK --") ||
        bare.startsWith("A PLAN FOR YOUR DECISION");
      return heading ? `<span class="report-heading">${esc(line)}</span>` : esc(line);
    });
    return `<pre class="report-text">${lines.join("\n")}</pre>`;
  }

  function panel(title, chips, body, extra = "") {
    return `<section class="panel ${extra}"><div class="panel-head">
      <span class="panel-title">${esc(title)}</span><span>${chips}</span></div>
      <div class="panel-body">${body}</div></section>`;
  }

  const chip = (text, kind = "") => `<span class="chip ${kind}">${esc(text)}</span>`;

  function setBusy(on, message) {
    busy = on;
    $("scanline").classList.toggle("active", on);
    $("debate-run").disabled = on;
    if (message !== undefined) $("status").textContent = message;
  }

  function errorPanel(error, suggestion) {
    return panel("Could not do that", "",
      `<div>${esc(error)}</div>${suggestion ? `<div class="dim">${esc(suggestion)}</div>` : ""}`,
      "error-panel");
  }

  async function refreshStatus() {
    const r = await api().debate_status();
    $("debate-status").textContent = r.ok ? r.detail : "";
  }

  async function run(query) {
    const text = (query ?? $("debate-query").value).trim();
    if (!text || busy) return;
    $("debate-query").value = text;
    const out = $("debate-out");
    out.innerHTML = panel("Working", "", `<div class="dim">Gathering the evidence, then letting the
      viewpoints argue. With your Claude login this takes about a minute.</div>`);
    setBusy(true, "Agents are debating…");
    try {
      const r = await api().debate(text, $("debate-depth").value);
      if (!r.ok) {
        out.innerHTML = errorPanel(r.error, r.suggestion);
        $("status").textContent = "Could not run the debate";
        return;
      }
      lastTicker = r.ticker;
      const [label, kind] = OUTCOMES[r.outcome] || OUTCOMES.standard;
      const chips = chip(label, kind) + (r.synthetic ? " " + chip("synthetic data", "bad") : "");
      const notes = [
        r.reason ? `Showing the standard report because ${r.reason}.` : "",
        ...r.notes,
      ].filter(Boolean);
      out.innerHTML =
        panel(`Debate — ${r.ticker}`, chips, reportBlock(r.report), "accent") +
        panel("How this was checked", chip(r.validation_ok ? "passed" : "check", r.validation_ok ? "good" : "warn"),
          `<div>${esc(r.validation)}</div><div class="dim">${esc(r.usage)}</div>` +
          (notes.length ? `<ul class="bullets">${notes.map((n) => `<li class="dim">${esc(n)}</li>`).join("")}</ul>` : "")) +
        planOffer();
      wirePlanOffer();
      $("status").textContent = "Done. The decision is yours.";
    } finally {
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------ the plan */

  function planOffer() {
    return `<div id="plan-zone">${panel("Want help planning an investment?", "",
      `<div class="dim">Only if you have already decided. This tool never places orders.</div>
       <div class="control-row" style="margin-top:10px">
         <button class="ghost small" id="plan-yes">Yes, help me plan</button>
         <button class="ghost small" id="plan-no">No thanks</button></div>`)}</div>`;
  }

  function wirePlanOffer() {
    $("plan-no").addEventListener("click", () => {
      $("plan-zone").innerHTML = panel("No plan", "", `<div class="dim">OK. The decision is yours.</div>`);
    });
    $("plan-yes").addEventListener("click", () => {
      $("plan-zone").innerHTML = panel("Your plan", "", `
        <div class="control-row">
          <div class="prompt" style="max-width:180px"><span class="prompt-sigil">$</span>
            <input type="text" id="plan-amount" placeholder="1000" autocomplete="off" /></div>
          <select id="plan-horizon" aria-label="How long">
            <option>3 months</option><option>6 months</option><option selected>1 year</option>
            <option>2 years</option><option>3 years</option><option>5 years</option></select>
          <div class="prompt" style="max-width:220px"><span class="prompt-sigil">↓</span>
            <input type="text" id="plan-loss" placeholder="most you could lose: $200 or 20%" autocomplete="off" /></div>
          <button class="primary" id="plan-go">Show options</button>
        </div><div id="plan-out"></div>`);
      $("plan-go").addEventListener("click", showPlan);
      $("plan-amount").focus();
    });
  }

  async function showPlan() {
    const amount = $("plan-amount").value.trim();
    const loss = $("plan-loss").value.trim() || "20%";
    if (!amount || !lastTicker || busy) return;
    setBusy(true, "Building plan options…");
    try {
      const r = await api().plan(lastTicker, amount, $("plan-horizon").value, loss, $("debate-depth").value);
      $("plan-out").innerHTML = r.ok
        ? reportBlock(r.report) + `<div class="dim">${esc(r.validation)}</div>`
        : `<div class="status-bad">${esc(r.error)}</div>`;
      $("status").textContent = r.ok ? "Plan options ready. The decision is yours." : "Could not build a plan";
    } finally {
      setBusy(false);
    }
  }

  /* -------------------------------------------------------------- wiring */

  function wire() {
    $("debate-run").addEventListener("click", () => run());
    $("debate-query").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
    document.querySelector('.tab[data-view="debate"]').addEventListener("click", refreshStatus);
  }

  window.Debate = {
    run(query) {
      document.querySelector('.tab[data-view="debate"]').click();
      run(query);
    },
  };

  if (window.pywebview && window.pywebview.api) wire();
  else window.addEventListener("pywebviewready", wire, { once: true });
})();
