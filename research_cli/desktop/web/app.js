/* Interface logic.
 *
 * Every figure on screen comes from the Python bridge, which got it from an
 * EvidenceBundle the validator has already checked. Nothing here computes a
 * statistic; formatting a percentage is the most arithmetic this file does.
 * A number calculated in JavaScript would sit outside the validator's reach,
 * which is the one guarantee the whole tool rests on.
 *
 * Depth is handled two ways at once. Layout differences live in CSS, keyed
 * off `data-depth` on <html>, so both versions render from the same markup.
 * Genuine wording differences -- where a beginner needs a sentence and an
 * analyst needs a table -- branch here, in `renderAttribution`.
 */

(() => {
  "use strict";

  const state = {
    depth: "beginner",
    stage: "home",
    view: "ask",
    labMode: "build",
    strategies: [],
    optionStrategies: [],
    indicators: [],
    credentials: [],
    lastTicker: "SPY",
    selectedPreset: null,
    busy: false,
    searchTimer: null,
  };

  const $ = (id) => document.getElementById(id);

  const THEMES = [
    ["deep-field", "Deep Field", "Near-black with a cold cyan signal."],
    ["phosphor", "Phosphor", "The amber monitor, warmed until it's readable."],
    ["tidewater", "Tidewater", "Slate and sea-glass. The calmest of the set."],
    ["oxide", "Oxide", "Charcoal and copper. Metal, not warning."],
    ["ultraviolet", "Ultraviolet", "Violet and magenta on near-black."],
    ["vellum", "Vellum", "Warm paper. The light one."],
  ];

  const DEPTHS = [
    ["beginner", "I'm new to this"],
    ["intermediate", "I know some"],
    ["analyst", "Analyst"],
  ];

  const EXAMPLES = [
    "why did SPY fall today",
    "why did Tesla drop yesterday",
    "is Apple worth investing in",
    "why did the stock market fall this week",
  ];

  /* ------------------------------------------------------------ helpers */

  function ready() {
    return new Promise((resolve) => {
      if (window.pywebview && window.pywebview.api) return resolve();
      window.addEventListener("pywebviewready", () => resolve(), { once: true });
    });
  }

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  const pct = (v, d = 1) => (v == null ? "n/a" : `${v.toFixed(d)}%`);
  const signed = (v, d = 2) => (v == null ? "n/a" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`);
  const rate = (v, d = 0) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
  const ratio = (v, d = 2) => (v == null ? "n/a" : v.toFixed(d));

  /** "4 times out of 5" reads faster than "80%" for someone new to this. */
  function outOfTen(fraction) {
    if (fraction == null) return "n/a";
    const n = Math.round(fraction * 10);
    if (n >= 10) return "almost every time";
    if (n <= 0) return "almost never";
    return `about ${n} times out of 10`;
  }

  function money(value) {
    if (value == null) return "n/a";
    for (const [size, suffix] of [[1e12, "T"], [1e9, "B"], [1e6, "M"]]) {
      if (Math.abs(value) >= size) return `$${(value / size).toFixed(1)}${suffix}`;
    }
    return `$${value.toFixed(0)}`;
  }

  function setBusy(busy, message) {
    state.busy = busy;
    $("scanline").classList.toggle("active", busy);
    document.querySelectorAll("button.primary").forEach((b) => (b.disabled = busy));
    if (message !== undefined) setStatus(message);
  }

  function setStatus(text, kind) {
    const node = $("status");
    if (!node) return;
    node.textContent = text;
    node.className = kind === "ok" ? "status-ok" : kind === "bad" ? "status-bad" : "";
  }

  function countUp(node, target, format, duration = 640) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || target == null) {
      node.textContent = format(target ?? 0);
      return;
    }
    const start = performance.now();
    // requestAnimationFrame stops firing entirely while the window is in the
    // background, which would strand the headline on whatever partial value
    // it had reached. The timer is the guarantee that the final figure lands.
    const settle = setTimeout(() => { node.textContent = format(target); }, duration + 250);
    const frame = (now) => {
      const t = Math.min((now - start) / duration, 1);
      // easeOutCubic: fast then settling, so it reads as arriving.
      node.textContent = format(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) return requestAnimationFrame(frame);
      clearTimeout(settle);
      node.textContent = format(target);
    };
    requestAnimationFrame(frame);
  }

  function panel({ title, chip, body, extra = "" }) {
    return `<section class="panel ${extra}">
      <div class="panel-head"><span class="panel-title">${esc(title)}</span>${chip || ""}</div>
      <div class="panel-body">${body}</div></section>`;
  }

  function card({ kicker, body, extra = "" }) {
    return `<section class="card ${extra}">
      ${kicker ? `<div class="card-kicker">${esc(kicker)}</div>` : ""}${body}</section>`;
  }

  function errorPanel(error, suggestion) {
    return panel({
      title: "Could not do that",
      body: `<div>${esc(error)}</div>${suggestion ? `<div class="dim" style="margin-top:8px">${esc(suggestion)}</div>` : ""}`,
      extra: "error-panel",
    });
  }

  function listPanel(title, items) {
    if (!items || !items.length) return "";
    return panel({
      title,
      body: `<ul class="bullets">${items.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`,
    });
  }

  function counterBlock(items, kind) {
    if (!items || !items.length) return "";
    const title = kind === "snapshot" ? "What this does not tell you" : "Worth knowing";
    if (state.depth === "beginner") {
      return card({
        kicker: title,
        extra: "warn-card",
        body: items
          .map((c) => `<div class="story-body"><strong>${esc(c.label)}.</strong> ${esc(c.detail)}</div>`)
          .join(""),
      });
    }
    return panel({
      title: kind === "snapshot" ? "What this does not tell you" : "Competing explanations",
      body: items
        .map((c) => `<div class="counter"><div class="counter-label">${esc(c.label)}</div>
                     <div class="counter-detail">${esc(c.detail)}</div></div>`)
        .join(""),
    });
  }

  /* ----------------------------------------------- attribution rendering */

  /** The honest version of "this factor matched N% of the time".
   *
   * Two things go wrong if this is written as a single fixed sentence. A
   * factor can clear the significance bar by moving *opposite* to the price,
   * in which case "this tells you something" reads as agreement when the
   * numbers say the reverse. And rounding to tenths can collapse 62% and 55%
   * into the same phrase, so the sentence would claim a gap the reader
   * cannot see. Percentages come back whenever the plain phrasing collides.
   */
  function factorSentence(f, ticker) {
    if (f.hit_rate == null || f.base_rate == null) return "";
    const collided = outOfTen(f.hit_rate) === outOfTen(f.base_rate);
    const hit = collided ? rate(f.hit_rate) : outOfTen(f.hit_rate);
    const base = collided ? rate(f.base_rate) : outOfTen(f.base_rate);
    const verdict = f.lift > 0
      ? "that is more often than usual, so this one does carry some information."
      : `that is <em>less</em> often than usual — when this happens, ${esc(ticker)} has tended to go the other way.`;
    return `On days that looked like today, ${esc(ticker)} moved the same way
      <strong>${hit}</strong>, against ${base} on an ordinary day. In other words ${verdict}`;
  }

  /** Beginner: one idea per card, prose first, a picture for every number. */
  function renderStory(view) {
    const o = view.observation;
    const strong = view.factors.filter((f) => f.beats);
    const weak = view.factors.filter((f) => !f.beats);
    let html = "";

    if (o) {
      const verb = o.direction === "up" ? "went up" : o.direction === "down" ? "went down" : "barely moved";
      const unusual = o.unusual
        ? "and it was a bigger move than it usually has in a day."
        : "by about as much as it usually does in a day.";
      html += card({
        body: `
          <div class="story-lead">${esc(view.ticker)} ${verb} ${o.return_pct == null ? "" : Math.abs(o.return_pct).toFixed(2) + "%"} — ${unusual}</div>
          <div class="story-body" style="margin-top:14px">
            On a normal day ${esc(view.ticker)} drifts about ${pct(o.daily_vol_pct, 2)}.
            ${o.unusual
              ? "Today was well outside that, which is the kind of day worth looking into."
              : "Moves this size happen constantly and usually have no particular explanation at all."}
          </div>
          <div id="move-bars" style="margin-top:16px"></div>`,
      });
    }

    if (!o || !o.unusual) {
      html += card({
        kicker: "The short answer",
        body: `<div class="story-body">Probably nothing in particular. Prices wander on their own,
               and a move this size is ordinary wandering. The things below happened on the same
               day — that is all anyone can actually see.</div>`,
      });
    }

    if (strong.length) {
      html += card({
        kicker: "What else was happening",
        extra: "good-card",
        body: strong
          .map(
            (f) => `<div class="factor-story">
              <div class="factor-name">${esc(f.label)}</div>
              <div class="story-body">${esc(f.what_happened)}.</div>
              <div class="story-body" style="margin-top:8px">${factorSentence(f, view.ticker)}</div>
              <div class="cmp" data-hit="${f.hit_rate}" data-base="${f.base_rate}"
                   data-label="${esc(view.ticker)}" style="margin-top:12px"></div>
            </div>`
          )
          .join(""),
      });
    }

    if (weak.length) {
      html += card({
        kicker: "Things that turned out not to matter",
        body: `<div class="story-body">${weak.length} other ${weak.length === 1 ? "thing" : "things"}
               happened too, but ${weak.length === 1 ? "its" : "their"} track record is no better than
               a coin flip, so ${weak.length === 1 ? "it isn't" : "they aren't"} worth reading into.</div>
          <button class="collapse-toggle" data-collapse="weak-factors" style="margin-top:10px">Show them anyway</button>
          <div class="collapse-body" id="weak-factors" hidden>
            ${weak.map((f) => `<div class="story-body" style="margin-top:8px">
              <strong>${esc(f.label)}</strong> — ${esc(f.what_happened)}. Matched
              ${outOfTen(f.hit_rate)} against a normal ${outOfTen(f.base_rate)}.</div>`).join("")}
          </div>`,
      });
    }

    if (view.macro && view.macro.length) {
      html += card({
        kicker: "Economic news that day",
        body: view.macro.map((m) => `<div class="story-body">${esc(m.description)}</div>`).join(""),
      });
    }

    html += counterBlock(view.counterevidence, view.kind);
    html += card({
      kicker: "Before you go",
      body: `<div class="story-body">Nothing here proves one thing made the other happen.
             These are things that happened at the same time, and how often that has
             lined up before. That is genuinely all anyone can measure.</div>`,
    });
    return `<div class="story">${html}</div>`;
  }

  /** Intermediate and analyst: the dense table view. */
  function renderTable(view) {
    const o = view.observation;
    let html = "";

    if (o) {
      html += panel({
        title: "What happened",
        chip: o.unusual ? '<span class="chip warn">unusual move</span>'
                        : '<span class="chip">within normal range</span>',
        body: `
          <div class="headline">
            <span class="ticker">${esc(view.ticker)}</span>
            <span class="delta ${o.direction}" id="headline-delta">--</span>
            <span class="headline-meta">${esc(view.period)}</span>
          </div>
          <p class="summary-text">${esc(o.summary)}</p>
          <div class="stat-grid">
            <div class="stat"><div class="stat-label">Move</div><div class="stat-value">${signed(o.return_pct)}</div></div>
            <div class="stat"><div class="stat-label">Typical day</div><div class="stat-value">${pct(o.daily_vol_pct, 2)}</div></div>
            <div class="stat"><div class="stat-label">Std deviations</div><div class="stat-value">${o.sigma == null ? "n/a" : (o.sigma >= 0 ? "+" : "") + o.sigma.toFixed(2)}</div><div class="stat-note">${esc(o.label)}</div></div>
            <div class="stat"><div class="stat-label">Close</div><div class="stat-value">${o.close == null ? "n/a" : o.close.toFixed(2)}</div></div>
          </div>`,
      });
    }

    const factorTable = (factors, title, note, extra) => {
      if (!factors || !factors.length) return "";
      return panel({
        title, extra: extra || "",
        body: `${note ? `<p class="dim">${esc(note)}</p>` : ""}
          <table><thead><tr>
            <th>Factor</th><th class="num">Observed</th><th class="num">Match</th>
            <th class="num">Base</th><th class="num">Lift</th><th class="num">Days</th><th>Reading</th>
          </tr></thead><tbody>${factors.map((f) => {
            const width = f.hit_rate == null ? 0 : Math.max(0, Math.min(100, f.hit_rate * 100));
            const basePos = f.base_rate == null ? 0 : Math.max(0, Math.min(100, f.base_rate * 100));
            return `<tr>
              <td><div>${esc(f.label)}</div>
                  <div class="dim" style="font-size:11.5px">${esc(f.what_happened)}</div>
                  ${f.hit_rate == null ? "" : `<div class="rate-track">
                     <div class="rate-fill ${f.beats ? "beats" : ""}" data-width="${width}"></div>
                     <div class="rate-base" style="left:${basePos}%"></div></div>`}</td>
              <td class="num">${f.observed == null ? "n/a" : (f.observed >= 0 ? "+" : "") + f.observed.toFixed(2) + esc(f.units || "")}</td>
              <td class="num">${rate(f.hit_rate)}</td>
              <td class="num dim">${rate(f.base_rate)}</td>
              <td class="num">${f.lift == null ? "n/a" : rate(f.lift, 1)}</td>
              <td class="num dim">${f.sample}</td>
              <td><span class="chip ${f.beats ? (f.lift > 0 ? "good" : "bad") : ""}">${esc(f.reading)}</span></td>
            </tr>`;
          }).join("")}</tbody></table>`,
      });
    };

    html += factorTable(view.factors, "What else happened",
      `Ranked by historical record over ${view.history_years || 4} years, not by how convincing it sounds.`, "accent");
    html += factorTable(view.mechanical, "What the move was made of",
      "Parts of the symbol itself — high match rates here are arithmetic, not evidence.");
    if (view.macro && view.macro.length) {
      html += listPanel("Economic data released around this day", view.macro.map((m) => m.description));
    }
    html += counterBlock(view.counterevidence, view.kind);
    html += listPanel("Limitations", view.limitations);
    html += listPanel("Assumptions made", view.warnings);
    return html;
  }

  function renderSnapshot(view) {
    const f = view.fundamentals;
    let html = "";
    if (state.depth === "beginner" && f) {
      html += card({
        body: `<div class="story-lead">${esc(f.name || view.ticker)} is worth about ${money(f.market_cap)} in total.</div>
          <div class="story-body" style="margin-top:12px">
            That is what the whole company costs at today's share price.
            ${f.profitable === false
              ? "It currently loses money, which changes how every number below should be read."
              : ""}
          </div>`,
      });
      html += card({
        kicker: "The numbers, in plain words",
        body: f.metrics.map((m) => `<div class="factor-story">
            <div class="factor-name">${esc(m.label)} — ${m.format === "rate" ? rate(m.value, 1) : ratio(m.value)}</div>
            ${m.gloss ? `<div class="story-body">${esc(m.gloss)}.</div>` : ""}
          </div>`).join(""),
      });
    } else if (f) {
      html += panel({
        title: "The numbers",
        chip: f.sector ? `<span class="chip">${esc(f.sector)}</span>` : "",
        body: `<div class="headline"><span class="ticker">${esc(view.ticker)}</span>
            <span class="headline-meta">${esc(f.name || "")} · ${money(f.market_cap)}</span></div>
          <div class="stat-grid">${f.metrics.map((m) => `<div class="stat">
            <div class="stat-label">${esc(m.label)}</div>
            <div class="stat-value">${m.format === "rate" ? rate(m.value, 2) : ratio(m.value)}</div>
            ${m.gloss ? `<div class="stat-note">${esc(m.gloss)}</div>` : ""}</div>`).join("")}</div>`,
      });
    }

    if (view.peers && view.peers.length) {
      html += panel({
        title: "Compared with similar companies",
        body: `<div id="peer-bars"></div>`,
      });
    }
    if (view.trend) {
      html += panel({
        title: "Price history",
        body: `<p class="summary-text">${esc(view.trend.summary)}</p>
               <div id="price-chart" style="min-height:240px"></div>`,
      });
    }
    if (view.news && view.news.length) {
      html += panel({
        title: "Recent headlines",
        chip: '<span class="chip">titles only, not read</span>',
        body: `<ul class="bullets">${view.news.map((n) =>
          `<li>${esc(n.title)} <span class="dim">— ${esc(n.publisher)}</span></li>`).join("")}</ul>`,
      });
    }
    html += counterBlock(view.counterevidence, view.kind);
    html += listPanel("Limitations", view.limitations);
    return html;
  }

  function renderAsk(payload) {
    const view = payload.view;
    const out = $("ask-out");
    let html = "";
    if (view.synthetic) {
      html += panel({
        title: "Offline mode", chip: '<span class="chip bad">synthetic</span>',
        body: "<div>These figures are generated, not real market data.</div>",
        extra: "error-panel",
      });
    }
    html += view.kind === "snapshot"
      ? renderSnapshot(view)
      : (state.depth === "beginner" ? renderStory(view) : renderTable(view));
    out.innerHTML = html;
    state.lastTicker = view.ticker;

    requestAnimationFrame(() => {
      out.querySelectorAll(".rate-fill").forEach((b) => (b.style.width = `${b.dataset.width}%`));
      const delta = $("headline-delta");
      if (delta && view.observation) {
        countUp(delta, view.observation.return_pct, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
      }
      // Beginner comparison bars.
      const moveBars = $("move-bars");
      if (moveBars && view.observation) {
        Charts.compare(moveBars, { rows: [
          { label: "today", value: Math.abs(view.observation.return_pct || 0),
            display: signed(view.observation.return_pct), highlight: true },
          { label: "a normal day", value: view.observation.daily_vol_pct || 0,
            display: pct(view.observation.daily_vol_pct, 2) },
        ]});
      }
      out.querySelectorAll(".cmp").forEach((node) => {
        const hit = parseFloat(node.dataset.hit), base = parseFloat(node.dataset.base);
        Charts.compare(node, { rows: [
          { label: "days like today", value: hit * 100, display: outOfTen(hit), highlight: true },
          { label: "any random day", value: base * 100, display: outOfTen(base) },
        ]});
      });
      out.querySelectorAll("[data-collapse]").forEach((button) => {
        button.addEventListener("click", () => {
          const target = $(button.dataset.collapse);
          const open = !target.hidden;
          target.hidden = open;
          button.textContent = open ? "Show them anyway" : "Hide";
        });
      });
      if (view.kind === "snapshot") {
        if (view.trend) drawPriceChart(view.ticker);
        const peers = $("peer-bars");
        if (peers && view.peers) {
          Charts.compare(peers, {
            rows: view.peers.filter((p) => p.subject != null && p.median != null).map((p) => ({
              label: p.label,
              value: Math.abs(p.subject) * (p.is_rate ? 100 : 1),
              reference: Math.abs(p.median) * (p.is_rate ? 100 : 1),
              display: p.is_rate ? rate(p.subject, 1) : ratio(p.subject),
              highlight: true,
            })),
          });
        }
      }
    });

    const v = payload.validation;
    setStatus(`${v.matched}/${v.checked} figures traced to the evidence${v.ok ? "" : " — CHECK FAILED"}`,
              v.ok ? "ok" : "bad");
  }

  async function drawPriceChart(ticker) {
    const container = $("price-chart");
    if (!container) return;
    const result = await window.pywebview.api.price_series(ticker, 365);
    if (!result.ok) { container.innerHTML = `<span class="dim">${esc(result.error)}</span>`; return; }
    Charts.line(container, { values: result.values, dates: result.dates, format: (v) => v.toFixed(0) });
  }

  /* ---------------------------------------------------------- backtests */

  function statsPanel(r) {
    return panel({
      title: "Statistics",
      body: `<table><thead><tr><th>Measure</th><th class="num">This rule</th>
        <th class="num">Buy and hold</th></tr></thead><tbody>
        ${Object.keys(r.stats).map((k) => `<tr><td>${esc(k)}</td>
          <td class="num">${esc(r.stats[k])}</td>
          <td class="num dim">${esc(r.benchmark_stats[k] ?? "")}</td></tr>`).join("")}
        </tbody></table>`,
    });
  }

  function curvePanel(r, title) {
    const ahead = r.excess_pct > 0;
    return panel({
      title, extra: "accent",
      chip: `<span class="chip ${ahead ? "good" : "warn"}">${ahead ? "ahead of" : "behind"} buy and hold</span>`,
      body: `<div class="headline">
          <span class="ticker">${esc(r.ticker)}</span>
          <span class="delta ${ahead ? "up" : "down"}" id="bt-delta">--</span>
          <span class="headline-meta">${esc(r.name || r.strategy || "your rule")}</span>
        </div>
        <div id="bt-chart" style="min-height:250px"></div>
        <div class="legend">
          <span><span class="legend-swatch" style="background:var(--accent)"></span>This rule</span>
          <span><span class="legend-swatch" style="background:var(--text-faint)"></span>Buy and hold</span>
        </div>
        <div id="bt-dd" style="min-height:140px"></div>
        <p class="dim">How far it fell below its own best point.</p>`,
    });
  }

  function renderBacktest(r, extraHtml = "") {
    const out = $("lab-out");
    let html = "";
    if (r.synthetic) {
      html += panel({ title: "Offline mode", chip: '<span class="chip bad">synthetic</span>',
        body: "<div>These prices are generated, not real.</div>", extra: "error-panel" });
    }
    html += extraHtml;
    html += curvePanel(r, "Out-of-sample result");
    html += statsPanel(r);
    if (r.strategy_notes && r.strategy_notes.length) {
      html += listPanel("What to keep in mind", r.strategy_notes);
    }
    out.innerHTML = html;

    requestAnimationFrame(() => {
      Charts.line($("bt-chart"), {
        values: r.equity, benchmark: r.benchmark, dates: r.dates,
        format: (v) => v.toFixed(0), height: 250,
      });
      Charts.drawdown($("bt-dd"), { values: r.equity, dates: r.dates });
      const d = $("bt-delta");
      if (d) countUp(d, r.excess_pct, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} pts`);
      const payoff = $("payoff-chart");
      if (payoff && r.payoff) Charts.payoff(payoff, { spot: r.spot, points: r.payoff });
    });
    setStatus("Tested on data the rule never saw while being chosen", "ok");
  }

  /* --------------------------------------------------------------- flows */

  async function runAsk(query) {
    const text = (query || "").trim();
    if (!text || state.busy) return;
    goStage("working");
    showView("ask");
    $("query").value = text;
    setBusy(true, "Gathering evidence…");
    try {
      const result = await window.pywebview.api.research(text, state.depth);
      if (!result.ok) {
        $("ask-out").innerHTML = errorPanel(result.error, result.suggestion);
        setStatus("Could not answer that", "bad");
      } else {
        renderAsk(result);
      }
    } finally { setBusy(false); }
  }

  async function runLab() {
    if (state.busy) return;
    const ticker = ($("lab-ticker").value || state.lastTicker || "SPY").trim().toUpperCase();
    const years = parseFloat($("lab-years").value);
    setBusy(true, "Running…");
    try {
      if (state.labMode === "build") {
        const text = $("strategy-text").value.trim();
        if (!text) { $("lab-out").innerHTML = errorPanel("Describe a strategy first.",
            "Try: buy when the 50 day average crosses above the 200 day"); return; }
        const r = await window.pywebview.api.custom_backtest(text, ticker, years, state.depth);
        if (!r.ok) { $("lab-out").innerHTML = errorPanel(r.error, r.suggestion); setStatus("Could not run that", "bad"); return; }
        let head = panel({
          title: "Your rule", extra: "accent",
          body: `<div class="story-body"><code>${esc(r.expression)}</code></div>
                 <div class="story-body" style="margin-top:10px">${esc(r.explanation)}</div>
                 <div class="story-body" style="margin-top:10px">${esc(r.behaviour.summary)}</div>
                 ${r.behaviour.warnings.map((w) => `<div class="story-body" style="color:var(--warn);margin-top:10px">${esc(w)}</div>`).join("")}
                 ${r.notes.map((n) => `<div class="dim" style="margin-top:8px">${esc(n)}</div>`).join("")}`,
        });
        renderBacktest(r, head);
      } else if (state.labMode === "preset") {
        const strategy = state.selectedPreset || "ma-crossover";
        const r = await window.pywebview.api.backtest(ticker, strategy, years);
        if (!r.ok) { $("lab-out").innerHTML = errorPanel(r.error, r.suggestion); return; }
        renderBacktest(r, "");
        const chosen = state.strategies.find((s) => s.id === strategy);
        if (chosen && chosen.has_surface) drawSurface(ticker, strategy, years);
      } else if (state.labMode === "options") {
        const strategy = state.selectedPreset || "covered-call";
        const r = await window.pywebview.api.option_backtest(ticker, strategy, Math.min(years, 10));
        if (!r.ok) { $("lab-out").innerHTML = errorPanel(r.error, r.suggestion); return; }
        const head = panel({
          title: "What this strategy does", extra: "accent",
          body: `<div class="story-body">${esc(r.caveat)}</div>
                 <div id="payoff-chart" style="min-height:230px;margin-top:14px"></div>
                 <p class="dim">What you'd make or lose at expiry, at each possible price.
                    Strike ${r.strike == null ? "" : r.strike.toFixed(2)}, now ${r.spot == null ? "" : r.spot.toFixed(2)}.</p>`,
        });
        renderBacktest(r, head);
      } else {
        const r = await window.pywebview.api.quant(ticker);
        if (!r.ok) { $("lab-out").innerHTML = errorPanel(r.error, r.suggestion); return; }
        renderQuant(r);
      }
    } finally { setBusy(false); }
  }

  async function drawSurface(ticker, strategy, years) {
    const out = $("lab-out");
    const holder = document.createElement("div");
    holder.innerHTML = panel({
      title: "Does the setting matter?",
      body: `<div class="dim">Measuring…</div>`,
    });
    out.appendChild(holder.firstElementChild);
    const r = await window.pywebview.api.parameter_surface(ticker, strategy, Math.min(years, 10));
    const target = out.lastElementChild.querySelector(".panel-body");
    if (!r.ok) { target.innerHTML = `<div class="dim">${esc(r.error)}</div>`; return; }
    target.innerHTML = `
      <p class="summary-text">${esc(r.verdict)}</p>
      <div id="surface-chart" style="min-height:350px"></div>
      <div class="legend"><span>Height and brightness are the risk-adjusted result.
        The dot marks the best cell.</span></div>
      <div class="stat-grid" style="margin-top:14px">
        <div class="stat"><div class="stat-label">${esc(r.x_name)} (best)</div><div class="stat-value">${r.best ? r.best.x : "n/a"}</div></div>
        <div class="stat"><div class="stat-label">${esc(r.y_name)} (best)</div><div class="stat-value">${r.best ? r.best.y : "n/a"}</div></div>
        <div class="stat"><div class="stat-label">Best Sharpe</div><div class="stat-value">${ratio(r.best_sharpe)}</div></div>
        <div class="stat"><div class="stat-label">Ruggedness</div><div class="stat-value">${ratio(r.ruggedness)}</div>
          <div class="stat-note">low means nearby settings agree</div></div>
      </div>
      ${r.notes.map((n) => `<div class="dim" style="margin-top:8px">${esc(n)}</div>`).join("")}`;
    requestAnimationFrame(() => {
      Charts.surface($("surface-chart"), {
        z: r.sharpe, xValues: r.x_values, yValues: r.y_values,
        xLabel: r.x_name, yLabel: r.y_name, best: r.best,
      });
    });
  }

  function renderQuant(r) {
    const v = r.volatility, g = r.regime, m = r.mean_reversion;
    $("lab-out").innerHTML = `
      ${panel({ title: "Experimental", chip: '<span class="chip alpha">alpha</span>',
        body: `<p class="summary-text">${esc(r.notice)}</p>`, extra: "alpha" })}
      ${v ? panel({ title: "How much it moves", chip: `<span class="chip">${esc(v.regime)}</span>`,
        body: `<div class="stat-grid">
          <div class="stat"><div class="stat-label">Recent</div><div class="stat-value">${pct(v.ewma)}</div>
            <div class="stat-note">weighted toward the last few weeks</div></div>
          <div class="stat"><div class="stat-label">Using the full bar</div><div class="stat-value">${pct(v.yang_zhang)}</div>
            <div class="stat-note">Yang-Zhang, the most efficient estimator</div></div>
          <div class="stat"><div class="stat-label">Forecast (${v.horizon}d)</div><div class="stat-value">${pct(v.forecast)}</div>
            <div class="stat-note">${esc(v.model)}</div></div>
          <div class="stat"><div class="stat-label">Long run</div><div class="stat-value">${pct(v.long_run)}</div>
            <div class="stat-note">where the model says it settles</div></div>
        </div>` }) : ""}
      ${g ? panel({ title: "Does it trend or drift back?", chip: `<span class="chip">${esc(g.classification)}</span>`,
        body: `<p class="summary-text">${esc(g.describe)}</p>
          <div class="stat-grid analyst-only">
            <div class="stat"><div class="stat-label">Hurst</div><div class="stat-value">${ratio(g.hurst, 3)}</div>
              <div class="stat-note">± ${ratio(g.hurst_error, 3)} · 0.5 is a random walk</div></div>
            <div class="stat"><div class="stat-label">Variance ratio</div><div class="stat-value">${ratio(g.variance_ratio, 3)}</div></div>
            <div class="stat"><div class="stat-label">VR z</div><div class="stat-value">${ratio(g.variance_ratio_z, 2)}</div>
              <div class="stat-note">beyond ±1.96 is significant</div></div>
          </div>` }) : ""}
      ${m ? panel({ title: "Does it come back to average?",
        chip: `<span class="chip ${m.reverting ? "" : "warn"}">${m.reverting ? "yes" : "no"}</span>`,
        body: `<p class="summary-text">${esc(m.describe)}</p>` }) : ""}
      ${r.failures.length ? listPanel("Models that did not run", r.failures) : ""}`;
    setStatus("Descriptions of the market, not signals", "");
  }

  /* --------------------------------------------------------- explore tab */

  async function runSearch(query) {
    const results = $("explore-results");
    if (!query.trim()) { results.innerHTML = ""; return; }
    const r = await window.pywebview.api.search(query, 7);
    if (!r.ok || !r.results.length) {
      results.innerHTML = `<div class="dim">No matches for “${esc(query)}”.</div>`;
      return;
    }
    results.innerHTML = r.results.map((hit, i) =>
      `<button class="search-hit" data-symbol="${esc(hit.symbol)}" style="animation-delay:${i * 30}ms">
         <span class="search-symbol">${esc(hit.symbol)}</span>
         <span class="search-name">${esc(hit.name)}</span>
         <span class="chip">${esc(hit.kind.toLowerCase())}</span>
       </button>`).join("");
    results.querySelectorAll("[data-symbol]").forEach((button) => {
      button.addEventListener("click", () => showChart(button.dataset.symbol));
    });
  }

  async function showChart(symbol) {
    setBusy(true, `Loading ${symbol}…`);
    try {
      const days = parseInt($("explore-range").value, 10);
      const r = await window.pywebview.api.price_bars(symbol, days);
      if (!r.ok) { $("explore-out").innerHTML = errorPanel(r.error, r.suggestion); return; }
      state.lastTicker = symbol;
      $("explore-out").innerHTML = panel({
        title: "Price", extra: "accent",
        chip: r.synthetic ? '<span class="chip bad">synthetic</span>' : "",
        body: `<div class="headline">
            <span class="ticker">${esc(r.ticker)}</span>
            <span class="delta ${r.change_pct >= 0 ? "up" : "down"}" id="ex-delta">--</span>
            <span class="headline-meta">over this period</span>
          </div>
          <div id="candle-chart" style="min-height:330px"></div>
          <div class="stat-grid" style="margin-top:14px">
            <div class="stat"><div class="stat-label">Last</div><div class="stat-value">${ratio(r.last)}</div></div>
            <div class="stat"><div class="stat-label">High</div><div class="stat-value">${ratio(r.high)}</div></div>
            <div class="stat"><div class="stat-label">Low</div><div class="stat-value">${ratio(r.low)}</div></div>
          </div>
          <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap">
            <button class="ghost small" data-act="ask">Why did it move?</button>
            <button class="ghost small" data-act="snapshot">Is it worth a look?</button>
            <button class="ghost small" data-act="lab">Test a strategy on it</button>
          </div>`,
      });
      requestAnimationFrame(() => {
        Charts.candles($("candle-chart"), { bars: r.bars });
        const d = $("ex-delta");
        if (d) countUp(d, r.change_pct, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`);
      });
      $("explore-out").querySelectorAll("[data-act]").forEach((b) => {
        b.addEventListener("click", () => {
          const act = b.dataset.act;
          if (act === "ask") runAsk(`why did ${symbol} move today`);
          else if (act === "snapshot") runAsk(`is ${symbol} good to invest`);
          else { $("lab-ticker").value = symbol; showView("lab"); }
        });
      });
      setStatus(`${r.bars.length} sessions`, "ok");
    } finally { setBusy(false); }
  }

  /* ------------------------------------------------------------ settings */

  function renderSettings() {
    $("settings-out").innerHTML = `
      ${panel({ title: "Theme", body: `<div class="theme-grid">${THEMES.map(([id, name, note]) =>
        `<button class="theme-card" data-theme-id="${id}"
                 aria-pressed="${document.documentElement.dataset.theme === id}">
           <div class="theme-name">${esc(name)}</div>
           <div class="theme-note">${esc(note)}</div>
           <div class="theme-swatches" data-swatch="${id}"></div>
         </button>`).join("")}</div>` })}
      ${panel({ title: "API keys", body:
        `<p class="summary-text">Stored in a file, not an environment variable — an app opened
         from Finder doesn't inherit your shell.</p>
         ${state.credentials.map((c) => `<div class="field-row">
            <label for="key-${esc(c.name)}">${esc(c.name)}</label>
            <input type="text" id="key-${esc(c.name)}"
              placeholder="${c.present ? "set (…" + esc(c.hint) + ")" : "not set"}" />
            <button data-save-key="${esc(c.name)}">Save</button>
          </div><div class="dim" style="margin-left:158px">${esc(c.unlocks)}</div>`).join("")}` })}
      ${panel({ title: "About", body:
        `<p class="summary-text">This reports what coincided with a price move and how often that
         has coincided before. It does not identify causes and does not recommend transactions.</p>
         <p class="dim">Every figure is checked against the underlying evidence before it is shown.</p>` })}`;

    // Paint each theme's own colours into its swatch strip.
    THEMES.forEach(([id]) => {
      const holder = document.querySelector(`[data-swatch="${id}"]`);
      if (!holder) return;
      const probe = document.createElement("div");
      probe.dataset.theme = id;
      probe.style.display = "none";
      document.body.appendChild(probe);
      const styles = getComputedStyle(probe);
      ["--bg", "--surface-2", "--accent", "--good", "--bad"].forEach((token) => {
        const swatch = document.createElement("span");
        swatch.className = "theme-swatch";
        swatch.style.background = styles.getPropertyValue(token).trim() || "#888";
        holder.appendChild(swatch);
      });
      probe.remove();
    });

    document.querySelectorAll("[data-theme-id]").forEach((button) => {
      button.addEventListener("click", () => { applyTheme(button.dataset.themeId); renderSettings(); });
    });
    document.querySelectorAll("[data-save-key]").forEach((button) => {
      button.addEventListener("click", async () => {
        const name = button.dataset.saveKey;
        const value = $(`key-${name}`).value.trim();
        if (!value) return;
        const r = await window.pywebview.api.save_key(name, value);
        if (r.ok) { setStatus(`${name} saved`, "ok"); await loadBootstrap(); renderSettings(); }
        else setStatus(r.error, "bad");
      });
    });
  }

  /* ---------------------------------------------------------- navigation */

  function goStage(stage) {
    state.stage = stage;
    document.documentElement.dataset.stage = stage;
  }

  function showView(name) {
    state.view = name;
    document.querySelectorAll(".tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.view === name)));
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("active", v.id === `view-${name}`));
    if (name === "settings") renderSettings();
    if (name === "lab" && !$("lab-ticker").value) $("lab-ticker").value = state.lastTicker;
  }

  function setLabMode(mode) {
    state.labMode = mode;
    state.selectedPreset = null;
    document.querySelectorAll(".lab-tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.lab === mode)));
    ["build", "preset", "options", "diagnose"].forEach((key) => {
      $(`lab-${key}`).hidden = key !== mode;
    });
    if (mode === "preset") renderPresets($("preset-grid"), state.strategies, (s) =>
      ({ id: s.id, name: s.id.replace(/-/g, " "), desc: s.description, alpha: s.alpha }));
    if (mode === "options") renderPresets($("option-grid"), state.optionStrategies, (s) =>
      ({ id: s.id, name: s.name, desc: s.description }));
  }

  function renderPresets(container, items, map) {
    container.innerHTML = items.map((raw) => {
      const s = map(raw);
      return `<button class="preset" data-preset="${esc(s.id)}" aria-pressed="false">
        <div class="preset-name">${esc(s.name)}${s.alpha ? ' <span class="chip alpha">alpha</span>' : ""}</div>
        <div class="preset-desc">${esc(s.desc)}</div></button>`;
    }).join("");
    container.querySelectorAll("[data-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedPreset = button.dataset.preset;
        container.querySelectorAll("[data-preset]").forEach((b) =>
          b.setAttribute("aria-pressed", String(b === button)));
        runLab();
      });
    });
  }

  function applyTheme(name) {
    document.documentElement.dataset.theme = name;
    try { localStorage.setItem("research.theme", name); } catch (_) { /* blocked storage */ }
  }

  function applyDepth(depth) {
    state.depth = depth;
    document.documentElement.dataset.depth = depth;
    try { localStorage.setItem("research.depth", depth); } catch (_) { /* blocked storage */ }
    document.querySelectorAll("[data-depth-option]").forEach((b) =>
      b.setAttribute("aria-checked", String(b.dataset.depthOption === depth)));
  }

  function buildDepthSwitch(container) {
    container.innerHTML = DEPTHS.map(([id, label]) =>
      `<button class="depth-option" role="radio" data-depth-option="${id}"
               aria-checked="${id === state.depth}">${esc(label)}</button>`).join("");
    container.querySelectorAll("[data-depth-option]").forEach((button) => {
      button.addEventListener("click", () => {
        applyDepth(button.dataset.depthOption);
        buildDepthSwitch($("depth-switch"));
        buildDepthSwitch($("home-depth"));
        // Re-render so the layout, not just the wording, follows the level.
        if (state.stage === "working" && state.view === "ask" && $("query").value.trim()) {
          runAsk($("query").value);
        }
      });
    });
  }

  function strategyHelp() {
    return `<p>Write it in plain words, or as a rule. Both end up in the same place.</p>
      <table style="margin-top:10px">
        <tr><td class="dim">buy when the 50 day average crosses above the 200 day</td>
            <td><code>sma(50) &gt; sma(200)</code></td></tr>
        <tr><td class="dim">buy when RSI is below 30</td><td><code>rsi(14) &lt; 30</code></td></tr>
        <tr><td class="dim">golden cross</td><td><code>sma(50) &gt; sma(200)</code></td></tr>
        <tr><td class="dim">buy the dip</td><td><code>zscore(63) &lt; -1.5</code></td></tr>
      </table>
      <p style="margin-top:12px">Available: ${state.indicators.map((i) => `<code>${esc(i.name)}(n)</code>`).join(", ")},
         plus <code>close</code>. Combine with <code>and</code>, <code>or</code>, and comparisons.</p>`;
  }

  async function loadBootstrap() {
    const info = await window.pywebview.api.bootstrap();
    state.strategies = info.strategies;
    state.optionStrategies = info.option_strategies || [];
    state.indicators = info.indicators || [];
    state.credentials = info.credentials;
    $("mode-badge").textContent = info.mock ? "OFFLINE — synthetic data" : "";
  }

  function wire() {
    $("home-go").addEventListener("click", () => runAsk($("home-query").value));
    $("home-query").addEventListener("keydown", (e) => {
      if (e.key === "Enter") runAsk($("home-query").value);
    });
    $("go-home").addEventListener("click", () => {
      goStage("home");
      $("home-query").value = "";
      $("home-query").focus();
    });
    $("run").addEventListener("click", () => runAsk($("query").value));
    $("query").addEventListener("keydown", (e) => { if (e.key === "Enter") runAsk($("query").value); });

    document.querySelectorAll(".tab").forEach((tab) =>
      tab.addEventListener("click", () => showView(tab.dataset.view)));
    document.querySelectorAll(".lab-tab").forEach((tab) =>
      tab.addEventListener("click", () => setLabMode(tab.dataset.lab)));

    $("lab-run").addEventListener("click", runLab);
    $("lab-ticker").addEventListener("keydown", (e) => { if (e.key === "Enter") runLab(); });
    $("strategy-help").addEventListener("click", () => {
      const box = $("strategy-help-box");
      box.hidden = !box.hidden;
      if (!box.hidden) box.innerHTML = strategyHelp();
    });

    // Feedback as you type, debounced: understanding what was written is a
    // different failure from the strategy performing badly.
    let typingTimer = null;
    $("strategy-text").addEventListener("input", () => {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(async () => {
        const text = $("strategy-text").value.trim();
        const node = $("strategy-feedback");
        if (!text) { node.textContent = ""; return; }
        const r = await window.pywebview.api.interpret_strategy(text, state.depth);
        node.textContent = r.ok ? `Reads as: ${r.expression}` : r.error;
        node.style.color = r.ok ? "var(--good)" : "var(--warn)";
      }, 420);
    });

    $("explore-query").addEventListener("input", (e) => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => runSearch(e.target.value), 260);
    });
    $("explore-range").addEventListener("change", () => {
      if (state.lastTicker) showChart(state.lastTicker);
    });

    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        const box = state.stage === "home" ? $("home-query") : $("query");
        box.focus(); box.select();
      }
    });
  }

  ready().then(async () => {
    let theme = "deep-field", depth = "beginner";
    try {
      theme = localStorage.getItem("research.theme") || theme;
      depth = localStorage.getItem("research.depth") || depth;
    } catch (_) { /* storage unavailable; defaults stand */ }
    applyTheme(theme);
    state.depth = depth;
    applyDepth(depth);

    $("home-examples").innerHTML = EXAMPLES.map((q) =>
      `<button class="example" data-example="${esc(q)}">${esc(q)}</button>`).join("");
    document.querySelectorAll("[data-example]").forEach((b) =>
      b.addEventListener("click", () => runAsk(b.dataset.example)));

    buildDepthSwitch($("depth-switch"));
    buildDepthSwitch($("home-depth"));
    wire();
    await loadBootstrap();
    setLabMode("build");
    goStage("home");
    $("home-query").focus();
    setStatus("Ready");
  });
})();
