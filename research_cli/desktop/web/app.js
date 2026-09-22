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
    // Whether anything has been asked yet. The home screen is a dead end
    // until something has: there is no earlier place to go back to.
    worked: false,
    lastBars: null,
    candlesStale: false,
    options: {
      ticker: null, expiry: null, chain: null, contract: null,
      side: "buy", contracts: 1, account: null, timer: null,
    },
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

  /* The typefaces offered in Settings. The ids match `data-font` in
   * theme.css, which owns the stacks and any adjustment a face needs; this
   * list only says what each one is called and why you might want it. */
  const FONTS = [
    ["system", "System Mono", "The default. Hinted by the OS, readable for hours."],
    ["classic", "Classic Terminal", "Monaco and Courier — the older machine."],
    ["pixel", "8-Bit Terminal", "Press Start 2P. Square pixels, no antialiasing."],
    ["book", "Book", "Proportional, not monospaced. The easiest to read at length."],
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
      state.lastBars = r.bars;
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
            <button class="ghost small" data-act="options">Trade options on it</button>
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
          else if (act === "options") { showView("options"); loadOptionExpiries(symbol); }
          else { $("lab-ticker").value = symbol; showView("lab"); }
        });
      });
      setStatus(`${r.bars.length} sessions`, "ok");
    } finally { setBusy(false); }
  }

  /* --------------------------------------------------------- options desk */
  /*
   * A brokerage screen, and it is organised the way one is: the chain is the
   * document, and the ticket, the greeks and the blotter are all views onto
   * whichever contract is selected in it.
   *
   * Everything shown as money comes from the Python side, including the cost
   * of an order, which is quoted by the account before it is placed rather
   * than multiplied out here. An order ticket that disagrees with the fill by
   * a commission is the sort of detail that destroys trust in a paper desk,
   * and the only way to guarantee it cannot happen is to have one
   * implementation of the arithmetic.
   */

  const OPTION_REFRESH_MS = 45_000;

  const usd = (value, decimals = 2) => {
    if (value == null) return "n/a";
    const body = Math.abs(value).toLocaleString(undefined, {
      minimumFractionDigits: decimals, maximumFractionDigits: decimals,
    });
    return `${value < 0 ? "-" : ""}$${body}`;
  };
  const signedUsd = (value, decimals = 2) =>
    value == null ? "n/a" : `${value >= 0 ? "+" : "-"}$${Math.abs(value).toLocaleString(undefined,
      { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
  const num = (value, decimals = 2) => (value == null ? "—" : value.toFixed(decimals));
  const ivText = (value) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);
  const countText = (value) => (value == null ? "—" : value.toLocaleString());

  function optionsBusy(busy, message) {
    $("scanline").classList.toggle("active", busy);
    if (message !== undefined) setStatus(message);
  }

  /** Load the expiries for a symbol, then the chain for one of them. */
  async function loadOptionExpiries(ticker, keepExpiry) {
    const symbol = (ticker || "").trim().toUpperCase();
    if (!symbol) return;
    state.options.ticker = symbol;
    $("opt-ticker").value = symbol;
    optionsBusy(true, `Loading ${symbol} chains…`);
    try {
      const r = await window.pywebview.api.option_expiries(symbol);
      if (!r.ok) {
        $("options-out").querySelector("#opt-chain .chain-error")?.remove();
        $("opt-chain").innerHTML =
          `<tbody><tr><td class="chain-error">${esc(r.error)} ${esc(r.suggestion || "")}</td></tr></tbody>`;
        setStatus(r.error, "bad");
        return;
      }
      const select = $("opt-expiry");
      select.innerHTML = r.expiries.map((e) =>
        `<option value="${esc(e.date)}">${esc(e.label)} · ${e.days}d</option>`).join("");
      // Default to the first expiry at least three weeks out: the front week
      // is mostly noise and decays too fast to learn anything from.
      const preferred = keepExpiry && r.expiries.some((e) => e.date === keepExpiry)
        ? keepExpiry
        : (r.expiries.find((e) => e.days >= 21) || r.expiries[r.expiries.length - 1] || {}).date;
      if (preferred) select.value = preferred;
      state.options.expiry = select.value;
      await loadChain();
    } finally { optionsBusy(false); }
  }

  async function loadChain() {
    const { ticker, expiry } = state.options;
    if (!ticker || !expiry) return;
    optionsBusy(true, `Loading ${ticker} ${expiry}…`);
    try {
      const r = await window.pywebview.api.option_chain(ticker, expiry);
      if (!r.ok) {
        $("opt-chain").innerHTML =
          `<tbody><tr><td class="chain-error">${esc(r.error)}</td></tr></tbody>`;
        setStatus(r.error, "bad");
        return;
      }
      state.options.chain = r;
      renderChain(r);
      if (state.options.contract) {
        const { strike, kind } = state.options.contract;
        selectContract(strike, kind, { quiet: true });
      }
      await refreshAccount();
      setStatus(
        `${r.ticker} ${usd(r.spot)} · ${r.days_to_expiry} days to expiry`,
        r.synthetic ? "bad" : "ok",
      );
    } finally { optionsBusy(false); }
  }

  function renderChain(chain) {
    const quoted = new Date(chain.quoted_at);
    $("opt-chain-chip") .innerHTML = chain.synthetic
      ? '<span class="chip bad">synthetic chain</span>'
      : `<span class="chip">quoted ${quoted.toLocaleTimeString()}</span>`;
    $("opt-chain-head").innerHTML = `
      <div class="chain-spot">
        <span class="ticker">${esc(chain.ticker)}</span>
        <span class="chain-price">${usd(chain.spot)}</span>
        <span class="headline-meta">${esc(chain.expiry)} · ${chain.days_to_expiry} days</span>
      </div>
      ${chain.warnings.map((w) => `<div class="dim chain-warning">${esc(w)}</div>`).join("")}`;

    const cell = (row, kind, field, text, extra = "") =>
      `<td class="side ${kind}${extra}" data-strike="${row.strike}" data-kind="${kind}"
           data-field="${field}">${text}</td>`;

    const body = chain.rows.map((row) => {
      const call = row.call || {};
      const put = row.put || {};
      const atm = Math.abs(row.strike - chain.atm_strike) < 1e-6;
      const callItm = call.in_the_money ? " itm" : "";
      const putItm = put.in_the_money ? " itm" : "";
      return `<tr class="chain-row${atm ? " atm" : ""}">
        ${cell(row, "call", "oi", countText(call.open_interest), callItm)}
        ${cell(row, "call", "vol", countText(call.volume), callItm)}
        ${cell(row, "call", "delta", num(call.delta, 2), callItm)}
        ${cell(row, "call", "iv", ivText(call.iv), callItm)}
        ${cell(row, "call", "bid", num(call.bid), callItm + " price")}
        ${cell(row, "call", "ask", num(call.ask), callItm + " price")}
        <td class="strike-col">${row.strike.toFixed(2)}</td>
        ${cell(row, "put", "bid", num(put.bid), putItm + " price")}
        ${cell(row, "put", "ask", num(put.ask), putItm + " price")}
        ${cell(row, "put", "iv", ivText(put.iv), putItm)}
        ${cell(row, "put", "delta", num(put.delta, 2), putItm)}
        ${cell(row, "put", "vol", countText(put.volume), putItm)}
        ${cell(row, "put", "oi", countText(put.open_interest), putItm)}
      </tr>`;
    }).join("");

    $("opt-chain").innerHTML = `
      <thead>
        <tr class="chain-sides">
          <th colspan="6">Calls</th><th class="strike-col">Strike</th><th colspan="6">Puts</th>
        </tr>
        <tr>
          <th>OI</th><th>Vol</th><th>Δ</th><th>IV</th><th>Bid</th><th>Ask</th>
          <th class="strike-col"></th>
          <th>Bid</th><th>Ask</th><th>IV</th><th>Δ</th><th>Vol</th><th>OI</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>`;

    $("opt-chain").querySelectorAll("[data-strike]").forEach((node) => {
      node.addEventListener("click", () =>
        selectContract(parseFloat(node.dataset.strike), node.dataset.kind));
    });

    // Open centred on the money rather than at the top of a hundred strikes,
    // which is where a chain is always read outward from. Scrolled by hand
    // rather than with scrollIntoView, which walks every ancestor and drags
    // the page itself down, pulling the account strip off the top.
    const scroller = $("opt-chain").closest(".chain-scroll");
    const atmRow = $("opt-chain").querySelector(".chain-row.atm");
    if (scroller && atmRow) {
      const frame = scroller.getBoundingClientRect();
      const row = atmRow.getBoundingClientRect();
      scroller.scrollTop += (row.top - frame.top) - (frame.height - row.height) / 2;
    }
  }

  async function selectContract(strike, kind, { quiet = false } = {}) {
    const { ticker, expiry } = state.options;
    if (!quiet) optionsBusy(true, "Pricing…");
    try {
      const r = await window.pywebview.api.option_contract(ticker, expiry, strike, kind);
      if (!r.ok) { setStatus(r.error, "bad"); return; }
      state.options.contract = { strike, kind, detail: r };
      renderContract(r);
    } finally { if (!quiet) optionsBusy(false); }
  }

  function renderContract(detail) {
    const panel = $("opt-contract-panel");
    panel.hidden = false;
    const q = detail.contract;
    $("opt-contract-title").textContent = detail.label;
    $("opt-contract-chip").innerHTML = detail.synthetic
      ? '<span class="chip bad">synthetic</span>'
      : `<span class="chip">${detail.days_to_expiry} days left</span>`;

    const greekTile = (label, value, note) =>
      `<div class="stat"><div class="stat-label">${esc(label)}</div>
        <div class="stat-value">${value}</div>
        <div class="stat-note">${esc(note)}</div></div>`;

    $("opt-contract-body").innerHTML = `
      <div class="stat-grid tiles-six">
        ${greekTile("Bid", num(q.bid), "what you would be paid")}
        ${greekTile("Ask", num(q.ask), "what it costs to buy")}
        ${greekTile("Spread", q.spread_pct == null ? "—" : `${(q.spread_pct * 100).toFixed(1)}%`,
          "of the midpoint, paid on the round trip")}
        ${greekTile("Implied vol", ivText(detail.iv), detail.iv_source || "")}
        ${greekTile("Break even", num(detail.break_even), "underlying, at expiry")}
        ${greekTile("Leverage", detail.leverage == null ? "—" : `${detail.leverage.toFixed(1)}x`,
          "move per 1% in the underlying")}
      </div>

      <div class="ticket" id="opt-ticket">
        <div class="ticket-sides" role="group" aria-label="Side">
          <button class="ticket-side" data-side="buy" aria-pressed="${state.options.side === "buy"}">Buy</button>
          <button class="ticket-side sell" data-side="sell" aria-pressed="${state.options.side === "sell"}">Sell</button>
        </div>
        <div class="ticket-size">
          <button class="step" data-step="-1" aria-label="One fewer contract">−</button>
          <input type="number" id="opt-contracts" min="1" max="500" step="1"
                 value="${state.options.contracts}" aria-label="Contracts" />
          <button class="step" data-step="1" aria-label="One more contract">+</button>
          <span class="dim">contracts</span>
        </div>
        <span class="spacer"></span>
        <div class="ticket-cost" id="opt-preview"></div>
        <button class="primary" id="opt-place">Place order</button>
      </div>
      ${detail.note ? `<div class="dim">${esc(detail.note)}</div>` : ""}

      ${detail.curves ? `
      <div class="stat-grid tiles-six">
        ${greekTile("Delta", num(q.delta, 3), "shares of exposure, per share")}
        ${greekTile("Gamma", num(q.gamma, 4), "delta gained per $1 move")}
        ${greekTile("Theta", num(q.theta, 3), "lost per day, per share")}
        ${greekTile("Vega", num(q.vega, 3), "per point of implied vol")}
        ${greekTile("Rho", num(q.rho, 3), "per point of interest rate")}
        ${greekTile("One contract", usd(detail.cost_per_contract), "at the ask, before commission")}
      </div>

      <div class="greek-grid">
        <figure><figcaption>Value against the underlying
          <span class="dim">— now, and the dashed line at expiry</span></figcaption>
          <div id="gc-price"></div></figure>
        <figure><figcaption>Delta <span class="dim">— how much it moves with the stock</span></figcaption>
          <div id="gc-delta"></div></figure>
        <figure><figcaption>Gamma <span class="dim">— how fast delta itself changes</span></figcaption>
          <div id="gc-gamma"></div></figure>
        <figure><figcaption>Theta <span class="dim">— decay per day</span></figcaption>
          <div id="gc-theta"></div></figure>
        <figure><figcaption>Vega <span class="dim">— sensitivity to implied volatility</span></figcaption>
          <div id="gc-vega"></div></figure>
        <figure><figcaption>Rho <span class="dim">— sensitivity to interest rates</span></figcaption>
          <div id="gc-rho"></div></figure>
      </div>` : ""}

      ${detail.history ? `
      <div class="compare-block">
        <div class="legend">
          <span><span class="legend-swatch" style="background:var(--accent)"></span>This contract</span>
          <span><span class="legend-swatch" style="background:var(--text-faint)"></span>${esc(detail.ticker)}</span>
        </div>
        <div id="opt-history" style="min-height:200px"></div>
        <p class="dim">Both indexed to 100 six months ago, which is the only way two
           series this far apart in size can share an axis. The option's line is
           <strong>modelled</strong> — there is no free source of historical option
           quotes, so each day is priced from that day's close at today's implied
           volatility. The level is an estimate; the point is how much harder the
           option moves than the stock.</p>
      </div>` : ""}`;

    wireTicket();
    if (detail.curves) requestAnimationFrame(() => drawGreekCurves(detail));
    if (detail.history) requestAnimationFrame(() => {
      Charts.line($("opt-history"), {
        values: detail.history.option_indexed,
        benchmark: detail.history.underlying_indexed,
        dates: detail.history.dates,
        format: (v) => v.toFixed(0),
        height: 200,
      });
    });
  }

  function drawGreekCurves(detail) {
    const c = detail.curves;
    const spot = detail.spot;
    const xFormat = (v) => v.toFixed(0);
    Charts.curve($("gc-price"), {
      xs: c.spots, ys: c.price, second: c.expiry, marker: spot, xFormat,
      format: (v) => v.toFixed(1), fill: true,
    });
    [["delta", 2], ["gamma", 3], ["theta", 2], ["vega", 2], ["rho", 2]].forEach(([name, places]) => {
      Charts.curve($(`gc-${name}`), {
        xs: c.spots, ys: c[name], marker: spot, xFormat,
        format: (v) => v.toFixed(places),
      });
    });
  }

  function wireTicket() {
    document.querySelectorAll("#opt-ticket [data-side]").forEach((button) => {
      button.addEventListener("click", () => {
        state.options.side = button.dataset.side;
        document.querySelectorAll("#opt-ticket [data-side]").forEach((b) =>
          b.setAttribute("aria-pressed", String(b.dataset.side === state.options.side)));
        previewOrder();
      });
    });
    document.querySelectorAll("#opt-ticket [data-step]").forEach((button) => {
      button.addEventListener("click", () => {
        const box = $("opt-contracts");
        box.value = Math.max(1, (parseInt(box.value, 10) || 1) + parseInt(button.dataset.step, 10));
        state.options.contracts = parseInt(box.value, 10);
        previewOrder();
      });
    });
    $("opt-contracts").addEventListener("input", () => {
      state.options.contracts = Math.max(1, parseInt($("opt-contracts").value, 10) || 1);
      previewOrder();
    });
    $("opt-place").addEventListener("click", placeOrder);
    previewOrder();
  }

  /** Ask the account what the order would do, before anyone commits to it. */
  async function previewOrder() {
    const selected = state.options.contract;
    const node = $("opt-preview");
    if (!selected || !node) return;
    const { ticker, expiry, side, contracts } = state.options;
    const r = await window.pywebview.api.option_preview(
      ticker, expiry, selected.strike, selected.kind, side, contracts);
    if (!r.ok) { node.innerHTML = `<span class="bad-text">${esc(r.error)}</span>`; return; }

    const debit = r.cash_effect < 0;
    node.innerHTML = `
      <div class="cost-line">
        <span class="${debit ? "bad-text" : "good-text"}">${signedUsd(r.cash_effect)}</span>
        <span class="dim">${debit ? "debit" : "credit"} at ${num(r.fill_price)} a share</span>
      </div>
      <div class="dim cost-detail">
        ${r.contracts} × 100 × ${num(r.fill_price)} + ${usd(r.commission)} commission
        ${r.collateral ? ` · ${usd(r.collateral)} held as collateral` : ""}
      </div>
      ${r.affordable ? "" : '<div class="bad-text">Not enough buying power.</div>'}`;
    $("opt-place").disabled = !r.affordable;
  }

  async function placeOrder() {
    const selected = state.options.contract;
    if (!selected) return;
    const { ticker, expiry, side, contracts } = state.options;
    optionsBusy(true, "Placing…");
    try {
      const r = await window.pywebview.api.option_order(
        ticker, expiry, selected.strike, selected.kind, side, contracts);
      if (!r.ok) { setStatus(r.error, "bad"); return; }
      renderAccount(r.account);
      setStatus(
        `${side === "buy" ? "Bought" : "Sold"} ${r.filled.contracts} ${r.filled.label} `
        + `at ${num(r.filled.price)} · ${signedUsd(r.filled.cash_effect)}`,
        "ok",
      );
      previewOrder();
    } finally { optionsBusy(false); }
  }

  async function refreshAccount() {
    const r = await window.pywebview.api.option_account();
    if (r.ok) renderAccount(r);
  }

  function renderAccount(account) {
    state.options.account = account;

    $("opt-account-strip").innerHTML = `
      <div class="account-figure"><span class="dim">Equity</span><strong>${usd(account.equity)}</strong></div>
      <div class="account-figure"><span class="dim">Buying power</span><strong>${usd(account.buying_power)}</strong></div>
      <div class="account-figure"><span class="dim">Open P&L</span>
        <strong class="${(account.unrealized || 0) >= 0 ? "good-text" : "bad-text"}">${signedUsd(account.unrealized)}</strong></div>
      <div class="account-figure"><span class="dim">Realized</span>
        <strong class="${(account.realized || 0) >= 0 ? "good-text" : "bad-text"}">${signedUsd(account.realized)}</strong></div>`;

    const exposure = account.exposure || {};
    $("opt-exposure-chip").innerHTML = account.positions.length
      ? `<span class="chip">delta ${num(exposure.delta, 0)} · theta ${signedUsd(exposure.theta)}/day`
        + ` · vega ${num(exposure.vega, 1)}</span>`
      : "";

    if (!account.positions.length) {
      $("opt-positions").innerHTML = `<p class="dim">Nothing open. Click a bid or an ask in the
        chain above to build an order — you start with ${usd(account.starting_cash, 0)}.</p>`;
    } else {
      $("opt-positions").innerHTML = `
        <table class="blotter"><thead><tr>
          <th>Contract</th><th class="num">Qty</th><th class="num">Avg</th><th class="num">Mark</th>
          <th class="num">Delta</th><th class="num">Theta/day</th><th class="num">Days</th>
          <th class="num">Open P&L</th><th></th>
        </tr></thead><tbody>
        ${account.positions.map((p) => `<tr>
          <td>${esc(p.label)}${p.assignment_risk
            ? ' <span class="chip warn" title="Short and in the money: a real account can be assigned at any time">assignment risk</span>'
            : ""}</td>
          <td class="num">${p.contracts > 0 ? "+" : ""}${p.contracts}</td>
          <td class="num">${num(p.average_price)}</td>
          <td class="num">${num(p.mark)}</td>
          <td class="num">${num(p.delta, 2)}</td>
          <td class="num">${p.theta == null ? "—" : signedUsd(p.theta * p.contracts * 100)}</td>
          <td class="num">${p.days_to_expiry}</td>
          <td class="num ${(p.unrealized || 0) >= 0 ? "good-text" : "bad-text"}">${signedUsd(p.unrealized)}</td>
          <td class="num"><button class="ghost small" data-close="${esc(p.id)}">Close</button></td>
        </tr>`).join("")}
        </tbody></table>
        ${account.collateral_held
          ? `<p class="dim">${usd(account.collateral_held)} of cash is held as collateral against
             short positions and cannot be used for anything else.</p>` : ""}
        ${account.stale.length
          ? `<p class="dim">No live quote for ${account.stale.map(esc).join(", ")}; marked at what
             they would settle for today, or at cost where even the underlying
             could not be priced.</p>` : ""}`;

      $("opt-positions").querySelectorAll("[data-close]").forEach((button) =>
        button.addEventListener("click", () => closePosition(button.dataset.close)));
    }

    const settled = account.expiries_settled || [];
    const ledger = account.ledger || [];
    $("opt-activity").innerHTML = `
      ${settled.length ? `<div class="settled">${settled.map((e) =>
        `<div class="story-body">${esc(e.position)} ${esc(e.outcome)} —
         ${signedUsd(e.realized)}, against ${esc(e.underlying.toFixed(2))}.</div>`).join("")}</div>` : ""}
      ${ledger.length ? `<table class="blotter"><tbody>
        ${ledger.map((entry) => `<tr>
          <td class="dim">${esc(new Date(entry.at).toLocaleString())}</td>
          <td>${esc(entry.note)}</td>
          <td class="num ${entry.cash_effect >= 0 ? "good-text" : "bad-text"}">${signedUsd(entry.cash_effect)}</td>
        </tr>`).join("")}</tbody></table>`
        : '<p class="dim">Nothing has happened yet.</p>'}
      <p class="dim">Fills cross the spread — you buy at the ask and sell at the bid — and
         commission is ${usd(0.65)} a contract each way, both of which a simulator that
         used the midpoint would hide from you.</p>`;
  }

  async function closePosition(id) {
    optionsBusy(true, "Closing…");
    try {
      const r = await window.pywebview.api.option_close(id);
      if (!r.ok) { setStatus(r.error, "bad"); return; }
      renderAccount(r.account);
      setStatus(`Closed ${r.closed.label} · ${signedUsd(r.closed.realized)} realized`,
                r.closed.realized >= 0 ? "ok" : "bad");
    } finally { optionsBusy(false); }
  }

  async function resetOptionAccount() {
    const r = await window.pywebview.api.option_reset();
    if (r.ok) { renderAccount(r.account); setStatus("Paper account reset", "ok"); }
  }

  /** Keep the screen live while it is the one being looked at. */
  function optionsPolling(on) {
    clearInterval(state.options.timer);
    state.options.timer = null;
    if (!on) return;
    state.options.timer = setInterval(() => {
      if (state.view !== "options" || document.hidden || state.busy) return;
      loadChain();
    }, OPTION_REFRESH_MS);
  }

  async function openOptionsDesk() {
    if (!state.options.ticker) {
      await loadOptionExpiries(state.lastTicker || "AAPL");
    } else {
      await refreshAccount();
    }
    optionsPolling(true);
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
      ${panel({ title: "Typeface", body: `<div class="theme-grid">${FONTS.map(([id, name, note]) =>
        `<button class="theme-card" data-font-id="${id}"
                 aria-pressed="${document.documentElement.dataset.font === id}">
           <div class="theme-name">${esc(name)}</div>
           <div class="theme-note">${esc(note)}</div>
           <div class="font-sample" data-face="${id}">AAPL 182.40 +1.2%</div>
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
    document.querySelectorAll("[data-font-id]").forEach((button) => {
      button.addEventListener("click", () => { applyFont(button.dataset.fontId); renderSettings(); });
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

  function activateView(name) {
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("active", v.id === `view-${name}`));
  }

  /** Move between the home screen and the working interface.
   *
   * The home screen is a view like any other, so entering the stage has to
   * activate it. Changing `data-stage` alone only hid the title bar, which
   * left whichever tab was open still on screen with nothing to navigate
   * by: the tabs were gone, home had not appeared, and the only way out was
   * to ask another question. Leaving the stage restores the tab you were on.
   */
  function goStage(stage) {
    state.stage = stage;
    document.documentElement.dataset.stage = stage;
    if (stage !== "home") state.worked = true;
    activateView(stage === "home" ? "home" : state.view);
  }

  function showView(name) {
    state.view = name;
    document.querySelectorAll(".tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.view === name)));
    if (state.stage === "home") goStage("working");
    else activateView(name);
    if (name === "settings") renderSettings();
    if (name === "explore" && state.candlesStale) requestAnimationFrame(redrawCandles);
    if (name === "lab" && !$("lab-ticker").value) $("lab-ticker").value = state.lastTicker;
    // The desk polls for quotes, so it only runs while it is the tab on screen.
    if (name === "options") openOptionsDesk();
    else optionsPolling(false);
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

  function applyFont(name) {
    document.documentElement.dataset.font = name;
    try { localStorage.setItem("research.font", name); } catch (_) { /* blocked storage */ }
    // Charts set their axis labels in the computed font, and an SVG does not
    // reflow the way text does, so the drawing has to be made again.
    requestAnimationFrame(redrawCandles);
  }

  /** Draw the candle chart again, or mark it as owed a redraw.
   *
   * A chart inside a hidden tab measures zero wide, and drawing into that
   * falls back to a default width -- producing an SVG built for a container
   * it was never in, which is then stretched to fit when the tab is shown.
   * Anything that invalidates the chart while it is off screen leaves this
   * flag behind instead, and `showView` settles it on the way in.
   */
  function redrawCandles() {
    const node = $("candle-chart");
    if (!node || !state.lastBars) return;
    if (!node.clientWidth) { state.candlesStale = true; return; }
    state.candlesStale = false;
    Charts.candles(node, { bars: state.lastBars });
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

    $("opt-ticker").addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadOptionExpiries($("opt-ticker").value);
    });
    $("opt-expiry").addEventListener("change", () => {
      state.options.expiry = $("opt-expiry").value;
      state.options.contract = null;
      $("opt-contract-panel").hidden = true;
      loadChain();
    });
    $("opt-refresh").addEventListener("click", () => loadChain());
    $("opt-reset").addEventListener("click", resetOptionAccount);

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
      // Going home is one click from anywhere; this is the way back, for a
      // brand button pressed by accident with an answer still behind it.
      if (e.key === "Escape" && state.stage === "home" && state.worked) goStage("working");
    });
  }

  ready().then(async () => {
    let theme = "deep-field", depth = "beginner", font = "system";
    try {
      theme = localStorage.getItem("research.theme") || theme;
      depth = localStorage.getItem("research.depth") || depth;
      font = localStorage.getItem("research.font") || font;
    } catch (_) { /* storage unavailable; defaults stand */ }
    applyTheme(theme);
    applyFont(font);
    state.depth = depth;
    applyDepth(depth);

    buildDepthSwitch($("depth-switch"));
    wire();
    await loadBootstrap();
    setLabMode("build");
    goStage("home");
    $("home-query").focus();
    setStatus("Ready");
  });
})();
