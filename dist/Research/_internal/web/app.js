/* Interface logic: talk to Python, render the result, never invent a number.
 *
 * Every figure on screen arrives from the Python bridge, which got it from
 * an EvidenceBundle that the validator has already checked. Nothing here
 * computes a statistic -- formatting a percentage is the most arithmetic
 * this file is allowed to do. A number calculated in JavaScript would sit
 * outside the validator's reach, which is the one guarantee the whole tool
 * rests on.
 */

(() => {
  "use strict";

  const state = { depth: "beginner", busy: false, strategies: [], credentials: [] };
  const $ = (id) => document.getElementById(id);

  /* ---------------------------------------------------------------- utils */

  /** Wait for pywebview to attach its bridge before calling anything. */
  function ready() {
    return new Promise((resolve) => {
      if (window.pywebview && window.pywebview.api) return resolve();
      window.addEventListener("pywebviewready", () => resolve(), { once: true });
    });
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : `${v.toFixed(d)}%`);
  const signed = (v, d = 2) => (v === null || v === undefined ? "n/a" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`);
  const rate = (v, d = 0) => (v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(d)}%`);
  const ratio = (v, d = 2) => (v === null || v === undefined ? "n/a" : v.toFixed(d));

  function money(value) {
    if (value === null || value === undefined) return "n/a";
    const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"]];
    for (const [size, suffix] of units) {
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
    node.textContent = text;
    node.className = kind === "ok" ? "status-ok" : kind === "bad" ? "status-bad" : "";
  }

  /**
   * Count a number up to its value.
   *
   * Only for headline figures. A settling number draws the eye once and
   * then holds still; doing it to every cell in a table would be a
   * fairground.
   */
  function countUp(node, target, format, duration = 620) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      node.textContent = format(target);
      return;
    }
    const start = performance.now();
    function frame(now) {
      const t = Math.min((now - start) / duration, 1);
      // easeOutCubic: fast then settling, which reads as arriving rather
      // than as a slot machine.
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = format(target * eased);
      if (t < 1) requestAnimationFrame(frame);
      else node.textContent = format(target);
    }
    requestAnimationFrame(frame);
  }

  function panel({ title, chip, body, extra = "" }) {
    return `
      <section class="panel ${extra}">
        <div class="panel-head">
          <span class="panel-title">${escapeHtml(title)}</span>
          ${chip || ""}
        </div>
        <div class="panel-body">${body}</div>
      </section>`;
  }

  function errorPanel(error, suggestion) {
    return panel({
      title: "Could not answer that",
      body: `<div>${escapeHtml(error)}</div>${
        suggestion ? `<div class="dim">${escapeHtml(suggestion)}</div>` : ""
      }`,
      extra: "error-panel",
    });
  }

  /* ------------------------------------------------------------ research */

  function renderObservation(view) {
    const o = view.observation;
    if (!o) return "";
    const direction = o.direction === "up" ? "up" : o.direction === "down" ? "down" : "flat";
    return panel({
      title: view.kind === "snapshot" ? "Overview" : "What happened",
      chip: o.unusual
        ? '<span class="chip warn">unusual move</span>'
        : '<span class="chip">within normal range</span>',
      body: `
        <div class="headline">
          <span class="ticker">${escapeHtml(view.ticker)}</span>
          <span class="delta ${direction}" id="headline-delta">--</span>
          <span class="headline-meta">${escapeHtml(view.period)}</span>
        </div>
        <p class="summary-text">${escapeHtml(o.summary)}</p>
        <div class="stat-grid">
          <div class="stat">
            <div class="stat-label">Move</div>
            <div class="stat-value">${signed(o.return_pct)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Typical day</div>
            <div class="stat-value">${pct(o.daily_vol_pct, 2)}</div>
            ${o.gloss ? `<div class="stat-note">${escapeHtml(o.gloss)}</div>` : ""}
          </div>
          <div class="stat">
            <div class="stat-label">Standard deviations</div>
            <div class="stat-value">${o.sigma === null ? "n/a" : (o.sigma >= 0 ? "+" : "") + o.sigma.toFixed(2)}</div>
            <div class="stat-note">${escapeHtml(o.label)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Close</div>
            <div class="stat-value">${o.close === null ? "n/a" : o.close.toFixed(2)}</div>
          </div>
        </div>`,
    });
  }

  function factorRows(factors) {
    return factors
      .map((f) => {
        const hit = f.hit_rate;
        const base = f.base_rate;
        const width = hit === null ? 0 : Math.max(0, Math.min(100, hit * 100));
        const basePos = base === null ? 0 : Math.max(0, Math.min(100, base * 100));
        const chipClass = f.beats ? (f.lift > 0 ? "good" : "bad") : "";
        return `
          <tr>
            <td>
              <div class="factor-label">${escapeHtml(f.label)}</div>
              <div class="factor-what">${escapeHtml(f.what_happened)}</div>
              ${
                hit === null
                  ? ""
                  : `<div class="rate-track" title="Match rate ${rate(hit)} vs base rate ${rate(base)}">
                       <div class="rate-fill ${f.beats ? "beats" : ""}" data-width="${width}"></div>
                       <div class="rate-base" style="left:${basePos}%"></div>
                     </div>`
              }
            </td>
            <td class="num">${f.observed === null ? "n/a" : (f.observed >= 0 ? "+" : "") + f.observed.toFixed(2) + escapeHtml(f.units || "")}</td>
            <td class="num">${rate(hit)}</td>
            <td class="num dim">${rate(base)}</td>
            <td class="num">${f.lift === null ? "n/a" : rate(f.lift, 1)}</td>
            <td class="num dim">${f.sample}</td>
            <td><span class="chip ${chipClass}">${escapeHtml(f.reading)}</span></td>
          </tr>`;
      })
      .join("");
  }

  function factorTable(factors, title, note, extra) {
    if (!factors || factors.length === 0) return "";
    return panel({
      title,
      body: `
        ${note ? `<p class="dim">${escapeHtml(note)}</p>` : ""}
        <table>
          <thead>
            <tr>
              <th>Factor</th><th class="num">Observed</th><th class="num">Match</th>
              <th class="num">Base</th><th class="num">Lift</th>
              <th class="num">Days</th><th>Reading</th>
            </tr>
          </thead>
          <tbody>${factorRows(factors)}</tbody>
        </table>`,
      extra: extra || "",
    });
  }

  function listPanel(title, items, klass) {
    if (!items || items.length === 0) return "";
    return panel({
      title,
      body: `<ul class="bullets">${items
        .map((t) => `<li>${escapeHtml(t)}</li>`)
        .join("")}</ul>`,
      extra: klass || "",
    });
  }

  function counterPanel(items, kind) {
    if (!items || items.length === 0) return "";
    return panel({
      title: kind === "snapshot" ? "What this does not tell you" : "Competing explanations",
      body: items
        .map(
          (c) => `<div class="counter">
                    <div class="counter-label">${escapeHtml(c.label)}</div>
                    <div class="counter-detail">${escapeHtml(c.detail)}</div>
                  </div>`
        )
        .join(""),
    });
  }

  function snapshotPanels(view) {
    let html = "";
    const f = view.fundamentals;
    if (f) {
      html += panel({
        title: "The numbers",
        chip: f.sector ? `<span class="chip">${escapeHtml(f.sector)}</span>` : "",
        body: `
          <div class="headline">
            <span class="ticker">${escapeHtml(view.ticker)}</span>
            <span class="headline-meta">${escapeHtml(f.name || "")} · ${money(f.market_cap)}</span>
          </div>
          ${
            f.profitable === false
              ? '<p class="summary-text">This company currently loses money, which changes how every figure below should be read.</p>'
              : ""
          }
          <div class="stat-grid">
            ${f.metrics
              .map(
                (m) => `<div class="stat">
                  <div class="stat-label">${escapeHtml(m.label)}</div>
                  <div class="stat-value">${
                    m.format === "rate" ? rate(m.value, 2) : ratio(m.value)
                  }</div>
                  ${m.gloss ? `<div class="stat-note">${escapeHtml(m.gloss)}</div>` : ""}
                </div>`
              )
              .join("")}
          </div>`,
      });
    }

    if (view.peers && view.peers.length) {
      html += panel({
        title: "Compared with similar companies",
        body: `<table>
          <thead><tr><th>Measure</th><th class="num">${escapeHtml(view.ticker)}</th>
            <th class="num">Typical peer</th><th class="num">Difference</th></tr></thead>
          <tbody>${view.peers
            .map(
              (p) => `<tr>
                <td>${escapeHtml(p.label)}</td>
                <td class="num">${p.is_rate ? rate(p.subject, 1) : ratio(p.subject)}</td>
                <td class="num dim">${p.is_rate ? rate(p.median, 1) : ratio(p.median)}</td>
                <td class="num">${p.vs_median_pct === null ? "n/a" : signed(p.vs_median_pct, 0)}</td>
              </tr>`
            )
            .join("")}</tbody></table>`,
      });
    }

    if (view.trend) {
      const t = view.trend;
      html += panel({
        title: "Price history",
        body: `
          <p class="summary-text">${escapeHtml(t.summary)}</p>
          <div id="price-chart" style="min-height:220px;"></div>
          <div class="stat-grid">
            <div class="stat"><div class="stat-label">1 month</div><div class="stat-value">${signed(t.return_1m, 1)}</div></div>
            <div class="stat"><div class="stat-label">3 months</div><div class="stat-value">${signed(t.return_3m, 1)}</div></div>
            <div class="stat"><div class="stat-label">1 year</div><div class="stat-value">${signed(t.return_1y, 1)}</div></div>
            <div class="stat"><div class="stat-label">Volatility</div><div class="stat-value">${pct(t.vol)}</div></div>
          </div>`,
      });
    }

    if (view.news && view.news.length) {
      html += panel({
        title: "Recent headlines",
        chip: '<span class="chip">titles only, not read</span>',
        body: `<ul class="bullets">${view.news
          .map(
            (n) =>
              `<li>${escapeHtml(n.title)} <span class="dim">— ${escapeHtml(n.publisher)}${
                n.published ? " · " + escapeHtml(n.published) : ""
              }</span></li>`
          )
          .join("")}</ul>`,
      });
    }
    return html;
  }

  function renderResearch(payload) {
    const view = payload.view;
    const out = $("research-out");
    let html = "";

    if (view.synthetic) {
      html += panel({
        title: "Offline mode",
        chip: '<span class="chip bad">synthetic</span>',
        body: '<div>These figures are generated, not real market data.</div>',
        extra: "error-panel",
      });
    }

    if (view.kind === "snapshot") {
      html += snapshotPanels(view);
    } else {
      html += renderObservation(view);
      html += factorTable(
        view.factors,
        "What else happened",
        "Ranked by how often each has coincided with moves like this one, not by how convincing it sounds.",
        "accent"
      );
      html += factorTable(
        view.mechanical,
        "What the move was made of",
        "These are parts of the symbol itself, so their high match rates are arithmetic, not evidence about why."
      );
      if (view.macro && view.macro.length) {
        html += listPanel("Economic data released around this day", view.macro.map((m) => m.description));
      }
    }

    html += counterPanel(view.counterevidence, view.kind);
    html += listPanel("Limitations", view.limitations);
    html += listPanel("Assumptions made", view.warnings);

    html += panel({
      title: "Reminder",
      body: `<p class="summary-text">${
        view.kind === "snapshot"
          ? `This has not said whether to own ${escapeHtml(view.ticker)}. How long you plan to hold, what else you own, and what you would do if it fell by half all change the answer, and none of them appear above.`
          : "This is a summary of evidence, not investment advice. Nothing above identifies what made the move happen; every factor listed is something that happened at the same time."
      }</p>`,
    });

    out.innerHTML = html;

    // Animate the bars in after layout, so the transition has a start value.
    requestAnimationFrame(() => {
      out.querySelectorAll(".rate-fill").forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      });
      const delta = $("headline-delta");
      if (delta && view.observation && view.observation.return_pct !== null) {
        countUp(delta, view.observation.return_pct, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
      }
    });

    if (view.kind === "snapshot" && view.trend) drawPriceChart(view.ticker);

    const v = payload.validation;
    setStatus(
      `${v.matched}/${v.checked} figures traced to the evidence${v.ok ? "" : " — VALIDATION FAILED"}`,
      v.ok ? "ok" : "bad"
    );
  }

  async function drawPriceChart(ticker) {
    const container = $("price-chart");
    if (!container) return;
    const result = await window.pywebview.api.price_series(ticker, 365);
    if (!result.ok) {
      container.innerHTML = `<span class="dim">Chart unavailable: ${escapeHtml(result.error)}</span>`;
      return;
    }
    Charts.line(container, {
      values: result.values, dates: result.dates, format: (v) => v.toFixed(0),
    });
  }

  /* ------------------------------------------------------------ backtest */

  function renderBacktest(r) {
    const rows = Object.keys(r.stats);
    const ahead = r.excess_pct > 0;
    $("backtest-out").innerHTML = `
      ${
        r.synthetic
          ? panel({ title: "Offline mode", chip: '<span class="chip bad">synthetic</span>',
                    body: "<div>These prices are generated, not real.</div>", extra: "error-panel" })
          : ""
      }
      ${panel({
        title: "Walk-forward result",
        chip: `<span class="chip ${ahead ? "good" : "warn"}">${
          ahead ? "ahead of" : "behind"
        } buy and hold</span>${r.alpha ? '<span class="chip alpha">alpha</span>' : ""}`,
        body: `
          <div class="headline">
            <span class="ticker">${escapeHtml(r.ticker)}</span>
            <span class="delta ${ahead ? "up" : "down"}" id="bt-delta">--</span>
            <span class="headline-meta">${escapeHtml(r.strategy)} · ${r.folds} out-of-sample periods</span>
          </div>
          <div id="bt-chart" style="min-height:240px;"></div>
          <div class="legend">
            <span><span class="legend-swatch" style="background:var(--accent)"></span>This rule</span>
            <span><span class="legend-swatch" style="background:var(--text-faint)"></span>Buy and hold</span>
          </div>
          <div id="bt-dd" style="min-height:130px;"></div>
          <p class="dim">Fall from the running peak.</p>`,
        extra: "accent",
      })}
      ${panel({
        title: "Statistics",
        body: `<table>
          <thead><tr><th>Measure</th><th class="num">This rule</th><th class="num">Buy and hold</th></tr></thead>
          <tbody>${rows
            .map(
              (k) => `<tr><td>${escapeHtml(k)}</td>
                      <td class="num">${escapeHtml(r.stats[k])}</td>
                      <td class="num dim">${escapeHtml(r.benchmark_stats[k] ?? "")}</td></tr>`
            )
            .join("")}</tbody></table>`,
      })}
      ${panel({
        title: "What this means",
        body: `
          <p class="summary-text">It beat buying and holding in ${r.folds_won} of ${r.folds} out-of-sample periods.</p>
          <p class="summary-text">Parameter choice was ${escapeHtml(r.stability)}.</p>
          <p class="summary-text">Trading costs of ${r.costs === null ? "n/a" : r.costs.toFixed(2)} are already subtracted from every figure.</p>
          <p class="dim">A backtest shows what a rule would have returned on data that has already happened, which is not what it will return next.</p>`,
      })}`;

    requestAnimationFrame(() => {
      Charts.line($("bt-chart"), {
        values: r.equity, benchmark: r.benchmark, dates: r.dates,
        format: (v) => v.toFixed(0), height: 240,
      });
      Charts.drawdown($("bt-dd"), { values: r.equity, dates: r.dates });
      const delta = $("bt-delta");
      if (delta) countUp(delta, r.excess_pct, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} pts`);
    });
    setStatus(`${r.folds} folds scored out of sample`, "ok");
  }

  /* --------------------------------------------------------------- quant */

  function renderQuant(r) {
    const v = r.volatility, g = r.regime, m = r.mean_reversion;
    $("quant-out").innerHTML = `
      ${panel({
        title: "Experimental",
        chip: '<span class="chip alpha">alpha</span>',
        body: `<p class="summary-text">${escapeHtml(r.notice)}</p>`,
        extra: "alpha",
      })}
      ${
        v
          ? panel({
              title: "Volatility",
              chip: `<span class="chip">${escapeHtml(v.regime)}</span>`,
              body: `
                <div class="stat-grid">
                  <div class="stat"><div class="stat-label">EWMA</div><div class="stat-value">${pct(v.ewma)}</div>
                    <div class="stat-note">recent, exponentially weighted</div></div>
                  <div class="stat"><div class="stat-label">Yang-Zhang</div><div class="stat-value">${pct(v.yang_zhang)}</div>
                    <div class="stat-note">uses the full OHLC bar</div></div>
                  <div class="stat"><div class="stat-label">Close to close</div><div class="stat-value">${pct(v.close_to_close)}</div>
                    <div class="stat-note">the textbook estimator</div></div>
                  <div class="stat"><div class="stat-label">Forecast (${v.horizon}d)</div><div class="stat-value">${pct(v.forecast)}</div>
                    <div class="stat-note">${escapeHtml(v.model)}</div></div>
                  <div class="stat"><div class="stat-label">Long run</div><div class="stat-value">${pct(v.long_run)}</div>
                    <div class="stat-note">where the model says it settles</div></div>
                  <div class="stat"><div class="stat-label">Persistence</div><div class="stat-value">${ratio(v.persistence, 3)}</div>
                    <div class="stat-note">how long a shock lingers</div></div>
                </div>
                ${v.notes.length ? `<ul class="bullets">${v.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : ""}`,
            })
          : ""
      }
      ${
        g
          ? panel({
              title: "Regime",
              chip: `<span class="chip">${escapeHtml(g.classification)}</span>`,
              body: `
                <p class="summary-text">${escapeHtml(g.describe)}</p>
                <div class="stat-grid">
                  <div class="stat"><div class="stat-label">Hurst</div><div class="stat-value">${ratio(g.hurst, 3)}</div>
                    <div class="stat-note">± ${ratio(g.hurst_error, 3)} · 0.5 is a random walk</div></div>
                  <div class="stat"><div class="stat-label">Variance ratio</div><div class="stat-value">${ratio(g.variance_ratio, 3)}</div>
                    <div class="stat-note">1.0 is a random walk</div></div>
                  <div class="stat"><div class="stat-label">VR z-statistic</div><div class="stat-value">${ratio(g.variance_ratio_z, 2)}</div>
                    <div class="stat-note">beyond ±1.96 is significant</div></div>
                </div>`,
            })
          : ""
      }
      ${
        m
          ? panel({
              title: "Mean reversion",
              chip: `<span class="chip ${m.reverting ? "" : "warn"}">${m.reverting ? "reverting" : "no reversion"}</span>`,
              body: `
                <p class="summary-text">${escapeHtml(m.describe)}</p>
                <div class="stat-grid">
                  <div class="stat"><div class="stat-label">Z-score</div><div class="stat-value">${m.z_score === null ? "n/a" : (m.z_score >= 0 ? "+" : "") + m.z_score.toFixed(2)}</div></div>
                  <div class="stat"><div class="stat-label">Half-life</div><div class="stat-value">${m.half_life === null ? "none" : m.half_life.toFixed(0) + "d"}</div></div>
                </div>`,
            })
          : ""
      }
      ${r.failures.length ? listPanel("Models that did not run", r.failures) : ""}`;
    setStatus("Experimental diagnostics — not a signal", "");
  }

  /* ------------------------------------------------------------ settings */

  function renderSettings() {
    $("settings-out").innerHTML = `
      ${panel({
        title: "API keys",
        body: `
          <p class="summary-text">Keys are stored in a file rather than an environment variable, because an app opened from Finder does not inherit your shell's environment.</p>
          ${state.credentials
            .map(
              (c) => `<div class="field-row" style="margin-top:12px;">
                <label for="key-${escapeHtml(c.name)}">${escapeHtml(c.name)}</label>
                <input type="text" id="key-${escapeHtml(c.name)}" placeholder="${
                  c.present ? "set (…" + escapeHtml(c.hint) + ")" : "not set"
                }" style="flex:1;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:6px 10px;" />
                <button data-save-key="${escapeHtml(c.name)}">Save</button>
              </div>
              <div class="dim" style="margin-left:148px;">${escapeHtml(c.unlocks)}</div>`
            )
            .join("")}`,
      })}
      ${panel({
        title: "About",
        body: `<p class="summary-text">This tool reports what coincided with a price move and how often that has coincided before. It does not identify causes and does not recommend transactions.</p>
               <p class="dim">Every figure shown is checked against the underlying evidence before it is displayed.</p>`,
      })}`;

    document.querySelectorAll("[data-save-key]").forEach((button) => {
      button.addEventListener("click", async () => {
        const name = button.dataset.saveKey;
        const field = $(`key-${name}`);
        const value = field.value.trim();
        if (!value) return;
        const result = await window.pywebview.api.save_key(name, value);
        if (result.ok) {
          field.value = "";
          setStatus(`${name} saved`, "ok");
          await loadBootstrap();
          renderSettings();
        } else {
          setStatus(result.error, "bad");
        }
      });
    });
  }

  /* --------------------------------------------------------------- flows */

  async function runResearch() {
    const query = $("query").value.trim();
    if (!query || state.busy) return;
    setBusy(true, "Gathering evidence…");
    try {
      const result = await window.pywebview.api.research(query, state.depth);
      if (!result.ok) {
        $("research-out").innerHTML = errorPanel(result.error, result.suggestion);
        setStatus("Could not answer that", "bad");
      } else {
        renderResearch(result);
      }
    } finally {
      setBusy(false);
    }
  }

  async function runBacktest() {
    const ticker = $("bt-ticker").value.trim();
    if (!ticker || state.busy) return;
    setBusy(true, "Running walk-forward backtest…");
    try {
      const result = await window.pywebview.api.backtest(
        ticker, $("bt-strategy").value, parseFloat($("bt-years").value)
      );
      if (!result.ok) {
        $("backtest-out").innerHTML = errorPanel(result.error, result.suggestion);
        setStatus("Backtest failed", "bad");
      } else {
        renderBacktest(result);
      }
    } finally {
      setBusy(false);
    }
  }

  async function runQuant() {
    const ticker = $("q-ticker").value.trim();
    if (!ticker || state.busy) return;
    setBusy(true, "Fitting models…");
    try {
      const result = await window.pywebview.api.quant(ticker);
      if (!result.ok) {
        $("quant-out").innerHTML = errorPanel(result.error, result.suggestion);
        setStatus("Analysis failed", "bad");
      } else {
        renderQuant(result);
      }
    } finally {
      setBusy(false);
    }
  }

  /* ----------------------------------------------------------- bootstrap */

  function showView(name) {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.setAttribute("aria-selected", String(tab.dataset.view === name));
    });
    document.querySelectorAll(".view").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${name}`);
    });
    ["research", "backtest", "quant"].forEach((key) => {
      const row = $(`control-${key}`);
      if (row) row.hidden = key !== name;
    });
    if (name === "settings") renderSettings();
  }

  function emptyState() {
    const examples = [
      "why did SPY fall today",
      "why did TSLA drop yesterday",
      "is AAPL good to invest",
      "why did the stock market fall this week",
      "should i buy KO",
    ];
    $("research-out").innerHTML = `
      <div class="empty">
        <h2>Ask why something moved, or what a company's figures look like.</h2>
        <p>This tool never tells you a cause and never tells you what to buy. It shows what
           happened at the same time, how often that has coincided with moves like this before,
           and what argues the other way.</p>
        <p>Every number shown is checked against the underlying evidence before it appears.</p>
        <div class="examples">
          ${examples.map((q) => `<button class="example" data-example="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join("")}
        </div>
      </div>`;
    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        $("query").value = button.dataset.example;
        runResearch();
      });
    });
  }

  async function loadBootstrap() {
    const info = await window.pywebview.api.bootstrap();
    state.strategies = info.strategies;
    state.credentials = info.credentials;

    const depth = $("depth");
    depth.innerHTML = info.depths
      .map((d) => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.label)}</option>`)
      .join("");
    depth.value = state.depth;

    $("bt-strategy").innerHTML = info.strategies
      .map(
        (s) =>
          `<option value="${escapeHtml(s.id)}">${escapeHtml(s.id)}${s.alpha ? " (alpha)" : ""}</option>`
      )
      .join("");
    updateStrategyHint();

    $("mode-badge").textContent = info.mock ? "OFFLINE — synthetic data" : "";
  }

  function updateStrategyHint() {
    const chosen = state.strategies.find((s) => s.id === $("bt-strategy").value);
    $("bt-hint").textContent = chosen ? chosen.description : "";
  }

  function applyTheme(name) {
    document.documentElement.dataset.theme = name;
    try {
      localStorage.setItem("research.theme", name);
    } catch (_) {
      /* private mode or blocked storage: the theme simply will not persist */
    }
  }

  function wire() {
    $("run").addEventListener("click", runResearch);
    $("query").addEventListener("keydown", (e) => {
      if (e.key === "Enter") runResearch();
    });
    $("run-backtest").addEventListener("click", runBacktest);
    $("bt-ticker").addEventListener("keydown", (e) => {
      if (e.key === "Enter") runBacktest();
    });
    $("bt-strategy").addEventListener("change", updateStrategyHint);
    $("run-quant").addEventListener("click", runQuant);
    $("q-ticker").addEventListener("keydown", (e) => {
      if (e.key === "Enter") runQuant();
    });
    $("depth").addEventListener("change", (e) => {
      state.depth = e.target.value;
      if ($("query").value.trim()) runResearch();
    });
    $("theme").addEventListener("change", (e) => applyTheme(e.target.value));
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => showView(tab.dataset.view));
    });
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        $("query").focus();
        $("query").select();
      }
    });
  }

  ready().then(async () => {
    let saved = "graphite";
    try {
      saved = localStorage.getItem("research.theme") || "graphite";
    } catch (_) {
      /* storage unavailable; fall back to the default theme */
    }
    applyTheme(saved);
    $("theme").value = saved;

    wire();
    await loadBootstrap();
    emptyState();
    setStatus("Ready");
  });
})();
