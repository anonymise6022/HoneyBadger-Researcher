/* Charts, drawn as inline SVG against the active theme's CSS variables.
 *
 * Hand-rolled rather than pulled from a library. The charts needed are few;
 * a library would need its own theming layer wired to these variables, which
 * is most of the work anyway; and bundling one into a desktop app means
 * shipping a megabyte of JavaScript to draw some lines. The 3D surface in
 * particular is a painter's-algorithm projection in about eighty lines,
 * which is a great deal less than a WebGL dependency and matches the rest of
 * the interface instead of looking imported.
 *
 * Colours come from CSS classes and computed custom properties, never
 * literals, so a theme change restyles every chart with no redraw.
 */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  }

  function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (value || "").trim() || fallback;
  }

  function scales(values, width, height, pad) {
    const finite = values.filter((v) => Number.isFinite(v));
    const min = finite.length ? Math.min(...finite) : 0;
    const max = finite.length ? Math.max(...finite) : 1;
    // A flat series would divide by zero; give it a nominal band so the line
    // renders through the middle rather than vanishing.
    const span = max - min || Math.abs(max) || 1;
    return {
      min, max, span,
      x: (i, n) => pad.left + (i / Math.max(n - 1, 1)) * (width - pad.left - pad.right),
      y: (v) => height - pad.bottom - ((v - min) / span) * (height - pad.top - pad.bottom),
    };
  }

  function linePath(values, sx, sy) {
    let started = false;
    return values
      .map((v, i) => {
        if (!Number.isFinite(v)) return "";
        const cmd = started ? "L" : "M";
        started = true;
        return `${cmd}${sx(i, values.length).toFixed(2)},${sy(v).toFixed(2)}`;
      })
      .join(" ");
  }

  function gridlines(svg, scale, width, height, pad, format, count = 4) {
    for (let i = 0; i <= count; i++) {
      const value = scale.min + (scale.span * i) / count;
      const y = scale.y(value);
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: "chart-grid",
      }));
      const label = el("text", {
        x: pad.left - 8, y: y + 3, class: "chart-axis-text", "text-anchor": "end",
      });
      label.textContent = format(value);
      svg.appendChild(label);
    }
  }

  function dateAxis(svg, dates, scale, width, height, pad) {
    if (!dates || !dates.length) return;
    const wanted = Math.max(2, Math.floor((width - pad.left - pad.right) / 120));
    const step = Math.max(1, Math.floor(dates.length / wanted));
    for (let i = 0; i < dates.length; i += step) {
      const label = el("text", {
        x: scale.x(i, dates.length), y: height - pad.bottom + 15,
        class: "chart-axis-text", "text-anchor": "middle",
      });
      label.textContent = (dates[i] || "").slice(0, 7);
      svg.appendChild(label);
    }
  }

  /** Animate a path drawing itself in, using its own measured length. */
  function animateDraw(node) {
    const length = node.getTotalLength ? node.getTotalLength() : 0;
    if (!length || !Number.isFinite(length)) return;
    node.style.setProperty("--len", length);
    node.classList.add("animate-draw");
  }

  /* ------------------------------------------------------- line + area */

  function line(container, opts) {
    const { values, dates, benchmark, format, height = 240, animate = true } = opts;
    container.innerHTML = "";
    if (!values || values.length < 2) {
      container.innerHTML = '<div class="chart-empty">Not enough data to draw this.</div>';
      return;
    }
    const width = container.clientWidth || 820;
    const pad = { top: 14, right: 14, bottom: 26, left: 62 };
    const all = benchmark ? values.concat(benchmark) : values;
    const scale = scales(all, width, height, pad);
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height,
      preserveAspectRatio: "none", role: "img",
    });

    gridlines(svg, scale, width, height, pad, format || ((v) => v.toFixed(0)));
    dateAxis(svg, dates, scale, width, height, pad);

    const last = values.length - 1;
    svg.appendChild(el("path", {
      class: "chart-area",
      d: `${linePath(values, scale.x, scale.y)} L${scale.x(last, values.length).toFixed(2)},${height - pad.bottom} L${scale.x(0, values.length).toFixed(2)},${height - pad.bottom} Z`,
    }));

    if (benchmark && benchmark.length === values.length) {
      svg.appendChild(el("path", {
        class: "chart-line benchmark", d: linePath(benchmark, scale.x, scale.y),
      }));
    }
    const primary = el("path", {
      class: "chart-line primary", d: linePath(values, scale.x, scale.y),
    });
    svg.appendChild(primary);
    container.appendChild(svg);
    if (animate) animateDraw(primary);
  }

  /* --------------------------------------------------------- drawdown */

  function drawdown(container, { values, dates, height = 130 }) {
    container.innerHTML = "";
    if (!values || values.length < 2) return;
    let peak = values[0];
    const depths = values.map((v) => {
      peak = Math.max(peak, v);
      return peak > 0 ? (v / peak - 1) * 100 : 0;
    });
    const width = container.clientWidth || 820;
    const pad = { top: 12, right: 14, bottom: 22, left: 62 };
    const scale = scales(depths.concat([0]), width, height, pad);
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height,
      preserveAspectRatio: "none", role: "img",
    });
    gridlines(svg, scale, width, height, pad, (v) => `${v.toFixed(0)}%`, 2);
    const last = depths.length - 1;
    svg.appendChild(el("path", {
      class: "chart-dd",
      d: `${linePath(depths, scale.x, scale.y)} L${scale.x(last, depths.length).toFixed(2)},${scale.y(0)} L${scale.x(0, depths.length).toFixed(2)},${scale.y(0)} Z`,
    }));
    container.appendChild(svg);
  }

  /* ------------------------------------------------------- candlestick */

  /** Enough decimals to be exact for the instrument, and no more. */
  function priceText(value) {
    if (value == null || !Number.isFinite(value)) return "n/a";
    const size = Math.abs(value);
    return value.toFixed(size >= 1 ? 2 : size >= 0.01 ? 4 : 6);
  }

  /** The readout is the one place here that builds markup from data. */
  function safe(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  /**
   * OHLC candles, for the Explore tab where the bar shape is the point.
   *
   * A candle states four prices as a shape, and a shape can be read to
   * within a few pixels at best. The open and the close are the two figures
   * people actually want off a chart, so hovering a session reads them back
   * exactly -- with the high, the low and the move across the session --
   * rather than leaving them to be guessed against the axis.
   */
  function candles(container, { bars, height = 320 }) {
    container.innerHTML = "";
    container.classList.remove("chart-hoverable");
    if (!bars || bars.length < 2) {
      container.innerHTML = '<div class="chart-empty">Not enough data to draw this.</div>';
      return;
    }
    const width = container.clientWidth || 820;
    const pad = { top: 14, right: 14, bottom: 26, left: 62 };
    const lows = bars.map((b) => b.l);
    const highs = bars.map((b) => b.h);
    const scale = scales(lows.concat(highs), width, height, pad);
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height, role: "img",
    });
    gridlines(svg, scale, width, height, pad, (v) => v.toFixed(0));
    dateAxis(svg, bars.map((b) => b.d), scale, width, height, pad);

    const slot = (width - pad.left - pad.right) / bars.length;
    const body = Math.max(1, Math.min(slot * 0.68, 11));
    const plot = height - pad.top - pad.bottom;
    bars.forEach((bar, i) => {
      const x = scale.x(i, bars.length);
      const rising = bar.c >= bar.o;
      const klass = rising ? "candle up" : "candle down";
      svg.appendChild(el("line", {
        x1: x, x2: x, y1: scale.y(bar.h), y2: scale.y(bar.l), class: `${klass} wick`,
      }));
      const top = scale.y(Math.max(bar.o, bar.c));
      const bottom = scale.y(Math.min(bar.o, bar.c));
      svg.appendChild(el("rect", {
        x: x - body / 2, y: top, width: body,
        height: Math.max(bottom - top, 1), class: klass, rx: 1,
      }));
    });

    // The hover marks: a band naming the session being read, and a rule at
    // its close so the figure in the readout can be placed against the axis.
    // Drawn after the candles and inert to the mouse, so they neither need a
    // second pass nor interrupt hit testing.
    const band = el("rect", {
      class: "candle-band", x: -slot, y: pad.top,
      width: Math.max(slot, 2), height: plot,
    });
    const closeRule = el("line", {
      class: "candle-close-rule", x1: pad.left, x2: width - pad.right, y1: 0, y2: 0,
    });
    const marks = el("g", { class: "candle-marks", "pointer-events": "none", opacity: 0 });
    marks.appendChild(band);
    marks.appendChild(closeRule);
    svg.appendChild(marks);

    // One transparent column per session, tiling the plot so every position
    // inside it belongs to exactly one candle. Letting the browser hit-test
    // these is simpler than mapping the pointer back through the viewBox by
    // hand, and it stays correct when the SVG is scaled to a container that
    // is no longer the width it was drawn at.
    const columns = el("g", { class: "candle-columns" });
    bars.forEach((bar, i) => {
      columns.appendChild(el("rect", {
        x: scale.x(i, bars.length) - slot / 2, y: pad.top, width: slot,
        height: plot, fill: "transparent", "data-index": i,
      }));
    });
    svg.appendChild(columns);
    container.appendChild(svg);

    const readout = document.createElement("div");
    readout.className = "chart-readout";
    readout.hidden = true;
    container.appendChild(readout);
    container.classList.add("chart-hoverable");

    const hide = () => {
      readout.hidden = true;
      marks.setAttribute("opacity", 0);
    };

    const show = (event) => {
      const column = event.target.closest && event.target.closest("[data-index]");
      if (!column) return hide();
      const bar = bars[Number(column.getAttribute("data-index"))];
      if (!bar) return hide();

      const move = bar.o ? (bar.c / bar.o - 1) * 100 : null;
      const way = bar.c >= bar.o ? "up" : "down";
      band.setAttribute("x", column.getAttribute("x"));
      closeRule.setAttribute("y1", scale.y(bar.c).toFixed(2));
      closeRule.setAttribute("y2", scale.y(bar.c).toFixed(2));
      marks.setAttribute("opacity", 1);

      readout.innerHTML = `
        <div class="readout-date">${safe(bar.d)}</div>
        <div class="readout-grid">
          <span>Open</span><span class="readout-value">${priceText(bar.o)}</span>
          <span>High</span><span class="readout-value">${priceText(bar.h)}</span>
          <span>Low</span><span class="readout-value">${priceText(bar.l)}</span>
          <span>Close</span><span class="readout-value ${way}">${priceText(bar.c)}</span>
          <span>Change</span><span class="readout-value ${way}">${
            move == null ? "n/a" : `${move >= 0 ? "+" : ""}${move.toFixed(2)}%`
          }</span>
        </div>`;
      readout.hidden = false;

      // Positioned from the column's own rendered box rather than from SVG
      // coordinates, which is the same reason the columns exist: it holds
      // whatever scale the drawing ends up at. Sides flip near the right
      // edge so the box never leaves the chart.
      const frame = container.getBoundingClientRect();
      const box = column.getBoundingClientRect();
      const centre = box.left - frame.left + box.width / 2;
      const gap = 14;
      const wide = centre + gap + readout.offsetWidth > frame.width;
      const left = wide ? centre - gap - readout.offsetWidth : centre + gap;
      const top = event.clientY - frame.top - readout.offsetHeight / 2;
      const clamp = (v, hi) => Math.max(4, Math.min(v, hi - 4));
      readout.style.left = `${clamp(left, frame.width - readout.offsetWidth)}px`;
      readout.style.top = `${clamp(top, frame.height - readout.offsetHeight)}px`;
    };

    svg.addEventListener("mousemove", show);
    svg.addEventListener("mouseleave", hide);
  }

  /* ------------------------------------------------------------ curve */

  /**
   * One quantity against an arbitrary numeric x axis.
   *
   * The greeks need this and the line chart cannot give it: a line chart's
   * x axis is a position in a series, which is right for dates and wrong for
   * a price. Delta against the underlying has to put the strike where the
   * strike is, or the shape of the curve says something false about where
   * the contract changes character.
   *
   * `marker` draws a vertical rule at one x value -- where the underlying
   * is now -- because every one of these curves is read as "where am I on
   * it, and which way am I about to move".
   */
  function curve(container, opts) {
    const {
      xs, ys, second, marker, format, xFormat, height = 132, zero = true, fill = false,
    } = opts;
    container.innerHTML = "";
    const points = (xs || []).map((x, i) => [x, ys ? ys[i] : null])
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    if (points.length < 2) {
      container.innerHTML = '<div class="chart-empty">No curve to draw.</div>';
      return;
    }
    const width = container.clientWidth || 300;
    const pad = { top: 10, right: 8, bottom: 20, left: 46 };

    const values = points.map((p) => p[1]).concat(second ? second.filter(Number.isFinite) : []);
    if (zero) values.push(0);
    const scale = scales(values, width, height, pad);
    const xMin = Math.min(...points.map((p) => p[0]));
    const xMax = Math.max(...points.map((p) => p[0]));
    const xSpan = xMax - xMin || 1;
    const px = (x) => pad.left + ((x - xMin) / xSpan) * (width - pad.left - pad.right);

    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height, role: "img",
    });
    gridlines(svg, scale, width, height, pad, format || ((v) => v.toFixed(2)), 2);

    if (zero && scale.min < 0 && scale.max > 0) {
      const y = scale.y(0);
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: "chart-zero",
      }));
    }

    const path = (series) => series
      .map(([x, y], i) => `${i ? "L" : "M"}${px(x).toFixed(2)},${scale.y(y).toFixed(2)}`)
      .join(" ");

    if (fill) {
      svg.appendChild(el("path", {
        class: "chart-area",
        d: `${path(points)} L${px(xMax).toFixed(2)},${scale.y(zero ? 0 : scale.min).toFixed(2)}`
           + ` L${px(xMin).toFixed(2)},${scale.y(zero ? 0 : scale.min).toFixed(2)} Z`,
      }));
    }
    if (second) {
      const other = xs.map((x, i) => [x, second[i]])
        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      if (other.length > 1) {
        svg.appendChild(el("path", { class: "chart-line benchmark", d: path(other) }));
      }
    }
    svg.appendChild(el("path", { class: "chart-line primary", d: path(points) }));

    if (Number.isFinite(marker) && marker >= xMin && marker <= xMax) {
      const x = px(marker);
      svg.appendChild(el("line", {
        x1: x, x2: x, y1: pad.top, y2: height - pad.bottom, class: "chart-marker",
      }));
    }

    // Three x labels: the ends and the middle. More than that is unreadable
    // at the size these are drawn, and the marker says where "now" is.
    const label = xFormat || ((v) => v.toFixed(0));
    [[xMin, "start"], [(xMin + xMax) / 2, "middle"], [xMax, "end"]].forEach(([value, anchorAt]) => {
      const text = el("text", {
        x: px(value), y: height - pad.bottom + 14,
        class: "chart-axis-text", "text-anchor": anchorAt,
      });
      text.textContent = label(value);
      svg.appendChild(text);
    });
    container.appendChild(svg);
  }

  /* ------------------------------------------------------------ bars */

  /** Horizontal comparison bars — used wherever two rates sit side by side. */
  function compare(container, { rows, height = 0 }) {
    container.innerHTML = "";
    if (!rows || !rows.length) return;
    const width = container.clientWidth || 600;
    const rowHeight = 30;
    const total = height || rows.length * rowHeight + 10;
    const labelWidth = 150;
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${total}`, height: total, role: "img",
    });
    const maxValue = Math.max(...rows.map((r) => Math.max(r.value, r.reference || 0)), 1);
    rows.forEach((row, i) => {
      const y = i * rowHeight + 8;
      const track = width - labelWidth - 70;
      const label = el("text", {
        x: 0, y: y + 11, class: "chart-axis-text", "text-anchor": "start",
      });
      label.textContent = row.label;
      svg.appendChild(label);
      svg.appendChild(el("rect", {
        x: labelWidth, y, width: track, height: 12, rx: 6, class: "bar-track",
      }));
      const filled = el("rect", {
        x: labelWidth, y, width: 0, height: 12, rx: 6,
        class: `bar-fill${row.highlight ? " highlight" : ""}`,
      });
      svg.appendChild(filled);
      if (row.reference != null) {
        const rx = labelWidth + (row.reference / maxValue) * track;
        svg.appendChild(el("line", {
          x1: rx, x2: rx, y1: y - 3, y2: y + 15, class: "bar-reference",
        }));
      }
      const value = el("text", {
        x: width - 4, y: y + 11, class: "chart-axis-text", "text-anchor": "end",
      });
      value.textContent = row.display ?? String(row.value);
      svg.appendChild(value);
      // Grow after mount so the transition has a start value to run from.
      requestAnimationFrame(() => {
        filled.setAttribute("width", String((row.value / maxValue) * track));
      });
    });
    container.appendChild(svg);
  }

  /* ----------------------------------------------------- 3D surface */

  /**
   * A parameter surface, projected and painted back to front.
   *
   * No WebGL and no dependency: an isometric projection plus the painter's
   * algorithm is enough for a 5x5 grid, and it inherits the theme's colours
   * the way every other chart here does. Cells are drawn from the far corner
   * forward so nearer quads overlap farther ones correctly.
   */
  function surface(container, opts) {
    const { z, xValues, yValues, xLabel, yLabel, height = 340, best } = opts;
    container.innerHTML = "";
    if (!z || !z.length || !z[0].length) {
      container.innerHTML = '<div class="chart-empty">No surface to draw.</div>';
      return;
    }
    const width = container.clientWidth || 700;
    const rows = z.length;
    const cols = z[0].length;

    const flat = z.flat().filter((v) => v != null && Number.isFinite(v));
    const min = flat.length ? Math.min(...flat) : 0;
    const max = flat.length ? Math.max(...flat) : 1;
    const span = max - min || 1;

    // Isometric projection with a vertical lift proportional to the value.
    const originX = width * 0.5;
    const originY = height * 0.74;
    const stepX = Math.min((width * 0.62) / Math.max(cols, rows), 46);
    const stepY = stepX * 0.55;
    const lift = height * 0.36;

    const project = (col, row, value) => {
      const normalized = value == null || !Number.isFinite(value) ? 0 : (value - min) / span;
      return [
        originX + (col - (cols - 1) / 2) * stepX - (row - (rows - 1) / 2) * stepX * 0.62,
        originY + (col - (cols - 1) / 2) * stepY + (row - (rows - 1) / 2) * stepY
          - normalized * lift,
      ];
    };

    const accent = cssVar("--accent", "#4fd6e8");
    const cold = cssVar("--surface-2", "#161b25");
    const svg = el("svg", {
      class: "chart surface", viewBox: `0 0 ${width} ${height}`, height, role: "img",
    });

    // Painter's algorithm: farthest row first so nearer quads paint over it.
    const quads = [];
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const corners = [
          [c, r, z[r][c]], [c + 1, r, z[r][c + 1]],
          [c + 1, r + 1, z[r + 1][c + 1]], [c, r + 1, z[r + 1][c]],
        ];
        const values = corners.map((k) => k[2]).filter((v) => v != null && Number.isFinite(v));
        if (values.length < 4) continue;  // a hole in the grid stays a hole
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        quads.push({
          depth: r + c,
          points: corners.map(([cc, rr, v]) => project(cc, rr, v)),
          t: (mean - min) / span,
        });
      }
    }
    quads.sort((a, b) => a.depth - b.depth);

    quads.forEach((quad, i) => {
      const poly = el("polygon", {
        points: quad.points.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
        fill: accent,
        "fill-opacity": (0.14 + quad.t * 0.72).toFixed(3),
        stroke: cold,
        "stroke-width": 0.6,
        class: "surface-cell",
      });
      poly.style.animationDelay = `${i * 7}ms`;
      svg.appendChild(poly);
    });

    // Mark the best cell, since the whole point is whether it stands alone.
    if (best && Number.isFinite(best.col) && Number.isFinite(best.row)) {
      const [bx, by] = project(best.col, best.row, z[best.row]?.[best.col]);
      svg.appendChild(el("circle", { cx: bx, cy: by, r: 4.5, class: "surface-best" }));
    }

    const xText = el("text", {
      x: width - 12, y: height - 8, class: "chart-axis-text", "text-anchor": "end",
    });
    xText.textContent = `${xLabel}: ${xValues[0]} → ${xValues[xValues.length - 1]}`;
    svg.appendChild(xText);
    const yText = el("text", { x: 12, y: height - 8, class: "chart-axis-text" });
    yText.textContent = `${yLabel}: ${yValues[0]} → ${yValues[yValues.length - 1]}`;
    svg.appendChild(yText);

    container.appendChild(svg);
  }

  /* ------------------------------------------------------- payoff */

  /** An option payoff diagram at expiry — the clearest way to show shape. */
  function payoff(container, { spot, points, height = 220 }) {
    container.innerHTML = "";
    if (!points || points.length < 2) return;
    const width = container.clientWidth || 600;
    const pad = { top: 16, right: 16, bottom: 28, left: 62 };
    const values = points.map((p) => p.y);
    const scale = scales(values.concat([0]), width, height, pad);
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height, role: "img",
    });
    gridlines(svg, scale, width, height, pad, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}`, 3);

    const zeroY = scale.y(0);
    svg.appendChild(el("line", {
      x1: pad.left, x2: width - pad.right, y1: zeroY, y2: zeroY, class: "chart-zero",
    }));
    svg.appendChild(el("path", {
      class: "chart-line primary", d: linePath(values, scale.x, scale.y),
    }));

    // Where the underlying is now, so the shape has a reference point.
    const spotIndex = points.findIndex((p) => p.x >= spot);
    if (spotIndex > 0) {
      const sx = scale.x(spotIndex, points.length);
      svg.appendChild(el("line", {
        x1: sx, x2: sx, y1: pad.top, y2: height - pad.bottom, class: "chart-marker",
      }));
      const label = el("text", {
        x: sx + 4, y: pad.top + 10, class: "chart-axis-text",
      });
      label.textContent = "now";
      svg.appendChild(label);
    }
    container.appendChild(svg);
  }

  return { line, drawdown, candles, compare, curve, surface, payoff };
})();
