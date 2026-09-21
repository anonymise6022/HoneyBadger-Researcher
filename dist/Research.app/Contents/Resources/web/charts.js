/* Inline SVG charts, drawn against the active theme's CSS variables.
 *
 * Hand-rolled rather than pulled from a charting library, for three
 * reasons. The charts needed are few and simple. A library would need its
 * own theming layer wired to these variables, which is most of the work
 * anyway. And bundling one into a desktop app means shipping a megabyte of
 * JavaScript to draw four lines.
 *
 * Every colour comes from a CSS class, never a literal, so a theme change
 * restyles the charts with no redraw.
 */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) {
      node.setAttribute(key, String(value));
    }
    return node;
  }

  /** Map data to pixel space. Returns the scales as closures. */
  function scales(values, width, height, pad) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    // A flat series would divide by zero; give it a nominal band so the
    // line renders through the middle instead of vanishing.
    const span = max - min || Math.abs(max) || 1;
    const x = (i, n) => pad.left + (i / Math.max(n - 1, 1)) * (width - pad.left - pad.right);
    const y = (v) => height - pad.bottom - ((v - min) / span) * (height - pad.top - pad.bottom);
    return { x, y, min, max, span };
  }

  function path(values, sx, sy) {
    return values
      .map((v, i) => `${i === 0 ? "M" : "L"}${sx(i, values.length).toFixed(2)},${sy(v).toFixed(2)}`)
      .join(" ");
  }

  /** Horizontal gridlines with value labels on the left. */
  function gridlines(svg, scale, width, height, pad, format, count = 4) {
    for (let i = 0; i <= count; i++) {
      const value = scale.min + (scale.span * i) / count;
      const y = scale.y(value);
      svg.appendChild(
        el("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: "chart-grid" })
      );
      const label = el("text", {
        x: pad.left - 6, y: y + 3, class: "chart-axis-text", "text-anchor": "end",
      });
      label.textContent = format(value);
      svg.appendChild(label);
    }
  }

  /** Date labels along the bottom, thinned so they never collide. */
  function dateAxis(svg, dates, scale, width, height, pad) {
    if (!dates || dates.length === 0) return;
    const wanted = Math.max(2, Math.floor((width - pad.left - pad.right) / 110));
    const step = Math.max(1, Math.floor(dates.length / wanted));
    for (let i = 0; i < dates.length; i += step) {
      const label = el("text", {
        x: scale.x(i, dates.length),
        y: height - pad.bottom + 14,
        class: "chart-axis-text",
        "text-anchor": "middle",
      });
      label.textContent = (dates[i] || "").slice(0, 7);
      svg.appendChild(label);
    }
  }

  /** Animate a path drawing itself in, using its own measured length. */
  function animate(node) {
    const length = node.getTotalLength ? node.getTotalLength() : 0;
    if (!length) return;
    node.style.setProperty("--len", length);
    node.classList.add("animate");
  }

  /**
   * A price or equity line, optionally with a dashed benchmark beside it.
   */
  function line(container, { values, dates, benchmark, format, height = 220 }) {
    container.innerHTML = "";
    if (!values || values.length < 2) {
      container.textContent = "Not enough data to draw a chart.";
      return;
    }
    const width = container.clientWidth || 820;
    const pad = { top: 12, right: 12, bottom: 24, left: 56 };
    const all = benchmark ? values.concat(benchmark) : values;
    const scale = scales(all, width, height, pad);

    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height,
      preserveAspectRatio: "none", role: "img",
    });

    const fmt = format || ((v) => v.toFixed(0));
    gridlines(svg, scale, width, height, pad, fmt);
    dateAxis(svg, dates, scale, width, height, pad);

    // Area beneath the primary line, which gives the eye a baseline to
    // read the shape against without adding another stroke to track.
    const area = el("path", {
      class: "chart-area",
      d:
        path(values, scale.x, scale.y) +
        ` L${scale.x(values.length - 1, values.length).toFixed(2)},${height - pad.bottom}` +
        ` L${scale.x(0, values.length).toFixed(2)},${height - pad.bottom} Z`,
    });
    svg.appendChild(area);

    if (benchmark && benchmark.length === values.length) {
      svg.appendChild(
        el("path", { class: "chart-line benchmark", d: path(benchmark, scale.x, scale.y) })
      );
    }
    const primary = el("path", {
      class: "chart-line primary", d: path(values, scale.x, scale.y),
    });
    svg.appendChild(primary);
    container.appendChild(svg);
    animate(primary);
  }

  /** Depth below the running peak, as a filled area under zero. */
  function drawdown(container, { values, dates, height = 120 }) {
    container.innerHTML = "";
    if (!values || values.length < 2) return;

    let peak = values[0];
    const depths = values.map((v) => {
      peak = Math.max(peak, v);
      return peak > 0 ? (v / peak - 1) * 100 : 0;
    });

    const width = container.clientWidth || 820;
    const pad = { top: 10, right: 12, bottom: 20, left: 56 };
    // Always include zero so the top of the chart is the peak line.
    const scale = scales(depths.concat([0]), width, height, pad);

    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${width} ${height}`, height,
      preserveAspectRatio: "none", role: "img",
    });
    gridlines(svg, scale, width, height, pad, (v) => `${v.toFixed(0)}%`, 2);

    svg.appendChild(
      el("path", {
        class: "chart-dd",
        d:
          path(depths, scale.x, scale.y) +
          ` L${scale.x(depths.length - 1, depths.length).toFixed(2)},${scale.y(0)}` +
          ` L${scale.x(0, depths.length).toFixed(2)},${scale.y(0)} Z`,
      })
    );
    container.appendChild(svg);
  }

  return { line, drawdown };
})();
