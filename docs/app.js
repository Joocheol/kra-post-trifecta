const formatter = new Intl.NumberFormat("en-US");

function compactNumber(value) {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

function renderBars(container, entries, valueKey, valueLabel) {
  const max = Math.max(...entries.map(([, value]) => value[valueKey] ?? value), 1);
  container.innerHTML = "";

  for (const [label, value] of entries) {
    const rawValue = value[valueKey] ?? value;
    const row = document.createElement("div");
    row.className = "bar-row";

    const name = document.createElement("strong");
    name.textContent = label;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max((rawValue / max) * 100, 1)}%`;
    track.appendChild(fill);

    const number = document.createElement("span");
    number.className = "value";
    number.textContent = `${formatter.format(rawValue)} ${valueLabel}`;

    row.append(name, track, number);
    container.appendChild(row);
  }
}

function renderSamples(samples) {
  const container = document.getElementById("sample-list");
  container.innerHTML = "";

  for (const sample of samples) {
    const item = document.createElement("div");
    item.className = "sample";

    const title = document.createElement("strong");
    title.textContent = sample.path;

    const meta = document.createElement("small");
    meta.textContent = `${sample.race_date} / ${sample.json_type}`;

    const keys = document.createElement("code");
    keys.textContent = sample.top_level_keys.join(", ");

    item.append(title, meta, keys);
    container.appendChild(item);
  }
}

function renderSchemaOptions(schemas) {
  const select = document.getElementById("schema-select");
  const columns = document.getElementById("schema-columns");
  const names = Object.keys(schemas).sort();

  function renderColumns(name) {
    columns.innerHTML = "";
    for (const column of schemas[name] ?? []) {
      const token = document.createElement("span");
      token.className = "column-token";
      token.textContent = column;
      columns.appendChild(token);
    }
  }

  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }

  select.addEventListener("change", () => renderColumns(select.value));
  if (names.length) {
    select.value = names[0];
    renderColumns(names[0]);
  }
}

async function loadAvailableDates() {
  const dateSelect = document.getElementById("date-select");
  const status = document.getElementById("odds-status");

  try {
    const response = await fetch("./api/dates");
    if (!response.ok) {
      throw new Error("API server unavailable");
    }
    const payload = await response.json();
    dateSelect.innerHTML = "";
    for (const date of payload.dates.reverse()) {
      const option = document.createElement("option");
      option.value = date;
      option.textContent = date;
      dateSelect.appendChild(option);
    }
    status.textContent = `${payload.dates.length} available parsed race dates.`;
    document.getElementById("load-odds").disabled = false;
  } catch (error) {
    const staticManifest = await loadStaticOddsManifest();
    dateSelect.innerHTML = "";
    for (const item of [...staticManifest.dates].reverse()) {
      const option = document.createElement("option");
      option.value = item.date;
      option.textContent = item.date;
      dateSelect.appendChild(option);
    }
    window.staticOddsManifest = staticManifest;
    status.textContent = `${staticManifest.dates.length} precomputed race dates.`;
    document.getElementById("load-odds").disabled = false;
  }
}

async function loadStaticOddsManifest() {
  const response = await fetch("./data/odds/manifest.json");
  if (!response.ok) {
    throw new Error("No local API or precomputed odds manifest is available.");
  }
  return response.json();
}

async function readGzipJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Static odds file not found: ${response.status}`);
  }
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).json();
}

function formatBoolean(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "";
}

function renderOddsRows(rows) {
  const body = document.getElementById("odds-body");
  body.innerHTML = "";

  for (const sourceRow of rows) {
    const row = Array.isArray(sourceRow)
      ? {
          race_id: sourceRow[0],
          meet: sourceRow[1],
          market: sourceRow[2],
          combination: sourceRow[3].join("-"),
          odds: sourceRow[4],
          is_hit: sourceRow[5],
          is_capped_odds: sourceRow[6],
          arrival_order: sourceRow[7],
        }
      : sourceRow;
    const tr = document.createElement("tr");
    const values = [
      row.race_id,
      row.meet,
      row.market,
      row.combination,
      row.odds == null ? "" : Number(row.odds).toLocaleString("en-US"),
      formatBoolean(row.is_hit),
      formatBoolean(row.is_capped_odds),
      row.arrival_order,
    ];

    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value ?? "";
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

async function loadOddsForSelection() {
  const button = document.getElementById("load-odds");
  const date = document.getElementById("date-select").value;
  const market = document.getElementById("market-select").value;
  const status = document.getElementById("odds-status");

  if (!date) return;

  button.disabled = true;
  status.textContent = `Loading odds for ${date}...`;

  try {
    const payload = await loadOddsPayload(date, market);
    renderOddsRows(payload.rows);
    const suffix = payload.truncated ? ` Showing first ${formatter.format(payload.returned)}.` : "";
    status.textContent = `${formatter.format(payload.total)} rows for ${payload.date}.${suffix}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function loadOddsPayload(date, market) {
  try {
    const response = await fetch(`./api/odds?date=${encodeURIComponent(date)}&market=${encodeURIComponent(market)}&limit=5000`);
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    return loadStaticOddsPayload(date, market);
  }
}

async function loadStaticOddsPayload(date, market) {
  const manifest = window.staticOddsManifest ?? await loadStaticOddsManifest();
  window.staticOddsManifest = manifest;
  const item = manifest.dates.find((entry) => entry.date === date);
  if (!item) {
    return { date, rows: [], returned: 0, total: 0, truncated: false };
  }
  const payload = await readGzipJson(`./${item.file}`);
  const filteredRows = market === "all" ? payload.rows : payload.rows.filter((row) => row[2] === market);
  const rows = filteredRows.slice(0, 5000);
  return {
    date,
    rows,
    total: filteredRows.length,
    returned: rows.length,
    truncated: filteredRows.length > rows.length,
  };
}

function wireOddsControls() {
  document.getElementById("load-odds").addEventListener("click", loadOddsForSelection);
  document.getElementById("market-select").addEventListener("change", loadOddsForSelection);
  document.getElementById("date-select").addEventListener("change", loadOddsForSelection);
}

function wireCoverageToggle(summary) {
  const buttons = document.querySelectorAll("[data-chart]");
  const chart = document.getElementById("year-chart");

  function render(mode) {
    for (const button of buttons) {
      button.classList.toggle("active", button.dataset.chart === mode);
    }

    if (mode === "raw") {
      renderBars(chart, Object.entries(summary.raw.by_year), null, "files");
      return;
    }

    renderBars(chart, Object.entries(summary.parsed.by_year), "rows", "rows");
  }

  for (const button of buttons) {
    button.addEventListener("click", () => render(button.dataset.chart));
  }
  render("raw");
}

async function init() {
  const response = await fetch("./data/site-summary.json");
  if (!response.ok) {
    throw new Error(`Failed to load site summary: ${response.status}`);
  }
  const summary = await response.json();

  setText("raw-files", formatter.format(summary.raw.file_count));
  setText("raw-size", `${summary.raw.size_mb.toLocaleString()} MB on disk`);
  setText("parquet-files", formatter.format(summary.parsed.file_count));
  setText("parquet-size", `${summary.parsed.size_mb.toLocaleString()} MB on disk`);
  setText("parsed-rows", compactNumber(summary.parsed.row_count));
  setText("generated-at", `Generated metadata: ${new Date(summary.generated_at).toLocaleString()}`);

  wireCoverageToggle(summary);
  renderBars(
    document.getElementById("kind-chart"),
    Object.entries(summary.parsed.by_kind).sort((a, b) => b[1].rows - a[1].rows),
    "rows",
    "rows",
  );
  renderSamples(summary.raw.samples);
  renderSchemaOptions(summary.parsed.schemas);
  wireOddsControls();
  await loadAvailableDates();
  await loadOddsForSelection();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="panel"><h1>Unable to load dataset metadata</h1><p>${error.message}</p></main>`;
});
