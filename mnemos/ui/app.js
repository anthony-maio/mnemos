const state = {
  settingsView: null,
  dashboard: null,
  selectedMemoryId: null,
  selectedMemoryMeta: null,
  selectedMemoryVersion: null,
  csrfToken: null,
};

async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (method !== "GET" && state.csrfToken) {
    headers["X-Mnemos-CSRF"] = state.csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    headers,
  });
  const contentType = response.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    throw new Error(typeof payload === "string" ? payload : JSON.stringify(payload));
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusPill(label, tone = "pass") {
  return `<span class="pill ${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function metricCard(label, value, detail = "") {
  return `
    <article class="metric-card">
      <p class="section-kicker">${escapeHtml(label)}</p>
      <h3>${escapeHtml(value)}</h3>
      <p>${escapeHtml(detail)}</p>
    </article>
  `;
}

function activeProvider() {
  return document.getElementById("llm-provider").value;
}

function providerSettings(settings) {
  const provider = activeProvider();
  return settings.providers?.[provider] || {};
}

function preferredHost(settings) {
  return settings.onboarding?.preferred_host || "claude-code";
}

function activePresetId(settings) {
  return settings.sleep?.recall_gated_plasticity_enabled ? "stable-memory" : "starter-sqlite";
}

function refreshProviderFields() {
  if (!state.settingsView) {
    return;
  }
  const provider = providerSettings(state.settingsView.settings);
  document.getElementById("provider-url").value = provider.base_url || "";
  document.getElementById("provider-key").value = "";
  document.getElementById("provider-key-status").textContent = provider.configured
    ? "A secret is already configured here. Leave blank to keep it."
    : "No secret saved yet for this provider.";
}

function fillForm(view) {
  state.settingsView = view;
  const { settings } = view;

  document.getElementById("onboarding-mode").value = settings.onboarding?.mode || "dev";
  document.getElementById("preferred-host").value = preferredHost(settings);
  document.getElementById("llm-provider").value = settings.llm?.provider || "mock";
  document.getElementById("llm-model").value = settings.llm?.model || "";
  document.getElementById("embedding-provider").value = settings.embedding?.provider || "simple";
  document.getElementById("embedding-model").value = settings.embedding?.model || "";
  document.getElementById("store-type").value = settings.storage?.type || "sqlite";
  document.getElementById("sqlite-path").value = settings.storage?.sqlite_path || "";
  document.getElementById("recall-gated-enabled").checked =
    settings.sleep?.recall_gated_plasticity_enabled || false;
  document.getElementById("recall-min-supporting-episodes").value =
    settings.sleep?.recall_min_supporting_episodes || 2;
  document.getElementById("recall-similarity-threshold").value =
    settings.sleep?.recall_similarity_threshold ?? 0.82;

  renderPresets(view.presets || [], activePresetId(settings));
  refreshProviderFields();
}

function renderPresets(presets, activeId) {
  document.getElementById("preset-list").innerHTML = presets
    .map(
      (preset) => `
      <button class="preset-card ${preset.id === activeId ? "active" : ""}" data-preset-id="${escapeHtml(preset.id)}">
        <div>
          <div class="preset-title">
            <strong>${escapeHtml(preset.label)}</strong>
            ${preset.recommended ? '<span class="pill warn">Recommended</span>' : ""}
          </div>
          <p>${escapeHtml(preset.description)}</p>
        </div>
      </button>
    `,
    )
    .join("");

  document.querySelectorAll(".preset-card").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.presetId));
  });
}

function applyPreset(presetId) {
  const preset = (state.settingsView?.presets || []).find((item) => item.id === presetId);
  if (!preset) {
    return;
  }
  document.getElementById("store-type").value = preset.payload?.storage?.type || "sqlite";
  document.getElementById("recall-gated-enabled").checked =
    preset.payload?.sleep?.recall_gated_plasticity_enabled || false;
  document.getElementById("recall-min-supporting-episodes").value =
    preset.payload?.sleep?.recall_min_supporting_episodes || 2;
  document.getElementById("recall-similarity-threshold").value =
    preset.payload?.sleep?.recall_similarity_threshold ?? 0.82;
  renderPresets(state.settingsView?.presets || [], presetId);
}

function collectPayload() {
  const provider = activeProvider();
  return {
    onboarding: {
      mode: document.getElementById("onboarding-mode").value,
      preferred_host: document.getElementById("preferred-host").value,
    },
    llm: {
      provider,
      model: document.getElementById("llm-model").value.trim(),
    },
    embedding: {
      provider: document.getElementById("embedding-provider").value,
      model: document.getElementById("embedding-model").value.trim(),
    },
    storage: {
      type: document.getElementById("store-type").value,
      sqlite_path: document.getElementById("sqlite-path").value.trim(),
    },
    sleep: {
      recall_gated_plasticity_enabled: document.getElementById("recall-gated-enabled").checked,
      recall_min_supporting_episodes: Number.parseInt(
        document.getElementById("recall-min-supporting-episodes").value || "2",
        10,
      ),
      recall_similarity_threshold: Number.parseFloat(
        document.getElementById("recall-similarity-threshold").value || "0.82",
      ),
    },
    providers: {
      [provider]: {
        base_url: document.getElementById("provider-url").value.trim(),
        api_key: document.getElementById("provider-key").value,
      },
    },
  };
}

function renderDashboard(dashboard) {
  state.dashboard = dashboard;
  const setup = dashboard.setup;
  const health = dashboard.health;
  const memory = dashboard.memory;
  const host = dashboard.host.current;
  const activity = dashboard.activity;
  const scope = dashboard.scope || {};
  if (scope.profile) document.getElementById("project-profile").value = scope.profile;
  const jobs = dashboard.jobs || {};

  document.getElementById("hero-status").innerHTML = [
    statusPill(`Setup: ${setup.status}`, setup.status === "ready" ? "pass" : setup.status === "degraded" ? "warn" : "fail"),
    statusPill(`Host: ${host.label} (${host.tier})`, host.tier === "Tier 1" ? "pass" : "warn"),
    statusPill(`Store: ${setup.store_type}`),
    statusPill(`Chunks: ${memory.count}${memory.count_at_least ? "+" : ""}`),
  ].join("");

  document.getElementById("hero-metrics").innerHTML = [
    metricCard("SQLite path", setup.active_sqlite_path || setup.sqlite_path || "Not configured", setup.active_sqlite_path ? "Observed running service" : "Configured path; active store not observed"),
    metricCard("Health checks", `${health.summary.pass} pass / ${health.summary.warn} warn`, "Readiness at a glance"),
    (() => {
      const maintenance = activity.last_consolidation;
      const counts = maintenanceCounts(maintenance, jobs);
      return metricCard(
        "Last maintenance",
        maintenance?.present
          ? counts.completed !== null
            ? `${counts.completed} completed`
            : maintenance.status || "Observed"
          : "Not run yet",
        maintenance?.present
          ? maintenanceDetail(maintenance, jobs)
          : "Run maintenance when you want to inspect durable jobs",
      );
    })(),
  ].join("");

  document.getElementById("dashboard-metrics").innerHTML = [
    metricCard("Current host", host.label, `${host.tier} · ${host.status}`),
    metricCard(
      "Visible memory",
      `${memory.count} chunks`,
      `project ${memory.counts_by_scope.project} · workspace ${memory.counts_by_scope.workspace} · global ${memory.counts_by_scope.global}`,
    ),
    metricCard(
      "Last smoke check",
      activity.last_smoke_test.present ? activity.last_smoke_test.status : "Not run yet",
      activity.last_smoke_test.present
        ? Object.entries(activity.last_smoke_test.steps || {})
            .map(([name, status]) => `${name}:${status}`)
            .join(" · ")
        : "Recommended once per setup change",
    ),
    metricCard(
      "Recall gating",
      memory.recall_gated.enabled ? "Enabled" : "Off",
      memory.recall_gated.enabled
        ? `${memory.recall_gated.stored_chunk_count} stored chunks carry recall support metadata`
        : "Use Stable Memory when you want more selective consolidation",
    ),
    metricCard(
      "Scope",
      scope.status || "Not configured",
      scope.profile ? `profile ${scope.profile}${scope.project?.id ? ` · project ${scope.project.id}` : ""}` : (scope.error || "Set an explicit profile and registered project"),
    ),
    metricCard(
      "Durable jobs",
      jobs.pending ?? "n/a",
      jobs.exhausted != null ? `${jobs.exhausted} exhausted · ${jobs.retrying || 0} retrying` : "Service job status not observed",
    ),
  ].join("");

  document.getElementById("next-actions").innerHTML = (dashboard.next_actions || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");

  document.getElementById("activity-cards").innerHTML = [
    renderActivityCard("Smoke check", activity.last_smoke_test),
    renderActivityCard("Consolidation", activity.last_consolidation, jobs, memory),
  ].join("");

  renderHostCards(dashboard.host.hosts || [], host.name);
  renderHealth(health);
  renderMemory(memory);
}

function renderActivityCard(label, activity, jobs = {}, memory = {}) {
  if (!activity?.present) {
    let detail = "No recent activity recorded in this UI session.";
    if (label === "Consolidation") {
      const counts = maintenanceCounts({}, jobs);
      if (memory.last_consolidation?.present) {
        detail = `${memory.last_consolidation.stored_chunk_count} consolidated fact evidence recorded.`;
      } else if (counts.completed !== null || counts.pending !== null || counts.exhausted !== null) {
        detail = `Queue: ${counts.completed ?? 0} completed, ${counts.pending ?? 0} pending, ${counts.exhausted ?? 0} exhausted.`;
      } else {
        detail = "No maintenance job or consolidated fact evidence recorded.";
      }
    }
    return `
      <article class="activity-card">
        <p class="section-kicker">${escapeHtml(label)}</p>
        <h3>Not run yet</h3>
        <p>${escapeHtml(detail)}</p>
      </article>
    `;
  }
  const counts = maintenanceCounts(activity, jobs);
  const details = [];
  if (counts.completed !== null) details.push(`${counts.completed} jobs completed`);
  if (counts.pending !== null) details.push(`${counts.pending} pending`);
  if (counts.exhausted !== null) details.push(`${counts.exhausted} exhausted`);
  if (activity.facts_preview?.length) details.push(`Fact evidence: ${activity.facts_preview.join(" · ")}`);
  if (activity.error) details.push(activity.error);
  if (details.length === 0) details.push("No durable job or fact evidence recorded.");
  return `
    <article class="activity-card">
      <p class="section-kicker">${escapeHtml(label)}</p>
      <h3>${escapeHtml(activity.status)}</h3>
      <p>${escapeHtml(details.join("; "))}</p>
    </article>
  `;
}

function renderHostCards(hosts, currentHostName) {
  document.getElementById("host-stack").innerHTML = hosts
    .map(
      (host) => `
      <article class="host-card ${host.name === currentHostName ? "current" : ""}" data-host="${escapeHtml(host.name)}">
        <div class="host-card-head">
          <div>
            <h3>${escapeHtml(host.label)}</h3>
            <p>${escapeHtml(host.summary)}</p>
          </div>
          <div class="status-cluster">
            <span class="pill ${host.tier === "Tier 1" ? "pass" : "warn"}">${escapeHtml(host.tier)}</span>
            ${host.recommended ? '<span class="pill warn">Primary</span>' : ""}
          </div>
        </div>
        <p class="host-best-path">${escapeHtml(host.best_path)}</p>
        <div class="host-actions">
          <button class="button button-ghost preview-host">Preview</button>
          <button class="button apply-host">Apply</button>
        </div>
      </article>
    `,
    )
    .join("");

  document.querySelectorAll(".preview-host").forEach((button) => {
    button.addEventListener("click", (event) => {
      const host = event.target.closest(".host-card").dataset.host;
      previewHost(host, "preview");
    });
  });
  document.querySelectorAll(".apply-host").forEach((button) => {
    button.addEventListener("click", (event) => {
      const host = event.target.closest(".host-card").dataset.host;
      previewHost(host, "apply");
    });
  });
}

function renderHealth(report) {
  const summary = document.getElementById("health-summary");
  const checks = document.getElementById("health-checks");
  const actions = document.getElementById("health-actions");

  summary.innerHTML = [
    statusPill(`Status: ${report.status}`, report.status === "ready" ? "pass" : report.status === "degraded" ? "warn" : "fail"),
    statusPill(`Profile: ${report.profile}`),
    statusPill(`Store: ${report.store_type}`),
    statusPill(`LLM: ${report.llm_provider}`),
    statusPill(`Embedding: ${report.embedding_provider}`),
    statusPill(`Provider configured: ${Object.values(report.provider_readiness?.configured || {}).filter((item) => item?.configured).length}`),
    statusPill(`Provider observed: ${report.service?.status || "not_observed"}`, report.service?.status === "running" ? "pass" : "warn"),
  ].join("");

  checks.innerHTML = report.checks
    .map(
      (check) => `
      <li>
        <span class="check-name">${escapeHtml(check.name)}</span>
        <span>${escapeHtml(check.message)}</span>
      </li>
    `,
    )
    .join("");

  actions.innerHTML = (report.next_actions || report.recommendations || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function renderMemory(snapshot) {
  const maintenance = state.dashboard?.activity?.last_consolidation;
  const counts = maintenanceCounts(maintenance, snapshot.jobs || {});
  let consolidationSummary = "No maintenance or consolidated fact evidence recorded";
  if (snapshot.last_consolidation.present) {
    consolidationSummary = `${snapshot.last_consolidation.stored_chunk_count} consolidated fact(s) recorded`;
  } else if (maintenance?.present) {
    consolidationSummary = counts.completed !== null
      ? `maintenance completed ${counts.completed} job(s)`
      : `maintenance status ${maintenance.status || "observed"}`;
  } else if (counts.pending !== null || counts.exhausted !== null) {
    consolidationSummary = `queue ${counts.pending ?? 0} pending, ${counts.exhausted ?? 0} exhausted`;
  }
  document.getElementById("memory-summary").innerHTML = `
    <strong>${escapeHtml(snapshot.count)}${snapshot.count_at_least ? "+" : ""}</strong> chunks visible.
    ${snapshot.count_at_least ? escapeHtml(snapshot.listing_note || "Preview limited to 500 facts.") : ""}
    ${escapeHtml(consolidationSummary)}.
  `;

  if (!state.selectedMemoryId && snapshot.recent.length > 0) {
    state.selectedMemoryId = snapshot.recent[0].id;
    state.selectedMemoryMeta = snapshot.recent[0];
    state.selectedMemoryVersion = snapshot.recent[0].version || 1;
  }

  document.getElementById("memory-list").innerHTML = snapshot.recent
    .map(
      (item) => `
      <button
        class="memory-item ${state.selectedMemoryId === item.id ? "selected" : ""}"
        data-memory-id="${escapeHtml(item.id)}"
        data-scope="${escapeHtml(item.scope)}"
        data-scope-id="${escapeHtml(item.scope_id || "")}"
        data-version="${escapeHtml(item.version || 1)}"
      >
        <div class="memory-meta">
          <span>${escapeHtml(item.scope)}${item.scope_id ? `:${escapeHtml(item.scope_id)}` : ""}</span>
          <span>${escapeHtml(item.stored_by || "unknown source")}</span>
          <span>accessed ${escapeHtml(item.access_count)}x</span>
        </div>
        <p>${escapeHtml(item.content)}</p>
      </button>
    `,
    )
    .join("");

  document.querySelectorAll(".memory-item").forEach((element) => {
    element.addEventListener("click", async () => {
      state.selectedMemoryId = element.dataset.memoryId;
      state.selectedMemoryMeta = {
        id: element.dataset.memoryId,
        scope: element.dataset.scope,
        scope_id: element.dataset.scopeId || null,
      };
      state.selectedMemoryVersion = element.dataset.version || 1;
      await refreshMemory();
    });
  });

  if (state.selectedMemoryId) {
    refreshMemoryDetail();
  } else {
    document.getElementById("memory-detail").textContent = "No stored memories yet.";
  }
}

function detailSection(title, body) {
  return `
    <div class="detail-section">
      <p class="section-kicker">${escapeHtml(title)}</p>
      ${body}
    </div>
  `;
}

async function refreshMemoryDetail() {
  if (!state.selectedMemoryId || !state.selectedMemoryMeta) {
    return;
  }
  try {
    const query = document.getElementById("memory-query").value.trim();
    const params = new URLSearchParams();
    if (query) {
      params.set("query", query);
      params.set("current_scope", state.selectedMemoryMeta.scope || "project");
      if (state.selectedMemoryMeta.scope_id) {
        params.set("scope_id", state.selectedMemoryMeta.scope_id);
      }
    }
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const payload = await api(`/api/memory/${encodeURIComponent(state.selectedMemoryId)}${suffix}`);
    state.selectedMemoryVersion = payload.version || state.selectedMemoryVersion || 1;
    setMemoryActionAvailability(true);

    const history = (payload.history || [])
      .map(
        (entry) => `
          <li>
            <span>Previous version ${escapeHtml(entry.version)}</span>
            <span>${escapeHtml(entry.created_at)}</span>
            <p>${escapeHtml(entry.content)}</p>
          </li>
        `,
      )
      .join("");
    const neighbors = (payload.graph?.neighbors || [])
      .map(
        (neighbor) => `
          <li>
            <span>${escapeHtml(neighbor.id)}</span>
            <span>${escapeHtml(neighbor.weight)}</span>
          </li>
        `,
      )
      .join("");
    const evidence = (payload.evidence || [])
      .map(
        (entry) => `
          <li>
            <span>${escapeHtml(entry.excerpt || "Recorded evidence")}</span>
            <span>${escapeHtml(entry.source_ref || entry.source_event_id || "")}</span>
            <span>${escapeHtml(entry.created_at || entry.occurred_at || "")}</span>
          </li>
        `,
      )
      .join("");
    const retrieval = payload.retrieval
      ? `
        <ul class="detail-list">
          <li><strong>Scope match:</strong> ${escapeHtml(payload.retrieval.scope_match)}</li>
          <li><strong>Semantic candidate:</strong> ${escapeHtml(payload.retrieval.in_semantic_candidates)}</li>
          <li><strong>Semantic similarity:</strong> ${escapeHtml(payload.retrieval.semantic_similarity ?? "n/a")}</li>
          <li><strong>Graph activated:</strong> ${escapeHtml(payload.retrieval.graph_activated)}</li>
        </ul>
        <ul class="detail-list">
          ${payload.retrieval.explanation.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      `
      : "<p>Add a query above to explain why this memory would come back.</p>";

    const consolidation = `
      <ul class="detail-list">
        <li><strong>Stored by:</strong> ${escapeHtml(payload.provenance?.stored_by || "unknown")}</li>
        <li><strong>Consolidated at:</strong> ${escapeHtml(payload.metadata?.consolidation_time || "n/a")}</li>
        <li><strong>Recall support:</strong> ${escapeHtml(payload.metadata?.recall_supporting_episodes ?? "n/a")}</li>
        <li><strong>Recall score:</strong> ${escapeHtml(payload.metadata?.recall_support_score ?? "n/a")}</li>
      </ul>
    `;

    document.getElementById("memory-detail").innerHTML = [
      detailSection(
        "Scope",
        `<h3>${escapeHtml(payload.scope)}${payload.scope_id ? `:${escapeHtml(payload.scope_id)}` : ""}</h3><p>${escapeHtml(payload.content)}</p>`,
      ),
      detailSection(
        "Why Stored",
        `
          <ul class="detail-list">
            <li><strong>Channel:</strong> ${escapeHtml(payload.provenance?.ingest_channel || "n/a")}</li>
            <li><strong>Reason:</strong> ${escapeHtml(payload.provenance?.encoding_reason || "n/a")}</li>
            <li><strong>Version:</strong> ${escapeHtml(payload.version)}</li>
            <li><strong>Access count:</strong> ${escapeHtml(payload.access_count)}</li>
          </ul>
        `,
      ),
      detailSection("Evidence", `<ul class="detail-list">${evidence || "<li>No linked evidence recorded.</li>"}</ul>`),
      detailSection("Consolidation", consolidation),
      detailSection("Why Retrieved", retrieval),
      detailSection("Revision History", `<ul class="detail-list">${history || "<li>No rewrites yet.</li>"}</ul>`),
      detailSection(
        "Graph",
        `
          <ul class="detail-list">
            <li><strong>Neighbors:</strong> ${escapeHtml(payload.graph?.neighbor_count || 0)}</li>
          </ul>
          <ul class="detail-list">${neighbors || "<li>No graph neighbors yet.</li>"}</ul>
        `,
      ),
      detailSection("Metadata", `<pre class="console">${escapeHtml(JSON.stringify(payload.metadata || {}, null, 2))}</pre>`),
    ].join("");
  } catch (error) {
    document.getElementById("memory-detail").textContent = error.message;
    setMemoryActionAvailability(false);
  }
}

function numericValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function maintenanceCounts(activity, jobs = {}) {
  const queue = activity?.maintenance || {};
  return {
    completed: numericValue(activity?.completed ?? queue.completed ?? jobs.completed),
    pending: numericValue(queue.pending ?? jobs.pending),
    exhausted: numericValue(queue.exhausted ?? jobs.exhausted),
  };
}

function maintenanceDetail(activity, jobs = {}) {
  const counts = maintenanceCounts(activity, jobs);
  const details = [];
  if (counts.pending !== null) details.push(`${counts.pending} pending`);
  if (counts.exhausted !== null) details.push(`${counts.exhausted} exhausted`);
  if (details.length > 0) return details.join(" · ");
  if (activity?.status) return `Status: ${activity.status}`;
  return "Run maintenance when you want to inspect durable jobs";
}

function setMemoryActionAvailability(enabled) {
  document.getElementById("correct-memory").disabled = !enabled;
  document.getElementById("forget-memory").disabled = !enabled;
}

async function correctMemory() {
  if (!state.selectedMemoryId) return;
  const content = window.prompt("Correct this memory:");
  if (content === null || !content.trim()) return;
  try {
    const result = await api(`/api/memory/${encodeURIComponent(state.selectedMemoryId)}/correct`, {
      method: "POST",
      body: JSON.stringify({ content: content.trim(), expected_version: state.selectedMemoryVersion }),
    });
    document.getElementById("memory-action-status").textContent =
      result.status === "committed" ? `Correction queued. Processing: ${result.processing_status || "pending"}.` : `Correction not accepted: ${result.status || "unknown"}.`;
    await refreshDashboard();
  } catch (error) {
    document.getElementById("memory-action-status").textContent = error.message;
  }
}

async function forgetMemory() {
  if (!state.selectedMemoryId || !window.confirm("Forget this memory?")) return;
  try {
    const result = await api(`/api/memory/${encodeURIComponent(state.selectedMemoryId)}/forget`, {
      method: "POST",
      body: JSON.stringify({ expected_version: state.selectedMemoryVersion }),
    });
    if (result.forgotten !== true) throw new Error(`Forget was not confirmed: ${result.status || "unknown"}`);
    state.selectedMemoryId = null;
    state.selectedMemoryMeta = null;
    state.selectedMemoryVersion = null;
    document.getElementById("memory-action-status").textContent = "Memory forgotten.";
    setMemoryActionAvailability(false);
    await refreshDashboard();
  } catch (error) {
    document.getElementById("memory-action-status").textContent = error.message;
  }
}

async function refreshSettings() {
  const view = await api("/api/settings");
  fillForm(view);
}

async function loadCsrfToken() {
  const payload = await api("/api/csrf");
  state.csrfToken = payload.csrf_token;
}

async function refreshDashboard() {
  const dashboard = await api("/api/dashboard");
  renderDashboard(dashboard);
}

async function refreshMemory() {
  if (!state.dashboard) {
    await refreshDashboard();
    return;
  }
  renderMemory(state.dashboard.memory);
}

async function save(scope) {
  const result = await api(`/api/settings/${scope}`, {
    method: "POST",
    body: JSON.stringify(collectPayload()),
  });
  document.getElementById("integration-preview").textContent = `Saved ${scope} settings to ${result.path}`;
  await refreshSettings();
  await refreshDashboard();
}

async function importSetup() {
  const result = await api("/api/import", { method: "POST", body: "{}" });
  fillForm(result);
  document.getElementById("integration-preview").textContent =
    `Imported existing setup from: ${result.sources.join(", ") || "no external host config found"}`;
  await refreshDashboard();
}

async function previewHost(host, action) {
  const result = await api(`/api/integrations/${host}/${action}`, {
    method: "POST",
    body: "{}",
  });
  document.getElementById("integration-preview").textContent = result.preview;
  if (action === "apply") {
    await refreshDashboard();
  }
}

async function smokeTest() {
  const result = await api("/api/smoke", { method: "POST", body: "{}" });
  document.getElementById("integration-preview").textContent = JSON.stringify(result, null, 2);
  await refreshDashboard();
}

async function runConsolidation() {
  const result = await api("/api/consolidate", { method: "POST", body: "{}" });
  document.getElementById("integration-preview").textContent = JSON.stringify(result, null, 2);
  await refreshDashboard();
}

function bindEvents() {
  for (const [id, enabled] of [["enable-project", true], ["pause-project", false]]) {
    document.getElementById(id).addEventListener("click", async () => {
      try {
        const result = await api("/api/project", {method: "POST", body: JSON.stringify({profile: document.getElementById("project-profile").value.trim(), enabled})});
        document.getElementById("integration-preview").textContent = JSON.stringify(result, null, 2);
        await refreshDashboard();
      } catch (error) {
        document.getElementById("integration-preview").textContent = error.message;
      }
    });
  }
  document.getElementById("save-global").addEventListener("click", () => save("global"));
  document.getElementById("save-project").addEventListener("click", () => save("project"));
  document.getElementById("import-button").addEventListener("click", importSetup);
  document.getElementById("smoke-button").addEventListener("click", smokeTest);
  document.getElementById("consolidate-button").addEventListener("click", runConsolidation);
  document.getElementById("refresh-health").addEventListener("click", refreshDashboard);
  document.getElementById("refresh-memory").addEventListener("click", refreshDashboard);
  document.getElementById("refresh-dashboard").addEventListener("click", refreshDashboard);
  document.getElementById("explain-memory").addEventListener("click", refreshMemoryDetail);
  document.getElementById("correct-memory").addEventListener("click", correctMemory);
  document.getElementById("forget-memory").addEventListener("click", forgetMemory);
  document.getElementById("llm-provider").addEventListener("change", refreshProviderFields);
}

async function init() {
  bindEvents();
  await loadCsrfToken();
  await refreshSettings();
  await refreshDashboard();
}

init().catch((error) => {
  document.getElementById("integration-preview").textContent = error.message;
});
