const state = {
  settingsView: null,
  dashboard: null,
  selectedMemoryId: null,
  selectedMemoryMeta: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
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

function statusPill(label, tone = "pass") {
  return `<span class="pill ${tone}">${label}</span>`;
}

function metricCard(label, value, detail = "") {
  return `
    <article class="metric-card">
      <p class="section-kicker">${label}</p>
      <h3>${value}</h3>
      <p>${detail}</p>
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
      <button class="preset-card ${preset.id === activeId ? "active" : ""}" data-preset-id="${preset.id}">
        <div>
          <div class="preset-title">
            <strong>${preset.label}</strong>
            ${preset.recommended ? '<span class="pill warn">Recommended</span>' : ""}
          </div>
          <p>${preset.description}</p>
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

  document.getElementById("hero-status").innerHTML = [
    statusPill(`Setup: ${setup.status}`, setup.status === "ready" ? "pass" : setup.status === "degraded" ? "warn" : "fail"),
    statusPill(`Host: ${host.label} (${host.tier})`, host.tier === "Tier 1" ? "pass" : "warn"),
    statusPill(`Store: ${setup.store_type}`),
    statusPill(`Chunks: ${memory.count}`),
  ].join("");

  document.getElementById("hero-metrics").innerHTML = [
    metricCard("SQLite path", setup.sqlite_path || "Not configured", "Canonical local store"),
    metricCard("Health checks", `${health.summary.pass} pass / ${health.summary.warn} warn`, "Readiness at a glance"),
    metricCard(
      "Last consolidation",
      activity.last_consolidation.present
        ? `${activity.last_consolidation.facts_extracted} facts`
        : "Not run yet",
      activity.last_consolidation.present
        ? `${activity.last_consolidation.recall_supported} promoted / ${activity.last_consolidation.recall_filtered} filtered`
        : "Run consolidation when you want to inspect long-term writes",
    ),
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
  ].join("");

  document.getElementById("next-actions").innerHTML = (dashboard.next_actions || [])
    .map((item) => `<li>${item}</li>`)
    .join("");

  document.getElementById("activity-cards").innerHTML = [
    renderActivityCard("Smoke check", activity.last_smoke_test),
    renderActivityCard("Consolidation", activity.last_consolidation),
  ].join("");

  renderHostCards(dashboard.host.hosts || [], host.name);
  renderHealth(health);
  renderMemory(memory);
}

function renderActivityCard(label, activity) {
  if (!activity?.present) {
    return `
      <article class="activity-card">
        <p class="section-kicker">${label}</p>
        <h3>Not run yet</h3>
        <p>No recent activity recorded in this UI session.</p>
      </article>
    `;
  }
  const detail = activity.facts_preview?.length
    ? activity.facts_preview.join(" · ")
    : activity.error || "No extracted facts recorded.";
  return `
    <article class="activity-card">
      <p class="section-kicker">${label}</p>
      <h3>${activity.status}</h3>
      <p>${detail}</p>
    </article>
  `;
}

function renderHostCards(hosts, currentHostName) {
  document.getElementById("host-stack").innerHTML = hosts
    .map(
      (host) => `
      <article class="host-card ${host.name === currentHostName ? "current" : ""}" data-host="${host.name}">
        <div class="host-card-head">
          <div>
            <h3>${host.label}</h3>
            <p>${host.summary}</p>
          </div>
          <div class="status-cluster">
            <span class="pill ${host.tier === "Tier 1" ? "pass" : "warn"}">${host.tier}</span>
            ${host.recommended ? '<span class="pill warn">Primary</span>' : ""}
          </div>
        </div>
        <p class="host-best-path">${host.best_path}</p>
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
  ].join("");

  checks.innerHTML = report.checks
    .map(
      (check) => `
      <li>
        <span class="check-name">${check.name}</span>
        <span>${check.message}</span>
      </li>
    `,
    )
    .join("");

  actions.innerHTML = (report.next_actions || report.recommendations || [])
    .map((item) => `<li>${item}</li>`)
    .join("");
}

function renderMemory(snapshot) {
  document.getElementById("memory-summary").innerHTML = `
    <strong>${snapshot.count}</strong> chunks visible.
    Latest consolidation: ${
      snapshot.last_consolidation.present
        ? `${snapshot.last_consolidation.stored_chunk_count} stored chunk(s)`
        : "none recorded in the current store"
    }.
  `;

  if (!state.selectedMemoryId && snapshot.recent.length > 0) {
    state.selectedMemoryId = snapshot.recent[0].id;
    state.selectedMemoryMeta = snapshot.recent[0];
  }

  document.getElementById("memory-list").innerHTML = snapshot.recent
    .map(
      (item) => `
      <button
        class="memory-item ${state.selectedMemoryId === item.id ? "selected" : ""}"
        data-memory-id="${item.id}"
        data-scope="${item.scope}"
        data-scope-id="${item.scope_id || ""}"
      >
        <div class="memory-meta">
          <span>${item.scope}${item.scope_id ? `:${item.scope_id}` : ""}</span>
          <span>${item.stored_by || "unknown source"}</span>
          <span>accessed ${item.access_count}x</span>
        </div>
        <p>${item.content}</p>
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
      <p class="section-kicker">${title}</p>
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

    const history = (payload.history || [])
      .map(
        (entry) => `
          <li>
            <span>v${entry.from_version} -> v${entry.to_version}</span>
            <span>${entry.changed_at}</span>
            <p>${entry.previous_content}</p>
            <p>${entry.new_content}</p>
          </li>
        `,
      )
      .join("");
    const neighbors = (payload.graph?.neighbors || [])
      .map(
        (neighbor) => `
          <li>
            <span>${neighbor.id}</span>
            <span>${neighbor.weight}</span>
          </li>
        `,
      )
      .join("");
    const retrieval = payload.retrieval
      ? `
        <ul class="detail-list">
          <li><strong>Scope match:</strong> ${payload.retrieval.scope_match}</li>
          <li><strong>Semantic candidate:</strong> ${payload.retrieval.in_semantic_candidates}</li>
          <li><strong>Semantic similarity:</strong> ${payload.retrieval.semantic_similarity ?? "n/a"}</li>
          <li><strong>Graph activated:</strong> ${payload.retrieval.graph_activated}</li>
        </ul>
        <ul class="detail-list">
          ${payload.retrieval.explanation.map((item) => `<li>${item}</li>`).join("")}
        </ul>
      `
      : "<p>Add a query above to explain why this memory would come back.</p>";

    const consolidation = `
      <ul class="detail-list">
        <li><strong>Stored by:</strong> ${payload.provenance?.stored_by || "unknown"}</li>
        <li><strong>Consolidated at:</strong> ${payload.metadata?.consolidation_time || "n/a"}</li>
        <li><strong>Recall support:</strong> ${payload.metadata?.recall_supporting_episodes ?? "n/a"}</li>
        <li><strong>Recall score:</strong> ${payload.metadata?.recall_support_score ?? "n/a"}</li>
      </ul>
    `;

    document.getElementById("memory-detail").innerHTML = [
      detailSection(
        "Scope",
        `<h3>${payload.scope}${payload.scope_id ? `:${payload.scope_id}` : ""}</h3><p>${payload.content}</p>`,
      ),
      detailSection(
        "Why Stored",
        `
          <ul class="detail-list">
            <li><strong>Channel:</strong> ${payload.provenance?.ingest_channel || "n/a"}</li>
            <li><strong>Reason:</strong> ${payload.provenance?.encoding_reason || "n/a"}</li>
            <li><strong>Version:</strong> ${payload.version}</li>
            <li><strong>Access count:</strong> ${payload.access_count}</li>
          </ul>
        `,
      ),
      detailSection("Consolidation", consolidation),
      detailSection("Why Retrieved", retrieval),
      detailSection("Revision History", `<ul class="detail-list">${history || "<li>No rewrites yet.</li>"}</ul>`),
      detailSection(
        "Graph",
        `
          <ul class="detail-list">
            <li><strong>Neighbors:</strong> ${payload.graph?.neighbor_count || 0}</li>
          </ul>
          <ul class="detail-list">${neighbors || "<li>No graph neighbors yet.</li>"}</ul>
        `,
      ),
      detailSection("Metadata", `<pre class="console">${JSON.stringify(payload.metadata || {}, null, 2)}</pre>`),
    ].join("");
  } catch (error) {
    document.getElementById("memory-detail").textContent = error.message;
  }
}

async function refreshSettings() {
  const view = await api("/api/settings");
  fillForm(view);
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
  document.getElementById("save-global").addEventListener("click", () => save("global"));
  document.getElementById("save-project").addEventListener("click", () => save("project"));
  document.getElementById("import-button").addEventListener("click", importSetup);
  document.getElementById("smoke-button").addEventListener("click", smokeTest);
  document.getElementById("consolidate-button").addEventListener("click", runConsolidation);
  document.getElementById("refresh-health").addEventListener("click", refreshDashboard);
  document.getElementById("refresh-memory").addEventListener("click", refreshDashboard);
  document.getElementById("refresh-dashboard").addEventListener("click", refreshDashboard);
  document.getElementById("explain-memory").addEventListener("click", refreshMemoryDetail);
  document.getElementById("llm-provider").addEventListener("change", refreshProviderFields);
}

async function init() {
  bindEvents();
  await refreshSettings();
  await refreshDashboard();
}

init().catch((error) => {
  document.getElementById("integration-preview").textContent = error.message;
});
