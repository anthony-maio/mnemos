const state = {
  settings: null,
  view: null,
  selectedMemoryId: null,
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

function activeProvider() {
  return document.getElementById("llm-provider").value;
}

function providerSettings(settings) {
  const provider = activeProvider();
  return settings.providers?.[provider] || {};
}

function refreshProviderFields() {
  if (!state.settings) {
    return;
  }
  const provider = providerSettings(state.settings);
  document.getElementById("provider-url").value = provider.base_url || "";
  document.getElementById("provider-key").value = "";
  document.getElementById("provider-key-status").textContent = provider.configured
    ? "A secret is already configured here. Leave blank to keep it."
    : "No secret saved yet for this provider.";
}

function fillForm(view) {
  state.view = view;
  state.settings = view.settings;
  const { settings, paths, warnings } = view;
  document.getElementById("onboarding-mode").value = settings.onboarding?.mode || "dev";
  document.getElementById("preferred-host").value =
    settings.onboarding?.preferred_host || "claude-code";
  document.getElementById("llm-provider").value = settings.llm?.provider || "mock";
  document.getElementById("llm-model").value = settings.llm?.model || "";
  document.getElementById("embedding-provider").value = settings.embedding?.provider || "simple";
  document.getElementById("embedding-model").value = settings.embedding?.model || "";
  document.getElementById("store-type").value = settings.storage?.type || "sqlite";
  document.getElementById("sqlite-path").value = settings.storage?.sqlite_path || "";
  document.getElementById("qdrant-path").value = settings.storage?.qdrant_path || "";
  document.getElementById("qdrant-url").value = settings.storage?.qdrant_url || "";
  refreshProviderFields();

  const heroStatus = document.getElementById("hero-status");
  heroStatus.innerHTML = [
    statusPill(`Global: ${paths.global_config}`),
    paths.project_config ? statusPill(`Project: ${paths.project_config}`, "warn") : "",
    ...warnings.map((warning) => statusPill(warning, "warn")),
  ].join("");
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
      qdrant_path: document.getElementById("qdrant-path").value.trim() || null,
      qdrant_url: document.getElementById("qdrant-url").value.trim() || null,
    },
    providers: {
      [provider]: {
        base_url: document.getElementById("provider-url").value.trim(),
        api_key: document.getElementById("provider-key").value,
      },
    },
  };
}

async function refreshSettings() {
  const view = await api("/api/settings");
  fillForm(view);
}

async function refreshHealth() {
  const report = await api("/api/health");
  const summary = document.getElementById("health-summary");
  const checks = document.getElementById("health-checks");
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
}

async function refreshMemory() {
  const snapshot = await api("/api/memory");
  document.getElementById("memory-summary").textContent = `${snapshot.count} chunks currently visible from the configured store.`;
  if (!state.selectedMemoryId && snapshot.recent.length > 0) {
    state.selectedMemoryId = snapshot.recent[0].id;
  }
  document.getElementById("memory-list").innerHTML = snapshot.recent
    .map(
      (item) => `
      <button class="memory-item ${state.selectedMemoryId === item.id ? "selected" : ""}" data-memory-id="${item.id}">
        <div class="memory-meta">
          <span>${item.scope}${item.scope_id ? `:${item.scope_id}` : ""}</span>
          <span>accessed ${item.access_count}x</span>
          <span>${item.updated_at}</span>
        </div>
        <p>${item.content}</p>
      </button>
    `,
    )
    .join("");
  document.querySelectorAll(".memory-item").forEach((element) => {
    element.addEventListener("click", async () => {
      state.selectedMemoryId = element.dataset.memoryId;
      await refreshMemory();
    });
  });
  if (state.selectedMemoryId) {
    await refreshMemoryDetail(state.selectedMemoryId);
  } else {
    document.getElementById("memory-detail").textContent = "No stored memories yet.";
  }
}

async function refreshMemoryDetail(chunkId) {
  try {
    const payload = await api(`/api/memory/${encodeURIComponent(chunkId)}`);
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
    document.getElementById("memory-detail").innerHTML = `
      <div class="detail-section">
        <p class="section-kicker">Scope</p>
        <h3>${payload.scope}${payload.scope_id ? `:${payload.scope_id}` : ""}</h3>
        <p>${payload.content}</p>
      </div>
      <div class="detail-section">
        <p class="section-kicker">Provenance</p>
        <ul class="detail-list">
          <li><strong>Stored by:</strong> ${payload.provenance?.stored_by || "unknown"}</li>
          <li><strong>Channel:</strong> ${payload.provenance?.ingest_channel || "n/a"}</li>
          <li><strong>Reason:</strong> ${payload.provenance?.encoding_reason || "n/a"}</li>
          <li><strong>Version:</strong> ${payload.version}</li>
          <li><strong>Access count:</strong> ${payload.access_count}</li>
        </ul>
      </div>
      <div class="detail-section">
        <p class="section-kicker">Revision History</p>
        <ul class="detail-list">${history || "<li>No rewrites yet.</li>"}</ul>
      </div>
      <div class="detail-section">
        <p class="section-kicker">Graph</p>
        <ul class="detail-list">
          <li><strong>Neighbors:</strong> ${payload.graph?.neighbor_count || 0}</li>
        </ul>
        <ul class="detail-list">${neighbors || "<li>No graph neighbors yet.</li>"}</ul>
      </div>
      <div class="detail-section">
        <p class="section-kicker">Metadata</p>
        <pre class="console">${JSON.stringify(payload.metadata || {}, null, 2)}</pre>
      </div>
    `;
  } catch (error) {
    document.getElementById("memory-detail").textContent = error.message;
  }
}

async function save(scope) {
  const result = await api(`/api/settings/${scope}`, {
    method: "POST",
    body: JSON.stringify(collectPayload()),
  });
  document.getElementById("integration-preview").textContent = `Saved ${scope} settings to ${result.path}`;
  await refreshSettings();
  await refreshHealth();
}

async function importSetup() {
  const result = await api("/api/import", { method: "POST", body: "{}" });
  fillForm(result);
  document.getElementById("integration-preview").textContent =
    `Imported existing setup from: ${result.sources.join(", ") || "no external host config found"}`;
}

async function previewHost(host, action) {
  const result = await api(`/api/integrations/${host}/${action}`, {
    method: "POST",
    body: "{}",
  });
  document.getElementById("integration-preview").textContent = result.preview;
}

async function smokeTest() {
  const result = await api("/api/smoke", { method: "POST", body: "{}" });
  document.getElementById("integration-preview").textContent = JSON.stringify(result, null, 2);
}

function bindEvents() {
  document.getElementById("save-global").addEventListener("click", () => save("global"));
  document.getElementById("save-project").addEventListener("click", () => save("project"));
  document.getElementById("import-button").addEventListener("click", importSetup);
  document.getElementById("smoke-button").addEventListener("click", smokeTest);
  document.getElementById("refresh-health").addEventListener("click", refreshHealth);
  document.getElementById("refresh-memory").addEventListener("click", refreshMemory);
  document.getElementById("llm-provider").addEventListener("change", refreshProviderFields);

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

async function init() {
  bindEvents();
  await refreshSettings();
  await refreshHealth();
  await refreshMemory();
}

init().catch((error) => {
  document.getElementById("integration-preview").textContent = error.message;
});
