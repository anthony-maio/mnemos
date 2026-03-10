const state = {
  settings: null,
  view: null,
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
  document.getElementById("memory-list").innerHTML = snapshot.recent
    .map(
      (item) => `
      <article class="memory-item">
        <div class="memory-meta">
          <span>${item.scope}${item.scope_id ? `:${item.scope_id}` : ""}</span>
          <span>accessed ${item.access_count}x</span>
          <span>${item.updated_at}</span>
        </div>
        <p>${item.content}</p>
      </article>
    `,
    )
    .join("");
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
