$ErrorActionPreference = 'Stop'

# This coordinator is independent of the failed lifecycle coordinator.  It
# owns only fresh v2 output/work/runtime paths and never reuses v1 artifacts.
$mnemosRoot = 'D:\Development\mnemos\mnemos\.worktrees\functional-local-memory'
$mnemosSnapshot = 'C:\Users\Ant\AppData\Local\Temp\mnemos-source-retrieval-comparison-v1'
$mnemosSource = 'fc72cc251aeafa161a1a390269b8dfa34d83cf2b'
$mnemosPython = Join-Path $mnemosRoot '.venv\Scripts\python.exe'
$mnemosRuntime = 'C:\Users\Ant\AppData\Local\Temp\mnemos-local-supplemental-v2'
$mnemosDataset = Join-Path $mnemosSnapshot 'benchmarks\local_memory\datasets\local-memory-heldout-v3.jsonl'
$mnemosDatasetSha = '036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc'
$mnemosEndpoint = 'http://127.0.0.1:11434'
$mnemosPins = @{
    'qwen3.5:9b' = '6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7'
    'nomic-embed-text:latest' = '0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f'
}
$mnemosSettings = @{
    model = 'qwen3.5:9b'
    embedding_model = 'nomic-embed-text:latest'
    repetitions = 3
    think = $false
    temperature = 0.0
    max_tokens = 512
    timeout_seconds = 120
    top_k = 5
    max_context_tokens = 1200
}
$mnemosStages = @(
    @{
        name = 'encoding'
        module = 'encoding_comparator_run'
        protocol = '2026-09-18-encoding-comparison-protocol-v1.md'
        protocol_sha = 'f403c5d49352a649a59a8aad38953a8f94699e0aa00596b35e32f617ff7520bf'
        output = 'encoding-comparison-local-v2'
        work = 'mnemos-encoding-comparison-local-v2'
    }
    @{
        name = 'retrieval'
        module = 'retrieval_comparison_run'
        protocol = '2026-09-18-retrieval-comparison-protocol-v1.md'
        protocol_sha = '3220dda551ddaf2774296107f2c8872ee7a7db99c71d05f8096ca46aaf3aa1ec'
        output = 'retrieval-comparison-local-v2'
        work = 'mnemos-retrieval-comparison-local-v2'
    }
)

function Save-MnemosState {
    $mnemosState.updated_at = [DateTime]::UtcNow.ToString('o')
    $mnemosTemp = Join-Path $mnemosRuntime 'state.tmp'
    [IO.File]::WriteAllText($mnemosTemp, ($mnemosState | ConvertTo-Json -Depth 20))
    Move-Item -LiteralPath $mnemosTemp -Destination (Join-Path $mnemosRuntime 'state.json') -Force
}

function Confirm-MnemosBinding {
    $mnemosHead = & git -C $mnemosSnapshot rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $mnemosHead -ne $mnemosSource) { throw 'Source commit mismatch.' }
    $mnemosDirty = & git -C $mnemosSnapshot status --porcelain
    if ($LASTEXITCODE -ne 0 -or $mnemosDirty) { throw 'Frozen source is dirty.' }
    if (-not (Test-Path -LiteralPath $mnemosDataset)) { throw 'Dataset is missing.' }
    if ((Get-FileHash -LiteralPath $mnemosDataset -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mnemosDatasetSha) { throw 'Dataset changed.' }
    foreach ($mnemosStage in $mnemosStages) {
        $mnemosProtocolPath = Join-Path $mnemosSnapshot ('docs\plans\' + $mnemosStage.protocol)
        if ((Get-FileHash -LiteralPath $mnemosProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mnemosStage.protocol_sha) { throw 'Protocol changed.' }
    }
    $mnemosModels = (Invoke-RestMethod -Uri ($mnemosEndpoint + '/api/tags') -TimeoutSec 30).models
    foreach ($mnemosName in $mnemosPins.Keys) {
        if (($mnemosModels | Where-Object name -eq $mnemosName).digest -ne $mnemosPins[$mnemosName]) { throw 'Model digest mismatch.' }
    }
}

function Read-MnemosJsonObject([string] $mnemosPath) {
    if (-not (Test-Path -LiteralPath $mnemosPath)) { throw "Missing stage artifact: $mnemosPath" }
    $mnemosValue = Get-Content -LiteralPath $mnemosPath -Raw | ConvertFrom-Json
    if ($null -eq $mnemosValue) { throw "Invalid stage artifact: $mnemosPath" }
    return $mnemosValue
}

function Confirm-MnemosStageTerminal($mnemosStage) {
    $mnemosManifestPath = Join-Path $mnemosStage.work_path 'manifest.json'
    try {
        $mnemosManifest = Read-MnemosJsonObject $mnemosManifestPath
        $mnemosStage.manifest_status = [string]$mnemosManifest.status
        if ($mnemosStage.manifest_status -eq 'completed') {
            $mnemosStage.terminal = 'completed'
            return $false
        }
        if ($mnemosStage.manifest_status -eq 'incomplete') {
            $mnemosStage.terminal = 'incomplete'
            $mnemosStage.status = 'finished_incomplete'
            return $true
        }
        throw "Stage $($mnemosStage.name) returned manifest status '$($mnemosStage.manifest_status)'; execution/provider failure must be reviewed before continuing."
    } catch {
        if (-not $mnemosStage.terminal) {
            $mnemosStage.status = 'failed'
            $mnemosStage.terminal = 'execution_failure'
            $mnemosStage.failure_reason = if ($mnemosStage.manifest_status) {
                'manifest_failure'
            } else {
                'missing_or_invalid_manifest'
            }
        }
        throw
    }
}

if (Test-Path -LiteralPath $mnemosRuntime) { throw 'Supplemental v2 runtime exists; inspect instead of overwriting.' }
foreach ($mnemosStage in $mnemosStages) {
    $mnemosStage.output_path = Join-Path $mnemosRoot ('docs\validation\2026-09-19\' + $mnemosStage.output)
    $mnemosStage.work_path = Join-Path 'C:\Users\Ant\AppData\Local\Temp' $mnemosStage.work
    if ((Test-Path -LiteralPath $mnemosStage.output_path) -or (Test-Path -LiteralPath $mnemosStage.work_path)) { throw 'Comparison v2 output/work directory already exists.' }
}
if (-not (Test-Path -LiteralPath $mnemosPython)) { throw 'Bound Python interpreter is missing.' }

Confirm-MnemosBinding
New-Item -ItemType Directory -Path $mnemosRuntime | Out-Null
$mnemosState = @{
    status = 'waiting_for_endpoint'
    pid = $PID
    started_at = [DateTime]::UtcNow.ToString('o')
    source_commit = $mnemosSource
    snapshot = $mnemosSnapshot
    endpoint = $mnemosEndpoint
    dataset_sha256 = $mnemosDatasetSha
    model_digests = $mnemosPins
    settings = $mnemosSettings
    stages = $mnemosStages
    any_incomplete = $false
    cost = 'unknown; provider usage and local GPU timing are recorded by the child reports when available'
    original_failed_run_preserved = $true
}
Save-MnemosState

try {
    foreach ($mnemosStage in $mnemosStages) {
        Confirm-MnemosBinding
        $mnemosArguments = @(
            '-u', '-m', ('benchmarks.local_memory.' + $mnemosStage.module),
            '--dataset', $mnemosDataset,
            '--protocol', (Join-Path $mnemosSnapshot ('docs\plans\' + $mnemosStage.protocol)),
            '--output-dir', $mnemosStage.output_path,
            '--work-dir', $mnemosStage.work_path,
            '--endpoint', $mnemosEndpoint,
            '--model', $mnemosSettings.model,
            '--embedding-model', $mnemosSettings.embedding_model,
            '--repetitions', [string]$mnemosSettings.repetitions,
            '--think', 'false'
        )
        $mnemosStage.command = @($mnemosPython) + $mnemosArguments
        $mnemosStage.status = 'running'
        $mnemosStage.started_at = [DateTime]::UtcNow.ToString('o')
        $mnemosState.status = 'running'
        Save-MnemosState

        foreach ($mnemosVariable in @([Environment]::GetEnvironmentVariables('Process').Keys)) {
            if ($mnemosVariable -eq 'PYTHONPATH' -or $mnemosVariable.StartsWith('MNEMOS_')) {
                [Environment]::SetEnvironmentVariable($mnemosVariable, $null, 'Process')
            }
        }
        Push-Location $mnemosSnapshot
        try {
            & $mnemosPython @mnemosArguments 1> (Join-Path $mnemosRuntime ($mnemosStage.name + '.stdout.log')) 2> (Join-Path $mnemosRuntime ($mnemosStage.name + '.stderr.log'))
            $mnemosStage.exit_code = $LASTEXITCODE
        } finally { Pop-Location }
        $mnemosStage.finished_at = [DateTime]::UtcNow.ToString('o')
        if ($mnemosStage.exit_code -ne 0) {
            $mnemosStage.terminal = 'execution_failure'
            $mnemosStage.failure_reason = 'child_nonzero_exit'
            $mnemosState.status = 'failed'
            $mnemosState.error = "Stage $($mnemosStage.name) exited with code $($mnemosStage.exit_code)."
            Save-MnemosState
            throw $mnemosState.error
        }
        $mnemosStageWasIncomplete = Confirm-MnemosStageTerminal $mnemosStage
        if ($mnemosStageWasIncomplete) {
            $mnemosState.any_incomplete = $true
            $mnemosState.status = 'stage_completed_incomplete'
        } else {
            $mnemosStage.status = 'finished'
            $mnemosState.status = 'stage_completed'
        }
        Save-MnemosState
    }
    $mnemosState.status = if ($mnemosState.any_incomplete) { 'finished_with_incomplete' } else { 'finished' }
    Save-MnemosState
} catch {
    if (-not $mnemosState.error) { $mnemosState.error = $_.Exception.Message }
    if ($mnemosState.status -ne 'failed') { $mnemosState.status = 'failed' }
    Save-MnemosState
    throw
}
