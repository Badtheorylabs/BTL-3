$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Server = Join-Path $BundleRoot "libexec\llama-server.exe"
$ModelName = "BTL-3-Compact-AVQ2.gguf"

function Find-Btl3Model {
    if ($env:BTL3_MODEL) { return $env:BTL3_MODEL }
    $Candidates = @(
        (Join-Path $BundleRoot "model\$ModelName"),
        (Join-Path (Get-Location) $ModelName)
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -PathType Leaf $Candidate) { return $Candidate }
    }
    return $null
}

function Get-GpuMemoryMiB {
    if ($env:BTL3_GPU_MEMORY_MIB) { return [int]$env:BTL3_GPU_MEMORY_MIB }
    $Value = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
        Select-Object -First 1
    if (-not $Value) { throw "NVIDIA GPU memory detection failed." }
    return [int](($Value -replace "[^0-9]", ""))
}

function Select-Context([int]$MemoryMiB) {
    if ($MemoryMiB -ge 96000) { return 131072 }
    if ($MemoryMiB -ge 48000) { return 98304 }
    if ($MemoryMiB -ge 28000) { return 65536 }
    if ($MemoryMiB -ge 20000) { return 32768 }
    return 16384
}

$Model = Find-Btl3Model
if (-not $Model) { throw "BTL-3 model not found. Set BTL3_MODEL to $ModelName." }
if (-not (Test-Path -PathType Leaf $Server)) {
    throw "Packaged llama-server is missing: $Server"
}
$MemoryMiB = Get-GpuMemoryMiB
$Context = if ($env:BTL3_CTX_SIZE) {
    [int]$env:BTL3_CTX_SIZE
} else {
    Select-Context $MemoryMiB
}
$HostName = if ($env:BTL3_HOST) { $env:BTL3_HOST } else { "127.0.0.1" }
$Port = if ($env:BTL3_PORT) { $env:BTL3_PORT } else { "8080" }
$Parallel = if ($env:BTL3_PARALLEL) { $env:BTL3_PARALLEL } else { "1" }
$Alias = if ($env:BTL3_MODEL_ALIASES) { $env:BTL3_MODEL_ALIASES } else { "BTL-3" }
$GpuLayers = if ($env:BTL3_GPU_LAYERS) { $env:BTL3_GPU_LAYERS } else { "99" }
$MaxTokens = if ($env:BTL3_MAX_TOKENS) { $env:BTL3_MAX_TOKENS } else { "2048" }
$RepeatPenalty = if ($env:BTL3_REPEAT_PENALTY) { $env:BTL3_REPEAT_PENALTY } else { "1.10" }
$RepeatLastN = if ($env:BTL3_REPEAT_LAST_N) { $env:BTL3_REPEAT_LAST_N } else { "512" }
$ChatTemplateKwargs = if ($env:BTL3_ENABLE_THINKING -in @("1", "true", "TRUE")) {
    '{"enable_thinking":true}'
} else {
    '{"enable_thinking":false}'
}
$env:PATH = "$(Join-Path $BundleRoot 'lib');$env:PATH"
$env:GGML_BACKEND_PATH = Join-Path $BundleRoot "lib\ggml-cuda.dll"
$Arguments = @(
    "--model", $Model, "--alias", $Alias, "--host", $HostName, "--port", $Port,
    "--ctx-size", $Context, "--parallel", $Parallel, "--n-gpu-layers", $GpuLayers,
    "--jinja", "--reasoning", "auto", "--reasoning-format", "deepseek",
    "--chat-template-kwargs", $ChatTemplateKwargs,
    "--cont-batching", "--n-predict", $MaxTokens, "--repeat-penalty", $RepeatPenalty,
    "--repeat-last-n", $RepeatLastN, "--cache-ram", "0", "--no-warmup", "--no-ui"
)
if ($env:BTL3_API_KEY) { $Arguments += @("--api-key", $env:BTL3_API_KEY) }
$Arguments += $args
if ($env:BTL3_PRINT_COMMAND -eq "1") {
    "executable=$Server"
    "model=$Model"
    "host=$HostName"
    "port=$Port"
    "ctx_size=$Context"
    "gpu_memory_mib=$MemoryMiB"
    "gpu_layers=$GpuLayers"
    "max_tokens=$MaxTokens"
    "repeat_penalty=$RepeatPenalty"
    "repeat_last_n=$RepeatLastN"
    "thinking_enabled=$($ChatTemplateKwargs -eq '{"enable_thinking":true}')"
    exit 0
}
& $Server @Arguments
exit $LASTEXITCODE
