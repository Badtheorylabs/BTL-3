param(
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$BuildDirectory = ""
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Source = Join-Path $Root "native\llama.cpp"
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $Source "build-btl3-cuda-windows"
}
$Output = [System.IO.Path]::GetFullPath($Output)
if (Test-Path $Output) { throw "Output already exists: $Output" }
if (-not $env:CUDA_PATH) { throw "CUDA Toolkit 13.0.x is required." }

cmake -S $Source -B $BuildDirectory -A x64 `
    -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real" `
    -DGGML_CUDA=ON `
    -DGGML_BACKEND_DL=ON `
    -DGGML_NATIVE=OFF `
    -DBUILD_SHARED_LIBS=ON `
    -DLLAMA_OPENSSL=OFF `
    -DLLAMA_BUILD_TESTS=ON `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_UI=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
cmake --build $BuildDirectory --config Release --target `
    llama-server llama-cli test-btl3-avq-cuda test-btl3-int4-cuda `
    test-btl3-vocab-cuda --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

$Stage = Join-Path $env:TEMP ("btl3-cuda-" + [guid]::NewGuid())
New-Item -ItemType Directory $Stage | Out-Null
try {
    $Bin = Join-Path $BuildDirectory "bin\Release"
    if (-not (Test-Path $Bin)) { $Bin = Join-Path $BuildDirectory "bin" }
    Copy-Item (Join-Path $Bin "llama-server.exe") $Stage
    Copy-Item (Join-Path $Bin "llama-cli.exe") $Stage
    Get-ChildItem $Bin -Filter "*.dll" | Copy-Item -Destination $Stage
    foreach ($Pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        Get-ChildItem (Join-Path $env:CUDA_PATH "bin") -Filter $Pattern |
            Copy-Item -Destination $Stage
    }
    $CudaEula = Join-Path $env:CUDA_PATH "EULA.txt"
    if (Test-Path $CudaEula) {
        Copy-Item $CudaEula (Join-Path $Stage "LICENSE.NVIDIA-CUDA")
    }
    python (Join-Path $Root "tools\build_cuda_bundle.py") `
        --target windows-x86_64 --source $Stage --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Bundle packaging failed." }
} finally {
    Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
}
