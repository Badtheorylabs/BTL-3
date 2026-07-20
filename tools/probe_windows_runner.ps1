param(
    [Parameter(Mandatory = $true)]
    [string]$Runner,
    [string]$Output = "runner-cli-contract.json"
)

$ErrorActionPreference = "Stop"
$runnerPath = (Resolve-Path $Runner).Path
$requiredFlags = @(
    "--model",
    "--port",
    "--host",
    "--no-webui",
    "--offline",
    "-np"
)

$helpText = (& $runnerPath --help 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Runner --help exited with code $LASTEXITCODE"
}
foreach ($flag in $requiredFlags) {
    if (-not $helpText.Contains($flag)) {
        throw "Runner --help lacks required flag: $flag"
    }
}

$contract = [ordered]@{
    schema_version = 1
    platform = "windows-amd64-cuda13"
    runner_sha256 = (Get-FileHash $runnerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    required_flags = $requiredFlags
}
$json = $contract | ConvertTo-Json -Depth 3
$outputPath = [IO.Path]::GetFullPath($Output)
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, $utf8NoBom)
Write-Output $outputPath
