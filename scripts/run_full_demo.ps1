param(
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 8000,
    [int]$StartupTimeoutSeconds = 120,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw ".venv is missing. Run scripts/setup_local_cuda.ps1 first."
}

$stdoutLog = Join-Path $repoRoot "tests\output\demo_full_uvicorn_stdout.log"
$stderrLog = Join-Path $repoRoot "tests\output\demo_full_uvicorn_stderr.log"

$env:YOLO_CONFIG_DIR = Join-Path $repoRoot ".cache\ultralytics"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $repoRoot ".cache\paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:PADDLE_HOME = Join-Path $repoRoot ".cache\paddle"

$serverProcess = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "main:app", "--host", $ServerHost, "--port", "$Port") `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

try {
    $healthUrl = "http://$ServerHost`:$Port/health"
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    $ready = $false

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if ($serverProcess.HasExited) {
            throw "Backend process exited before readiness. Check tests/output/demo_full_uvicorn_stderr.log"
        }
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
            if ($response.status -eq "healthy") {
                $ready = $true
                break
            }
        } catch {
            # Server may still be warming models.
        }
    }

    if (-not $ready) {
        throw "Backend startup timeout. Check tests/output/demo_full_uvicorn_stderr.log"
    }

    & $python scripts\demo_terminal.py --server "ws://$ServerHost`:$Port/ws/vision" @DemoArgs
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
