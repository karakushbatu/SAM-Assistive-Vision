$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw ".venv is missing. Run scripts/setup_local_cuda.ps1 first."
}

& $python -m uvicorn main:app --host 0.0.0.0 --port 8000
