param(
    [switch]$DownloadSmallWhisper = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$hf = Join-Path $repoRoot ".venv\Scripts\hf.exe"

& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt

if ($DownloadSmallWhisper) {
    & $hf download Systran/faster-whisper-small | Out-Host
}

& $python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
