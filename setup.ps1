# CrossMind setup — installs dependencies on Python 3.12
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Installing CrossMind dependencies (Python 3.12)..." -ForegroundColor Cyan
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
Write-Host "Setup complete. Run: .\run.ps1" -ForegroundColor Green
