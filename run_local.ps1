$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  throw 'The project virtual environment is missing. Create it with: python -m venv .venv'
}

if ([string]::IsNullOrWhiteSpace($env:DATABASE_URL)) {
  $env:DATABASE_URL = 'sqlite:///./attendai_demo.db'
  Write-Host 'Using the local attendai_demo.db database.'
} else {
  Write-Host 'Using the configured database URL.'
}
if ([string]::IsNullOrWhiteSpace($env:FACE_STRICT)) {
  $env:FACE_STRICT = 'true'
}

Write-Host "Starting AttendAI at http://127.0.0.1:8765"
Write-Host 'Face matching is required for camera attendance. Manual attendance remains available in the admin page.'
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
exit $LASTEXITCODE
