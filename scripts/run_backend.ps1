param(
  [string]$Host = "127.0.0.1",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  throw "No se encontró el Python del venv en: $python. Crea el venv en .venv o ajusta la ruta."
}

Set-Location $repoRoot

& $python -m uvicorn backend.app.main:app --reload --host $Host --port $Port
