param(
  [switch]$RecreateVenv,
  [string]$RawDataPath
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

if ($RecreateVenv -and (Test-Path (Join-Path $repoRoot '.venv'))) {
  Remove-Item -Recurse -Force (Join-Path $repoRoot '.venv')
}

if (-not (Test-Path $venvPython)) {
  Write-Host 'Creando entorno virtual (.venv)...'
  python -m venv .venv
}

Write-Host 'Usando Python:' $venvPython

& $venvPython -m pip install --upgrade pip

if (Test-Path (Join-Path $repoRoot 'backend\requirements.txt')) {
  & $venvPython -m pip install -r (Join-Path $repoRoot 'backend\requirements.txt')
} elseif (Test-Path (Join-Path $repoRoot 'requirements.txt')) {
  & $venvPython -m pip install -r (Join-Path $repoRoot 'requirements.txt')
}

# Asegurar deps para train_pipeline
& $venvPython -m pip install lightgbm openpyxl

Write-Host 'Ejecutando train pipeline...'
$trainScript = (Join-Path $repoRoot 'backend\models\train_pipeline.py')
if ($RawDataPath -and $RawDataPath.Trim().Length -gt 0) {
  & $venvPython $trainScript --raw-data $RawDataPath
} else {
  & $venvPython $trainScript
}
