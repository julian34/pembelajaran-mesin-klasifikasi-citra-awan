param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$reloreProject = Split-Path -Parent $PSScriptRoot
$relorePython = Join-Path $reloreProject '.venv-relore/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $relorePython)) {
    & $Python -m venv (Join-Path $reloreProject '.venv-relore')
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan virtual environment gagal; gunakan -Python dengan path Python >= 3.10.' }
}
& $relorePython -m pip install --disable-pip-version-check -r (Join-Path $reloreProject 'requirements-relore.txt')
if ($LASTEXITCODE -ne 0) { throw 'Instalasi Relore gagal.' }
. "$PSScriptRoot/relore-env.ps1"
& (Join-Path $reloreProject '.venv-relore/Scripts/relored.exe') migrate
if ($LASTEXITCODE -ne 0) { throw 'Migrasi database Relore gagal.' }
& $relorePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Pemeriksaan dependensi gagal.' }
Write-Output 'Relore siap. Lihat docs/RELORE.md untuk server dan pengindeksan GitHub.'
