. "$PSScriptRoot/relore-env.ps1"
$reloreExe = Join-Path $reloreProject '.venv-relore/Scripts/relored.exe'
if (-not (Test-Path -LiteralPath $reloreExe)) {
    throw 'Jalankan scripts/setup-relore.ps1 terlebih dahulu.'
}
Push-Location $reloreProject
try {
    & $reloreExe @args
    $reloreExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $reloreExit
