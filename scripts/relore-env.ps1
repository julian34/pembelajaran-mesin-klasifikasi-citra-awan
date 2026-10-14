$ErrorActionPreference = 'Stop'
$reloreProject = Split-Path -Parent $PSScriptRoot
$reloreConfig = Get-Content -LiteralPath (Join-Path $reloreProject 'configs/relore.json') -Raw | ConvertFrom-Json
$reloreData = Join-Path $reloreProject '.relore'
New-Item -ItemType Directory -Path $reloreData -Force | Out-Null
# Always use this project's local index and clone storage.
$env:RELORE_DATABASE_URL = 'sqlite:///' + ((Join-Path $reloreData 'index.db') -replace '\\', '/')
$env:RELORE_CLONE_ROOT = Join-Path $reloreData 'clones'
$env:RELORE_API = $reloreConfig.api
$env:RELORE_REPO = $reloreConfig.repo
$env:PYTHONUTF8 = '1'
