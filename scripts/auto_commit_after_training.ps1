<#
.SYNOPSIS
  Pantau artefak hasil pelatihan sebuah eksperimen, lalu commit otomatis begitu
  pelatihan selesai (file `notes.md` muncul & stabil). Push opsional.

.DESCRIPTION
  `write_outputs` (scripts/train.py) menulis artefak ke experiments/<Exp>/ dengan
  urutan: config.yaml -> metrics.json -> training_history.csv ->
  classification_report.txt -> confusion_matrix.png -> notes.md (TERAKHIR).
  Jadi kemunculan notes.md yang stabil = penanda andal bahwa pelatihan + evaluasi
  test sudah tuntas. Skrip ini menunggu penanda itu, lalu `git add -A` + commit
  (pesan memuat test macro-F1 dari metrics.json). checkpoints/*.pt & data/ tetap
  di-ignore oleh .gitignore, jadi yang masuk hanya artefak ringan + kode/dokumen.

  Jalankan di terminal TERPISAH dan biarkan terbuka selama pelatihan (bisa berjam-jam).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\auto_commit_after_training.ps1
  # menunggu exp_005_densenet121, commit saja (tanpa push)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\auto_commit_after_training.ps1 -Exp exp_005_densenet121 -Push
  # commit lalu push ke origin/<branch saat ini>
#>
param(
  [string]$Exp = "exp_005_densenet121",
  [int]$IntervalSec = 60,
  [switch]$Push,
  [string]$Message = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot      # scripts\.. = akar repo
Set-Location $repo

$marker      = Join-Path $repo "experiments\$Exp\notes.md"
$metricsPath = Join-Path $repo "experiments\$Exp\metrics.json"

Write-Host "[auto-commit] repo   : $repo"
Write-Host "[auto-commit] exp    : $Exp"
Write-Host "[auto-commit] penanda: $marker"
Write-Host "[auto-commit] menunggu pelatihan selesai (cek tiap ${IntervalSec}s). Ctrl+C untuk batal."

function Get-Stamp([string]$p) {
  if (Test-Path $p) {
    $f = Get-Item $p
    return ("{0}:{1}" -f $f.Length, $f.LastWriteTimeUtc.Ticks)
  }
  return ""
}

# Tunggu sampai notes.md ADA dan STABIL (ukuran+mtime tak berubah antar dua pemeriksaan).
while ($true) {
  if (Test-Path $marker) {
    $s1 = Get-Stamp $marker; $m1 = Get-Stamp $metricsPath
    Start-Sleep -Seconds $IntervalSec
    $s2 = Get-Stamp $marker; $m2 = Get-Stamp $metricsPath
    if ($s1 -ne "" -and $s1 -eq $s2 -and $m1 -eq $m2) {
      Write-Host ("[auto-commit] {0} artefak terdeteksi & stabil." -f (Get-Date -Format "HH:mm:ss"))
      break
    }
    Write-Host ("[auto-commit] {0} artefak masih ditulis, tunggu lagi..." -f (Get-Date -Format "HH:mm:ss"))
  } else {
    Write-Host ("[auto-commit] {0} belum ada artefak; menunggu..." -f (Get-Date -Format "HH:mm:ss"))
    Start-Sleep -Seconds $IntervalSec
  }
}

# Ambil test macro-F1 untuk pesan commit.
$mf1 = "?"
try {
  $m = Get-Content $metricsPath -Raw | ConvertFrom-Json
  if ($m.test -ne $null -and $null -ne $m.test.macro_f1) {
    $mf1 = "{0:N4}" -f [double]$m.test.macro_f1
  }
} catch {
  Write-Host "[auto-commit] (peringatan) gagal membaca macro_f1 dari metrics.json: $_"
}

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = "exp($Exp): hasil pelatihan + persiapan ResNet-18 & perbaikan dokumen`n`n" +
             "Test macro-F1 $Exp = $mf1.`n" +
             "Termasuk: artefak experiments/$Exp/, reorganisasi docs per-arsitektur, " +
             "notebook 06 ResNet-18 + config exp_008 + diagram, dan pembaruan status/total data (31.851)."
}

Write-Host "[auto-commit] menyiapkan commit..."
git add -A

$staged = git diff --cached --name-only
if (-not $staged) {
  Write-Host "[auto-commit] tidak ada perubahan untuk di-commit. Selesai."
  exit 0
}
Write-Host "[auto-commit] berkas yang akan di-commit:"
$staged | ForEach-Object { Write-Host ("    " + $_) }

$tmp = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tmp -Value $Message -Encoding utf8
git commit -F $tmp
$code = $LASTEXITCODE
Remove-Item $tmp -Force
if ($code -ne 0) {
  Write-Host "[auto-commit] commit GAGAL (git keluar $code). Periksa pesan di atas."
  exit $code
}
Write-Host "[auto-commit] commit dibuat (test macro-F1 $mf1)."

if ($Push) {
  $branch = (git branch --show-current).Trim()
  Write-Host "[auto-commit] push ke origin/$branch ..."
  git push origin $branch
  if ($LASTEXITCODE -ne 0) { Write-Host "[auto-commit] push GAGAL."; exit $LASTEXITCODE }
  Write-Host "[auto-commit] push selesai."
} else {
  Write-Host "[auto-commit] push dilewati (tambahkan -Push bila ingin otomatis push)."
}

Write-Host "[auto-commit] SELESAI."
