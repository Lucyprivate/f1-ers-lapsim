param(
    [ValidateSet('diagnose-v5','run-v5','verify-v5','test-v5')][string]$Mode = 'diagnose-v5',
    [string]$PythonPath = '',
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) {
    foreach ($candidate in @('.venv/Scripts/python.exe', '../../work/venv/Scripts/python.exe')) {
        if (Test-Path -LiteralPath $candidate) { $PythonPath = (Resolve-Path -LiteralPath $candidate).Path; break }
    }
}
if (-not $PythonPath) { $PythonPath = (Get-Command python -ErrorAction Stop).Source }
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
if ($Mode -eq 'test-v5') {
    $v5Temp = Join-Path $PSScriptRoot '.cache/test-tmp'
    New-Item -ItemType Directory -Path $v5Temp -Force | Out-Null
    $env:TEMP = $v5Temp
    $env:TMP = $v5Temp
    & $PythonPath -m unittest discover -s tests -p 'test_v5*.py' -v
} elseif ($Offline) {
    & $PythonPath -m f1ers_v5 $Mode --offline
} else {
    & $PythonPath -m f1ers_v5 $Mode
}
exit $LASTEXITCODE
