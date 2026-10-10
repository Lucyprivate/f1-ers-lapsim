param(
    [ValidateSet('audit-v6','run-v6','verify-v6','test-v6')][string]$Mode='audit-v6',
    [string]$PythonPath='', [switch]$Offline
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) {
    foreach ($candidate in @('.venv/Scripts/python.exe','../../work/venv/Scripts/python.exe')) {
        if (Test-Path -LiteralPath $candidate) {$PythonPath=(Resolve-Path -LiteralPath $candidate).Path;break}
    }
}
if (-not $PythonPath) {$PythonPath=(Get-Command python -ErrorAction Stop).Source}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$v6Temp=Join-Path $PSScriptRoot '.cache/test-tmp'
New-Item -ItemType Directory -Path $v6Temp -Force | Out-Null
$env:TEMP=$v6Temp
$env:TMP=$v6Temp
if ($Mode -eq 'test-v6') { & $PythonPath -m unittest discover -s tests -p 'test_v6*.py' -v }
else { & $PythonPath -m f1ers_v6 $Mode --offline }
exit $LASTEXITCODE
