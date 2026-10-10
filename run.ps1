param(
    [ValidateSet('run-v4','audit-v4','verify-v4','restore-artifacts','test-v4','run-v3','audit-v3','verify-v3','run','run-v2','verify-v2','test','fetch','audit','serve')]
    [string]$Mode = 'run-v4',
    [int]$Episodes = 60,
    [int]$BcEpochs = 600,
    [string]$PythonPath = '',
    [int]$Port = 8765,
    [string]$Archive = '',
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) {
    $localCandidates = @(
        (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
        (Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe')
    )
    foreach ($candidate in $localCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $PythonPath = (Resolve-Path -LiteralPath $candidate).Path
            break
        }
    }
}
if (-not $PythonPath) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $PythonPath = $pythonCommand.Source }
}
if (-not $PythonPath) { throw 'Python environment missing. Follow README setup or supply -PythonPath.' }
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
switch ($Mode) {
    'run-v4' {
        if ($Offline) { & $PythonPath -m f1ers_v4 run-v4 --offline }
        else { & $PythonPath -m f1ers_v4 run-v4 }
    }
    'audit-v4' { & $PythonPath -m f1ers_v4 audit-v4 }
    'verify-v4' { & $PythonPath -m f1ers_v4 verify-v4 }
    'restore-artifacts' {
        if (-not $Archive) { throw 'Supply -Archive with the frozen v3 ZIP path.' }
        & $PythonPath -m f1ers_v4 restore-artifacts --archive $Archive
    }
    'test-v4' {
        $stageV4TestTemp = Join-Path $PSScriptRoot '.cache/test-tmp'
        New-Item -ItemType Directory -Path $stageV4TestTemp -Force | Out-Null
        $env:TEMP = $stageV4TestTemp
        $env:TMP = $stageV4TestTemp
        & $PythonPath -m unittest discover -s tests -p 'test_v4*.py' -v
    }
    'run-v3' { & $PythonPath -m f1ers_v3 run-v3 }
    'audit-v3' { & $PythonPath -m f1ers_v3 audit-v3 }
    'verify-v3' { & $PythonPath -m f1ers_v3 verify-v3 }
    'run' { & $PythonPath -m f1ers run --episodes $Episodes --bc-epochs $BcEpochs }
    'run-v2' { & $PythonPath -m f1ers run-v2 }
    'verify-v2' { & $PythonPath -m f1ers verify-v2 }
    'test' {
        $stageTestTemp = Join-Path $PSScriptRoot '.cache/test-tmp'
        New-Item -ItemType Directory -Path $stageTestTemp -Force | Out-Null
        $env:TEMP = $stageTestTemp
        $env:TMP = $stageTestTemp
        & $PythonPath -m unittest discover -s tests -v
    }
    'serve' { & $PythonPath -m http.server $Port --bind 127.0.0.1 --directory results }
    'fetch' { & $PythonPath -m f1ers fetch }
    'audit' { & $PythonPath -m f1ers audit }
}
exit $LASTEXITCODE
