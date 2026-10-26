param(
    [ValidateSet("audit","overview","explain","recover")]
    [string]$Mode="explain",
    [int]$MaxReplays=1,
    [string]$Python=""
)
$ErrorActionPreference="Stop"
Push-Location $PSScriptRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) { $Python=Join-Path $PWD ".venv_gpu\Scripts\python.exe" }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Pass -Python with the full path to your existing GPU Python." }
    $env:PYTHONPATH=Join-Path $PWD "src"
    $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
    if ($Mode -eq "recover") {
        Write-Host "Explicit recovery enabled for at most $MaxReplays missing models."
        & $Python experiments/06_interpretation_precision_fix.py --mode explain --replay-missing --max-replays $MaxReplays
    } else {
        Write-Host "Existing checkpoints only; no model training."
        & $Python experiments/06_interpretation_precision_fix.py --mode $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Inference stopped. Preserve all checkpoints; inspect inference_precision_checks and the error above." }
} finally { Pop-Location }
