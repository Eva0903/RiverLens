param(
    [ValidateSet("audit","overview","explain","recover")]
    [string]$Mode="overview",
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
        Write-Host "EXPLICIT MODEL RECOVERY: at most $MaxReplays original models will be trained to their saved best epochs."
        & $Python experiments/06_interpretation_random.py --config configs/interpretation_random.json --mode explain --replay-missing --max-replays $MaxReplays
    } else {
        & $Python experiments/06_interpretation_random.py --config configs/interpretation_random.json --mode $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Interpretation failed. See the error above; preserve existing results and checkpoints." }
} finally { Pop-Location }
