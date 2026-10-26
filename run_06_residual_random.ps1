param(
    [ValidateSet("audit", "smoke", "run")]
    [string]$Mode = "audit",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $Python = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "GPU Python not found. Pass -Python with the full path to the Python executable used for 04."
    }
    $env:PYTHONPATH = Join-Path $PWD "src"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $config = "configs/residual_random_gpu.json"
    if ($Mode -eq "smoke") { $config = "configs/residual_random_smoke_gpu.json" }
    if ($Mode -eq "audit") {
        & $Python experiments/06_residual_random.py --config $config --audit-only
    } else {
        & $Python experiments/06_residual_random.py --config $config
    }
    if ($LASTEXITCODE -ne 0) { throw "06 failed. See the error above; do not delete the completed results." }
} finally {
    Pop-Location
}
