param(
    [ValidateSet("audit","smoke","run")]
    [string]$Mode="audit",
    [string]$Python="",
    [int]$MaxNewOrigins=0
)
$ErrorActionPreference="Stop"
$oldPythonPath=$env:PYTHONPATH
$oldCublas=$env:CUBLAS_WORKSPACE_CONFIG
Push-Location $PSScriptRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) { $Python=Join-Path $PWD ".venv_gpu\Scripts\python.exe" }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Pass -Python with your GPU Python executable." }
    if ($MaxNewOrigins -lt 0) { throw "MaxNewOrigins must be zero (all) or positive." }
    $env:PYTHONPATH=Join-Path $PWD "src"
    $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
    $arguments=@("experiments/07_shap_kernel.py","--mode",$Mode)
    if ($MaxNewOrigins -gt 0) { $arguments+=@("--max-new-origins",[string]$MaxNewOrigins) }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Kernel SHAP stopped. Preserve checkpoints and origin caches; inspect the error above." }
} finally {
    Pop-Location
    $env:PYTHONPATH=$oldPythonPath
    $env:CUBLAS_WORKSPACE_CONFIG=$oldCublas
}
