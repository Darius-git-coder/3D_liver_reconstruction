param(
    [ValidateSet("smoke", "short", "final")]
    [string]$Stage = "smoke",
    [string]$Config = "",
    [switch]$SkipTrain,
    [switch]$SkipEval
)

$pythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$resolvedConfig = $Config
if (-not $resolvedConfig) {
    $localConfig = "configs/ablation_${Stage}.local.json"
    if (Test-Path $localConfig) {
        $resolvedConfig = $localConfig
    }
}
if (-not $resolvedConfig) {
    switch ($Stage) {
        "smoke" { $resolvedConfig = "configs/ablation_smoke.example.json" }
        "short" { $resolvedConfig = "configs/ablation_short.example.json" }
        "final" { $resolvedConfig = "configs/ablation_final.example.json" }
    }
}

$argsList = @("scripts/run_experiment_suite.py", "--config", $resolvedConfig)
if ($SkipTrain) {
    $argsList += "--skip_train"
}
if ($SkipEval) {
    $argsList += "--skip_eval"
}

& $pythonExe @argsList
