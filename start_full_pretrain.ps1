param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $PSScriptRoot "data/tinystories_full/training_config.json"
$config = Get-Content -LiteralPath $configPath -Raw -Encoding utf8 | ConvertFrom-Json
$manifestPath = Join-Path $PSScriptRoot "data/tinystories_full/tokenized_manifest.json"
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestHash -ne $config.data_manifest_sha256) {
    throw "Prepared data manifest has changed. Run tools.prepare_full_data again."
}
$pythonPath = $config.python
$trainingArguments = @("-u", "-m", "codes.pretrain") + @($config.arguments)
if ($CheckOnly) {
    $trainingArguments += "--dry-run"
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonPath @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pretraining command exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
