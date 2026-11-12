# Carry a kernel's results into its next version:
#   kernel output -> results\<gate> -> private dataset harukadia/bitcycle-<gate>-results -> push.
# A new version starts with an empty disk and reads only that dataset, so a version that
# failed or was cut never loses what earlier versions finished.
#
#   .\kaggle\sync.ps1 gate1                    # after a version ended with work left
#   .\kaggle\sync.ps1 gate2 -Ref some/branch   # run a branch of bitcycle instead of main
param(
    [Parameter(Mandatory)][ValidateSet("gate1", "gate2")][string]$Gate,
    [string]$Ref = "main"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$kernel = "harukadia/bitcycle-$Gate"
$dataset = "$kernel-results"
$results = Join-Path (Split-Path $PSScriptRoot) "results\$Gate"
$tmp = Join-Path ([IO.Path]::GetTempPath()) "bitcycle-$Gate"

if ((kaggle kernels status $kernel) -match "RUNNING|QUEUED") { throw "$kernel is still running" }
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $results, "$tmp\out", "$tmp\data", "$tmp\push" | Out-Null

# A version that failed before its first result has no output: results\<gate> then stays as it is.
kaggle kernels output $kernel -p "$tmp\out" | Out-Null
if (Test-Path "$tmp\out\results\$Gate") { Copy-Item "$tmp\out\results\$Gate\*" $results -Force }

Get-ChildItem $results -File | Copy-Item -Destination "$tmp\data"
@{ title = "bitcycle-$Gate-results"; id = $dataset; licenses = @(@{ name = "apache-2.0" }) } |
    ConvertTo-Json | Set-Content "$tmp\data\dataset-metadata.json"
if ((kaggle datasets status $dataset 2>&1 | Out-String) -match "ready") {
    kaggle datasets version -p "$tmp\data" -m "results after $kernel"
} else {
    kaggle datasets create -p "$tmp\data"
}
if ($LASTEXITCODE) { throw "dataset upload failed" }
do { Start-Sleep 10 } until ((kaggle datasets status $dataset 2>&1 | Out-String) -match "ready")

Copy-Item "$PSScriptRoot\$Gate\*" "$tmp\push"
(Get-Content "$tmp\push\run.py" -Raw).Replace('"BITCYCLE_REF", "main"', "`"BITCYCLE_REF`", `"$Ref`"") |
    Set-Content "$tmp\push\run.py" -NoNewline
kaggle kernels push -p "$tmp\push"
if ($LASTEXITCODE) { throw "kernel push failed" }
