[CmdletBinding()]
param(
    [ValidateSet("Adapter", "Full")]
    [string]$Scope = "Adapter",
    [string]$DataRoot = "data",
    [string]$RunRoot = "",
    [string]$Python = "python",
    [string]$Device = "",
    [switch]$Install,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SourceRoot = Join-Path $RepositoryRoot "src"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $SourceRoot
}
else {
    $env:PYTHONPATH = "$SourceRoot$([System.IO.Path]::PathSeparator)$env:PYTHONPATH"
}

function Invoke-Python {
    $Arguments = $args
    Write-Host ("> {0} {1}" -f $Python, ($Arguments -join " ")) -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Resolve-PythonPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    $Resolved = & $Python -c `
        "from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())" `
        $PathValue
    if ($LASTEXITCODE -ne 0) {
        throw "Could not canonicalise path: $PathValue"
    }
    return [string]$Resolved
}

Push-Location $RepositoryRoot
try {
    if ($Install) {
        Invoke-Python -m pip install --upgrade pip
        Invoke-Python -m pip install -e ".[dev]"
    }

    $DataRootPath = Resolve-PythonPath ([System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($RepositoryRoot, $DataRoot)
    ))
    if ((Split-Path -Leaf $DataRootPath) -ne "data") {
        throw "DataRoot must name the supplied 'data' directory: $DataRootPath"
    }
    if ((-not $DryRun) -and (-not (Test-Path -LiteralPath $DataRootPath -PathType Container))) {
        throw "Controlled processed-data directory is missing: $DataRootPath"
    }
    if ([string]::IsNullOrWhiteSpace($RunRoot)) {
        $Stamp = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
        $RunRoot = Join-Path "retraining_runs" ("{0}_{1}" -f $Scope.ToLowerInvariant(), $Stamp)
    }
    $RunRootPath = Resolve-PythonPath ([System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($RepositoryRoot, $RunRoot)
    ))
    $DataPrefix = $DataRootPath.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    if (($RunRootPath -eq $DataRootPath) -or $RunRootPath.StartsWith(
        $DataPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "RunRoot must be outside the read-only data directory."
    }
    New-Item -ItemType Directory -Force -Path $RunRootPath | Out-Null

    Write-Host "M14.3-v2 $Scope protocol retraining" -ForegroundColor Green
    Write-Host "Repository: $RepositoryRoot"
    Write-Host "Data root: $DataRootPath"
    Write-Host "Run root: $RunRootPath"

    Invoke-Python scripts/capture_environment.py `
        --output (Join-Path $RunRootPath "environment.json")
    Invoke-Python -m pytest -q -p no:cacheprovider

    $Arguments = @(
        "-m", "ecg_pcg_denoise.retrain",
        "--scope", $Scope.ToLowerInvariant(),
        "--data-root", $DataRootPath,
        "--run-root", $RunRootPath,
        "--python", $Python
    )
    if (-not [string]::IsNullOrWhiteSpace($Device)) {
        $Arguments += @("--device", $Device)
    }
    if ($DryRun) {
        $Arguments += "--dry-run"
    }
    Invoke-Python @Arguments
    Write-Host "M14.3-v2 $Scope retraining workflow completed." -ForegroundColor Green
}
finally {
    Pop-Location
}
