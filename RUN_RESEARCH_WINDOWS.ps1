<#
 Giorno--Tsallis HMM-6 | Reverse Attribution | APARCH family
 Windows PowerShell 5.1+ / PowerShell 7+

 Examples:
   .\RUN_RESEARCH_WINDOWS.ps1
   .\RUN_RESEARCH_WINDOWS.ps1 -Mode Research -Observations 3000 -Replications 50 -Scenarios 'calibration_anchor,tail_only' -Models aparch
   .\RUN_RESEARCH_WINDOWS.ps1 -Mode CompetitorsOnly -Output 'results\GI_HMM6_APARCH' -Models all

 The DGP and recovery stage never supplies true latent states/parameters to an estimator.
 The competitor stage is blind to those states until ex-post scientific reporting.
#>
[CmdletBinding()]
param(
    [ValidateSet('Pilot','Research','GenerateOnly','CompetitorsOnly')]
    [string]$Mode = 'Pilot',
    [ValidateRange(100,2000000)][int]$Observations = 750,
    [ValidateRange(1,100000)][int]$Replications = 2,
    [ValidateRange(1,64)][int]$HMMStarts = 4,
    [ValidateRange(1,64)][int]$CompetitorStarts = 2,
    [ValidateRange(20,20000)][int]$HMMMaxIter = 220,
    [ValidateRange(20,20000)][int]$CompetitorMaxIter = 160,
    [string]$Scenarios = 'calibration_anchor,tail_only',
    [string]$Models = 'all',
    [string]$Output = 'results\GI_HMM6_APARCH',
    [int]$Seed = 20261028,
    [ValidateRange(0,200000)][int]$LimitPaths = 0,
    [string]$Python = 'python',
    [switch]$InstallDependencies,
    [switch]$NoResume,
    [switch]$SkipReport
)

$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'

$root = $PSScriptRoot
if (-not $root) { throw 'Run this launcher from a saved .ps1 file.' }
$required = @('00_check_environment.py',
              '01_generate_and_recover_hmm6.py','02_blind_econometric_competitors.py',
              '03_summarize_aparch_experiment.py','giorno_models.py','requirements.txt')
foreach ($f in $required) { if (-not (Test-Path (Join-Path $root $f))) { throw "Missing package file: $f" } }

# Resolve the study location relative to the package, not the calling directory.
if ([System.IO.Path]::IsPathRooted($Output)) { $studyDir = $Output }
else { $studyDir = Join-Path $root $Output }
New-Item -ItemType Directory -Force -Path $studyDir | Out-Null
$studyDir = (Resolve-Path $studyDir).Path
$env:PYTHONPATH = $root

$aparchGroup = 'GARCH11-t,GJR11-t,APARCH11-t,APARCH11-power-symmetric-t,APARCH11-delta2-asym-t,APARCH11-delta2-symmetric-t,APARCH11-delta1-asym-t,APARCH11-delta1-symmetric-t'
if ($Models -eq 'aparch') { $selectedModels = $aparchGroup }
else { $selectedModels = $Models }
$resumeArg = @()
if (-not $NoResume) { $resumeArg = @('--resume') }

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    Write-Host ('  > ' + $Python + ' ' + ($Arguments -join ' ')) -ForegroundColor DarkGray
    & $Python @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $commandLabel = if ($Arguments.Count -gt 0) { $Arguments[0] } else { '(no arguments)' }
        throw "Python returned exit code $exitCode while running $commandLabel. Look ABOVE this line for the original Python error. Existing study data have not been removed."
    }
}

$transcribing = $false
Push-Location $root
try {
    try {
        Start-Transcript -Path (Join-Path $studyDir 'WINDOWS_RUN_LOG.txt') -Append -ErrorAction Stop | Out-Null
        $transcribing = $true
    } catch { Write-Warning 'Could not open transcript; program logs still appear in this console.' }

    Write-Host ''
    Write-Host '==============================================================' -ForegroundColor Cyan
    Write-Host ' Giorno--Tsallis | HMM-6 Known-DGP Reverse Attribution' -ForegroundColor Cyan
    Write-Host ' APARCH: unrestricted, power-only, asymmetry-only, restrictions' -ForegroundColor Cyan
    Write-Host '==============================================================' -ForegroundColor Cyan
    Write-Host "Mode: $Mode | Study: $studyDir"
    Write-Host "Sample: $Observations obs x $Replications replications | DGPs: $Scenarios"
    Write-Host "Competitors: $selectedModels"
    Write-Host 'NOTE: Only synthetic DGPs; no market-price files required.' -ForegroundColor Yellow
    Write-Host ''

    Invoke-CheckedPython @('--version')
    if ($InstallDependencies) { Invoke-CheckedPython @('-m','pip','install','-r','requirements.txt') }
    Invoke-CheckedPython @('00_check_environment.py')

    if ($Mode -ne 'CompetitorsOnly') {
        Write-Host '[Stage 1] Generate known HMM-6 data and recover parameters' -ForegroundColor Green
        $gen = @('01_generate_and_recover_hmm6.py','--out',$studyDir,
                 '--n',"$Observations",'--reps',"$Replications",'--starts',"$HMMStarts",
                 '--maxiter',"$HMMMaxIter",'--scenarios',$Scenarios,'--seed',"$Seed") + $resumeArg
        Invoke-CheckedPython $gen
    }
    if ($Mode -ne 'GenerateOnly') {
        Write-Host '[Stage 2] Fit blind econometric models, including APARCH restrictions' -ForegroundColor Green
        $fit = @('02_blind_econometric_competitors.py','--study',$studyDir,
                 '--models',$selectedModels,'--starts',"$CompetitorStarts",'--maxiter',"$CompetitorMaxIter",
                 '--seed',"$Seed",'--limit-paths',"$LimitPaths") + $resumeArg
        Invoke-CheckedPython $fit
        if (-not $SkipReport) {
            Write-Host '[Stage 3] Create attribution and APARCH restriction report' -ForegroundColor Green
            Invoke-CheckedPython @('03_summarize_aparch_experiment.py','--study',$studyDir)
        }
    }
    Write-Host "`nSUCCESS. All saved outputs: $studyDir" -ForegroundColor Green
    Write-Host "Summary: $studyDir\APARCH_RESEARCH_SUMMARY.md" -ForegroundColor Green
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    Pop-Location
}
