# Reproducibility protocol (version 0.9.0)

## Supported software

The recorded Windows research environment was **Python 3.11.15**, NumPy **1.26.4**, pandas **2.3.3**, SciPy **1.17.1**, Numba **0.59.1**. Scripts use NumPy/SciPy/pandas/Numba and do not depend on MetaTrader5 for the synthetic DGP study. These recorded package versions are evidence from the user's Windows run, **not** a newly tested lock-file or guarantee of compatibility on all platforms.

Create your own environment, then `python -m pip install -r requirements.txt`. Large runs can require substantial CPU, RAM and disk space.

## A. Quick release integrity check

```powershell
python .\tools\verify_release.py
python .\code\simulation\04_verify_aparch_nesting.py
```

The verifier checks released CSV record counts, model/DGP coverage, and critical artifacts. This is a **package-level sanity check**, not proof of theoretical correctness.

## B. Pilot / smoke test

```powershell
cd .\code\simulation
.\RUN_RESEARCH_WINDOWS.ps1
```

The launcher defaults to a small pilot. A complete two-path example, including observed synthetic sequences, truth fields, fitted HMM parameters, competitor fit JSON, and one-step forecasts, is under `examples/smoke_outputs/`. It is illustrative and is **not** a subset of the full production experiment unless explicitly matched by identifiers and seeds.

## C. Recreate the 4,000-path experiment

From `code/simulation` on Windows PowerShell:

```powershell
.\RUN_RESEARCH_WINDOWS.ps1 `
  -Mode Research `
  -Observations 6000 `
  -Replications 500 `
  -HMMStarts 10 `
  -CompetitorStarts 5 `
  -HMMMaxIter 400 `
  -CompetitorMaxIter 320 `
  -Scenarios all `
  -Models all `
  -Seed 20261028 `
  -Output 'results\FULL_T6000_R500'
```

The experiment uses **600 discarded burn-in observations**, a chronological 80%/20% train/test split, **8 DGP scenarios**, and **19 competitors**. The original full study was recorded as completed on **28 October 2026 at 23:44:12**, on Windows. Original files are retained locally by the researcher. `--resume` is the default when using this launcher. **Do not overwrite a prior run with changed scientific settings.**

Output includes `study_design.json`, `DGP_MANIFEST.csv`, `HMM6_PARAMETER_RECOVERY_ALL.csv`, `HMM6_RECOVERY_SUMMARY.csv`, `datasets/`, `hmm6_fits/`, and `blind_competitor_results/` with `ALL_COMPETITOR_SUMMARIES.csv`, `ALL_FITTED_PARAMETERS.csv`, individual model fits and observation-level scores.

## D. Repeat the independent numerical audit (locally)

For the **full** audit including paired statistical uncertainty, run in the repository root:

```powershell
python .\code\audit\audit_hmm6_final.py `
  --study '.\code\simulation\results\FULL_T6000_R500' `
  --out '.\results\FULL_T6000_R500_AUDIT'
```

The full local input `blind_competitor_results/ALL_COMPETITOR_SUMMARIES.csv` is required for independently calculating paired Monte Carlo bootstrap intervals and the 11 penalized EGARCH paths. Those raw 76,000 records **are not included in this reduced GitHub archive**, although their previously computed summary tables are included.

For the *available* aggregates only:

```powershell
python .\code\audit\audit_hmm6_final.py --source-dir '.\data\simulated\summary' --out '.\results\ARCHIVED_TABLE_AUDIT'
```

Interpret bootstrap CIs as **MC intervals for means across independent simulated paths**. They are neither individual-fit CIs nor a general solution to nonregular HMM testing.

## E. The finite-window omitted-Gamma-jump sensitivity experiment

`code/theorem/GT_Proposition_A_Targeted_v3.py` runs the observed-HMM score/impulse sensitivity experiment **separately** from the 4,000-path unshocked DGP study.

A lightweight exploratory run:

```powershell
python .\code\theorem\GT_Proposition_A_Targeted_v3.py `
  --paths 200 --length 15 --seed 20261026 --out '.\results\V3_SMOKE' `
  --nodes 20 --maxiter 80
```

The **manuscript's** V3 experiment uses `--paths 15000 --length 40 --nodes 48` for **two seeds**, 20261026 and 20261027, with epsilon and Gamma-scale grids documented in the manuscript. It is computationally expensive. Historical V1 and V2 full scripts/output archives are not present in this release. Do not misrepresent the V3 source as an independent replication of *every* numerical table in the manuscript.

## F. Eight market/timeframe exercises

`code/empirical/GT_GARCH_HMM_Joint_Experiment.py` fits a common GARCH(1,1) process with three different innovation structures. Run `--help` for data-format options. The WIN branch expects MT5-style `<DATE>`, `<TIME>`, `<CLOSE>` tab-separated data; the U.S. branch expects `datetime`, `Close` CSV. The code excludes non-consecutive within-session observations and audits gaps. **Raw underlying market inputs are not distributed** here.

`code/empirical/GT_Compare_All8.py` consolidates saved run summaries and runs observation-level paired tests **only when valid matched forecast records exist**. It must not produce significance claims from aggregates alone. Example with supplied summary-only table:

```powershell
python .\code\empirical\GT_Compare_All8.py `
  --summaries '.\data\market\summary\GT_GARCH_HMM_All_8_Verified_Summaries.csv' `
  --base-dir '.\results\market_runs' `
  --out '.\results\market_comparison'
```

If missing market forecast files, the script reports their absence and emits descriptive aggregate comparisons only. U.S. asset identity remains **provisional**.

## G. Freeze the final public record

Before peer-review submission, publish output hash manifests for the full study, software versions, code commit SHA, actual market source/ticker provenance, exact source timestamps and market treatment of gaps/overnights, plus the resolved numerical-penalty cases. Never silently alter an existing generated dataset to improve fit.
