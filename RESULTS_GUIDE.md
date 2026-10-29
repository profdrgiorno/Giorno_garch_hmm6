# Results inventory and how to read it

## Synthetic Monte Carlo (uncontaminated)

Core complete-summary artifacts in `data/simulated/summary/`:

| File | Description |
|:--|:--|
| `study_design_PUBLIC.json` | Exact DGP parameters, 8 scenarios and common run settings; absolute local path removed. |
| `DGP_MANIFEST.csv` | **4,000** simulated-path IDs, true state parameters, sample sizes, per-path seeds. |
| `HMM6_PARAMETER_RECOVERY_ALL.csv` | **4,000** HMM results, likelihood and fitted-vs-true parameter measurements. |
| `HMM6_RECOVERY_SUMMARY.csv` | Grouped HMM recovery results. |
| `MODEL_RANKING_WITH_MC_SE.csv` | **152** scenario/model rankings (8 × 19); original mean scores include flagged EGARCH penalties. |
| `APARCH_COEFFICIENT_ATTRIBUTION.csv` | Recovered coefficients in the misspecified APARCH family. |
| `APARCH_VS_GARCH_MATCHED_PATHS.csv` | Selected path-matched APARCH-vs-GARCH performance differences. |
| `PATH_MATCHED_HMM6_VS_19_MODELS.csv` | **152** aggregate HMM/competitor pair comparisons with MC uncertainty from the original 76,000 path-level results. |
| `SCENARIO_RECOVERY_AUDIT.csv` | Persistence and classification checks, including majority-class baseline. |
| `PARAMETER_RECOVERY_MC_INTERVALS.csv` | Parameter bias, RMSE and intervals for **MC averages**, not single-fit coverage. |
| `HMM_NUMERICAL_FLAGS.csv` | HMM case in which optimizer likelihood is below likelihood at known true parameter. |
| `EGARCH_AND_OTHER_NUMERICAL_PENALTY_PATHS.csv` | **11** path identifiers affected by invalid EGARCH-recursion penalty log scores. |
| `MODEL_RANKING_NUMERICAL_FLAGS.csv` | Model-level flags for contaminated aggregate rankings. |

The 19 candidates comprise Static-t, independent t-mixture, AR2, AR3, MA1, ARMA11, ARMA12, GARCH11-normal, GARCH11-t, GARCH12-t, GJR11-t, EGARCH11-t, APARCH11-t and five APARCH restrictions, and HN11-t.

### Selected HMM-vs-symmetric-power APARCH comparisons

The following are **mean per-observation OOS log score gains** (HMM minus comparator) with path-bootstrap 95% intervals. **Positive** favors HMM. Values are descriptive of this prespecified DGP and estimator implementation.

| True DGP | Mean difference | 95% path-bootstrap interval |
|:--|--:|:--|
| Calibration anchor | +0.028336 | [0.027814, 0.028851] |
| Full HMM-6 | +0.005766 | [0.005451, 0.006073] |
| Scale-only | +0.004371 | [0.004102, 0.004651] |
| Tail-only | +0.000018 | [−0.000101, 0.000131] |
| Weak-tail | −0.000130 | [−0.000207, −0.000052] |

### Numerical penalties

The custom EGARCH recursion can emit a sentinel log score of **−1e8** after numerical recursion failure. The resulting invalid values are not legitimate predictive densities. The flagged 11 cases must be independently checked and, if needed, selectively re-estimated. Do not compare their raw scenario averages with the valid models or quietly delete failures to change ranking.

## Market exercises (distinct GARCH-based model)

`data/market/summary/GT_GARCH_HMM_All_8_Verified_Log_Scores.csv` has **eight** market–timeframe rows; `..._VaR.csv` has **24** rows for three risk levels each. WIN and the **provisional U.S. series** are studied at M1/M5/M10/M15. GARCH–HMM is descriptively best in **five** of the eight log-score comparisons, but margins vary widely.

`paired_tests_available_only.csv` reflects only those market cells with valid matched observation-level forecasts. Original instrument identity, market closing times, data acquisition, exchange calendar alignment and corporate-action adjustment for the U.S. series remain to be confirmed. Do not advertise an independently reproducible eight-cell market-data experiment on the basis of aggregate tables alone.

## Outputs that are *not* present

The original large-run `datasets/` full return sequences, `hmm6_fits/` individual predicted probabilities, **76,000** individual competitor JSON outputs and `blind_competitor_results/ALL_COMPETITOR_SUMMARIES.csv` are retained on the researcher's Windows machine, not in this GitHub package. The complete two-path demonstrator is provided under `examples/smoke_outputs/`.
