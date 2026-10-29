# Data dictionary and units

## Simulation manifest

- `id`: unique string `scenario__rep0000` etc.; `rep`: path-specific repetition index (0–499).
- `n`, `n_train`: observations in full path and train window; test length `n − n_train` (6,000 and 4,800 in the production study).
- `seed`: path-specific deterministic simulation seed.
- `true_p00`, `true_p11`: HMM state persistence probabilities; `true_rho = p00 + p11 − 1`.
- `true_nu0`, `true_nu1`: Student-t degrees of freedom; `true_s0`, `true_s1`: Student-t *scale*, not variance.
- `true_pi0`: stationary occupancy of regime 0.

## HMM parameter recovery

- `aligned_*`: fitted parameters post-hoc state-label alignment to the known simulation truth; this alignment was **not used in fitting**.
- `oos_mean_logscore`: mean one-step test log predictive density, larger is better.
- `train_loglik`, `true_parameter_train_loglik`: training observed HMM likelihood at optimizer estimate and at known DGP parameters.
- `state_classification_accuracy_test`: state accuracy conditional on true state labels; compare with `constant_state_majority_accuracy` before calling an HMM classifier useful.
- `optimizer_success`: optimizer termination flag, not sufficient to prove identification.

## Paired model comparisons

- `mean_hmm_minus_competitor`: MC average of matched-path differences in per-observation OOS mean log predictive densities; positive favors HMM.
- `paired_mc_se`: standard error across independent simulated path differences.
- `bootstrap_ci95_lo`, `bootstrap_ci95_hi`: 95% bootstrap interval across independent simulated paths, **not** parameter confidence intervals or real-market inferential guarantees.
- `valid_paired_paths`, `penalized_paths`: valid scores and numerically invalid paths. Never treat the −1e8 sentinel as a genuine density.

## Intraday market files

- `market`: `WIN` or provisionally labeled `US` (not independently confirmed SPY/SPX).
- `timeframe`: M1, M5, M10 or M15, minute grids.
- `test_n`: number of valid holdout return observations.
- `garch_t_log_score`, `garch_mix_t_log_score`, `garch_hmm_log_score`: average log predictive densities for distinct common-volatility models.
- `confidence_pct`: 95, 99 or 99.5 for one-sided VaR; reported loss is pinball/quantile loss, smaller is better.

## Interpretation of Student-t parameters

State-conditioned Student-t *variance* equals \(s_j^2\nu_j/(\nu_j-2)\) for \(\nu_j>2\). The Tsallis/q-Gaussian shape relabeling \(q_j=(\nu_j+3)/(\nu_j+1)\) satisfies \(1<q_j<5/3\). It is not a directly observed entropy measure.
