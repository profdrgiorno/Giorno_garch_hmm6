# Giorno--Tsallis HMM-6 / APARCH reverse-attribution run

## Execution inventory
- Scored model fits: **38**; hard failures without scores: **0**.
- Optimizer non-convergence flags among scored fits: **2**.
- DGP conditions with scored fits: calibration_anchor, tail_only.
- All competitors fitted to training returns without access to DGP hidden states.
- This is a synthetic known-DGP study, **not** evidence of real-market forecasting value.

## Scientific questions
- If true conditional variance is constant (tail-only HMM), does a blind volatility estimator manufacture time-varying variance?
- If true tail states persist, does APARCH misinterpret them as leverage or a nonquadratic power?
- Does permitting asymmetric innovations in the variance equation improve held-out density scores, and how large is the incremental gain over plain GARCH?
- Does the unrestricted APARCH model consistently beat its correctly nested restrictions after complexity and simulation uncertainty are considered?

## Identifiability and inference limitations
- Non-HMM models do not recover true HMM transition probabilities; their coefficients are reverse-attribution proxies.
- Numerical optimizer success is not evidence of correct HMM state identification.
- MC standard errors are descriptive over independent simulation paths, and should not be called valid hypothesis tests with only a handful of replications.
- Model likelihood approximations are implemented in the local custom code. Validate against independent software before publication.
- The ARMA and GARCH competitors here are not jointly estimated ARMA--GARCH models.
- Fixed APARCH parameters are NOT counted among free coefficients for AIC/BIC.

## Saved reports
- `MODEL_RANKING_WITH_MC_SE.csv`: mean score, Monte Carlo SE, convergence, fitted variance behavior.
- `APARCH_VS_GARCH_MATCHED_PATHS.csv`: path-matched density score differences versus GARCH--Student-t.
- `APARCH_COEFFICIENT_ATTRIBUTION.csv`: mean/sd of APARCH and other comparable fitted coefficients.
- `blind_competitor_results/`: full per-observation forecasts, model parameters, optimizer diagnostics.
- `HMM6_PARAMETER_RECOVERY_ALL.csv`: truth-vs-estimate comparison and state recovery.

## Principle
**We will not change the evidence to fit the theory. We will improve the theory to explain the evidence.**
