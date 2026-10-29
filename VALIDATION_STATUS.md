# Outstanding verification before unconditional peer-review submission

This record is intentionally conservative. Completing the large Monte Carlo **does not** establish independent confirmation of all source code and empirical inputs.

| Issue | Current evidence | Action to close |
|:--|:--|:--|
| EGARCH recursion penalties | 11 offending path IDs; invalid −1e8 forecast penalty scored as log likelihood | Inspect variance and likelihood recursion, validate independently, re-estimate *only affected paths*, preserve original flagged values |
| HMM local optimization | `tail_only__rep0192` achieved likelihood below likelihood at true DGP parameter | Retry with additional starts/diagnostics; retain original failure |
| Weak tail identification | Tail-only/weak-tail 500-rep recovery is poor despite optimizer convergence | Explain as empirical limitation and present majority-state benchmark |
| No-persistence false positives | IID-mixture has positive mean estimated Markov persistence | Do not assert generic HMM distinguishability |
| Finite-window score theorem | Conditional theoretical result with observed-information assumptions | Proof and code-level finite-difference/Hessian review; V1/V2 raw artifacts incomplete |
| U.S. instrument provenance | Aggregates labeled U.S., asset identity not fully established | Verify source ticker, time zone, adjustment, exact sampling and overnight filtering |
| U.S. bootstrap | Earlier gap-aware bootstrap contained two flagged optimizer cases | Resolve and disclose sensitivity |
| Market paired uncertainty | Valid matched observation records available for only part of eight cells | Avoid significance claims for other cells until actual forecasts are supplied |
| Source independent replication | Custom HMM/ARMA/APARCH/GARCH MLE | Cross-check against independent reference implementations and optimizer starts |
| Repository provenance | Full per-path 4k DGP/76k fit records retained off-GitHub | Archive hashes and input/output manifests for an independent verifier |

**No alteration of observations, seeds, scoring windows, or negative results is authorized to produce a favorable model ranking.**
