# Scientific methodology and interpretation

## Separate the research questions

**Theory:** The omitted downward Bernoulli–Gamma shock is a perturbation to a *centered two-state Student-t HMM* fitted by the **observed likelihood**. The sensitivity derivative for a **fixed finite** observation window and fixed Gamma parameters is defined only under local identification, differentiability, and non-singular observed information. Use **full-sequence smoothed probabilities** for the jump incidence score.

**Uncontaminated recovery:** The 4,000-path known-DGP study tests whether six baseline HMM parameters can be recovered and what alternative, misspecified models infer. Its DGPs do **not** have omitted jumps; this cannot be cited as empirical validation of the Gamma shock derivative.

**Market prediction:** The common-GARCH–HMM comparison is a **different specification** in which regime-dependent standardized Student-t tails enter a conditional-volatility recursion. It is not the same six-parameter model. The goal is predictive density and VaR, not a claim that historical Gamma shocks were observed.

## State alignment and identification

The states are exchangeable under label permutations. Oracle truth alignment is used **only after fitting** for simulation recovery evaluation. It must never enter optimization or out-of-sample forecasts. If emission distributions are identical, transition probabilities are not identified. With weak state separation, numerical optimizer success is insufficient to establish reliable recovery.

The two-state persistence functional is \(\rho=p_{00}+p_{11}-1\). For the IID-mixture control, \(\rho=0\). Positive *estimated* persistence in this null scenario quantifies a false-regime failure, not genuine latent temporal dependence.

## Known-DGP coverage

- Calibration anchor: strong scale and tail differences.
- Full HMM-6: scale and tail differences.
- Scale-only: common Student-t degrees of freedom, different variances.
- Tail-only: same mean and conditional variance, different tail shapes.
- Weak-tail: nearby state-specific tail shapes.
- Weak-persistence: low state autocorrelation.
- IID mixture: two emission laws but independent state assignments.
- Single Student-t: identical emissions, unidentified transition parameters.

**Key result:** strong scale separation aids parameter recovery. Tail-only and weak-tail switching remain difficult at T=6,000; the IID mixture produces positive average estimated persistence. These limitations are central to the paper.

## Reverse attribution, not parameter recovery in the wrong model

A model with no Markov chain cannot estimate \(p_{00}\) or \(p_{11}\). Examine its estimated AR/MA coefficients, volatility power, leverage and forecast density to understand how it *approximates* data from an HMM. APARCH power \(\delta=2\), asymmetry \(\gamma=0\) nests a GARCH(1,1) recursion. EGARCH and HN-style recursions are **separate** comparators.

## Statistical interpretation

Each DGP has 500 independent simulation paths, with 1,200 chronologically held-out observations per path. Path-level paired differences are the correct basis for uncertainty in mean predictive performance. The supplied bootstrap intervals describe variation across generated paths; do not present them as individual parameter confidence intervals or tests of actual market jump occurrence. Eight empirical timeframe-cell rankings alone are descriptive and vulnerable to data snooping and multiple comparisons.

## Scientific restraint

Do **not** claim: universal dominance of HMM tails, global Gamma-jump identification, conventional chi-square jump boundary inference, a universal sign change in Tsallis q, that GARCH approximations 'recover' the hidden transition parameters, economic trading profits, or that all eight intraday cells have validated paired tests.
