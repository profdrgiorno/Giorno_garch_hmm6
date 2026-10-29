# Giorno–Tsallis HMM-6: observed-likelihood sensitivity and identification

**Research manuscript · Version 0.9.0 (referee-review draft) · 29 October 2026**  
**Author:** J. Giorno, PhD  
**Status:** Public-facing reproducibility **working package**, not a certification that every empirical result has been independently reproduced.

> **Scientific principle:** We will not change the evidence to fit the theory. We will improve the theory to explain the evidence.

## Read the paper

- **[Manuscript (PDF)](paper/HMM6_Integrated_Referee_Draft.pdf)** — full integrated paper with theorem, proofs, simulations and market results.
- **[LaTeX source](paper/HMM6_Integrated_Referee_Draft.tex)** — editable manuscript (`pdflatex` × 2).
- **[Detailed reproducibility instructions](docs/REPRODUCIBILITY.md)** — Windows/PowerShell and terminal commands, seeds, settings and audit steps.
- **[Results inventory and interpretation](docs/RESULTS_GUIDE.md)** — tables and limitations.
- **[Open validation issues](docs/VALIDATION_STATUS.md)** — issues that must not be hidden before submission.

## Main research question

How does the pseudo-true parameter vector of a *centered, two-state Student-t hidden Markov model* move when rare **downward Bernoulli–Gamma shocks** are omitted from its observed-data likelihood? How reliably can the resulting latent-tail mechanism be identified against familiar mean and volatility specifications?

The baseline is a six-parameter model

\[
\psi=(p_{00},p_{11},\nu_0,\nu_1,s_0,s_1)^\top,
\qquad
q_j=1+\frac{2}{\nu_j+1},\quad \nu_j>2.
\]

Its central **fixed-window local sensitivity** result, conditional on identification and regularity, is

\[
\left.\frac{d\psi_T^*(\varepsilon)}{d\varepsilon}\right|_{0+}
=I_T^{-1}\,\mathbb{E}_0[S_T(Y) W_T(Y)] .
\]

Here \(I_T\) is the *observed* HMM Fisher information, and \(W_T\) uses **full-sequence smoothed state probabilities**. The theorem is local in the omitted-shock incidence probability; it **does not** prove global jump identification or actual jumps in observed financial markets.

## Evidence at a glance

| Component | Completed evidence | Scientific boundary |
|:--|:--|:--|
| Known DGP Monte Carlo | **8 DGPs × 500 replications = 4,000** independent paths of 6,000 observations | Uncontaminated HMM-6 recovery, not a test of the Gamma-jump theorem |
| Blind reverse attribution | **19** conventional estimator specifications × 4,000 paths = **76,000** scored fits | Estimates of misspecified models are *proxies*, not HMM transition estimates |
| Paired MC comparison | Paired OOS log-score gains and bootstrap CIs across the 500 independent paths per DGP | Some EGARCH forecasts require numerical repair |
| Market comparison | WIN and a provisionally identified **U.S. series**, at **M1, M5, M10, M15** | GARCH–HMM is top by *descriptive mean* log score in **5 of 8** cells; paired observations available only in part |
| Theorem sensitivity experiment | Independent finite-window, fixed-Gamma-direction Monte Carlo | Separate from the 4,000-path no-jump reverse-attribution experiment |

### Honest findings

The HMM recovers persistence well when regimes differ sufficiently in **scale**, but **not** in the equal-variance tail-only and weak-tail designs. A true IID mixture produces **false positive fitted Markov persistence**. Those unfavorable outcomes are part of the scientific result, not exclusions. See [the scientific interpretation](docs/METHODOLOGY.md).

## Repository map

```text
paper/                 Integrated manuscript (PDF + LaTeX), editorial status
code/simulation/       Two-stage HMM-6 DGP + 19-model estimation, APARCH restrictions, Windows launcher
code/theorem/          Numerical finite-window omitted-shock sensitivity experiment
code/empirical/        Common-GARCH/tail-mixture/HMM M1–M15 estimation and comparison tools
code/audit/            Independent read-only Monte Carlo auditor
data/simulated/summary/ 4,000-path manifest, HMM fits, 19-model summary rankings and paired audits
data/market/summary/   Eight intraday market/frequency density and VaR tables
examples/smoke_outputs/ Complete two-path demonstration, **not** the 500-replication dataset
reports/               Independent numerical audit PDF
docs/                  Methods, data dictionary, reproduction, provenance, outstanding checks
tools/                 Release verification and table rendering
.github/workflows/     Lightweight source/package CI; no full research recomputation
```

## Start here (Windows)

Install **Python 3.11**. In a terminal in the repository root:

```powershell
python -m pip install -r requirements.txt
python .\tools\verify_release.py
python .\code\simulation\04_verify_aparch_nesting.py
```

A small independent pilot (writes to `code/simulation/results/`):

```powershell
cd .\code\simulation
.\RUN_RESEARCH_WINDOWS.ps1
```

The launcher **defaults to a two-replication pilot**. It is *not* the long full study. To regenerate the full research design see [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); **do not casually repeat the full 76,000-model run** if the outputs already exist on your Windows computer.

To check the **included archived aggregates** without refitting:

```powershell
python .\code\audit\audit_hmm6_final.py --source-dir .\data\simulated\summary --out .\results\aggregate_reaudit
```

That command cannot recreate the full 76,000-row *path-level* input because those raw per-path summaries were not available in this upload. The original paired-audit outputs are supplied as CSV and are auditable against the original path-level files when mounted separately.

## Important reproducibility limits

1. This repository contains the **complete generating and estimating code** and 4,000-path HMM parameter-recovery **summaries**, but **not the 4,000 raw price/return paths, 76,000 individual competitor fits, or full observation-level forecasts**. Recreate them using the documented seeds or add the original `FULL_T6000_R500` folder to your local checkout. Detailed caveats: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
2. Intraday original market-price files are **not redistributed**. The U.S. instrument identity and timestamps still require formal provenance verification. The public eight-cell tables alone are insufficient for replicating every paired test.
3. **11 EGARCH path scores** contain implementation-generated numerical penalties and **must not be treated as genuine predictive log densities**. One HMM fit reached a likelihood below the likelihood at the known generating parameters. These numerical issues are disclosed and flagged.
4. The 19-model comparison uses **custom MLE code**; independent implementation verification is outstanding. Mean and variance families were not universally estimated as joint ARMA–GARCH specifications; the Student-t Heston–Nandi-style recursion is **not** risk-neutral option pricing.
5. A six-parameter centered HMM, a gap-aware empirical model, and a GARCH–HMM are **different** specifications. Do not merge their parameter counts or scores.

## Citation, reuse and contact

Please cite the working paper using [CITATION.cff](CITATION.cff). This release does **not** assert that the article is accepted or has a DOI. [License status](LICENSE_NOTICE.md): no public software/data license has been chosen yet; publication on GitHub by itself grants no general reuse permission. Contributions and issue reports are welcome under [CONTRIBUTING.md](CONTRIBUTING.md).
