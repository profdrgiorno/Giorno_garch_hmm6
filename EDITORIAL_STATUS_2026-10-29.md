# HMM-6 integrated referee-review draft (29 October 2026)

This manuscript is a careful integration of the existing observed-likelihood finite-window theorem, the 4,000-path known-DGP audit, and eight intraday GARCH--HMM forecasting comparisons (WIN and provisionally identified U.S. series, M1/M5/M10/M15). It preserves separate parameters and estimands for each experiment.

**This is a referee-review draft, not yet an unqualified submission-ready manuscript.** Outstanding issues are targeted EGARCH numerical repair (11 penalized paths), one HMM training-likelihood optimizer anomaly, provenance/timestamp verification for the U.S. input, two flagged U.S. null-bootstrap fits, and exhaustive proof/source consistency checks. Results do not verify historical Gamma jumps or market profit.

Source tables supplied with this package are copied byte-for-byte from the current audit artifacts. No proprietary commercial market indicators or projects are included. All eight intraday comparisons are descriptive; observation-level paired inference exists for only three cells. The 4,000-path experiment uses independent simulated sequences and is separate from the finite-window Gamma-jump derivative validation.

Edit the `.tex` source and run `pdflatex` twice; the generated PDF accompanies it.
