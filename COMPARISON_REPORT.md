# Eight-experiment GARCH / mixture / HMM comparison

**Scope:** two markets, four sampling frequencies, three jointly estimated GARCH-based models.
**Instrument note:** US label remains provisional until source identity is independently verified.

## Descriptive model rankings

Density winners (eight groups): GARCH–Student-t 2/8, GARCH + iid Student-t mixture 1/8, GARCH–HMM 5/8.
Pinball winners (24 groups): GARCH–Student-t 6/24, GARCH + iid Student-t mixture 7/24, GARCH–HMM 11/24.

| Market | Freq | N test | Best density | HMM − t logscore | HMM − iid mixture logscore |
|---|---:|---:|---|---:|---:|
| US | M1 | 15,339 | GARCH–HMM | +0.001101 | +0.000995 |
| US | M5 | 3,002 | GARCH–HMM | +0.000511 | +0.000896 |
| US | M10 | 1,493 | GARCH–HMM | +0.002247 | +0.002224 |
| US | M15 | 964 | GARCH–HMM | +0.001347 | +0.000343 |
| WIN | M1 | 17,123 | GARCH–HMM | +0.003417 | +0.002786 |
| WIN | M5 | 3,412 | GARCH + iid Student-t mixture | +0.000171 | -0.000722 |
| WIN | M10 | 1,694 | GARCH–Student-t | -0.000760 | -0.000640 |
| WIN | M15 | 1,122 | GARCH–Student-t | -0.000048 | +0.000141 |

## Inferential status

Observation-level matched forecasts available and evaluated for **3/8** groups.

- US M1: Paired tests available
- US M5: summary_only; statistical comparisons unavailable
- US M10: summary_only; statistical comparisons unavailable
- US M15: Paired tests available
- WIN M1: summary_only; statistical comparisons unavailable
- WIN M5: summary_only; statistical comparisons unavailable
- WIN M10: summary_only; statistical comparisons unavailable
- WIN M15: Paired tests available

The summary CSVs cannot support paired HAC or day-bootstrap inference. 
The script computes such tests only for genuine matched forecast rows; missing records are NOT imputed. 
The Holm column, if present, adjusts only tests currently available, not the complete prespecified eight-frequency family.
Day-resampling treats calendar dates as blocks and assumes sufficient cross-day independence; HAC bandwidth and multiple-comparison design still need methodological validation.

## Limits and next tests

- All results use a **single** chronological holdout at each frequency; no repeated-origin conclusions are justified.
- Cross-frequency log scores and VaR losses refer to different horizon distributions, and should not be directly ranked by their raw magnitude.
- Tail states that do not persist or reach optimization bounds are not compelling evidence for Markov dependence.
- A static Student-t model **without GARCH** is not included in these current eight experiments.
- To test Markov persistence, compare HMM against independent mixture, inspect transition eigenvalues and use nonstandard-valid inference.
- Before final claims, obtain all eight holdout_forecasts.csv and run_metadata.json files, check duplicate data across frequencies, and repeat rolling-origin evaluations.
