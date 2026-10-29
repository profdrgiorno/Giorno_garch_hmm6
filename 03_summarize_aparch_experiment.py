#!/usr/bin/env python3
"""Create reproducible APARCH reverse-attribution summaries from saved fits.

No selection of DGPs or models by outcome. Reports failed fits and MC uncertainty.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    args=p.parse_args()
    study=Path(args.study)
    comp=study/'blind_competitor_results'/'ALL_COMPETITOR_SUMMARIES.csv'
    if not comp.exists():p.error(f'Missing {comp}; run Stage 2 first.')
    df=pd.read_csv(comp)
    if df.empty or 'oos_mean_logscore' not in df:
        p.error('No scored competitor fits found.')
    df['oos_mean_logscore']=pd.to_numeric(df.oos_mean_logscore,errors='coerce')
    df['optimizer_success']=df['optimizer_success'].astype('boolean')
    good=df[df.oos_mean_logscore.notna()].copy()
    if good.empty:p.error('No finite scores to report.')
    rows=[]
    for (scen,model),g in good.groupby(['scenario','model']):
        n=len(g);scores=g.oos_mean_logscore.to_numpy(float)
        rows.append(dict(scenario=scen,model=model,n_paths=n,
            mean_logscore=float(np.mean(scores)),
            MC_SE_mean_logscore=float(np.std(scores,ddof=1)/np.sqrt(n)) if n>1 else np.nan,
            mean_estimated_variance_CV=float(g.estimated_variance_CV_test.mean()),
            mean_variance_MSE_vs_truth=float(g.variance_MSE_vs_true_test.mean()),
            convergence_fraction=float(g.optimizer_success.mean())))
    report=pd.DataFrame(rows).sort_values(['scenario','mean_logscore'],ascending=[True,False])
    report.to_csv(study/'MODEL_RANKING_WITH_MC_SE.csv',index=False)
    wide=good.pivot_table(index=['id','scenario','rep'],columns='model',values='oos_mean_logscore',aggfunc='first')
    pairs=[]
    if 'GARCH11-t' in wide.columns:
        for model in sorted([m for m in wide.columns if m.startswith('APARCH')]):
            matched=wide[['GARCH11-t',model]].dropna()
            if matched.empty:continue
            d=(matched[model]-matched['GARCH11-t']).rename('difference').reset_index()
            for scen,g in d.groupby('scenario'):
                z=g['difference'].to_numpy(float);n=len(z)
                pairs.append({'scenario':scen,'alternative':model,'reference':'GARCH11-t',
                    'matched_paths':n,'mean_delta_logscore':float(z.mean()),
                    'MC_SE_delta':float(z.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan,
                    'fraction_positive':float((z>0).mean())})
    pd.DataFrame(pairs,columns=['scenario','alternative','reference','matched_paths',
                                  'mean_delta_logscore','MC_SE_delta','fraction_positive']).to_csv(
                                     study/'APARCH_VS_GARCH_MATCHED_PATHS.csv',index=False)
    parkeys=['alpha_power','beta_power','gamma_power','delta','leverage','alpha1','beta1','gamma_z','nu']
    attr=[]
    for (scen,model),g in good.groupby(['scenario','model']):
        if not (model.startswith('APARCH') or model in ('GJR11-t','GARCH11-t','EGARCH11-t')):continue
        a={'scenario':scen,'model':model,'paths':len(g)}
        for key in parkeys:
            if key in g.columns and g[key].notna().any():
                vals=pd.to_numeric(g[key],errors='coerce').dropna().to_numpy(float)
                a[f'mean_{key}']=float(np.mean(vals))
                a[f'sd_{key}']=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan
        attr.append(a)
    pd.DataFrame(attr).to_csv(study/'APARCH_COEFFICIENT_ATTRIBUTION.csv',index=False)
    error_rows=df[df.oos_mean_logscore.isna()]
    nr=int((good.optimizer_success == False).sum())
    scenlist=sorted(good.scenario.unique())
    lines=['# Giorno--Tsallis HMM-6 / APARCH reverse-attribution run', '',
        '## Execution inventory',
        f'- Scored model fits: **{len(good)}**; hard failures without scores: **{len(error_rows)}**.',
        f'- Optimizer non-convergence flags among scored fits: **{nr}**.',
        f'- DGP conditions with scored fits: {", ".join(scenlist)}.',
        '- All competitors fitted to training returns without access to DGP hidden states.',
        '- This is a synthetic known-DGP study, **not** evidence of real-market forecasting value.',
        '', '## Scientific questions',
        '- If true conditional variance is constant (tail-only HMM), does a blind volatility estimator manufacture time-varying variance?',
        '- If true tail states persist, does APARCH misinterpret them as leverage or a nonquadratic power?',
        '- Does permitting asymmetric innovations in the variance equation improve held-out density scores, and how large is the incremental gain over plain GARCH?',
        '- Does the unrestricted APARCH model consistently beat its correctly nested restrictions after complexity and simulation uncertainty are considered?',
        '', '## Identifiability and inference limitations',
        '- Non-HMM models do not recover true HMM transition probabilities; their coefficients are reverse-attribution proxies.',
        '- Numerical optimizer success is not evidence of correct HMM state identification.',
        '- MC standard errors are descriptive over independent simulation paths, and should not be called valid hypothesis tests with only a handful of replications.',
        '- Model likelihood approximations are implemented in the local custom code. Validate against independent software before publication.',
        '- The ARMA and GARCH competitors here are not jointly estimated ARMA--GARCH models.',
        '- Fixed APARCH parameters are NOT counted among free coefficients for AIC/BIC.',
        '', '## Saved reports',
        '- `MODEL_RANKING_WITH_MC_SE.csv`: mean score, Monte Carlo SE, convergence, fitted variance behavior.',
        '- `APARCH_VS_GARCH_MATCHED_PATHS.csv`: path-matched density score differences versus GARCH--Student-t.',
        '- `APARCH_COEFFICIENT_ATTRIBUTION.csv`: mean/sd of APARCH and other comparable fitted coefficients.',
        '- `blind_competitor_results/`: full per-observation forecasts, model parameters, optimizer diagnostics.',
        '- `HMM6_PARAMETER_RECOVERY_ALL.csv`: truth-vs-estimate comparison and state recovery.',
        '', '## Principle',
        '**We will not change the evidence to fit the theory. We will improve the theory to explain the evidence.**', '']
    (study/'APARCH_RESEARCH_SUMMARY.md').write_text('\n'.join(lines),encoding='utf8')
    print(f'Reports written to {study.resolve()}')

if __name__=='__main__':main()
