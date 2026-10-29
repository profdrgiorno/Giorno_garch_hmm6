#!/usr/bin/env python3
"""Independent, read-only audit of Giorno--Tsallis HMM-6 known-DGP Monte Carlo.

Requires: numpy, pandas, scipy. Does not alter or re-estimate research outputs.

Example (Windows PowerShell):
  python .\audit_hmm6_final.py --study "results\\FULL_T6000_R500" --out ".\\AUDIT_FINAL"

For received conversation exports (with names ending in '(2)'):
  python audit_hmm6_final.py --source-dir /mnt/data --out /mnt/data/HMM6_Final_Independent_Audit_2026_10_29
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

SENTINEL_ALERT = -1000.0  # mean log scores below this are not plausible valid density fits here
PRIMARY = ['Static-t','GARCH11-t','APARCH11-power-symmetric-t']
PARAMETERS = ['p00','p11','nu0','nu1','s0','s1']


def locate(root: Path, stem: str, required=True) -> Path | None:
    possible = [root/stem.replace('.csv','(2).csv'),root/stem,root/stem.replace('.csv','(1).csv')]
    for path in possible:
        if path.exists(): return path
    if required: raise FileNotFoundError(f'{stem} not found under {root}')
    return None


def safe_float(x):
    return float(x) if pd.notna(x) else float('nan')


def stats_mean(x):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    return n,(float(np.mean(x)) if n else np.nan), (float(np.std(x,ddof=1)/np.sqrt(n)) if n>1 else np.nan)


def boot_mean_ci(x, rng, rep=1200):
    a=np.asarray(x,float)
    a=a[np.isfinite(a)]
    if len(a)<2:return (np.nan,np.nan)
    # Chunking avoids very large allocations on commodity Windows machines.
    b=[]
    for k in range(0,rep,200):
        size=min(200,rep-k)
        idx=rng.integers(0,len(a),(size,len(a)))
        b.extend(a[idx].mean(axis=1))
    return tuple(map(float,np.quantile(b,[.025,.975])))


def write_md_table(frame, columns=None, digits=5):
    if columns is not None:frame=frame[columns]
    if frame.empty:return '(No rows)'
    return frame.to_markdown(index=False,floatfmt=f'.{digits}f')


def run(study: Path, out: Path, full_competitors_path: Path | None):
    out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(locate(study,'DGP_MANIFEST.csv'))
    hmm=pd.read_csv(locate(study,'HMM6_PARAMETER_RECOVERY_ALL.csv'))
    ranking=pd.read_csv(locate(study,'MODEL_RANKING_WITH_MC_SE.csv'))
    aparch=pd.read_csv(locate(study,'APARCH_COEFFICIENT_ATTRIBUTION.csv'))
    apmatched_path=locate(study,'APARCH_VS_GARCH_MATCHED_PATHS.csv',required=False)
    apmatched=pd.read_csv(apmatched_path) if apmatched_path is not None else None
    assert manifest.id.is_unique and hmm.id.is_unique,'Duplicate dataset identifiers present'
    assert set(manifest.id)==set(hmm.id),'Manifest and HMM recovery dataset IDs do not match'
    assert np.isfinite(hmm.oos_mean_logscore).all(), 'HMM predictive log scores contain NaN/Inf'
    assert len(ranking)==ranking[['scenario','model']].drop_duplicates().shape[0], 'Duplicate aggregated scenario/model entries'
    rawcounts=manifest.groupby('scenario')['id'].nunique().rename('n_paths')
    hmmcounts=hmm.groupby('scenario')['id'].nunique().rename('n_hmm_fits')
    runsize=int(manifest.n.iloc[0]);test_size=int((manifest.n-manifest.n_train).iloc[0])
    rng=np.random.default_rng(20261029)
    merged=hmm.merge(manifest[['id','true_rho','true_pi0','n']],on='id',how='inner',validate='one_to_one')
    merged['train_loglik_gain_vs_truth']=merged['train_loglik']-merged['true_parameter_train_loglik']
    merged['negative_ll_gap']=merged['train_loglik_gain_vs_truth'] < -1.e-5
    scenario_rows=[]; param_rows=[]
    for scen,g in merged.groupby('scenario',sort=True):
        pi0=float(g.true_pi0.iloc[0]); baseline=max(pi0,1-pi0)
        ident=str(g.identification.iloc[0]);is_id=ident!='unidentified'
        rho=np.asarray(g.estimated_rho,float)
        ci=boot_mean_ci(rho,rng)
        ss=np.asarray(g.oos_mean_logscore,float)
        _,avscore,sescore=stats_mean(ss)
        scenario_rows.append(dict(scenario=scen,n_paths=len(g),model_identification=ident,
           true_rho=float(g.true_rho.iloc[0]) if is_id else np.nan,
           estimated_rho_mean=float(np.mean(rho)) if is_id else np.nan,
           estimated_rho_mc_se=stats_mean(rho)[2] if is_id else np.nan,
           estimated_rho_ci95_lo=ci[0] if is_id else np.nan,
           estimated_rho_ci95_hi=ci[1] if is_id else np.nan,
           state_accuracy_mean=safe_float(g.state_classification_accuracy_test.mean()) if is_id else np.nan,
           constant_state_majority_accuracy=baseline if is_id else np.nan,
           accuracy_minus_majority=safe_float(g.state_classification_accuracy_test.mean())-baseline if is_id else np.nan,
           oos_logscore_mean=avscore,oos_logscore_mc_se=sescore,
           optimization_successes=int(g.optimizer_success.sum()),
           ll_below_true_count=int(g.negative_ll_gap.sum()),
           min_train_ll_gain_vs_true=float(g.train_loglik_gain_vs_truth.min()),
           nu0_ge99=int((g.aligned_nu0>=99).sum()),nu1_ge99=int((g.aligned_nu1>=99).sum()),
           oracle_label_swap_share=float(g.used_oracle_label_swap_for_evaluation.mean())))
        for par in PARAMETERS:
            est=g[f'aligned_{par}'].to_numpy(float)
            true=float(g[f'true_{par}'].iloc[0])
            if not is_id:continue
            diff=est-true
            ci=boot_mean_ci(diff,rng)
            param_rows.append(dict(scenario=scen,parameter=par,true=true,mean_est=float(np.mean(est)),
                  bias=float(np.mean(diff)),rmse=float(np.sqrt(np.mean(diff**2))),
                  mc_se_bias=stats_mean(diff)[2],bias_bootstrap_ci95_lo=ci[0],bias_bootstrap_ci95_hi=ci[1],
                  near_upper_bound_ge99=int((est>=99).sum()) if par.startswith('nu') else 0))
    scen_df=pd.DataFrame(scenario_rows)
    prm_df=pd.DataFrame(param_rows)
    scen_df.to_csv(out/'SCENARIO_RECOVERY_AUDIT.csv',index=False)
    prm_df.to_csv(out/'PARAMETER_RECOVERY_MC_INTERVALS.csv',index=False)
    numflags=merged.loc[merged.negative_ll_gap,['id','scenario','rep','train_loglik','true_parameter_train_loglik','train_loglik_gain_vs_truth']].copy()
    numflags.to_csv(out/'HMM_NUMERICAL_FLAGS.csv',index=False)

    # High penalty paths in the raw EGARCH score cannot be genuine log densities.
    rank=ranking.copy()
    rank['numerically_suspect_aggregate']=((rank.mean_logscore<SENTINEL_ALERT) | (rank.MC_SE_mean_logscore>1000))
    rank.to_csv(out/'MODEL_RANKING_NUMERICAL_FLAGS.csv',index=False)
    comparison=[]
    for scen, g in rank.groupby('scenario'):
        s=scen_df.set_index('scenario').loc[scen]
        for _,m in g.iterrows():
            diff=float(s.oos_logscore_mean-m.mean_logscore)
            se_max=float(s.oos_logscore_mc_se+m.MC_SE_mean_logscore)
            comparison.append(dict(scenario=scen,model=m.model,n_paths=int(m.n_paths),
               hmm_mean_logscore=s.oos_logscore_mean,comp_mean_logscore=m.mean_logscore,
               hmm_minus_comp_mean_score=diff,
               SE_difference_upper_bound_Cauchy=se_max,
               lower_95_using_SE_upper_bound=diff-1.96*se_max,
               upper_95_using_SE_upper_bound=diff+1.96*se_max,
               comparison_valid_without_refit=not bool(m["numerically_suspect_aggregate"]),
               needs_path_paired_inference=True))
    cmp_df=pd.DataFrame(comparison)
    cmp_df.to_csv(out/'MODEL_COMPARISONS_PRELIM_UNPAIRED_BOUNDS.csv',index=False)

    aparch_rows=[]
    for _,a in aparch.iterrows():
        if str(a.model) not in ['APARCH11-t','APARCH11-power-symmetric-t']:
            continue
        n=int(a.paths)
        md=safe_float(a.mean_delta); sd=safe_float(a.sd_delta)
        mg=safe_float(a.mean_gamma_power);sg=safe_float(a.sd_gamma_power)
        aparch_rows.append(dict(scenario=a.scenario,model=a.model,paths=n,mean_delta=md,
                   mc_se_delta=sd/math.sqrt(n),mean_gamma=mg,mc_se_gamma=sg/math.sqrt(n) if np.isfinite(sg) else np.nan))
    apdf=pd.DataFrame(aparch_rows)
    apdf.to_csv(out/'APARCH_POWER_ASYMMETRY_AUDIT.csv',index=False)

    paired_df=None
    offenders=None
    if full_competitors_path and full_competitors_path.exists():
        comp=pd.read_csv(full_competitors_path)
        if not set(['id','scenario','model','oos_mean_logscore','hmm6_minus_competitor_logscore']).issubset(comp.columns):
            raise ValueError('ALL_COMPETITOR_SUMMARIES missing required columns')
        if comp.duplicated(['id','model']).any():raise ValueError('Duplicate competitor ID/model fits')
        exp=len(manifest)*len(ranking.model.unique())
        if len(comp)!=exp:
            print(f'WARNING: competitor summaries {len(comp)} versus {exp} expected; partial or different run.')
        comp['invalid_numeric_score']=comp.oos_mean_logscore<SENTINEL_ALERT
        offenders=comp[comp.invalid_numeric_score][['id','scenario','rep','model','oos_mean_logscore','optimizer_success']].copy()
        offenders.to_csv(out/'EGARCH_AND_OTHER_NUMERICAL_PENALTY_PATHS.csv',index=False)
        rows=[]
        for (scen,model),g in comp.groupby(['scenario','model']):
            nbad=int(g.invalid_numeric_score.sum())
            # EGARCH must be repaired if even one penalized result entered the original aggregate.
            valid=g.loc[~g.invalid_numeric_score].copy()
            vals=valid.hmm6_minus_competitor_logscore.to_numpy(float)
            n,mean,se=stats_mean(vals)
            lo,hi=boot_mean_ci(vals,rng)
            rows.append(dict(scenario=scen,model=model,valid_paired_paths=n,
                 penalized_paths=nbad,original_path_count=len(g),
                 mean_hmm_minus_competitor=mean,paired_mc_se=se,
                 bootstrap_ci95_lo=lo,bootstrap_ci95_hi=hi,
                 model_inference_status=('REPAIR_REQUIRED' if nbad else 'DESCRIPTIVE_PAIRED_VALID'),
                 optimizer_nonsuccess_count=int((g.optimizer_success==False).sum()) if 'optimizer_success' in g else np.nan))
        paired_df=pd.DataFrame(rows)
        paired_df.to_csv(out/'PATH_MATCHED_HMM6_VS_19_MODELS.csv',index=False)

    lines=[]
    def add(s=''):lines.append(s)
    add('# HMM-6 — independent numerical, uncertainty and theory audit')
    add('**Date:** 29 October 2026  |  **Status:** Independent audit; not a publication acceptance certificate.')
    add('')
    add(f'**Inputs:** {len(manifest):,} distinct paths, {len(rawcounts)} DGPs, {runsize:,} observations per path ({test_size:,} OOS).')
    add(f'**HMM fits:** {len(hmm):,} recorded; **models:** {len(ranking.model.unique())} specifications and {len(ranking)} scenario-model aggregates.')
    add('The full blind-model path-level file was '+('available; paired uncertainty is calculated.' if paired_df is not None else 'not included in these uploaded aggregates; use the companion Windows audit on the original study directory.'))
    add('')
    add('## 1. Recovery and empirical identification')
    add(write_md_table(scen_df[['scenario','true_rho','estimated_rho_mean','estimated_rho_ci95_lo','estimated_rho_ci95_hi','state_accuracy_mean','constant_state_majority_accuracy','nu0_ge99']].round(5).rename(columns={'scenario':'DGP','true_rho':'True rho','estimated_rho_mean':'Mean rho','estimated_rho_ci95_lo':'CI low','estimated_rho_ci95_hi':'CI high','state_accuracy_mean':'State acc.','constant_state_majority_accuracy':'Majority acc.','nu0_ge99':'nu0 >=99'}),digits=4))
    add('')
    add('State-classification accuracy uses oracle label alignment **only ex post** for evaluation. The constant-state classifier assumes known true stationary state probabilities; it is an easy benchmark, not an operational forecasting algorithm.')
    add('The bootstrap intervals describe Monte Carlo uncertainty of the *mean estimate across simulated paths*, not per-dataset confidence-interval coverage and not finite-T confidence intervals for a single fitted parameter.')
    add('')
    add('### Recovery conclusions')
    add('- Calibration-anchor, full-HMM and scale-only DGPs show close recovery of transition persistence; they do **not** justify universal tail parameter identification.')
    add('- Tail-only and weak-tail DGPs have severe transition/tail recovery problems even at T=6,000; a simple majority-state label benchmark exceeds the recorded classifier accuracy.')
    add('- IID-mixture data have true zero persistence but a positive average fitted rho; this is **false persistence**, not evidence of true hidden-state dynamics.')
    add('- The single-Student-t DGP has observationally identical emissions: transitions and regime identity are unidentifiable. Do not report their nominal recovery, standard tests or spurious rho as a conventional identifiable estimand.')
    add('')
    add('## 2. Numerical implementation audit')
    add(f'- HMM optimizer `success=True`: {int(hmm.optimizer_success.sum())}/{len(hmm)}. One or more fits still below the likelihood at the true parameter vector: **{len(numflags)}**. IDs: '+(', '.join(numflags.id.tolist()) if len(numflags) else '(none)')+'.')
    add('- These are finite local maximization quality checks; maximization over all admissible parameters must be no worse than evaluating the truth.')
    suspect=rank.loc[rank.numerically_suspect_aggregate,['scenario','model','mean_logscore']].sort_values('scenario')
    add('- **EGARCH numerical penalty contamination:** The custom implementation assigns a `-1e8` per-observation penalty after an invalid conditional-variance recursion. The scenario aggregates listed below have extreme numerical penalties, as revealed by their mean scores; they are **invalid predictive comparisons** without path-level repair.')
    add('')
    add(write_md_table(suspect.round(3),digits=3))
    add('')
    add('- Penalties are implementation artifacts, not evidence of economically catastrophic volatility forecasts. Use the raw per-path file to flag/re-estimate the affected EGARCH paths; never relabel the penalty as a genuine log density.')
    add('- Parameter estimates can sit at bounds with apparent numerical convergence; report upper-bound frequencies and restart stability.')
    add('')
    add('## 3. Predictive-density comparison')
    primary_cmp=cmp_df[(cmp_df.model.isin(PRIMARY))].copy().sort_values(['scenario','model'])
    add(write_md_table(primary_cmp[['scenario','model','hmm_minus_comp_mean_score','SE_difference_upper_bound_Cauchy','lower_95_using_SE_upper_bound','upper_95_using_SE_upper_bound']].round(5),digits=5))
    add('')
    add('Difference = HMM-6 minus competitor; positive favors HMM-6. The displayed uncertainty is a **Cauchy upper bound** on the paired standard error computed from marginal MC SEs, not an exact paired CI. It is deliberately conservative, and the 95% endpoints assume approximate normality. Do not report these as final paired hypothesis tests. All paths are matched; calculate the actual 500 path-wise differences using `ALL_COMPETITOR_SUMMARIES.csv`.')
    if paired_df is not None:
        add('')
        add('### Exact matched-path uncertainty (independent simulation paths)')
        p=paired_df[paired_df.model.isin(PRIMARY)].copy()
        add(write_md_table(p[['scenario','model','mean_hmm_minus_competitor','paired_mc_se','bootstrap_ci95_lo','bootstrap_ci95_hi']].round(6),digits=6))
        add(f'Flagged mean-logscore numerical penalty fits: {len(offenders)}. EGARCH scenario rankings with penalties are not inference-valid.')
    add('')
    add('## 4. APARCH reverse attribution')
    add(write_md_table(apdf[['scenario','model','mean_delta','mc_se_delta','mean_gamma','mc_se_gamma']].round(4),digits=4))
    add('')
    add('In the strong calibration DGP, estimated delta is close to one; in tail-only DGP, its mean is close to two, while leverage estimates have high cross-replication dispersion. A mean leverage near zero is not proof of correct coverage or zero effects in every path.')
    add('')
    add('## 5. Theoretical reconciliation — manuscript-ready boundaries')
    add('- **Target baseline:** six-parameter centered two-state Student-t HMM, psi=(p00,p11,nu0,nu1,s0,s1). A state-specific mean model adds two parameters and is not HMM-6. With nu_j>2, q_j=(nu_j+3)/(nu_j+1) lies in (1,5/3), **not** (1,3).')
    add('**Valid local result, for fixed finite T:** under interior/local identification, score regularity, positive observed Fisher information I_T and a locally unique pseudo-true branch, the derivative of psi*_T(epsilon) at epsilon=0+ is I_T^{-1} E_0[S_T W_T]; W_T is constructed with full-sequence *smoothed* state weights and a fixed Gamma shock direction (k,theta).')
    add('- **Do not claim:** global Gamma jump identification, a universal sign reversal in fitted q, standard chi-square Sup-LR law at epsilon=0, automatic T-to-infinity convergence, or 6-parameter recovery when the two emissions are identical.')
    add('- **Critical separation:** the 4,000-path experiment evaluates the **uncontaminated** HMM-6 estimator and its blind competitors. It does NOT validate the omitted-Bernoulli-Gamma derivative theorem; that requires the separate finite-window impulse/derivative experiment and its own numerical error assessment.')
    add('- **Data caveats:** synthetic return-level results do not demonstrate any real-market historical Gamma jumps, risk-neutral pricing or economic welfare improvement. The empirical gap-aware HMM may have eight or more free parameters and must not be labeled a six-parameter model.')
    add('- **Manuscript reconciliation:** the earlier 26-Oct proposal PDF describes an eight-parameter mean-switching baseline and asserts global jump identification, standard boundary inference and a generic q sign-reversal. Those assertions are not supported by these data. The later GT_HMM6_Paper.pdf contains the narrower six-parameter finite-window theorem and is the defensible starting point, subject to checking its proofs and independent numerical evidence.')
    add('- **Publication blocker:** verify EGARCH implementation; compute exact paired uncertainty; document tail-only weak identification and false persistence, and decide whether to report a conditional identification region rather than a universal applicability claim.')
    add('')
    add('## 6. Submission-ready wording (suggested)')
    add('> In 4,000 independent length-6,000 synthetic samples across eight generating regimes, the centered two-state Student-t HMM recovered transition persistence accurately when regimes were separated by conditional scale. Recovery deteriorated markedly when hidden regimes differed chiefly in tail shape, and an unrestricted HMM inferred positive average persistence under an IID-mixture control. These findings qualify the practical scope of latent-tail identification. The reverse-attribution study covered 19 conventional model specifications; numerical penalty contamination in selected EGARCH recursions requires repair before final comparative inference. The finite-window omitted-shock score sensitivity theorem is a separate local result and is not inferred from these no-jump Monte Carlo experiments.')
    add('')
    add('## 7. Remaining tasks with no new global Monte Carlo requirement')
    add('1. Run the exact paired audit on the original `FULL_T6000_R500/blind_competitor_results/ALL_COMPETITOR_SUMMARIES.csv`; retain all failures, do not let penalty constants enter density scores.')
    add('2. Independently verify EGARCH recursion and Student-t likelihood against a trusted reference implementation; selectively rerun **only** affected numerical cases (if needed).')
    add('3. Check all six-parameter HMM theoretical assumptions and observational identity / label conventions. Report finite-T theorem separately from global or long-run conjectures.')
    add('4. Consolidate the validated DGP tables with the existing finite-window Gamma-shock numerical experiment. Include negative and null outcomes explicitly.')
    add('5. Freeze experiment scripts, seeds, dates, versions, and input/output hashes in the public reproducibility archive.')
    add('')
    (out/'HMM6_FINAL_AUDIT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:')
    for p in sorted(out.glob('*.csv')):print('  ',p.name)
    print('  HMM6_FINAL_AUDIT_REPORT.md')
    print(f'Numerically suspect EGARCH scenario summaries: {len(suspect)}; HMM loglike-below-truth: {len(numflags)}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,help='Full study output folder; looks for its CSVs and blind_competitor_results')
    p.add_argument('--source-dir',type=Path,help='Export folder with CSV files; handles (2) file suffixes')
    p.add_argument('--out',type=Path,default=Path('HMM6_FINAL_AUDIT'))
    p.add_argument('--all-competitors',type=Path,help='Override path to complete ALL_COMPETITOR_SUMMARIES.csv')
    args=p.parse_args()
    if not args.study and not args.source_dir:p.error('Supply --study or --source-dir')
    study=(args.study or args.source_dir).resolve()
    path=args.all_competitors
    if path is None and args.study:
        path=study/'blind_competitor_results'/'ALL_COMPETITOR_SUMMARIES.csv'
    run(study,args.out,path)

if __name__=='__main__':main()
