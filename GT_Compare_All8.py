#!/usr/bin/env python3
"""Compare eight WIN/US M1/M5/M10/M15 GARCH-t, iid-mixture, GARCH-HMM runs.

Never infer statistical significance from aggregated model_summary.csv data.
Paired tests are computed ONLY from matching holdout_forecasts.csv observations.
Dependencies: numpy, scipy. Standard-library CSV/JSON for I/O.
"""
from __future__ import annotations
import argparse, csv, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import norm

MARKETS = ('US', 'WIN')
FREQUENCIES = ('M1', 'M5', 'M10', 'M15')
MODELS = ('garch_t', 'garch_mix_t', 'garch_hmm')
LABELS = {'garch_t': 'GARCH–Student-t', 'garch_mix_t': 'GARCH + iid Student-t mixture', 'garch_hmm': 'GARCH–HMM'}
LEVELS = ('95.0', '99.0', '99.5')
PAIRS = [('garch_hmm','garch_mix_t'), ('garch_hmm','garch_t'), ('garch_mix_t','garch_t')]

def csv_rows(p):
    with open(p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, cols):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr=csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)

def holm(pvals):
    n=len(pvals)
    order=sorted(range(n),key=lambda i:pvals[i]);out=[None]*n;high=0.
    for j,i in enumerate(order):
        high=max(high,min(1.,(n-j)*pvals[i]));out[i]=high
    return out

def nw_z(d):
    """Bartlett HAC; automatic bandwidth is a descriptive choice, not an oracle."""
    x=np.asarray(d,dtype=float)
    n=x.size;mean=float(x.mean());z=x-mean
    lag=max(1,min(n-1, int(round(4*(n/100)**(2/9)))))
    longvar=float(np.dot(z,z)/n)
    for k in range(1,lag+1):
        longvar += 2*(1-k/(lag+1))*float(np.dot(z[k:],z[:-k])/n)
    se=math.sqrt(max(longvar,0)/n)
    zstat=mean/se if se>0 else float('nan')
    pval=float(2*norm.sf(abs(zstat))) if math.isfinite(zstat) else float('nan')
    return mean,se,zstat,pval,lag

def day_boot_ci(x,days,rng,B):
    x=np.asarray(x,float)
    dates=np.asarray(days)
    uniq,inverse=np.unique(dates,return_inverse=True)
    ns=np.bincount(inverse)
    if len(uniq)<6:
        return float('nan'),float('nan'),len(uniq)
    sums=np.bincount(inverse,weights=x)
    ind=rng.integers(0,len(uniq),size=(B,len(uniq)))
    dist=np.sum(sums[ind],axis=1)/np.sum(ns[ind],axis=1)
    return float(np.quantile(dist,.025)),float(np.quantile(dist,.975)),len(uniq)

def evaluate_forecast(filepath,market,timeframe,B,seed):
    rows=csv_rows(filepath)
    if not rows: raise ValueError('Forecast CSV is empty')
    needed=['timestamp'] + [f'{m}_logdensity' for m in MODELS]
    needed += [f'{m}_loss_{a.replace(".","p")}' for m in MODELS for a in LEVELS]
    missing=set(needed)-set(rows[0])
    if missing: raise ValueError('Missing columns: '+str(sorted(missing)))
    rng=np.random.default_rng(seed)
    out=[]
    days=[r['timestamp'][:10] for r in rows]
    def compare(key,larger,smaller):
        a=np.array([float(row[f'{larger}_{key}']) for row in rows])
        b=np.array([float(row[f'{smaller}_{key}']) for row in rows])
        # For log score higher is better; for pinball loss lower is better.
        d=a-b if key=='logdensity' else b-a
        if not np.isfinite(d).all():raise ValueError('Nonfinite paired values')
        mean,se,z,p,lag=nw_z(d)
        ci_lo,ci_hi,n_days=day_boot_ci(d,days,rng,B)
        rel=(100*mean/np.mean(b)) if (key!='logdensity' and np.mean(b)!=0) else None
        return dict(market=market,timeframe=timeframe,n=len(rows),n_days=n_days,
                    score='mean_log_density' if key=='logdensity' else key,
                    comparison=f'{larger} vs {smaller}',
                    mean_gain=float(mean),relative_gain_pct=rel,
                    hac_se=se,hac_z=z,hac_two_sided_p=p,hac_lags=lag,
                    day_boot_ci_low=ci_lo,day_boot_ci_high=ci_hi,
                    day_boot_reps=B)
    for larger,smaller in PAIRS:
        out.append(compare('logdensity',larger,smaller))
        for lev in LEVELS:
            out.append(compare('loss_'+lev.replace('.','p'),larger,smaller))
    return out,len(rows)

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--summaries',required=True,help='GT_GARCH_HMM_All_8_Verified_Summaries.csv')
    pa.add_argument('--base-dir',default='.',help='Directory of GT_GARCH_HMM_US_M1, ...')
    pa.add_argument('--forecast',action='append',default=[],help='Market:timeframe=path e.g. US:M1=holdout_forecasts.csv')
    pa.add_argument('--out',default='GT_ALL8_COMPARISON')
    pa.add_argument('--bootstrap',type=int,default=1500)
    pa.add_argument('--seed',type=int,default=20261027)
    args=pa.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    full=csv_rows(args.summaries)
    coll=defaultdict(dict)
    for r in full:
        g=(r['market'],r['timeframe']);m=r['model']
        if (g[0] not in MARKETS) or (g[1] not in FREQUENCIES):continue
        coll[g].setdefault(m,{})[r['confidence_pct'] or 'density']=r
    if len(coll)!=8:raise ValueError(f'Expected 8 market-timeframes, found {len(coll)}: {list(coll)}')
    density=[];var=[]
    for market in MARKETS:
        for tf in FREQUENCIES:
            ms=coll[(market,tf)]
            if set(ms)!=set(MODELS):raise ValueError(f'Missing models for {(market,tf)}')
            scored={m:float(ms[m]['density']['mean_log_score']) for m in MODELS}
            row=dict(market=market,timeframe=tf,n=int(ms['garch_t']['density']['n']),
                     best_model=max(MODELS,key=lambda x:scored[x]),
                     hmm_minus_garch_t=scored['garch_hmm']-scored['garch_t'],
                     hmm_minus_iid_mix=scored['garch_hmm']-scored['garch_mix_t'],
                     iid_mix_minus_garch_t=scored['garch_mix_t']-scored['garch_t'])
            for m in MODELS:row[m+'_logscore']=scored[m]
            density.append(row)
            for lev in LEVELS:
                losses={m:float(ms[m][lev]['mean_pinball_loss']) for m in MODELS}
                breach={m:100*float(ms[m][lev]['exceedance_rate']) for m in MODELS}
                vr=dict(market=market,timeframe=tf,confidence_pct=float(lev),
                        target_breach_pct=100-float(lev),n=row['n'],
                        best_pinball_model=min(MODELS,key=lambda x:losses[x]),
                        hmm_vs_iid_mix_loss_reduction_pct=100*(losses['garch_mix_t']-losses['garch_hmm'])/losses['garch_mix_t'],
                        hmm_vs_garch_t_loss_reduction_pct=100*(losses['garch_t']-losses['garch_hmm'])/losses['garch_t'])
                for m in MODELS:
                    vr[m+'_pinball']=losses[m];vr[m+'_breach_pct']=breach[m]
                var.append(vr)
    write_csv(out/'density_comparison.csv',density,list(density[0]))
    write_csv(out/'VaR_comparison.csv',var,list(var[0]))
    explicit={}
    for arg in args.forecast:
        tag,sep,filename=arg.partition('=')
        if not sep:raise ValueError('Use --forecast US:M1=path')
        market,tf=tag.split(':')
        explicit[(market,tf)]=Path(filename)
    paired=[];audit=[];diag=[]
    for market in MARKETS:
        for tf in FREQUENCIES:
            g=(market,tf)
            folder=Path(args.base_dir)/f'GT_GARCH_HMM_{market}_{tf}'
            f=explicit.get(g, folder/'holdout_forecasts.csv')
            record=dict(market=market,timeframe=tf,forecast_file=str(f),
                        observation_level_available=f.is_file(),paired_test_completed=False,
                        status='summary_only; statistical comparisons unavailable')
            if f.is_file():
                try:
                    tests,n=evaluate_forecast(f,market,tf,args.bootstrap,args.seed)
                    expected=next(x['n'] for x in density if x['market']==market and x['timeframe']==tf)
                    if n!=expected:raise ValueError(f'Forecast rows {n} != summary test n {expected}. Wrong file/timeframe?')
                    raw=csv_rows(f)
                    for model in MODELS:
                        observed=float(np.mean([float(rr[f'{model}_logdensity']) for rr in raw]))
                        desired=float(coll[g][model]['density']['mean_log_score'])
                        if abs(observed-desired)>1e-5:
                            raise ValueError(f'{model} mean logscore mismatch: forecasts={observed}, summary={desired}')
                    paired.extend(tests);record['paired_test_completed']=True
                    record['status']='Paired tests available'
                except Exception as e:record['status']='ERROR: '+str(e)
            audit.append(record)
            meta=folder/'run_metadata.json'
            if meta.is_file():
                raw=json.load(open(meta,encoding='utf-8'))
                hmm=raw.get('models',{}).get('garch_hmm',{})
                if hmm:
                    p00=hmm.get('p00');p11=hmm.get('p11')
                    diag.append(dict(market=market,timeframe=tf,p00=p00,p11=p11,
                                     markov_second_eigen=(p00+p11-1) if (p00 is not None and p11 is not None) else None,
                                     nu0=hmm.get('nu0'),nu1=hmm.get('nu1'),
                                     max_nu_near_65=any((v is not None and v>=64.9) for v in (hmm.get('nu0'),hmm.get('nu1'))),
                                     train_aic_hmm=hmm.get('aic_train'),train_aic_iid_mix=raw.get('models',{}).get('garch_mix_t',{}).get('aic_train')))
    write_csv(out/'observation_files_audit.csv',audit,list(audit[0]))
    if paired:
        finite=[i for i,r in enumerate(paired) if math.isfinite(r['hac_two_sided_p'])]
        adjusted=holm([paired[i]['hac_two_sided_p'] for i in finite])
        for i,v in zip(finite,adjusted):paired[i]['holm_p_available_tests_only']=v
        write_csv(out/'paired_tests_available_only.csv',paired,list(paired[0]))
    if diag:write_csv(out/'transition_diagnostics_available.csv',diag,list(diag[0]))
    dwin=Counter(x['best_model'] for x in density)
    vwin=Counter(x['best_pinball_model'] for x in var)
    cases=sum(row['paired_test_completed'] for row in audit)
    lines=[
        '# Eight-experiment GARCH / mixture / HMM comparison',
        '',
        '**Scope:** two markets, four sampling frequencies, three jointly estimated GARCH-based models.',
        '**Instrument note:** US label remains provisional until source identity is independently verified.',
        '',
        '## Descriptive model rankings',
        '',
        f'Density winners (eight groups): '+', '.join(f'{LABELS[m]} {dwin[m]}/8' for m in MODELS)+'.',
        f'Pinball winners (24 groups): '+', '.join(f'{LABELS[m]} {vwin[m]}/24' for m in MODELS)+'.',
        '',
        '| Market | Freq | N test | Best density | HMM − t logscore | HMM − iid mixture logscore |',
        '|---|---:|---:|---|---:|---:|',
    ]
    for x in density:
        lines.append(f"| {x['market']} | {x['timeframe']} | {x['n']:,} | {LABELS[x['best_model']]} | {x['hmm_minus_garch_t']:+.6f} | {x['hmm_minus_iid_mix']:+.6f} |")
    lines.extend(['','## Inferential status','',f'Observation-level matched forecasts available and evaluated for **{cases}/8** groups.',''])
    for row in audit:
        lines.append(f"- {row['market']} {row['timeframe']}: {row['status']}")
    lines.extend(['','The summary CSVs cannot support paired HAC or day-bootstrap inference. ',
                  'The script computes such tests only for genuine matched forecast rows; missing records are NOT imputed. ',
                  'The Holm column, if present, adjusts only tests currently available, not the complete prespecified eight-frequency family.',
                  'Day-resampling treats calendar dates as blocks and assumes sufficient cross-day independence; HAC bandwidth and multiple-comparison design still need methodological validation.',
                  '', '## Limits and next tests', '',
                  '- All results use a **single** chronological holdout at each frequency; no repeated-origin conclusions are justified.',
                  '- Cross-frequency log scores and VaR losses refer to different horizon distributions, and should not be directly ranked by their raw magnitude.',
                  '- Tail states that do not persist or reach optimization bounds are not compelling evidence for Markov dependence.',
                  '- A static Student-t model **without GARCH** is not included in these current eight experiments.',
                  '- To test Markov persistence, compare HMM against independent mixture, inspect transition eigenvalues and use nonstandard-valid inference.',
                  '- Before final claims, obtain all eight holdout_forecasts.csv and run_metadata.json files, check duplicate data across frequencies, and repeat rolling-origin evaluations.',
                  ''])
    (out/'COMPARISON_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Results written to',out)
    print('Density winners',dict(dwin),'VaR winners',dict(vwin),'Paired groups',cases,'/8')
    for a in audit:print(a['market'],a['timeframe'],a['status'])

if __name__=='__main__':main()
