#!/usr/bin/env python3
"""01 - Simulate known six-parameter centered Student-t HMMs and recover HMM-6.
Saves complete paths, true latent states, true conditional variance, metadata,
train-only fitted parameters, out-of-sample scores, and cross-replication recovery.

Example:
  python 01_generate_and_recover_hmm6.py --n 1713 --reps 10 --starts 6 --out experiment
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from giorno_models import (make_scenario_table,simulate_hmm6,fit_hmm6,hmm_forward,
                           hmm_test_predict,ensure_dir,save_json,hmm_stationary_pi0)

KEYS=['p00','p11','nu0','nu1','s0','s1']

def align_for_evaluation(fitted, truth):
    """Oracle label reconciliation is ONLY for Monte Carlo evaluation, never for fitting."""
    a=np.array([fitted[k] for k in KEYS],float)
    b=np.array([truth[k] for k in KEYS],float)
    sw=np.array([a[1],a[0],a[3],a[2],a[5],a[4]])
    def score(v):
        # transformed distance so ν and scale are comparably weighted
        # p mismatch is not reliable when emissions are identical (flagged).
        return float(np.sum((np.log(v[2:4])-np.log(b[2:4]))**2)
                     +np.sum((np.log(v[4:])-np.log(b[4:]))**2))
    if score(sw)<score(a):return dict(zip(KEYS,sw)),True
    return dict(zip(KEYS,a)),False


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',default='Giorno_HMM6_DGP_Study')
    ap.add_argument('--n',type=int,default=1713,help='Observations in each generated path')
    ap.add_argument('--reps',type=int,default=3,help='Independent replications per DGP')
    ap.add_argument('--burn',type=int,default=600)
    ap.add_argument('--train-fraction',type=float,default=.8)
    ap.add_argument('--starts',type=int,default=5,help='HMM optimizer restarts')
    ap.add_argument('--maxiter',type=int,default=250)
    ap.add_argument('--seed',type=int,default=20261028)
    ap.add_argument('--scenarios',default='all',help='Comma-separated names or all')
    ap.add_argument('--resume',action='store_true',help='Continue from saved per-path outputs; requires identical design')
    args=ap.parse_args()
    if args.n<100 or args.burn<0 or args.reps<1 or not .55<=args.train_fraction<=.95:
        ap.error('n>=100, reps>=1, burn>=0, train fraction between .55 and .95')
    out=ensure_dir(args.out);ds=ensure_dir(out/'datasets');res=ensure_dir(out/'hmm6_fits')
    options=make_scenario_table()
    scenarios=list(options) if args.scenarios=='all' else [v.strip() for v in args.scenarios.split(',')]
    unknown=set(scenarios)-set(options)
    if unknown: ap.error(f'Unknown scenarios {unknown}; choose {list(options)}')
    design_file=out/'study_design.json'
    if args.resume and design_file.exists():
        with open(design_file,encoding='utf8') as f: saved=json.load(f)
        old=saved['command_line_options']
        for k in ['n','reps','burn','train_fraction','seed','scenarios']:
            if old[k]!=getattr(args,k):
                ap.error(f'Cannot resume with changed {k}: saved={old[k]} vs new={getattr(args,k)}')
    else:
        save_json(design_file,{'command_line_options':vars(args),'scenarios':options,
                 'model':'six parameters (p00,p11,nu0,nu1,s0,s1); centered Student t scale emissions',
                 'identifiability_warning':'identical states -> transition probabilities not identified',
                 'label_alignment':'truth used only for ex post evaluation, never to initialize optimizer'})
    random_seeds=np.random.SeedSequence(args.seed).spawn(len(scenarios)*args.reps)
    manifest=[];rows=[]
    done=set()
    if args.resume and (out/'DGP_MANIFEST.csv').exists() and (out/'HMM6_PARAMETER_RECOVERY_ALL.csv').exists():
        old_m=pd.read_csv(out/'DGP_MANIFEST.csv')
        old_r=pd.read_csv(out/'HMM6_PARAMETER_RECOVERY_ALL.csv')
        done={v for v in set(old_m.id)&set(old_r.id)
              if (out/'hmm6_fits'/f'{v}__fit.json').exists()
              and (out/'datasets'/f'{v}.csv').exists()}
        manifest=old_m[old_m.id.isin(done)].to_dict('records')
        rows=old_r[old_r.id.isin(done)].to_dict('records')
        print(f'Resuming Script 01: {len(done)} paths already finished.',flush=True)
    start_wall=time.time()
    for sidx,scen in enumerate(scenarios):
        pars=options[scen]
        for rep in range(args.reps):
            seed=int(random_seeds[sidx*args.reps+rep].generate_state(1)[0])
            ident=f'{scen}__rep{rep:04d}'
            if ident in done: continue
            data=simulate_hmm6(args.n,args.burn,pars,seed)
            ntrain=int(len(data)*args.train_fraction)
            data['sample']=np.where(data.t<ntrain,'train','test')
            path=ds/f'{ident}.csv';data.to_csv(path,index=False,float_format='%.12g')
            row={'id':ident,'scenario':scen,'rep':rep,'seed':seed,'n':args.n,'n_train':ntrain,
                 'dataset_path':str(path.relative_to(out)),'identification':pars['identification'],
                 'true_rho':pars['p00']+pars['p11']-1.,
                 'true_pi0':hmm_stationary_pi0(pars['p00'],pars['p11']),
                 **{f'true_{k}':pars[k] for k in KEYS}}
            manifest.append(row)
            t0=time.time()
            fitted=fit_hmm6(data['return'].values[:ntrain],starts=args.starts,maxiter=args.maxiter,
                            seed=(seed+1)%(2**32))
            fpars=fitted['params']
            truth_aligned,eval_swapped=align_for_evaluation(fpars,pars)
            params=np.array([fpars[k] for k in KEYS])
            y=data['return'].to_numpy()
            logs,pred,filtered=hmm_test_predict(y[:ntrain],y[ntrain:],params)
            trainlog,trainpred,trainfiltered=hmm_forward(y[:ntrain],params)
            pred_all=np.r_[trainpred,pred]
            filtered_all=np.r_[trainfiltered,filtered]
            if eval_swapped:
                pred_all=1-pred_all;filtered_all=1-filtered_all
            data['estimated_prob_true_state0_pred']=pred_all
            data['estimated_prob_true_state0_filtered']=filtered_all
            data['hmm6_oos_logscore']=np.r_[np.full(ntrain,np.nan),logs]
            data.to_csv(res/f'{ident}__hmm6_predictions.csv',index=False,float_format='%.12g')
            # Accuracy is meaningful only if the states produce distinct emissions.
            cls=float(np.mean((filtered_all[ntrain:]<.5).astype(int)==data.true_state.values[ntrain:])) if pars['identification']!='unidentified' else np.nan
            loss=float(np.mean((filtered_all[ntrain:]-(1-data.true_state.values[ntrain:]))**2)) if pars['identification']!='unidentified' else np.nan
            rec={'id':ident,'scenario':scen,'rep':rep,'identification':pars['identification'],
                 'train_loglik':fitted['loglik_train'],
                 'true_parameter_train_loglik':hmm_forward(y[:ntrain],
                     np.array([pars[k] for k in KEYS],float))[0],
                 'train_BIC':-2*fitted['loglik_train']+6*np.log(ntrain),
                 'oos_mean_logscore':float(np.mean(logs)),
                 'state_classification_accuracy_test':cls,'state_Brier_test':loss,
                 'optimizer_success':fitted['success'],'seconds':time.time()-t0,
                 'used_oracle_label_swap_for_evaluation':eval_swapped,
                 'estimated_rho':fpars['p00']+fpars['p11']-1.}
            for k in KEYS:
                rec[f'estimated_{k}']=fpars[k]
                rec[f'aligned_{k}']=truth_aligned[k]
                rec[f'true_{k}']=pars[k]
                rec[f'error_{k}']=(truth_aligned[k]-pars[k]) if pars['identification']!='unidentified' else np.nan
            rows.append(rec)
            save_json(res/f'{ident}__fit.json',{'DGP':row,'fit':fitted,
                     'oracle_alignment_only_for_evaluation':eval_swapped,
                     'estimator_did_not_use_truth':True})
            # Checkpoints: interrupted long Monte Carlo runs can resume safely.
            pd.DataFrame(manifest).to_csv(out/'DGP_MANIFEST.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'HMM6_PARAMETER_RECOVERY_ALL.csv',index=False)
            print(f'[01] {ident}: HMM train LL={fitted["loglik_train"]:.2f}; test mean logscore={np.mean(logs):.5f}; converged={fitted["success"]}; {time.time()-t0:.1f}s',flush=True)
    pd.DataFrame(manifest).to_csv(out/'DGP_MANIFEST.csv',index=False)
    fitted_df=pd.DataFrame(rows)
    fitted_df.to_csv(out/'HMM6_PARAMETER_RECOVERY_ALL.csv',index=False)
    summary=[]
    for scen,g in fitted_df.groupby('scenario'):
        a={'scenario':scen,'reps':len(g),'convergence_fraction':g.optimizer_success.mean(),
           'oos_mean_logscore':g.oos_mean_logscore.mean(),
           'mean_state_classification_accuracy':g.state_classification_accuracy_test.mean(),
           'mean_estimated_rho':g.estimated_rho.mean() if options[scen]['identification']!='unidentified' else np.nan}
        for k in KEYS:
            a[f'true_{k}']=float(options[scen][k]);a[f'mean_estimated_{k}']=g[f'aligned_{k}'].mean()
            a[f'bias_{k}']=g[f'error_{k}'].mean()
            a[f'rmse_{k}']=np.sqrt(np.nanmean(g[f'error_{k}']**2)) if options[scen]['identification']!='unidentified' else np.nan
        summary.append(a)
    pd.DataFrame(summary).to_csv(out/'HMM6_RECOVERY_SUMMARY.csv',index=False)
    print(f'\nSaved {len(manifest)} complete DGP paths and HMM results in {out.resolve()} ({time.time()-start_wall:.1f}s)',flush=True)

if __name__=='__main__':main()
