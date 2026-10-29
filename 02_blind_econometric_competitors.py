#!/usr/bin/env python3
"""02 - Blind econometric estimators on paths from 01_generate_and_recover_hmm6.py.

Fits Student-t, IID Student-t mixture, ARMA-t, GARCH-t, GJR-t, EGARCH-t,
APARCH-t and an HN-style asymmetric GARCH-t variance recursion by ML.
All are fitted to TRAIN observations only. Out-of-sample log densities use
one-step updates from *past* observed returns and frozen fitted coefficients.
The true HMM parameter/state columns are not accessed during fitting.
They are read only afterwards for reverse-attribution diagnostics.

This is a self-contained estimator implementation, not the 'arch' package;
validate first with the included smoke test before research-scale use.
No Heston--Nandi closed-form option pricing is claimed for Student-t errors.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t
from giorno_models import ensure_dir,save_json

# Numba types: 0=GARCH11, 1=GARCH12, 2=GJR11,
# 3=EGARCH11, 4=APARCH11, 5=HN11, 6=GARCH11_NORMAL
GARCH_SPECS={
 'GARCH11-t':0, 'GARCH12-t':1, 'GJR11-t':2,
 'EGARCH11-t':3, 'APARCH11-t':4, 'HN11-t':5,
 'GARCH11-normal':6,
}
# All restricted APARCH models reduce the optimization dimension; fixed
# coefficients are never left unidentified in a full parameter vector.
APARCH_VARIANTS = {
    'APARCH11-power-symmetric-t': {'gamma_power': 0.0},
    'APARCH11-delta2-asym-t': {'delta': 2.0},
    'APARCH11-delta2-symmetric-t': {'gamma_power': 0.0, 'delta': 2.0},
    'APARCH11-delta1-asym-t': {'delta': 1.0},
    'APARCH11-delta1-symmetric-t': {'gamma_power': 0.0, 'delta': 1.0},
}
APARCH_FULL_NAMES = ['mu','omega_power','alpha_power','beta_power','gamma_power','delta','nu']
APARCH_ALL_KEYS = ['APARCH11-t', *APARCH_VARIANTS.keys()]

MEAN_SPECS={'Static-t':(0,0),'AR2-t':(2,0),'AR3-t':(3,0),
            'MA1-t':(0,1),'ARMA11-t':(1,1),'ARMA12-t':(1,2)}
CORE=['Static-t','IIDmix-t','ARMA11-t','GARCH11-t','GJR11-t','EGARCH11-t','APARCH11-t','HN11-t']
ALL=['Static-t','IIDmix-t','AR2-t','AR3-t','MA1-t','ARMA11-t','ARMA12-t',
     'GARCH11-normal','GARCH11-t','GARCH12-t','GJR11-t','EGARCH11-t',
     *APARCH_ALL_KEYS,'HN11-t']

@njit(cache=True)
def t_logpdf_mu_var(err, h, nu, const):
    return const-.5*np.log(h) - (nu+1.)*.5*np.log1p(err*err/((nu-2.)*h))

@njit(cache=True)
def _garch_filter(y,pars,typ,initial_var,logconst,ehabs):
    n=len(y)
    scores=np.empty(n,np.float64)
    hs=np.empty(n,np.float64)
    errs=np.empty(n,np.float64)
    mu=pars[0];omega=pars[1];alpha=pars[2];beta=pars[3]
    # optional fields indexed by family
    beta2=pars[4] if typ==1 else 0.0
    gamma=pars[4] if typ in (2,3,4,5) else 0.0
    power=pars[5] if typ==4 else 2.0
    nu=pars[-1] if typ!=6 else 8.0
    hprev=initial_var
    hprev2=initial_var
    eprev=0.0
    loghprev=np.log(initial_var)
    for t in range(n):
        if t==0:
            h=initial_var
        elif typ==3:  # EGARCH
            zprev=eprev/np.sqrt(hprev)
            lh=omega+beta*loghprev+alpha*(abs(zprev)-ehabs)+gamma*zprev
            if lh<-24 or lh>24:
                for v in range(t,n):scores[v]=-1e8;hs[v]=1e-20;errs[v]=0.
                return scores,hs,errs
            h=np.exp(lh)
        elif typ==4:  # APARCH
            q=omega+alpha*(abs(eprev)-gamma*eprev)**power+beta*hprev**(power*.5)
            h=max(q,1e-20)**(2./power)
        elif typ==5:  # Heston-Nandi style variance equation
            zprev=eprev/np.sqrt(hprev)
            h=omega+beta*hprev+alpha*(zprev-gamma*np.sqrt(hprev))**2
        else:
            h=omega+alpha*eprev**2+beta*hprev+beta2*hprev2
            if typ==2 and eprev<0.:h+=gamma*eprev**2
        if not np.isfinite(h) or h<1e-14 or h>1e14:
            for v in range(t,n):scores[v]=-1e8;hs[v]=1e-20;errs[v]=0.
            return scores,hs,errs
        err=y[t]-mu
        if typ==6:score=-.5*(np.log(2.*np.pi*h)+err*err/h)
        else: score=t_logpdf_mu_var(err,h,nu,logconst)
        scores[t]=score
        hs[t]=h
        errs[t]=err
        eprev=err
        hprev2=hprev
        hprev=h
        loghprev=np.log(h)
    return scores,hs,errs

@njit(cache=True)
def _arma_filter(y,pars,p,q,logconst):
    n=len(y)
    mu=pars[0]
    sig=pars[-2]
    nu=pars[-1]
    v=sig*sig
    errs=np.empty(n,np.float64)
    scores=np.empty(n,np.float64)
    hs=np.empty(n,np.float64)
    for t in range(n):
        conditional=mu
        for k in range(p):
            if t>k: conditional+=pars[1+k]*(y[t-k-1]-mu)
        for k in range(q):
            if t>k: conditional+=pars[1+p+k]*errs[t-k-1]
        e=y[t]-conditional
        errs[t]=e
        hs[t]=v
        scores[t]=t_logpdf_mu_var(e,v,nu,logconst)
    return scores,hs,errs


def _t_const(nu):
    return float(gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log((nu-2)*np.pi))

def _t_eabs(nu):
    return float(2*np.sqrt(nu-2)*np.exp(gammaln((nu+1)/2)-gammaln(nu/2))/((nu-1)*np.sqrt(np.pi)))


def expand_aparch(name, compact):
    """Convert identified free coefficients into the full APARCH recursion vector.

    Fixed-power and symmetric restrictions are *not* estimated and do not
    consume degrees of freedom in AIC/BIC.
    """
    if name not in APARCH_VARIANTS:
        return np.asarray(compact, dtype=float)
    fixed=APARCH_VARIANTS[name]
    it=iter(compact)
    return np.array([fixed[k] if k in fixed else next(it)
                     for k in APARCH_FULL_NAMES], dtype=float)


def aparch_persistence(full):
    """beta + alpha E[(|z|-gamma*z)^delta] for standardized Student-t z."""
    alpha,beta,gamma,delta,nu = full[2],full[3],full[4],full[5],full[6]
    if nu <= max(2.,delta):
        return float('inf')
    log_abs_moment=(.5*delta*np.log(nu-2.)
                    + gammaln((delta+1.)/2.)
                    + gammaln((nu-delta)/2.)
                    - .5*np.log(np.pi) - gammaln(nu/2.))
    multiplier=((1.-gamma)**delta+(1.+gamma)**delta)/2.
    return float(beta+alpha*np.exp(log_abs_moment)*multiplier)


def spec_bounds(name,y):
    v=max(np.var(y),.01)
    mu=float(np.mean(y)); sd=float(np.sqrt(v))
    if name in APARCH_VARIANTS:
        full_names, full_starts, full_bounds = spec_bounds('APARCH11-t', y)
        fixed = APARCH_VARIANTS[name]
        free = [(n, v, b) for n,v,b in zip(full_names, full_starts, full_bounds) if n not in fixed]
        return [x[0] for x in free], [x[1] for x in free], [x[2] for x in free]
    if name in MEAN_SPECS:
        p,q=MEAN_SPECS[name]
        names=['mu']+[f'AR{i+1}' for i in range(p)]+[f'MA{i+1}' for i in range(q)]+['sigma','nu']
        starts=[mu]+[0.0]*p+[0.0]*q+[sd,8.]
        bounds=[(mu-3*sd,mu+3*sd)]+[(-.96,.96)]*(p+q)+[(sd*.08,sd*4),(2.1,120.)]
        return names,starts,bounds
    if name=='IIDmix-t':
        names=['mu','mix_w0','sigma0','sigma1','nu0','nu1']
        starts=[mu,.75,sd*.75,sd*1.25,15.,5.]
        bounds=[(mu-3*sd,mu+3*sd),(.02,.98),(sd*.08,sd*4),(sd*.08,sd*4),(2.1,120.),(2.1,120.)]
        return names,starts,bounds
    typ=GARCH_SPECS[name]
    if typ==0: return ['mu','omega','alpha1','beta1','nu'],[mu,.07*v,.09,.84,8.],[(mu-3*sd,mu+3*sd),(1e-6,.8*v),(.00001,.7),(.00001,.998),(2.1,120.)]
    if typ==1: return ['mu','omega','alpha1','beta1','beta2','nu'],[mu,.06*v,.09,.82,.02,8.],[(mu-3*sd,mu+3*sd),(1e-6,.8*v),(.00001,.7),(.00001,.998),(0.,.75),(2.1,120.)]
    if typ==2: return ['mu','omega','alpha1','beta1','leverage','nu'],[mu,.08*v,.06,.82,.1,8.],[(mu-3*sd,mu+3*sd),(1e-6,.8*v),(0.,.7),(.00001,.998),(-.3,.8),(2.1,120.)]
    if typ==3: return ['mu','logomega','alpha_absz','beta_logh','gamma_z','nu'],[mu,.03*np.log(v),.17,.93,-.08,8.],[(mu-3*sd,mu+3*sd),(-3.,3.),(-.5,1.),(.02,.999),(-1.,1.),(2.1,120.)]
    if typ==4: return ['mu','omega_power','alpha_power','beta_power','gamma_power','delta','nu'],[mu,.07*v,.08,.84,.14,2.,8.],[(mu-3*sd,mu+3*sd),(1e-6,max(.8*v**1.5,1e-3)),(0.,.6),(.001,.99),(-.9,.9),(.7,3.),(2.1,120.)]
    if typ==5: return ['mu','omega','alpha_hn','beta_hn','gamma_hn','nu'],[mu,.2*v,.08,.8,.3,8.],[(mu-3*sd,mu+3*sd),(1e-6,.9*v),(.00001,.8),(.00001,.995),(-2.5,2.5),(2.1,120.)]
    if typ==6: return ['mu','omega','alpha1','beta1'],[mu,.07*v,.09,.84],[(mu-3*sd,mu+3*sd),(1e-6,.8*v),(.00001,.7),(.00001,.998)]
    raise ValueError(name)


def eval_model(name,y,params,initvar):
    pars=np.asarray(params,dtype=np.float64)
    if name=='IIDmix-t':
        mu,w,s0,s1,nu0,nu1=pars
        s0*=np.sqrt((nu0-2)/nu0)
        s1*=np.sqrt((nu1-2)/nu1)
        l0=(_t_const(nu0)-np.log(pars[2])
            -.5*(nu0+1.)*np.log1p(((y-mu)/pars[2])**2/(nu0-2.)))
        l1=(_t_const(nu1)-np.log(pars[3])
            -.5*(nu1+1.)*np.log1p(((y-mu)/pars[3])**2/(nu1-2.)))
        logs=np.logaddexp(np.log(w)+l0,np.log1p(-w)+l1)
        vv=w*pars[2]**2+(1-w)*pars[3]**2
        return logs,np.full(len(y),vv),y-mu
    if name in MEAN_SPECS:
        p,q=MEAN_SPECS[name]
        return _arma_filter(np.asarray(y,dtype=float),pars,p,q,_t_const(pars[-1]))
    if name in APARCH_VARIANTS:
        pars=expand_aparch(name,pars)
        typ=4
    else:
        typ=GARCH_SPECS[name]
    nu=pars[-1] if typ!=6 else 8.
    return _garch_filter(np.asarray(y,dtype=float),pars,typ,float(initvar),
                         _t_const(nu) if typ!=6 else 0.,_t_eabs(nu) if typ==3 else 0.)


def valid_parameters(name,p):
    if name.startswith('AR') or name.startswith('MA'):
        nump,numq=MEAN_SPECS[name]
        ar=p[1:1+nump];ma=p[1+nump:1+nump+numq]
        if nump:
            r=np.roots(np.r_[-ar[::-1],1.])
            if np.any(np.abs(r)<1.005):return False
        if numq:
            r=np.roots(np.r_[ma[::-1],1.])
            if np.any(np.abs(r)<1.005):return False
        return True
    if name in APARCH_VARIANTS or name=='APARCH11-t':
        full=expand_aparch(name,p)
        return bool(np.all(np.isfinite(full)) and abs(full[4])<1.0 and aparch_persistence(full)<.995)
    if name not in GARCH_SPECS:return True
    typ=GARCH_SPECS[name]
    a,b=p[2],p[3]
    if typ in (0,6) and a+b>=.995:return False
    if typ==1 and a+b+p[4]>=.995:return False
    if typ==2 and (a+b+.5*p[4]>=.995 or a+p[4]<0):return False
    if typ==5 and b+a*p[4]*p[4]>=.995:return False
    return True


def fit_model(name,y,starts=2,maxiter=160,seed=2026):
    """Blind fit: no DGP truth is accessed. All params are training-only."""
    y=np.asarray(y,dtype=float)
    v=float(np.var(y));v=max(v,1e-4)
    names,x0,bounds=spec_bounds(name,y)
    x0=np.asarray(x0,dtype=float)
    rng=np.random.default_rng(seed)
    def nll(p):
        if not valid_parameters(name,p):return 1e8
        try:
            l,_,_=eval_model(name,y,p,v)
            if not np.all(np.isfinite(l)): return 1e8
            loss=-float(np.sum(l))
            return loss if np.isfinite(loss) else 1e8
        except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):return 1e8
    best=None; all_runs=[]
    constraints=[]
    if name in GARCH_SPECS:
        typ=GARCH_SPECS[name]
        if typ in (0,6):constraints=[{'type':'ineq','fun':lambda p:.9948-p[2]-p[3]}]
        elif typ==1:constraints=[{'type':'ineq','fun':lambda p:.9948-p[2]-p[3]-p[4]}]
        elif typ==2:constraints=[{'type':'ineq','fun':lambda p:.9948-p[2]-p[3]-.5*p[4]},
                                 {'type':'ineq','fun':lambda p:p[2]+p[4]}]
        elif typ==5:constraints=[{'type':'ineq','fun':lambda p:.9948-p[3]-p[2]*p[4]**2}]
    if name in APARCH_VARIANTS or name=='APARCH11-t':
        constraints=[{'type':'ineq', 'fun':lambda p:.9948-aparch_persistence(expand_aparch(name,p))}]
    for i in range(starts):
        initial=x0.copy()
        if i:
            noise=rng.normal(0,.12,len(x0))
            for k in range(len(initial)):
                initial[k]=initial[k]+noise[k]*max(abs(initial[k]), .15 if 'nu' not in names[k] else 3.)
        initial=np.clip(initial,np.array(bounds)[:,0]+1e-6,np.array(bounds)[:,1]-1e-6)
        fit=minimize(nll,initial,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':maxiter,'ftol':5e-9})
        all_runs.append({'start':i,'method':'L-BFGS-B','success':bool(fit.success),'nll':float(fit.fun),'nit':int(fit.nit),'message':str(fit.message)})
        if best is None or fit.fun<best.fun:best=fit
        if constraints and (not fit.success or fit.fun>=1e7):
            refine=minimize(nll, fit.x, method='SLSQP',bounds=bounds,constraints=constraints,
                            options={'maxiter':maxiter,'ftol':1e-7})
            all_runs.append({'start':i,'method':'SLSQP','success':bool(refine.success),
                             'nll':float(refine.fun),'nit':int(refine.nit),'message':str(refine.message)})
            if refine.fun < best.fun-1e-7 or (refine.success and not best.success and abs(refine.fun-best.fun)<1e-5):
                best=refine
    pars=best.x
    logpdf,vpath,err=eval_model(name,y,pars,v)
    # Including the initialized first observation is consistent across competitors.
    return {'model':name,'parameter_names':names,'params':dict(zip(names,map(float,pars))),
            'fixed_parameters':APARCH_VARIANTS.get(name,{}),
            'train_loglik':float(np.sum(logpdf)),'BIC':float(-2*np.sum(logpdf)+len(pars)*np.log(len(y))),
            'AIC':float(-2*np.sum(logpdf)+2*len(pars)),
            'optimizer_success':bool(best.success),'starts':all_runs,
            'training_initvar':v}


def safe_corr(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<15 or np.std(x[ok])<1e-9 or np.std(y[ok])<1e-9:return float('nan')
    return float(np.corrcoef(x[ok],y[ok])[0,1])


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study',default='Giorno_HMM6_DGP_Study',help='Directory generated by Script 01')
    ap.add_argument('--out',default=None,help='Defaults to STUDY/blind_competitor_results')
    ap.add_argument('--models',default='core',help='core, all, or comma-separated model keys')
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--maxiter',type=int,default=180)
    ap.add_argument('--limit-paths',type=int,default=0,help='Limit number of datasets for a pilot (0=all)')
    ap.add_argument('--seed',type=int,default=20261028)
    ap.add_argument('--resume',action='store_true',help='Skip fully completed paths in an interrupted study')
    args=ap.parse_args()
    study=Path(args.study);manifest_path=study/'DGP_MANIFEST.csv'
    if not manifest_path.exists():ap.error(f'No manifest: {manifest_path}; run 01 first')
    manifest=pd.read_csv(manifest_path)
    if args.limit_paths>0:manifest=manifest.iloc[:args.limit_paths]
    models=(CORE if args.models=='core' else ALL if args.models=='all'
            else ['GARCH11-t','GJR11-t',*APARCH_ALL_KEYS] if args.models=='aparch'
            else [s.strip() for s in args.models.split(',')])
    invalid=set(models)-set(ALL)
    if invalid:ap.error(f'Unknown model keys {invalid}; choose from {ALL}')
    out=ensure_dir(args.out or (study/'blind_competitor_results'))
    obs_dir=ensure_dir(out/'per_path_forecasts')
    fit_dir=ensure_dir(out/'per_path_fits')
    save_json(out/'competitor_study_design.json',{'options':vars(args),'models':models,
       'architecture':'Student-t errors, fixed train-only parameters, causal one-step test log densities',
       'inference_limitation':'All fits are custom MLE; compare convergence/restarts; not ARMA+GARCH joint',
       'hn_warning':'HN-style variance recursion with Student-t innovations is not closed-form Heston-Nandi option pricing',
       'true_info_used_only_after_fit':True,
       'APARCH_restrictions':APARCH_VARIANTS,
       'APARCH_note':'delta=2,gamma=0 is GARCH; delta=2,asymmetric is GJR-like by coefficient mapping; EGARCH/HN are not nested' })
    all_rows=[];all_params=[]
    completed_ids=set()
    old_summary=out/'ALL_COMPETITOR_SUMMARIES.csv'
    old_params=out/'ALL_FITTED_PARAMETERS.csv'
    if args.resume and old_summary.exists():
        previous=pd.read_csv(old_summary)
        completed_ids={ident for ident,g in previous.groupby('id')
                       if set(models).issubset(set(g.loc[g['oos_mean_logscore'].notna(),'model']))
                       and (obs_dir/f'{ident}__blind_forecasts.csv').exists()
                       and all((fit_dir/f'{ident}__{m}__fit.json').exists() for m in models)}
        all_rows=previous[previous.id.isin(completed_ids)].to_dict('records')
        if old_params.exists():
            pp=pd.read_csv(old_params)
            all_params=pp[pp.id.isin(completed_ids)].to_dict('records')
        print(f'Resuming: {len(completed_ids)} complete paths retained.',flush=True)
    for idx,entry in enumerate(manifest.itertuples(index=False)):
        if entry.id in completed_ids:
            print(f'[02] SKIP already-complete {entry.id}',flush=True)
            continue
        fp=study/entry.dataset_path
        df=pd.read_csv(fp)
        y=df['return'].to_numpy(float)
        ntrain=int(entry.n_train)
        if not np.isfinite(y).all() or ntrain<30:raise ValueError(f'Invalid data {fp}')
        summary=[]
        output=pd.DataFrame({'t':df.t,'sample':df['sample'],'actual':y,
            'true_state':df.true_state,
            'true_conditional_variance':df.true_pred_cond_variance,
            'true_state1_probability':1.-df.true_state0_prob_pred})
        preds_hmm=study/'hmm6_fits'/f'{entry.id}__hmm6_predictions.csv'
        hmm_log=np.full(len(y),np.nan)
        if preds_hmm.exists():
            hmm=pd.read_csv(preds_hmm)
            if len(hmm)==len(df) and np.max(np.abs(hmm['return'].to_numpy()-y))<1e-7:
                hmm_log=hmm['hmm6_oos_logscore'].to_numpy()
                output['HMM6_logscore']=hmm_log
            else:raise ValueError(f'HMM predictions not aligned for {entry.id}')
        for mindex,name in enumerate(models):
            t0=time.time()
            model_seed=(args.seed+idx*15427+mindex*653)%(2**32)
            try:
                fit=fit_model(name,y[:ntrain],starts=args.starts,maxiter=args.maxiter,seed=model_seed)
                params=np.array([fit['params'][k] for k in fit['parameter_names']],float)
                log_scores,h,innov=eval_model(name,y,params,fit['training_initvar'])
                score=float(np.mean(log_scores[ntrain:]))
                vtrue=output['true_conditional_variance'].to_numpy()[ntrain:]
                vfit=h[ntrain:]
                variability=float(np.std(vfit)/np.mean(vfit)) if np.mean(vfit)>0 else np.nan
                row={'id':entry.id,'scenario':entry.scenario,'rep':entry.rep,'model':name,
                     'n_train':ntrain,'n_test':len(y)-ntrain,
                     'train_BIC':fit['BIC'],'train_AIC':fit['AIC'],'train_loglik':fit['train_loglik'],
                     'oos_mean_logscore':score,
                     'hmm6_minus_competitor_logscore':(float(np.nanmean(hmm_log[ntrain:]-log_scores[ntrain:])) if np.isfinite(hmm_log[ntrain:]).any() else np.nan),
                     'optimizer_success':fit['optimizer_success'],'seconds':time.time()-t0,
                     'estimated_variance_CV_test':variability,
                     'variance_MSE_vs_true_test':float(np.mean((vfit-vtrue)**2)),
                     'corr_variance_to_true_tailprob_test':safe_corr(vfit,output['true_state1_probability'].to_numpy()[ntrain:]),
                     'corr_variance_to_true_variance_test':safe_corr(vfit,vtrue),
                     'train_stdret_abs_gt_2':float(np.mean(np.abs(innov[:ntrain]/np.sqrt(np.maximum(h[:ntrain],1e-12)))>2.)),
                     'test_stdret_abs_gt_2':float(np.mean(np.abs(innov[ntrain:]/np.sqrt(np.maximum(h[ntrain:],1e-12)))>2.))}
                for k,v in fit['params'].items():row[k]=v
                for k,v in fit.get('fixed_parameters',{}).items():
                    row[k]=v
                    row[f'{k}_is_fixed']=True
                summary.append(row)
                all_rows.append(row)
                output[f'{name}__logscore']=log_scores
                output[f'{name}__pred_variance']=h
                output[f'{name}__innovation']=innov
                save_json(fit_dir/f'{entry.id}__{name}__fit.json',fit)
                all_params.append({'id':entry.id,'scenario':entry.scenario,'model':name,
                                   **fit['params'],**fit.get('fixed_parameters',{})})
                print(f'[02] {entry.id:29} {name:20} logscore={score:.5f}  converged={fit["optimizer_success"]}  {time.time()-t0:.1f}s',flush=True)
            except Exception as exc:
                all_rows.append({'id':entry.id,'scenario':entry.scenario,'rep':entry.rep,'model':name,'error':repr(exc)})
                print(f'[02] FAILURE {entry.id} {name}: {exc}',flush=True)
        output.to_csv(obs_dir/f'{entry.id}__blind_forecasts.csv',index=False,float_format='%.12g')
        pd.DataFrame(summary).to_csv(fit_dir/f'{entry.id}__blind_summary.csv',index=False)
        pd.DataFrame(all_rows).to_csv(out/'ALL_COMPETITOR_SUMMARIES.csv',index=False)
    report=pd.DataFrame(all_rows)
    # Rebuild this table from per-fit JSON: robust if an interrupted run is resumed
    # after ALL_COMPETITOR_SUMMARIES was checkpointed but before parameter aggregation.
    all_params=[]
    for record in all_rows:
        pth=fit_dir/f'{record["id"]}__{record["model"]}__fit.json'
        if pth.exists():
            with open(pth,encoding='utf8') as handle:
                fitted_json=json.load(handle)
            all_params.append({'id':record['id'],'scenario':record['scenario'],
                               'model':record['model'],**fitted_json['params'],**fitted_json.get('fixed_parameters',{})})
    pd.DataFrame(all_params).to_csv(out/'ALL_FITTED_PARAMETERS.csv',index=False)
    complete=report[report.get('oos_mean_logscore',pd.Series(dtype=float)).notna()].copy()
    if not complete.empty:
        agg=complete.groupby(['scenario','model'],as_index=False).agg(
          paths=('id','count'),mean_oos_logscore=('oos_mean_logscore','mean'),
          mean_BIC=('train_BIC','mean'),mean_hmm_minus_model=('hmm6_minus_competitor_logscore','mean'),
          mean_variance_CV=('estimated_variance_CV_test','mean'),
          mean_variance_MSE=('variance_MSE_vs_true_test','mean'),
          mean_corr_estimated_variance_to_tail_state=('corr_variance_to_true_tailprob_test','mean'),
          converged_fraction=('optimizer_success','mean'))
        agg.to_csv(out/'REVERSE_ATTRIBUTION_BY_DGP.csv',index=False)
        structural=['mu','omega_power','AR1','AR2','AR3','MA1','MA2','alpha1','alpha_power',
                    'alpha_absz','alpha_hn','beta1','beta2','beta_logh',
                    'beta_power','beta_hn','gamma_z','gamma_power','gamma_hn',
                    'leverage','nu','delta','sigma','mix_w0']
        coeffs=[]
        for (scen,model),group in complete.groupby(['scenario','model']):
            record={'scenario':scen,'model':model,'replications':len(group)}
            for key in structural:
                if key in group and group[key].notna().any():
                    vals=pd.to_numeric(group[key],errors='coerce').dropna()
                    if len(vals):
                        record[f'mean_{key}']=float(vals.mean())
                        record[f'std_{key}']=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
            coeffs.append(record)
        pd.DataFrame(coeffs).to_csv(out/'COEFFICIENT_ATTRIBUTION_BY_DGP.csv',index=False)
    print(f'\nSaved blind model fits + causal observations in: {out.resolve()}',flush=True)

if __name__=='__main__':main()
