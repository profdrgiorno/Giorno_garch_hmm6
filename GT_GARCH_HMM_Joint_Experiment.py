#!/usr/bin/env python3
"""Giorno--Tsallis: joint common-GARCH + latent Student-t tail-regime experiment.

Three genuinely JOINT maximum-likelihood models, on *identical* intraday returns:
  M0: centered GARCH(1,1) with one unit-variance Student-t innovation.
  M1: common GARCH(1,1) with an i.i.d. two-Student-t mixture (no persistence).\n  M2: common GARCH(1,1) with a two-state HMM for unit-variance Student-t shape nu_j.

The HMM states change tail shape, NOT the innovation's conditional variance.
Estimation is from training data only; test filtering/volatility use only past returns.
Only exactly spaced within-session observed close-to-close returns are included.
Overnight and irregular return intervals are audited and excluded, never silently
regarded as consecutive minute observations. Sessions/gaps reset both filters.
At M1/M5/M10/M15 prices are selected directly at scheduled timestamps.
This is not evidence for historical Gamma jumps and does not estimate HMM-9.

Dependencies: numpy pandas scipy numba
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import t as student_t
from numba import njit

LEVELS=(0.05,0.01,0.005)


def parse_minutes(s):
    h,m=map(int,s.split(':'))
    if not (0<=h<=23 and 0<=m<=59): raise ValueError(f'Invalid clock time: {s}')
    return 60*h+m


def load_returns(file_path,fmt,minutes,rth_start,rth_end):
    path=Path(file_path)
    if fmt=='win':
        df=pd.read_csv(path,sep='\t')
        needed=['<DATE>','<TIME>','<CLOSE>']
        if any(x not in df for x in needed): raise ValueError(f'WIN CSV must have {needed}')
        ts=pd.to_datetime(df['<DATE>'].astype(str)+' '+df['<TIME>'].astype(str),
                          format='%Y.%m.%d %H:%M:%S',errors='coerce')
        price=pd.to_numeric(df['<CLOSE>'],errors='coerce')
    else:
        df=pd.read_csv(path)
        if not {'datetime','Close'}.issubset(df.columns):
            raise ValueError('US CSV must contain datetime and Close columns')
        ts=pd.to_datetime(df['datetime'],errors='coerce')
        price=pd.to_numeric(df['Close'],errors='coerce')
    raw_count=len(df)
    raw=pd.DataFrame({'timestamp':ts,'close':price}).dropna()
    raw=raw[raw.close>0].sort_values('timestamp')
    ndup=int(raw.timestamp.duplicated().sum())
    if ndup:
        warnings.warn(f'{ndup} duplicate timestamps, keeping the last observed price for each timestamp')
        raw=raw.drop_duplicates('timestamp',keep='last')
    rth_a=parse_minutes(rth_start);rth_b=parse_minutes(rth_end)
    if rth_b<=rth_a: raise ValueError('Session end must exceed session start on same local date')
    clock=raw.timestamp.dt.hour*60+raw.timestamp.dt.minute
    sel=(clock>=rth_a)&(clock<=rth_b)&(((clock-rth_a)%minutes)==0)
    chosen=raw.loc[sel].copy().reset_index(drop=True)
    if len(chosen)<200: raise ValueError('Too few scheduled observed prices at this frequency')
    deltam=(chosen.timestamp-chosen.timestamp.shift(1)).dt.total_seconds()/60.0
    dates=chosen.timestamp.dt.normalize()
    same_day=dates.eq(dates.shift(1))
    valid=(deltam==float(minutes))&same_day
    chosen['return_pct_log']=100.0*np.log(chosen.close/chosen.close.shift(1))
    valid_returns=chosen.loc[valid,['timestamp','return_pct_log']].copy().reset_index(drop=True)
    if len(valid_returns)<150: raise ValueError('Too few exact intraday returns')
    # The preceding price at an irregular boundary is not interpreted as a return;
    # a new segment begins whenever two surviving returns are not consecutive.
    vm=(valid_returns.timestamp-valid_returns.timestamp.shift(1)).dt.total_seconds()/60.0
    vday=valid_returns.timestamp.dt.normalize()
    starts=~((vm==float(minutes))&vday.eq(vday.shift(1)))
    starts.iloc[0]=True
    valid_returns['segment_start']=starts.astype(bool)
    meta={'raw_rows':int(raw_count),'valid_prices':int(len(raw)),
          'duplicate_timestamps':ndup,'scheduled_prices':int(len(chosen)),
          'valid_intraday_returns':int(len(valid_returns)),
          'excluded_day_boundary_price_intervals':int((~same_day).sum()),
          'excluded_nonconsecutive_same_day_price_intervals':int((same_day & ~valid).sum()),
          'segments':int(starts.sum()),'timeframe_minutes':minutes,
          'rth_start':rth_start,'rth_end':rth_end,
          'units':'percent log returns: 100*log(close_t/close_(t-delta))',
          'limitations':'Time zone and instrument identity follow source; not independently verified. Only exact observed intraday intervals included.'}
    return valid_returns,meta


@njit(cache=True)
def log_t_unitvar(x,h,nu):
    # x is demeaned return; h is conditional VARIANCE, not Student-t scale squared.
    sc2=h*(nu-2.0)/nu
    z2=x*x/sc2
    return (math.lgamma((nu+1.0)/2.0)-math.lgamma(nu/2.0)
            -0.5*math.log(math.pi*nu*sc2)
            -0.5*(nu+1.0)*math.log1p(z2/nu))


@njit(cache=True)
def loglik_path(x, start, vbar, alpha, beta, nu0, nu1, p00, p11, hmm):
    pi0=(1.0-p11)/(2.0-p00-p11) if hmm else 1.0
    omega=vbar*(1.0-alpha-beta)
    hprev=vbar
    posterior0=pi0
    prev=0.0
    ll=0.0
    for t in range(x.shape[0]):
        if start[t]:
            h=vbar
            pred0=pi0
        else:
            h=omega+alpha*prev*prev+beta*hprev
            pred0=posterior0*p00+(1.0-posterior0)*(1.0-p11) if hmm else 1.0
        l0=log_t_unitvar(x[t],h,nu0)
        if hmm:
            l1=log_t_unitvar(x[t],h,nu1)
            a=math.log(pred0)+l0
            b=math.log1p(-pred0)+l1
            c=max(a,b)
            l=c+math.log(math.exp(a-c)+math.exp(b-c))
            posterior0=math.exp(a-l)
        else:
            l=l0
        ll+=l
        prev=x[t]
        hprev=h
    return ll


def unpack(params,hmm):
    vbar=float(np.exp(params[0]))
    persistence=0.995*float(expit(params[1]))
    share=float(expit(params[2]))
    alpha=persistence*share
    beta=persistence*(1-share)
    if hmm:
        nu0,nu1=float(params[3]),float(params[4])
        p00,p11=float(expit(params[5])),float(expit(params[6]))
    else:
        nu0=nu1=float(params[3]);p00=p11=0.5
    return vbar,alpha,beta,nu0,nu1,p00,p11


def pack(vbar,alpha,beta,nu0,nu1=None,p00=.96,p11=.96):
    persistence=max(1e-5,min(.9949,alpha+beta))
    share=max(.001,min(.999,alpha/max(alpha+beta,1e-8)))
    base=[np.log(vbar),float(logit(persistence/.995)),float(logit(share)),nu0]
    if nu1 is not None:
        base.extend([nu1,float(logit(p00)),float(logit(p11))])
    return np.array(base,dtype=np.float64)


def fit_model(x,starts,hmm,maxiter=220,nstarts=3,seed=42):
    rng=np.random.default_rng(seed)
    empvar=max(float(np.var(x)),1e-8)
    bounds=[(np.log(empvar*0.01),np.log(empvar*100)),(-5,7),(-7,7),
            (2.05,65.0)]
    if hmm: bounds.extend([(2.05,65.0),(-5.0,8.0),(-5.0,8.0)])
    objective=lambda p: -loglik_path(x,starts,*unpack(p,hmm),hmm)/len(x)
    if hmm:
        seeds=[(3.2,12.,.97,.97),(8.,3.1,.96,.99),(4.5,20.,.90,.94),
               (2.8,6.5,.995,.89),(8.,16.,.80,.98)]
    else:
        seeds=[(3.5,),(6.5,),(12.,),(25.,)]
    raw=[]
    for i in range(nstarts):
        setting=seeds[i%len(seeds)]
        if i>=len(seeds):
            if hmm: setting=(float(rng.uniform(2.8,9)),float(rng.uniform(4,25)),
                             float(rng.uniform(.80,.995)),float(rng.uniform(.80,.995)))
            else: setting=(float(rng.uniform(3,20)),)
        alpha,beta=(.05,.90) if i%2==0 else (.10,.82)
        init=pack(empvar,alpha,beta,*setting)
        res=minimize(objective,init,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':maxiter,'ftol':1e-10,'gtol':3e-6,'maxls':35,'maxcor':15})
        raw.append(res)
        print(f'  {"GARCH-HMM" if hmm else "GARCH-t"} start {i+1}/{nstarts}: '
              f'loglike={-res.fun*len(x):.4f}, success={res.success}, iter={res.nit}',flush=True)
    best=min(raw,key=lambda r:r.fun)
    p=np.array(best.x)
    vbar,alpha,beta,nu0,nu1,p00,p11=unpack(p,hmm)
    if hmm and nu0>nu1: # consistent state0 = heavier tail; swap transition self-probabilities
        nu0,nu1=nu1,nu0
        p00,p11=p11,p00
    ret={'vbar':vbar,'omega':vbar*(1-alpha-beta),'alpha':alpha,'beta':beta,
         'persistence':alpha+beta,'nu0':nu0,'nu1':nu1,
         'q0':(nu0+3)/(nu0+1),'q1':(nu1+3)/(nu1+1),
         'p00':p00 if hmm else None,'p11':p11 if hmm else None,
         'n_parameters':7 if hmm else 4,
         'loglik_train':-float(best.fun)*len(x),'mean_loglik_train':-float(best.fun),
         'aic_train':-2*(-float(best.fun)*len(x))+2*(7 if hmm else 4),
         'optimization_success':bool(best.success),
         'optimizer_message':str(best.message),'gradient_inf_norm':float(np.max(abs(best.jac))),
         'starts':[{'success':bool(r.success),'loglik':float(-r.fun*len(x)),
                    'nit':int(r.nit),'message':str(r.message)} for r in raw]}
    return ret


def fit_iid_mixture(x,starts,maxiter=220,nstarts=3,seed=42):
    """GARCH with i.i.d. two-t mixture: tests flexible tails without Markov persistence.

    A transition matrix with identical rows [pi, 1-pi] gives independent latent states.
    This model has six free parameters vs seven in the full GARCH-HMM.
    """
    rng=np.random.default_rng(seed+1)
    empvar=max(float(np.var(x)),1e-8)
    bounds=[(np.log(empvar*.01),np.log(empvar*100)),(-5,7),(-7,7),
            (2.05,65.0),(2.05,65.0),(-5.0,5.0)]
    seeds=[(3.2,12.,.5),(8.,3.1,.5),(2.8,18.,.2),
           (12.,3.1,.75),(5.,25.,.35)]
    def obj(par):
        vbar=float(np.exp(par[0]))
        pers=.995*float(expit(par[1]))
        share=float(expit(par[2]))
        alpha,beta=pers*share,pers*(1-share)
        w=float(expit(par[5]))
        return -loglik_path(x,starts,vbar,alpha,beta,float(par[3]),float(par[4]),w,1-w,True)/len(x)
    fits=[]
    for i in range(nstarts):
        setting=seeds[i%len(seeds)]
        a,b=(.05,.90) if i%2==0 else (.10,.82)
        full=pack(empvar,a,b,setting[0],setting[1],setting[2],1-setting[2])
        init=np.array(full[:6])
        res=minimize(obj,init,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':maxiter,'ftol':1e-10,'gtol':3e-6,'maxls':35})
        fits.append(res)
        print(f'  GARCH-iid-mixture start {i+1}/{nstarts}: '
              f'loglike={-res.fun*len(x):.4f}, success={res.success}, iter={res.nit}',flush=True)
    best=min(fits,key=lambda z:z.fun)
    pp=best.x
    vbar=float(np.exp(pp[0]));pers=.995*float(expit(pp[1]))
    share=float(expit(pp[2]));a=pers*share;b=pers*(1-share)
    nu0=float(pp[3]);nu1=float(pp[4]);w=float(expit(pp[5]))
    if nu0>nu1:nu0,nu1=nu1,nu0;w=1-w
    return {'vbar':vbar,'omega':vbar*(1-a-b),'alpha':a,'beta':b,
            'persistence':a+b,'nu0':nu0,'nu1':nu1,
            'q0':(nu0+3)/(nu0+1),'q1':(nu1+3)/(nu1+1),
            'p00':w,'p11':1-w,'iid_mixture_weight0':w,'n_parameters':6,
            'loglik_train':float(-best.fun*len(x)),
            'mean_loglik_train':float(-best.fun),
            'aic_train':-2*float(-best.fun*len(x))+12,
            'optimization_success':bool(best.success),
            'optimizer_message':str(best.message),
            'gradient_inf_norm':float(np.max(abs(best.jac))),
            'starts':[{'success':bool(r.success),'loglik':float(-r.fun*len(x)),
                       'nit':int(r.nit),'message':str(r.message)} for r in fits]}


@njit(cache=True)
def pred_path(x,starts,vbar,alpha,beta,nu0,nu1,p00,p11,hmm):
    pi0=(1.0-p11)/(2.0-p00-p11) if hmm else 1.0
    omega=vbar*(1.0-alpha-beta)
    n=len(x)
    vol=np.empty(n)
    weights=np.empty(n)
    lp=np.empty(n)
    hprev=vbar
    prev=0.0
    posterior0=pi0
    for i in range(n):
        if starts[i]:
            h=vbar
            pred0=pi0
        else:
            h=omega+alpha*prev*prev+beta*hprev
            pred0=posterior0*p00+(1.0-posterior0)*(1.0-p11) if hmm else 1.0
        l0=log_t_unitvar(x[i],h,nu0)
        if hmm:
            l1=log_t_unitvar(x[i],h,nu1)
            a=math.log(pred0)+l0
            b=math.log1p(-pred0)+l1
            c=max(a,b)
            l=c+math.log(math.exp(a-c)+math.exp(b-c))
            posterior0=math.exp(a-l)
        else:
            l=l0
        vol[i]=h
        weights[i]=pred0
        lp[i]=l
        hprev=h
        prev=x[i]
    return vol,weights,lp


def forecast_params(x,starts,p):
    return pred_path(x,starts,p['vbar'],p['alpha'],p['beta'],p['nu0'],p['nu1'],
                     p['p00'] if p['p00'] is not None else .5,
                     p['p11'] if p['p11'] is not None else .5,p['p00'] is not None)


def quantiles(vol,w,p,alphas=LEVELS):
    s0=np.sqrt(vol*(p['nu0']-2.0)/p['nu0'])
    s1=np.sqrt(vol*(p['nu1']-2.0)/p['nu1'])
    out={}
    for a in alphas:
        if p['p00'] is None:
            out[a]=student_t.ppf(a,p['nu0'])*s0
            continue
        q0=student_t.ppf(a,p['nu0'])*s0
        q1=student_t.ppf(a,p['nu1'])*s1
        low=np.minimum(q0,q1)
        high=np.maximum(q0,q1)
        for _ in range(37):
            middle=.5*(low+high)
            cdf=w*student_t.cdf(middle/s0,p['nu0'])+(1-w)*student_t.cdf(middle/s1,p['nu1'])
            low=np.where(cdf<a,middle,low)
            high=np.where(cdf<a,high,middle)
        out[a]=.5*(low+high)
    return out


def hac_se(a,lags):
    a=np.asarray(a,dtype=float)
    x=a-a.mean();n=len(a)
    long_var=float(x@x/n)
    for k in range(1,min(lags,n-1)+1):
        long_var+=2*(1-k/(lags+1))*float(x[k:]@x[:-k])/n
    return math.sqrt(max(long_var,0.0)/n)


def daily_bootstrap(diffs,dates,B=500,seed=20261027):
    if B==0:return (float('nan'),float('nan'))
    dayid=pd.factorize(pd.Series(dates).dt.normalize(),sort=False)[0]
    ids=np.unique(dayid)
    # Resample whole trading days, preserving all within-day serial dependencies.
    sums=np.bincount(dayid,weights=diffs)
    counts=np.bincount(dayid)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(ids),size=(B,len(ids)))
    means=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return tuple(float(z) for z in np.quantile(means,[.025,.975]))


def evaluate(ret,mu,models,split,outdir,args):
    x=ret.return_pct_log.to_numpy(dtype=np.float64)-mu
    st=ret.segment_start.to_numpy(dtype=bool)
    times=ret.timestamp
    f=ret.iloc[split:][['timestamp','return_pct_log','segment_start']].copy().reset_index(drop=True)
    rows=[];sco={};qs={}
    for name,p in models.items():
        vols,weights,lp=forecast_params(x,st,p)
        f[f'{name}_predictive_variance']=vols[split:]
        f[f'{name}_pred_p_heavy']=weights[split:] if p['p00'] is not None else np.nan
        f[f'{name}_logdensity']=lp[split:]
        sco[name]=lp[split:]
        qq=quantiles(vols[split:],weights[split:],p)
        qs[name]=qq
        rows.append({'model':name,'confidence_pct':np.nan,'n':len(f),
                     'mean_log_score':float(lp[split:].mean())})
        for a in LEVELS:
            tag=str(100*(1-a)).replace('.','p')
            q=mu+qq[a]
            actual=f.return_pct_log.to_numpy()
            loss=(a-(actual<q).astype(float))*(actual-q)
            f[f'{name}_VaR_{tag}']=-q
            f[f'{name}_loss_{tag}']=loss
            f[f'{name}_breach_{tag}']=(actual<q)
            rows.append({'model':name,'confidence_pct':100*(1-a),'n':len(f),
                         'mean_log_score':float(lp[split:].mean()),
                         'mean_pinball_loss':float(loss.mean()),
                         'exceedance_rate':float((actual<q).mean()),
                         'breach_count':int((actual<q).sum()),
                         'target_exceedance_rate':float(a),
                         'mean_VaR_pct_log':float((-q).mean())})
    pd.DataFrame(rows).to_csv(outdir/'model_summary.csv',index=False)
    comparison=[]
    for baseline,candidate in [('garch_t','garch_mix_t'),('garch_t','garch_hmm'),('garch_mix_t','garch_hmm')]:
        d=sco[candidate]-sco[baseline]
        se=hac_se(d,args.hac_lags)
        comparison.append({'comparison':f'{candidate} minus {baseline} log score',
                           'mean_difference':float(d.mean()),'hac_se':float(se),
                           'z':float(d.mean()/se) if se>0 else None,
                           'daily_bootstrap_ci95':daily_bootstrap(d,times.iloc[split:],args.bootstrap,args.seed)})
        for a in LEVELS:
            tag=str(100*(1-a)).replace('.','p')
            diff=f[f'{baseline}_loss_{tag}'].to_numpy()-f[f'{candidate}_loss_{tag}'].to_numpy()
            se=hac_se(diff,args.hac_lags)
            comparison.append({'comparison':f'{baseline} minus {candidate} pinball loss ({100*(1-a):g}% VaR)',
                               'mean_difference':float(diff.mean()),
                               'relative_gain_pct':float(100*diff.mean()/f[f'{baseline}_loss_{tag}'].mean()),
                               'hac_se':float(se),'z':float(diff.mean()/se) if se>0 else None,
                               'daily_bootstrap_ci95':daily_bootstrap(diff,times.iloc[split:],args.bootstrap,args.seed)})
    pd.DataFrame(comparison).to_csv(outdir/'paired_forecast_tests.csv',index=False)
    f.to_csv(outdir/'holdout_forecasts.csv',index=False)
    print('\nHOLDOUT MODEL SUMMARY')
    print(pd.DataFrame(rows).dropna(subset=['confidence_pct'])
          [['model','confidence_pct','mean_pinball_loss','exceedance_rate','mean_log_score']]
          .to_string(index=False))
    print('\nPAIRED FORECAST TESTS (pointwise, no multiplicity adjustment):')
    print(pd.DataFrame(comparison)[['comparison','mean_difference','z','daily_bootstrap_ci95']].to_string(index=False))
    return comparison


def simulate_example(n=1800,seed=42):
    rng=np.random.default_rng(seed)
    nu0,nu1=3.6,14.
    p00,p11=.97,.94
    varbar=.014
    alpha,beta=.065,.91
    x=np.empty(n,dtype=float)
    st=np.zeros(n,dtype=bool)
    state=0
    hprev=varbar
    times=[]
    day=pd.Timestamp('2026-02-02')
    for i in range(n):
        if i%100==0:
            st[i]=True
            h=varbar
            state=int(rng.uniform()>.5)
            day=pd.Timestamp('2026-02-02')+pd.Timedelta(days=i//100)
        else:
            h=varbar*(1-alpha-beta)+alpha*x[i-1]**2+beta*hprev
            if state==0 and rng.uniform()>p00:state=1
            elif state==1 and rng.uniform()>p11:state=0
        nu=nu0 if state==0 else nu1
        x[i]=math.sqrt(h*(nu-2)/nu)*rng.standard_t(df=nu)
        hprev=h
        times.append(day+pd.Timedelta(hours=9,minutes=i%100))
    out=pd.DataFrame({'timestamp':times,'return_pct_log':x,'segment_start':st})
    return out,{'synthetic':True,'true_params':{'nu0':nu0,'nu1':nu1,'p00':p00,
                                                'p11':p11,'varbar':varbar,'alpha':alpha,'beta':beta}}


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv',help='WIN or US price CSV')
    ap.add_argument('--format',choices=['win','us'],default='win')
    ap.add_argument('--market',default='WIN')
    ap.add_argument('--timeframe',type=int,default=15,choices=[1,5,10,15])
    ap.add_argument('--rth-start',help='HH:MM, default WIN 09:00; US 09:30')
    ap.add_argument('--rth-end',help='HH:MM, default WIN 17:00; US 16:00')
    ap.add_argument('--train-frac',type=float,default=.80)
    ap.add_argument('--max-observations',type=int,default=0,
                    help='For QUICK DIAGNOSTICS only: retain last N intraday returns; not publishable')
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--maxiter',type=int,default=220)
    ap.add_argument('--hac-lags',type=int,default=20)
    ap.add_argument('--bootstrap',type=int,default=400)
    ap.add_argument('--seed',type=int,default=20261027)
    ap.add_argument('--out',default='GT_GARCH_HMM_RESULTS')
    ap.add_argument('--self-test',action='store_true',help='End-to-end synthetic smoke test')
    args=ap.parse_args()
    if not (0.5<=args.train_frac<=.95):ap.error('--train-frac must be between .5 and .95')
    if args.starts<1 or args.maxiter<1:ap.error('--starts and --maxiter must be positive')
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        ret,audit=simulate_example(n=1600,seed=args.seed)
    else:
        if not args.csv:ap.error('--csv is required outside --self-test')
        start=args.rth_start or ('09:00' if args.format=='win' else '09:30')
        end=args.rth_end or ('17:00' if args.format=='win' else '16:00')
        ret,audit=load_returns(args.csv,args.format,args.timeframe,start,end)
    if args.max_observations:
        ret=ret.iloc[-args.max_observations:].copy().reset_index(drop=True)
        ret.loc[0,'segment_start']=True
    # Split by complete trading day, preventing within-day train/test overlap.
    days=ret.timestamp.dt.normalize()
    uniq=np.sort(days.unique())
    cutoff=max(1,min(len(uniq)-1,int(np.floor(args.train_frac*len(uniq)))))
    ix=np.flatnonzero(days>=uniq[cutoff])
    if len(ix)==0:raise ValueError('No holdout observations')
    split=int(ix[0])
    if split<100 or len(ret)-split<60:
        raise ValueError(f'Insufficient train/test returns ({split}/{len(ret)-split}); provide a larger sample')
    train=ret.return_pct_log.to_numpy(dtype=np.float64)[:split]
    mu=float(train.mean()) # same TRAIN-ONLY location, fixed for both models
    x=np.ascontiguousarray(train-mu,dtype=np.float64)
    starts=np.ascontiguousarray(ret.segment_start.iloc[:split].to_numpy(dtype=np.bool_))
    print(f'Market={args.market} | timeframe=M{args.timeframe} | N={len(ret)} | '
          f'train={split} | test={len(ret)-split} | segments={int(ret.segment_start.sum())}',flush=True)
    print('Fitting train-only joint likelihoods; no test returns enter estimation.',flush=True)
    garch_t=fit_model(x,starts,hmm=False,nstarts=args.starts,maxiter=args.maxiter,seed=args.seed)
    garch_mix_t=fit_iid_mixture(x,starts,nstarts=args.starts,maxiter=args.maxiter,seed=args.seed)
    garch_hmm=fit_model(x,starts,hmm=True,nstarts=args.starts,maxiter=args.maxiter,seed=args.seed)
    models={'garch_t':garch_t,'garch_mix_t':garch_mix_t,'garch_hmm':garch_hmm}
    compare=evaluate(ret,mu,models,split,dest,args)
    metadata={'model':'common GARCH(1,1) + 2-state variance-standardized Student-t tail HMM',
              'market_label_user_supplied':args.market,'source_csv':args.csv,
              'timeframe_minutes':args.timeframe,'data_audit':audit,
              'date_train_end':str(ret.timestamp.iloc[split-1]),
              'date_test_start':str(ret.timestamp.iloc[split]),
              'training_return_mean':mu,'train_n':split,'test_n':len(ret)-split,
              'models':models,'notes':[
                  'HMM does NOT switch GARCH volatility: only standardized Student-t nu differs by state.',
                  'GARCH-iid-mixture benchmark has no Markov persistence: distinguishes mixture-fit from regime effects.',
                  'Each valid intraday segment resets GARCH recursion and HMM stationary prior.',
                  'Overnight and irregular intervals excluded and audited; not directly comparable to gap-aware historical HMM-6 test.',
                  'AIC/BIC and naive likelihood-ratio chi-squared tests are NOT proof of tail-regime identification.',
                  'If nu0 equals nu1, HMM transition parameters are unidentified; need nonstandard bootstrap for null.',
                  'HAC and day-block intervals are pointwise exploratory, and day bootstrap may understate across-day dependence.',
                  'No evidence for Gamma jumps or HMM-9 is inferred from this study.',
                  'Chronological holdout was never used for fitting; only past returns update prediction.'
              ]}
    (dest/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    print(f'\nSaved {dest}/model_summary.csv, holdout_forecasts.csv, '
          f'paired_forecast_tests.csv, run_metadata.json',flush=True)

if __name__=='__main__':
    main()
