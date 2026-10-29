#!/usr/bin/env python3
"""Giorno-Tsallis Proposition A: observed-HMM population sensitivity Monte Carlo.

Simulation uses independent baseline sample paths, exact HMM forward likelihood,
stationary chain initialization, deterministic generalized Gauss-Laguerre Gamma
convolution, and importance reweighting of the SAME Monte Carlo paths to estimate
pseudo-true HMM-6 parameters under contamination. No HMM-9 estimator is used.

Theoretical derivative uses D=I^{-1} E_0[S W] where W is obtained from
exact smoothed probabilities under the uncontaminated observed HMM.
The pseudo-true finite-epsilon derivative uses direct expectation under the
contaminated HMM (via likelihood-ratio importance weights). The two approaches
are logically distinct. This is a simulation study, not an analytical proof.

Requires numpy scipy. Example:
  python GT_Proposition_A_Final_MC.py --paths 5000 --length 30 --seed 20261026 --out gt_mc
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.special import roots_genlaguerre, gamma

PAR_NAMES = ('p00','p11','nu0','nu1','s0','s1')
BOUNDS = [(0.70,0.995),(0.70,0.995),(2.3,25.0),(2.3,25.0),(0.12,1.1),(0.7,2.7)]
DEFAULT_BETA = np.array([0.92,0.90,4.5,8.0,0.45,1.30])


def transition(b):
    return np.array([[b[0],1-b[0]],[1-b[1],b[1]]])


def stationary(b):
    denom=2-b[0]-b[1]
    return np.array([(1-b[1])/denom,(1-b[0])/denom])


def emissions(y,b):
    return np.stack([student_t.pdf(y,df=b[2+j],scale=b[4+j]) for j in range(2)],axis=-1)


def convolution(y,b,shape,scale,nodes):
    x,w=roots_genlaguerre(nodes,shape-1)
    w=w/gamma(shape)
    # Avoid giant arrays by iterating over quadrature nodes
    h=np.zeros(y.shape+(2,))
    for xx,ww in zip(x,w):
        h += ww*emissions(y+scale*xx,b)
    return h


def forward(y,b,E=None, smoothing=False):
    """Full observed likelihood, vectorized over Monte Carlo paths."""
    P=transition(b)
    if E is None: E=emissions(y,b)
    m,T=y.shape
    a=np.empty((T,m,2)) if smoothing else None
    scales=np.empty((T,m)) if smoothing else None
    alpha=stationary(b)[None,:]*E[:,0,:]
    c=alpha.sum(axis=1)
    if np.any(c<=0): raise FloatingPointError('Zero forward scaling; rescale dataset')
    ll=np.log(c)
    alpha=alpha/c[:,None]
    if smoothing: a[0],scales[0]=alpha,c
    for u in range(1,T):
        alpha=(alpha@P)*E[:,u,:]
        c=alpha.sum(axis=1)
        if np.any(c<=0): raise FloatingPointError('Zero forward scaling')
        ll+=np.log(c)
        alpha/=c[:,None]
        if smoothing: a[u],scales[u]=alpha,c
    if not smoothing: return ll
    back=np.ones((m,2))
    smooth=np.empty_like(a)
    smooth[-1]=a[-1]
    for u in range(T-2,-1,-1):
        back=(back*E[:,u+1,:])@P.T / scales[u+1,:,None]
        z=a[u]*back
        smooth[u]=z/z.sum(axis=1)[:,None]
    return ll,smooth.transpose(1,0,2)


def simulate_base(n,T,b,rng):
    state=np.empty((n,T),dtype=np.int8)
    state[:,0]=(rng.random(n)>stationary(b)[0]).astype(np.int8)
    for u in range(1,T):
        u0=rng.random(n)
        state[:,u]=np.where(state[:,u-1]==0,(u0> b[0]),(u0<= b[1])).astype(np.int8)
    yy=np.empty((n,T))
    for j in (0,1):
        idx=state==j
        yy[idx]=student_t.rvs(df=b[2+j],size=int(idx.sum()),random_state=rng)*b[4+j]
    return yy


def numeric_score(y,b,steps=1e-4):
    S=np.empty((len(y),6))
    for i in range(6):
        h=steps*max(abs(b[i]),0.2)
        bp=b.copy(); bm=b.copy();bp[i]+=h;bm[i]-=h
        S[:,i]=(forward(y,bp)-forward(y,bm))/(2*h)
    return S


def weighted_fit(y,w,guess,maxiter, gradient_step=2e-5):
    """Score-gradient L-BFGS-B in scaled coordinates; no silent near-flat stop."""
    w=np.asarray(w,float); w=w/np.mean(w)
    scale=np.array([0.05,0.05,2.0,3.0,0.3,0.7])
    bounds_z=[(lo/s,hi/s) for (lo,hi),s in zip(BOUNDS,scale)]
    def fg(z):
        x=z*scale
        ll=forward(y,x)
        sc=numeric_score(y,x,steps=gradient_step)
        return -np.dot(w,ll)/len(y), -(w[:,None]*sc).mean(axis=0)*scale
    res=minimize(fg,np.asarray(guess)/scale,method='L-BFGS-B',jac=True,bounds=bounds_z,
                 options={'maxiter':maxiter,'ftol':1e-14,'gtol':2e-9,'maxls':50,'maxcor':15,'maxfun':maxiter*40})
    res.x=res.x*scale
    res['gradient_physical']=np.asarray(res.jac)/scale
    res['gradient_inf']=float(np.max(np.abs(res['gradient_physical'])))
    return res


def numerical_hessian(fn, b, steps=1e-3):
    h=steps*np.maximum(np.abs(b),0.2)
    H=np.empty((6,6)); f0=fn(b)
    for i in range(6):
        xp=b.copy(); xm=b.copy(); xp[i]+=h[i]; xm[i]-=h[i]
        H[i,i]=(fn(xp)-2*f0+fn(xm))/h[i]**2
        for j in range(i):
            xpp=b.copy();xpm=b.copy();xmp=b.copy();xmm=b.copy()
            xpp[i]+=h[i];xpp[j]+=h[j]
            xpm[i]+=h[i];xpm[j]-=h[j]
            xmp[i]-=h[i];xmp[j]+=h[j]
            xmm[i]-=h[i];xmm[j]-=h[j]
            H[i,j]=H[j,i]=(fn(xpp)-fn(xpm)-fn(xmp)+fn(xmm))/(4*h[i]*h[j])
    return H

def as_float_dict(arr):
    return {k:float(v) for k,v in zip(PAR_NAMES,arr)}


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    b=DEFAULT_BETA.copy()
    y=simulate_base(args.paths,args.length,b,rng)
    base_ll=forward(y,b)
    print(f'Baseline: N={args.paths} T={args.length}, full observed-HMM forward likelihood',flush=True)
    fit0=weighted_fit(y,np.ones(len(y)),b,args.maxiter)
    print(f'Baseline fit: success={fit0.success}, \u0394 from truth={np.max(np.abs(fit0.x-b)):.3g}',flush=True)
    S=numeric_score(y,b)
    S_atfit=numeric_score(y,fit0.x)
    Hess=numerical_hessian(lambda bb:-forward(y,bb).mean(),fit0.x)
    print(f"  empirical Hessian cond={np.linalg.cond(Hess):.3g}, min eig={np.linalg.eigvalsh(Hess)[0]:.3g}",flush=True)
    # Use baseline covariance of score; finite Monte Carlo bias from nonzero
    # sample-mean score is retained as diagnostic.
    I=np.cov(S,rowvar=False)
    if np.linalg.eigvalsh(I)[0]<=0: raise RuntimeError('Information estimate not positive definite')
    rows=[]; summaries=[]
    for shape,theta in [(args.shape,x) for x in args.thetas]:
        print(f'Gamma shape={shape}, scale={theta}',flush=True)
        f=emissions(y,b)
        h=convolution(y,b,shape,theta,args.nodes)
        _,sm=forward(y,b,E=f,smoothing=True)
        W=np.sum(sm*(h/f-1),axis=(1,2))
        B=np.mean((S-S.mean(axis=0))*W[:,None],axis=0)
        d=np.linalg.solve(I,B)
        # Finite-sample exact implicit derivative for the weighted empirical
        # objective; compare to fitted epsilon perturbations separately from
        # asymptotic population covariance-information calculation.
        B_emp=np.mean((S_atfit-S_atfit.mean(axis=0))*W[:,None],axis=0)
        d_emp=np.linalg.solve(Hess,B_emp)
        # LR identity: W = derivative of full contaminated observed likelihood
        ep_probe=1e-6
        lrp=forward(y,b,E=(1-ep_probe)*f+ep_probe*h)
        err=np.max(np.abs((lrp-base_ll)/ep_probe-W))
        print(f'  mean W={W.mean():.5g}, W-score max FD error={err:.4g}, cond(I)={np.linalg.cond(I):.4g}',flush=True)
        # Weight each baseline path by the exact contaminated / baseline
        # observed sequence likelihood; no jump draws inside optimization.
        record={'theta':theta,'shape':shape,'theory_d':as_float_dict(d),'empirical_implicit_d':as_float_dict(d_emp),
                'theory_dq': {'q0':float(-2*d[2]/(b[2]+1)**2),'q1':float(-2*d[3]/(b[3]+1)**2)},
                'mean_W':float(W.mean()),'max_W_derivative_error':float(err),
                'information_condition':float(np.linalg.cond(I)),'results':[]}
        for eps in args.epsilons:
            l1=forward(y,b,E=(1-eps)*f+eps*h)
            lr=l1-base_ll
            w=np.exp(lr-np.max(lr))
            ess=float(w.sum()**2/np.dot(w,w))
            # A first-order warm start resolves extremely small parameter movements.
            warm=np.clip(fit0.x+eps*d_emp,
                         [lo+1e-7 for lo,hi in BOUNDS],
                         [hi-1e-7 for lo,hi in BOUNDS])
            fit=weighted_fit(y,w,warm,args.maxiter)
            # Compare to baseline start: keep higher likelihood to detect local minima.
            if args.compare_starts:
                fit_alt=weighted_fit(y,w,fit0.x,args.maxiter)
                if fit_alt.fun<fit.fun: fit=fit_alt
            obs=(fit.x-fit0.x)/eps
            q0=(fit0.x[[2,3]]+3)/(fit0.x[[2,3]]+1)
            qe=(fit.x[[2,3]]+3)/(fit.x[[2,3]]+1)
            # Exact weighted empirical score transport at baseline, with no re-optimization.
            b_transport=np.average(S_atfit,axis=0,weights=w)
            score_transport_error=float(np.linalg.norm(b_transport/eps-B_emp))
            rec={'epsilon':float(eps),'ESS':ess,'optimizer_success':bool(fit.success),
                 'score_transport_error':score_transport_error,
                 'optimizer_message':str(fit.message),'optimizer_gradient_inf':fit.gradient_inf, 'weighted_nll':float(fit.fun),
                 'finite_difference':as_float_dict(obs),
                 'finite_difference_q':{'q0':float((qe[0]-q0[0])/eps), 'q1':float((qe[1]-q0[1])/eps)},
                 'theory_vs_fd_l2':float(np.linalg.norm(d-obs)),
                 'empirical_implicit_vs_fd_l2':float(np.linalg.norm(d_emp-obs))}
            record['results'].append(rec)
            rows.append({'theta':theta,'epsilon':eps,'ESS':ess,'success':fit.success,
                         **{f'theory_{name}':d[ii] for ii,name in enumerate(PAR_NAMES)},
                         **{f'finite_{name}':obs[ii] for ii,name in enumerate(PAR_NAMES)},
                         'theory_vs_fd_l2':rec['theory_vs_fd_l2'],
                         'empirical_implicit_vs_fd_l2':rec['empirical_implicit_vs_fd_l2'],
                         'score_transport_error':score_transport_error})
            print(f'  eps={eps:g} ESS={ess:.0f}/{len(y)}, fit={fit.success}, ||d_pop-d_fd||={rec["theory_vs_fd_l2"]:.4g}, ||d_emp-d_fd||={rec["empirical_implicit_vs_fd_l2"]:.4g}',flush=True)
        summaries.append(record)
    # Targeted zero-contamination extrapolation: d_FD(eps) = d_emp + c*eps + O(eps^2).
    # These are sample-path diagnostics, not confidence intervals or proof.
    for rr in summaries:
        ee=np.array([v['epsilon'] for v in rr['results']],dtype=float)
        dd=np.array([[v['finite_difference'][name] for name in PAR_NAMES] for v in rr['results']])
        lim=np.array([np.polyfit(ee,dd[:,j],1)[1] for j in range(6)])
        de=np.array([rr['empirical_implicit_d'][name] for name in PAR_NAMES])
        rr['zero_epsilon_linear_extrapolation']=as_float_dict(lim)
        rr['zero_epsilon_extrapolation_vs_empirical_implicit_l2']=float(np.linalg.norm(lim-de))
        rr['zero_epsilon_extrapolation_vs_empirical_implicit_by_parameter']=as_float_dict(lim-de)
        print(f"  LINEAR eps->0 extrapolation: ||d_extrap-d_emp||={np.linalg.norm(lim-de):.5g}",flush=True)
        print(f"    nu0 extrap={lim[2]:.6g} implicit={de[2]:.6g}; nu1 extrap={lim[3]:.6g} implicit={de[3]:.6g}",flush=True)
    result={'design':{'seed':args.seed,'paths':args.paths,'length':args.length,'Gamma_shape':args.shape,
                      'theta_scales':args.thetas,'epsilon_values':args.epsilons,'quadrature_nodes':args.nodes,
                      'baseline_parameters':as_float_dict(b),'model':'centered 2-state Student-t HMM6, stationary initialization'},
            'baseline_fit':{'parameters':as_float_dict(fit0.x),'success':bool(fit0.success),'message':str(fit0.message), 'gradient_inf':fit0.gradient_inf,
                            'sample_mean_score':as_float_dict(S.mean(axis=0)),
                            'empirical_hessian_condition':float(np.linalg.cond(Hess)),
                            'empirical_hessian_min_eig':float(np.linalg.eigvalsh(Hess)[0])},
            'runs':summaries,'notes':[ 'Finite MC precision is limited; assess convergence in paths and epsilon.',
              'Population derivative uses score-covariance information at the true baseline; the empirical-implicit derivative uses the sample Hessian at the sample baseline optimum.',
              'Weighted contaminated pseudo-true fit uses importance sampling from the baseline HMM.',
              'Fixed-T, no real-data or full HMM9 identification claim.',
              'Quadrature approximates Gamma convolution deterministically.',
              'Small-sample weak identification or optimizer convergence can obscure parameter derivative convergence; check Hessian condition and refit with larger paths.',
              'No universal sign claim for tail-shape bias.',
              'Targeted V3: smaller epsilon, selected theta values, and linear zero-epsilon extrapolation. Retains all HMM6 likelihood, score, and optimizer calculations of V2; requires numerical audit.' ]}
    (out/'proposition_A_mc_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with open(out/'proposition_A_mc_table.csv','w',newline='',encoding='utf-8') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('SAVED',out/'proposition_A_mc_results.json',flush=True)
    print('SAVED',out/'proposition_A_mc_table.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=int,default=15000)
    p.add_argument('--length',type=int,default=40)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--out',default='GT_PropA_Targeted_v3')
    p.add_argument('--shape',type=float,default=2.0)
    p.add_argument('--thetas',nargs='+',type=float,default=[0.20,0.90])
    p.add_argument('--epsilons',nargs='+',type=float,default=[0.001,0.0025,0.005])
    p.add_argument('--nodes',type=int,default=48)
    p.add_argument('--maxiter',type=int,default=500)
    p.add_argument('--compare-starts',action='store_true',help='Repeat weighted fit from baseline as an optimizer audit')
    args=p.parse_args()
    if args.paths<100 or args.length<3 or args.shape<=0 or any(x<=0 for x in args.thetas) or any(not 0<x<1 for x in args.epsilons):
        p.error('invalid simulation design')
    run(args)
if __name__=='__main__': main()
