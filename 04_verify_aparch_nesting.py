#!/usr/bin/env python3
"""Check algebraic nesting numerically without fitting or observing DGP truth."""
import numpy as np
from importlib import import_module
m=import_module('02_blind_econometric_competitors')
rng=np.random.default_rng(188)
y=rng.standard_t(8,200)
v=float(np.var(y))
# Same intercept, alpha, beta, Student-t shape and conditional variance initialization.
sym=[0.03,0.07,0.09,0.84,9.0]
garch=m.eval_model('GARCH11-t',y,sym,v)
aparch=m.eval_model('APARCH11-delta2-symmetric-t',y,sym,v)
for k,(a,b) in enumerate(zip(garch,aparch)):
    if not np.allclose(a,b,rtol=0,atol=1e-10):
        raise AssertionError(f'APARCH(2,0) != GARCH in object {k}')
# Restricted parameterization fixes coefficients, not just labels.
for key in m.APARCH_ALL_KEYS:
    names,values,bounds=m.spec_bounds(key,y)
    assert len(names)==len(values)==len(bounds)
    full=m.expand_aparch(key,values)
    assert np.isfinite(full).all() and len(full)==7
    assert all(b[0]<=x<=b[1] for x,b in zip(values,bounds))
    _=m.eval_model(key,y,values,v)
print('PASS: APARCH(delta=2,gamma=0) nests GARCH exactly; all APARCH variants evaluated.')
