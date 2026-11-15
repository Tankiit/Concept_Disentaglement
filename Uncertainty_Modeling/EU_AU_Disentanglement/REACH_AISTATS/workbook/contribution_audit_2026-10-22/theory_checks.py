"""Numerical checks of explicit algebraic examples, not formal proofs."""
from pathlib import Path
import json
import numpy as np

actual=np.array([[1.,0.],[0.,1.]])
declared=np.array([[1.,0.],[.5,.5]])
w=np.array([.8,.2]);p=w@actual
fitted=np.linalg.solve(declared.T,p)
np.testing.assert_allclose(fitted,[.6,.4])
np.testing.assert_allclose(fitted@declared,p)
true_target=w[-1]*actual[-1,0]/p[0]
wrong_target=fitted[-1]*declared[-1,0]/p[0]
np.testing.assert_allclose([true_target,wrong_target],[0,.25])
for t in [0,.1,.5,1]:
    a=actual.copy();a[-1]=(1-t)*actual[-1]+t*declared[-1]
    np.testing.assert_allclose(np.abs(w@a-p).sum()/2,.2*t*np.abs(declared[-1]-actual[-1]).sum()/2,atol=1e-12)
out=dict(exact_mismatch_compatible=True,actual_target=true_target,wrong_point_target=wrong_target,
         fitted_weights=fitted.tolist(),identity_residual_formula_verified=True)
(Path(__file__).resolve().parent/'theory_checks.json').write_text(json.dumps(out,indent=2))
print(out)
