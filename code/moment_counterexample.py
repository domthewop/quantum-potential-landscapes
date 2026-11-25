"""Matching mass/mean/variance but opposite QC curvature at x=0."""
import json, numpy as np
from pathlib import Path

pA=np.array([-.8,.8]);wA=np.array([.5,.5]);
pB=np.array([-1.6,0,1.6]);wB=np.array([.125,.75,.125]);

def curvature(points,weights,sigma=1.):
    k=weights*np.exp(-.5*(points/sigma)**2)
    k/=k.sum()
    m2=np.sum(k*points**2);m4=np.sum(k*points**4)
    return 1/sigma**2 - 2*m2/sigma**4 + (m4-m2**2)/(2*sigma**6)

def descr(p,w):
    return {'support':p.tolist(),'weights':w.tolist(),'mass':float(sum(w)),
           'mean':float(sum(w*p)),'raw_second_moment':float(sum(w*p*p)),
           'potential_curvature_at_origin':float(curvature(p,w))}
report={'dataset_A':descr(pA,wA),'dataset_B':descr(pB,wB)}
Path(__file__).with_name('moment_counterexample.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
