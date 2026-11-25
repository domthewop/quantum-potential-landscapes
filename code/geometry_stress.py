#!/usr/bin/env python3
"""Falsification study for quantum-potential compression.

2-D roots are NUMERICAL candidates, not interval certificates.
1-D mode detection is grid sign-change detection (not formally certified).
No results should be described as proof of topology preservation.
"""
from __future__ import annotations
import argparse,csv,math,time,sys,platform
from pathlib import Path
import numpy as np
from scipy.optimize import root
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
from interval_verifier import Potential

SIGMA={'compact':.65,'overlap':.70,'anisotropic':.75,'three':.68,'moons':.26,'near_bifurcation':1.0,'null':.85}

def sample_case(name,n,seed):
 rng=np.random.default_rng(seed)
 if name=='compact':
  y=rng.choice(2,n); c=np.array([[-1.13,.02],[1.13,-.02]])
  X=c[y]+rng.normal(0,.035,(n,2))
  domain=(-1.75,1.75,-.9,.9)
 elif name=='overlap':
  y=rng.choice(2,n); c=np.array([[-.79,0],[.79,0]])
  X=c[y]+rng.normal(0,.30,(n,2));domain=(-2.3,2.3,-1.4,1.4)
 elif name=='three':
  y=rng.choice(3,n); c=np.array([[-1.3,-.7],[1.3,-.7],[.02,1.25]])
  X=c[y]+rng.normal(0,.10,(n,2));domain=(-1.95,1.95,-1.3,1.85)
 elif name=='anisotropic':
  y=rng.choice(3,n); c=np.array([[-1.35,-.65],[1.2,-.55],[.15,1.15]])
  trans=[np.array([[.6,.34],[.0,.14]]),np.array([[.23,-.21],[.05,.58]]),np.array([[.58,-.23],[.0,.13]])]
  X=np.array([c[z]+trans[z]@rng.normal(size=2) for z in y]);domain=(-2.6,2.6,-1.9,2.15)
 elif name=='moons':
  from sklearn.datasets import make_moons
  X,y=make_moons(n_samples=n,noise=.055,random_state=seed);domain=(-1.65,2.65,-.9,1.5)
 elif name=='near_bifurcation':
  y=rng.choice(2,n);c=np.array([[-.715,0],[.715,0]])
  X=c[y]+rng.normal(0,.135,(n,2));domain=(-1.65,1.65,-.95,.95)
 elif name=='null':
  y=np.zeros(n,dtype=int);X=rng.normal(0,.55,(n,2));domain=(-2.0,2.0,-2.0,2.0)
 else:raise ValueError(name)
 return X,y,domain

def rounded_potential(X,sigma,weights=None):
 if weights is None:weights=np.ones(len(X))
 return Potential([(repr(float(x)),repr(float(y)),repr(float(w))) for (x,y),w in zip(X,weights)],str(sigma))

def coreset(X,m,method,seed):
 if method=='centroids':
  km=KMeans(n_clusters=m,random_state=seed,n_init=3,max_iter=60).fit(X)
  counts=np.bincount(km.labels_,minlength=m).astype(float)
  return km.cluster_centers_,counts
 if method=='uniform':
  rg=np.random.default_rng(seed+10001);p=rg.choice(len(X),m,replace=False)
  return X[p],np.ones(m)*len(X)/m
 if method=='medoids':
  km=KMeans(n_clusters=m,random_state=seed,n_init=2,max_iter=40).fit(X)
  from scipy.spatial.distance import cdist
  dist=cdist(km.cluster_centers_,X)
  indices=np.argmin(dist,axis=1)
  return X[indices],np.bincount(km.labels_,minlength=m).astype(float)
 raise ValueError(method)

def numeric_roots(pot,domain,grid=7):
 # Numerical root discovery, without claims of completeness.
 x0,x1,y0,y1=domain;roots=[];kinds=[]
 for x in np.linspace(x0+.07*(x1-x0),x1-.07*(x1-x0),grid):
  for y in np.linspace(y0+.07*(y1-y0),y1-.07*(y1-y0),grid):
   try:
    so=root(lambda z:pot.numeric(z)[1],[x,y],jac=lambda z:pot.numeric(z)[2],method='hybr',tol=1e-9)
    loc=so.x
    if not np.isfinite(loc).all() or not (x0<loc[0]<x1 and y0<loc[1]<y1):continue
    _,g,H=pot.numeric(loc)
    if not np.isfinite(g).all() or np.linalg.norm(g)>3e-6:continue
    if min(abs(np.linalg.eigvalsh(H)))<1e-4:kind='degenerate'
    elif np.linalg.det(H)<0:kind='saddle'
    elif H[0,0]>0:kind='minimum'
    else:kind='maximum'
    if not any(np.linalg.norm(loc-v)<3e-3 for v in roots):roots.append(loc);kinds.append(kind)
   except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):pass
 return np.array(roots).reshape(-1,2),kinds

def compare_roots(r0,t0,r1,t1):
 if len(r0)!=len(r1):return (False,float('nan'),0)
 if len(r0)==0:return (True,0.,0)
 from scipy.spatial.distance import cdist
 C=cdist(r0,r1)
 C+=np.array([[10000 if a!=b else 0 for b in t1] for a in t0])
 ii,jj=linear_sum_assignment(C)
 same=all(t0[i]==t1[j] for i,j in zip(ii,jj))
 return (same,float(max(np.linalg.norm(r0[i]-r1[j]) for i,j in zip(ii,jj))) if same else float('nan'),sum(x=='minimum' for x in t1))

def roots_study(n=256,seeds=(0,1,2),ms=(8,16,32),cases=None):
 if cases is None:cases=['compact','overlap','three','anisotropic','moons','near_bifurcation','null']
 rows=[]
 for case in cases:
  for seed in seeds:
   X,y,domain=sample_case(case,n,seed)
   sig=SIGMA[case];t=time.perf_counter();ref=rounded_potential(X,sig)
   r0,k0=numeric_roots(ref,domain);buildref=time.perf_counter()-t
   # Cross-check grid stability of reference numerical search
   rcheck,tcheck=numeric_roots(ref,domain,grid=10)
   root_ref_stable,_,_=compare_roots(r0,k0,rcheck,tcheck)
   print('roots',case,seed,'N',n,'ref',len(r0),'stable',root_ref_stable,flush=True)
   for m in ms:
    for method in ('centroids','uniform','medoids'):
     t=time.perf_counter();Z,w=coreset(X,m,method,seed);build=time.perf_counter()-t
     p=rounded_potential(Z,sig,w)
     t=time.perf_counter();r,k=numeric_roots(p,domain);rtime=time.perf_counter()-t
     match,maxloc,_=compare_roots(r0,k0,r,k)
     # Field and gradient errors on a held-out grid; no shared evaluation points for landmark construction
     rng=np.random.default_rng(10000+seed)
     Q=np.column_stack([rng.uniform(domain[0],domain[1],50),rng.uniform(domain[2],domain[3],50)])
     vs=[];gs=[]
     for q in Q:
      v,g,_=ref.numeric(q);u,h,_=p.numeric(q)
      vs.append(abs(u-v));gs.append(np.linalg.norm(g-h))
     rows.append({'case':case,'seed':seed,'N':n,'bandwidth':sig,'m':m,'method':method,
                  'reference_count':len(r0),'reference_types':str({t:k0.count(t) for t in set(k0)}),
                  'reference_stable_to_grid':root_ref_stable,'approx_count':len(r),
                  'types_match':match,'max_location_error':maxloc,
                  'mean_V_error':float(np.mean(vs)),'max_V_error':float(np.max(vs)),
                  'mean_grad_error':float(np.mean(gs)),'build_s':build,'root_s':rtime,'reference_root_s':buildref})
 return rows

def count_modes_1d(X,sigma,grid):
 # use positive KDE gradient / negative QC gradient roots on a uniform grid
 diff=grid[:,None]-X[None,:]
 w=np.exp(-.5*(diff/sigma)**2)
 D=np.sum(w,axis=1);A=np.sum(w*diff*diff,axis=1)
 Dp=np.sum(-w*diff/(sigma*sigma),axis=1)
 Ap=np.sum(w*diff*(2-diff*diff/(sigma*sigma)),axis=1)
 qcgrad=(Ap/D-A*Dp/D**2)/(2*sigma*sigma)
 # maxima of KDE: derivative from + to -, minima of QC: derivative from - to +
 # filter endpoints, avoid arbitrary far-tail noise
 qx=np.where((qcgrad[:-1]<0)&(qcgrad[1:]>=0))[0]
 dx=np.where((Dp[:-1]>0)&(Dp[1:]<=0))[0]
 q=(grid[qx]+grid[qx+1])/2;d=(grid[dx]+grid[dx+1])/2
 def well_sep(loc):
  if len(loc)<2:return False
  return bool(max(loc)-min(loc)>=.35)
 return well_sep(q),well_sep(d),len(q),len(d)

def detection_study(n=200,seeds=range(60)):
 sigma=1.0;grid=np.linspace(-2.4,2.4,1001)
 rows=[]
 for kind in ['single_gaussian','two_a055','two_a070','two_a080','two_a095','two_a120']:
  for seed in seeds:
   rng=np.random.default_rng(99000+seed)
   if kind=='single_gaussian':X=rng.normal(0,.55,n)
   else:
    a=float(kind[5:])/100 if False else float('0.'+kind.split('a')[1][1:]) if False else None
    vals={'two_a055':.55,'two_a070':.70,'two_a080':.80,'two_a095':.95,'two_a120':1.2}
    a=vals[kind];X=rng.choice([-a,a],n)+rng.normal(0,.14,n)
   qc,kde,qcount,dcount=count_modes_1d(X,sigma,grid)
   rows.append({'case':kind,'seed':seed,'N':n,'sigma':sigma,'QC_detect_split':int(qc),'KDE_detect_split':int(kde),'QC_modes':qcount,'KDE_modes':dcount})
 return rows

def save_csv(path,rows):
 with open(path,'w',newline='') as f:
  wr=csv.DictWriter(f,list(rows[0].keys()));wr.writeheader();wr.writerows(rows)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=256)
 ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--m',default='8,16,32')
 ap.add_argument('--skip-root',action='store_true');ap.add_argument('--skip-detection',action='store_true')
 ap.add_argument('--cases',default='compact,overlap,three,anisotropic,moons,near_bifurcation,null')
 args=ap.parse_args();BASE.mkdir(exist_ok=True)
 if not args.skip_root:
  rows=roots_study(args.n,range(args.seeds),tuple(map(int,args.m.split(','))),args.cases.split(','))
  save_csv(BASE/'numerical_topology_stress.csv',rows)
 if not args.skip_detection:
  rr=detection_study();save_csv(BASE/'qc_kde_resolution.csv',rr)
 print('DONE',flush=True)
