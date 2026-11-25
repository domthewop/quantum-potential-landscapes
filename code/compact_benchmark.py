#!/usr/bin/env python3
"""Reproducible comparison: full anchor interval verification vs group-moment verification."""
import argparse, time, csv, json, traceback
from decimal import Decimal
from pathlib import Path
import numpy as np
from mpmath import iv
from interval_verifier import Potential,certify,case_data
from moment_enclosures import MomentPotential,read_groups

def synthetic_two_wells(n, seed=20):
    if n==8: return case_data('two_wells')[:3]
    assert n%2==0
    rng=np.random.default_rng(seed)
    a=[]
    for center in [(-1.13,.02),(1.13,-.02)]:
        for _ in range(n//2):
            x,y=np.array(center)+rng.normal(0,.028,size=2)
            a.append((f'{x:.10f}',f'{y:.10f}','1'))
    return '0.65',(-1.75,1.75,-.90,.90),a

def run(n, seed, reference=True, approximate=True):
    sig,domain,raw=synthetic_two_wells(n,seed)
    rows=[]
    if reference:
        t=time.monotonic();pot=Potential(raw,sig)
        result=certify(pot,domain)
        rows.append({'n':n,'seed':seed,'method':'full reference','landmarks':n,'build_s':0,'verify_s':result['elapsed_s'],
                     'total_s':round(time.monotonic()-t,3), 'certified':result['certified'],
                     'candidate_count':result['candidate_count'], 'unresolved':result['global'].get('unresolved',-1),
                     'split':result['global'].get('split',-1)})
        print(rows[-1],flush=True)
    if approximate:
        t=time.monotonic();groups=read_groups(raw,2,seed=seed)
        build=time.monotonic()-t
        pot=MomentPotential(groups,sig)
        try:
            result=certify(pot,domain)
            rows.append({'n':n,'seed':seed,'method':'moment certificate','landmarks':2,'build_s':round(build,4),'verify_s':result['elapsed_s'],
                          'total_s':round(time.monotonic()-t,3),'certified':result['certified'],
                          'candidate_count':result['candidate_count'],'unresolved':result['global'].get('unresolved',-1),
                          'split':result['global'].get('split',-1)})
        except Exception as e:
            rows.append({'n':n,'seed':seed,'method':'moment certificate','landmarks':2,'build_s':round(build,4),'verify_s':round(time.monotonic()-t-build,3),
                         'total_s':round(time.monotonic()-t,3),'certified':False,'candidate_count':-1,'unresolved':-1,'split':-1,'error':str(e)})
        print(rows[-1],flush=True)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',default='8,32,80')
    p.add_argument('--seed',type=int,default=20)
    p.add_argument('--out',default='results.csv')
    p.add_argument('--only',choices=['full','moment','both'],default='both')
    args=p.parse_args()
    rows=[]
    for n in [int(i) for i in args.sizes.split(',')]:
        row=run(n,args.seed,args.only in ('both','full'),args.only in ('both','moment'))
        rows+=row
        with open(args.out,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader();writer.writerows(rows)
    print('Saved',args.out,flush=True)
