#!/usr/bin/env python3
"""One isolated phase-eight 2D interval verification job, external timeout enforced by orchestrator."""
import argparse, json, sys, time, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from moment_enclosures import MomentPotential,read_groups
from interval_verifier import Potential,certify,case_data
from geometry_stress import sample_case,SIGMA
import numpy as np
from mpmath import iv
ap=argparse.ArgumentParser(); ap.add_argument('--case',required=True)
ap.add_argument('--n',type=int,default=24); ap.add_argument('--m',type=int,default=2)
ap.add_argument('--seed',type=int,default=0); ap.add_argument('--method',choices=['moment','direct'],default='moment')
ap.add_argument('--out',required=True); args=ap.parse_args(); iv.dps=32
if args.case=='canonical_three':
 sigma,domain,raw,_=case_data('three_wells')
elif args.case=='canonical_two':
 sigma,domain,raw,_=case_data('two_wells')
elif args.case=='compact_big':
 from compact_benchmark import synthetic_two_wells
 sigma,domain,raw=synthetic_two_wells(args.n, args.seed)
else:
 X,_,domain=sample_case(args.case,args.n,args.seed)
 sigma=SIGMA[args.case]
 raw=[(repr(float(x)),repr(float(y)),'1') for x,y in X]
start=time.monotonic()
if args.method=='direct':pot=Potential(raw,str(sigma)); n_landmarks=len(raw)
else:
 groups=read_groups(raw,args.m,args.seed)
 pot=MomentPotential(groups,sigma); n_landmarks=len(groups)
build=time.monotonic()-start
res=certify(pot,domain)
out=dict(case=args.case,n=len(raw),m=n_landmarks,seed=args.seed,method=args.method,
         build_s=round(build,4),elapsed_s=round(time.monotonic()-start,3),
         certified=res['certified'],n_roots=res['candidate_count'],
         critical_types=[r.get('type') for r in res['root_boxes']],unresolved=res['global'].get('unresolved'),
         global_info=res['global'],root_boxes=res['root_boxes'])
Path(args.out).write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['root_boxes','global_info']}),flush=True)
