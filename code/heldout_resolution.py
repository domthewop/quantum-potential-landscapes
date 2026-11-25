#!/usr/bin/env python3
"""Matched bandwidth selection, fixed false-split rule, disjoint validation/test seeds.
Reports power/FPR without treating either as a ground-truth topology proof.
"""
from pathlib import Path
import sys,csv, numpy as np
BASE=Path(__file__).resolve().parent
from geometry_stress import count_modes_1d
BWS=[.44,.52,.60,.66,.72,.80,.90,1.0,1.15]
NULL=['gauss_small','gauss_wide','student_t3','skew']
ALT=['mix_065','mix_080','mix_095','mix_120']

def draw(case, seed, n=160):
    r=np.random.default_rng(519100+seed)
    if case=='gauss_small':return r.normal(0,.45,n)
    if case=='gauss_wide':return r.normal(0,1.0,n)
    if case=='student_t3':return r.standard_t(3,n)*.5
    if case=='skew':return (r.lognormal(0,.7,n)-np.exp(.7**2/2))*.7
    a=float(case.split('_')[1])/100
    return r.choice([-a,a],size=n)+r.normal(0,.14,n)

def run():
    grid=np.linspace(-3,3,501)
    rows=[]
    for case in NULL+ALT:
        for split,seed in [('validation',i) for i in range(20)]+[('test',i) for i in range(100,140)]:
            x=draw(case,seed)
            for bw in BWS:
                q,k,_,_=count_modes_1d(x,bw,grid)
                rows.extend([{'case':case,'class':'null' if case in NULL else 'alternative','split':split,'seed':seed,'bandwidth':bw,'method':'QC','detected_split':int(q)},
                             {'case':case,'class':'null' if case in NULL else 'alternative','split':split,'seed':seed,'bandwidth':bw,'method':'KDE','detected_split':int(k)}])
        print('completed',case,flush=True)
    with open(BASE/'heldout_detections.csv','w',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    summ=[]
    for method in ['QC','KDE']:
        res=[]
        for bw in BWS:
            val=[r for r in rows if r['method']==method and r['bandwidth']==bw and r['split']=='validation']
            fp=np.mean([r['detected_split'] for r in val if r['class']=='null'])
            power=np.mean([r['detected_split'] for r in val if r['class']=='alternative'])
            res.append((fp,power,bw))
        valid=[v for v in res if v[0]<=.05]
        if not valid:
            fp,pow,bw=min(res,key=lambda t:(t[0],-t[1]))
        else:fp,pow,bw=max(valid,key=lambda t:(t[1],-t[0],t[2]))
        test=[r for r in rows if r['method']==method and r['bandwidth']==bw and r['split']=='test']
        fpr=np.mean([r['detected_split'] for r in test if r['class']=='null'])
        tpr=np.mean([r['detected_split'] for r in test if r['class']=='alternative'])
        per={case:np.mean([r['detected_split'] for r in test if r['case']==case]) for case in NULL+ALT}
        summ.append({'method':method,'selected_bandwidth':bw,'validation_fpr':fp,'validation_tpr':pow,'test_fpr':fpr,'test_tpr':tpr,'per_case_test_rates':str(per)})
    with open(BASE/'heldout_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,summ[0]);w.writeheader();w.writerows(summ)
    print('SUMMARY',summ,flush=True)
if __name__=='__main__':run()
