#!/usr/bin/env python3
"""Verified 2D critical-point audit for weighted Gaussian quantum potentials.

Uses mpmath.iv outward interval arithmetic, numerical root proposals, Krawczyk
inclusion/exclusion, and exhaustive subdivision of a rectangular analysis domain.
Certification pertains ONLY to the explicitly defined finite Gaussian mixtures and
closed domain. This is not a full QC segmentation or saddle-connection certificate.
"""
import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import root
from mpmath import iv

iv.dps = 45

# All input anchors and weights use decimal strings: explicit fixed rational inputs.

def case_data(name):
    if name == 'two_wells':
        centers=[('-1.13','0.02'),('1.13','-0.02')]
        offs=[('-0.06','-0.04'), ('-0.04','0.05'),('0.04','-0.05'),('0.06','0.04')]
        sig='0.65'; domain=(-1.75,1.75,-0.90,0.90)
    elif name == 'near_bifurcation':
        centers=[('-0.72','0'),('0.72','0')]
        offs=[('-0.06','-0.04'), ('-0.04','0.05'),('0.04','-0.05'),('0.06','0.04')]
        sig='1.0';domain=(-1.5,1.5,-0.8,0.8)
    elif name == 'bifurcation_flip':
        centers=[('-0.71','0'),('0.71','0')]
        offs=[('-0.18','-0.04'),('-0.126','0.05'),('0.126','-0.05'),('0.18','0.04')]
        sig='1.0';domain=(-1.3,1.3,-0.65,0.65)
    elif name == 'three_wells':
        centers=[('-1.3','-0.7'),('1.3','-0.7'),('0.02','1.25')]
        offs=[('-0.04','-0.035'),('-0.02','0.045'),('0.02','-0.045'),('0.04','0.035')]
        sig='0.68';domain=(-1.95,1.95,-1.25,1.85)
    else:
        raise ValueError(name)
    anchors=[]
    from decimal import Decimal
    for cx,cy in centers:
        for ox,oy in offs:
            anchors.append((str(Decimal(cx)+Decimal(ox)),str(Decimal(cy)+Decimal(oy))))
    approx=[(c[0],c[1],str(len(offs))) for c in centers]
    exact=[(x,y,'1') for x,y in anchors]
    return sig, domain, exact, approx

class Potential:
    def __init__(self, anchors, sigma):
        self.coords = np.array([(float(a),float(b)) for a,b,_ in anchors], dtype=float)
        self.weights=np.array([float(w) for _,_,w in anchors],dtype=float)
        self.sig=float(sigma)
        self.acoords=[(iv.mpf(a),iv.mpf(b),iv.mpf(w)) for a,b,w in anchors]
        self.s2=iv.mpf(str(sigma))**2
        self.iv2=iv.mpf('2')
    def numeric(self,x):
        r=np.asarray(x)[None,:]-self.coords
        q=np.sum(r*r,axis=1)
        s2=self.sig**2
        w=self.weights*np.exp(-q/(2*s2))
        D=w.sum(); A=(w*q).sum()
        Dg=-(w[:,None]*r).sum(axis=0)/s2
        Ag=(w[:,None]*r*(2-q[:,None]/s2)).sum(axis=0)
        rr=r[:,:,None]*r[:,None,:]
        Id=np.eye(2)
        Dh=np.sum(w[:,None,None]*(rr/s2**2-Id/s2),axis=0)
        Ah=np.sum(w[:,None,None]*(Id[None,:,:]*(2-q[:,None,None]/s2)+rr*(q[:,None,None]/s2**2-4/s2)),axis=0)
        v=A/(2*s2*D)
        g=(Ag/D-A*Dg/D**2)/(2*s2)
        H=(Ah/D-(np.outer(Ag,Dg)+np.outer(Dg,Ag)+A*Dh)/D**2+2*A*np.outer(Dg,Dg)/D**3)/(2*s2)
        return float(v),g,H
    def interval(self, box):
        x,y=box
        z=iv.mpf(0)
        one=iv.mpf(1)
        D=z;A=z
        Dg=[z,z];Ag=[z,z]
        Dh=[[z,z],[z,z]];Ah=[[z,z],[z,z]]
        s2=self.s2
        for ax,ay,mul in self.acoords:
            r=[x-ax,y-ay]
            q=r[0]**2+r[1]**2
            w=mul*iv.exp(-q/(2*s2))
            D+=w; A+=w*q
            for j in range(2):
                Dg[j]+=-w*r[j]/s2
                Ag[j]+=w*r[j]*(2-q/s2)
                for k in range(2):
                    delta=one if j==k else z
                    Dh[j][k]+=w*(r[j]*r[k]/(s2*s2)-delta/s2)
                    Ah[j][k]+=w*(delta*(2-q/s2)+r[j]*r[k]*(q/(s2*s2)-4/s2))
        g=[(Ag[j]/D-A*Dg[j]/(D*D))/(2*s2) for j in range(2)]
        H=[[(Ah[j][k]/D-(Ag[j]*Dg[k]+Dg[j]*Ag[k]+A*Dh[j][k])/(D*D)+2*A*Dg[j]*Dg[k]/(D*D*D))/(2*s2) for k in range(2)] for j in range(2)]
        V=A/(2*s2*D)
        return V,g,H

class HomotopyPotential:
    """Uniform interval enclosure of V_t=(1-t)*Vref+t*Vcompressed, t in [0,1].

    At every spatial query, interval hull encloses the convex combination
    pointwise, including its spatial gradient and Hessian.  Numeric root
    proposals use t=0.5, but all verification tests range over every t.
    """
    def __init__(self,ref,comp):
        self.ref=ref;self.comp=comp
    def numeric(self,x):
        v0,g0,H0=self.ref.numeric(x)
        v1,g1,H1=self.comp.numeric(x)
        return .5*(v0+v1),.5*(g0+g1),.5*(H0+H1)
    @staticmethod
    def hull(a,b):
        return iv.mpf([min(a.a,b.a),max(a.b,b.b)])
    def interval(self,box):
        v0,g0,H0=self.ref.interval(box)
        v1,g1,H1=self.comp.interval(box)
        return (self.hull(v0,v1),
                [self.hull(g0[i],g1[i]) for i in range(2)],
                [[self.hull(H0[i][j],H1[i][j]) for j in range(2)] for i in range(2)])

def box_iv(bounds):
    return [iv.mpf([str(bounds[0]),str(bounds[1])]),iv.mpf([str(bounds[2]),str(bounds[3])])]

def midpoint(bounds):
    return np.array([(bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2])

def excludes_zero(z):
    return bool(z.a>0 or z.b<0)

def interior(z,l,u):
    return bool(z.a > iv.mpf(str(l)) and z.b < iv.mpf(str(u)))

def abs_upper(z):
    """Outward-enclosed upper endpoint, avoiding float in proof decisions."""
    return max(abs(z.a),abs(z.b))

def krawczyk(pot,bounds,use_numeric_jac=True):
    xm=midpoint(bounds)
    _,fnum,Hnum=pot.numeric(xm)
    if np.linalg.cond(Hnum)>1e11:
        return 'unknown',None,None
    try:
        C=np.linalg.inv(Hnum)
    except np.linalg.LinAlgError:
        return 'unknown',None,None
    # Intervalize preconditioner elements as exact decimal representations of rounded floats.
    C=[[iv.mpf(repr(float(C[i,j]))) for j in range(2)] for i in range(2)]
    V,fg,H=pot.interval(box_iv(bounds))
    xmi=[iv.mpf(repr(float(x))) for x in xm]
    _,fm,_=pot.interval(xmi)
    M=[[iv.mpf(int(i==j))-sum((C[i][k]*H[k][j] for k in range(2)),iv.mpf(0)) for j in range(2)] for i in range(2)]
    delta=[box_iv(bounds)[i]-xmi[i] for i in range(2)]
    K=[xmi[i]-sum((C[i][k]*fm[k] for k in range(2)),iv.mpf(0)) + sum((M[i][j]*delta[j] for j in range(2)),iv.mpf(0)) for i in range(2)]
    max_norm=max(sum((abs_upper(M[i][j]) for j in range(2)),iv.mpf(0)) for i in range(2))
    inclusion=all(interior(K[i],bounds[2*i],bounds[2*i+1]) for i in range(2)) and max_norm < iv.mpf(1)
    if inclusion: return 'unique', K, float(max_norm.b)
    # No roots if K is disjoint from box in some coordinate.
    for i in range(2):
        if K[i].b < iv.mpf(str(bounds[2*i])) or K[i].a > iv.mpf(str(bounds[2*i+1])):
            return 'exclude',K,float(max_norm.b)
    return 'unknown',K,float(max_norm.b)

def root_candidates(pot,domain):
    x0,x1,y0,y1=domain
    roots=[]
    for x in np.linspace(x0,x1,13):
        for y in np.linspace(y0,y1,11):
            sol=root(lambda z:pot.numeric(z)[1], [x,y], jac=lambda z:pot.numeric(z)[2],method='hybr',tol=1e-11)
            if np.linalg.norm(pot.numeric(sol.x)[1])>1e-7:continue
            a,b=sol.x
            if not (x0<a<x1 and y0<b<y1):continue
            if not any(np.linalg.norm(sol.x-r)<1e-4 for r in roots):
                roots.append(sol.x)
    return sorted(roots,key=lambda p:(p[0],p[1]))

def init_box(x,domain,radius=.09):
    x0,x1,y0,y1=domain
    return (max(x0,float(x[0])-radius),min(x1,float(x[0])+radius),max(y0,float(x[1])-radius),min(y1,float(x[1])+radius))

def overlap(a,b):
    return (max(a[0],b[0])<min(a[1],b[1]) and max(a[2],b[2])<min(a[3],b[3]))

def subtract_box(a,b):
    if not overlap(a,b):return [a]
    ix0=max(a[0],b[0]);ix1=min(a[1],b[1]);iy0=max(a[2],b[2]);iy1=min(a[3],b[3])
    outs=[]
    if a[0]<ix0:outs.append((a[0],ix0,a[2],a[3]))
    if ix1<a[1]:outs.append((ix1,a[1],a[2],a[3]))
    if a[2]<iy0:outs.append((ix0,ix1,a[2],iy0))
    if iy1<a[3]:outs.append((ix0,ix1,iy1,a[3]))
    return outs

def rootbox_solve(pot,x,domain):
    best=None
    for radius in [.0005,.001,.002,.005,.01,.02,.035,.06,.10,.15,.22]:
        b=init_box(x,domain,radius)
        status,K,norm=krawczyk(pot,b)
        if status=='unique':
            best={'status':status,'box':b,'norm':norm,'enclosure':[[float(K[i].a),float(K[i].b)] for i in range(2)]}
    if best is not None: return best
    return {'status':'unresolved','box':init_box(x,domain,.10)}

def classify(pot,bounds):
    _,_,H=pot.interval(box_iv(bounds))
    # For symmetric 2x2 Hessian, definitive positive/negative definiteness or saddle
    # using interval signs of H00 and determinant; symmetry evaluated separately.
    det=H[0][0]*H[1][1]-H[0][1]*H[1][0]
    if det.a>0 and H[0][0].a>0:return 'minimum'
    if det.a>0 and H[0][0].b<0:return 'maximum'
    if det.b<0:return 'saddle'
    return 'undetermined'

def exclude_complement(pot, domain, rootboxes, max_boxes=50000, min_width=5e-5):
    remaining=[domain]
    for b in rootboxes:
        next_rem=[]
        for r in remaining:next_rem.extend(subtract_box(r,b))
        remaining=next_rem
    queue=deque(remaining)
    stats={'gradient_excluded':0,'krawczyk_excluded':0,'split':0,'max_queue':len(queue)}
    unresolved=[]
    while queue:
        stats['max_queue']=max(len(queue),stats['max_queue'])
        b=queue.popleft()
        _,g,_=pot.interval(box_iv(b))
        if excludes_zero(g[0]) or excludes_zero(g[1]):
            stats['gradient_excluded']+=1;continue
        status,_,_=krawczyk(pot,b)
        if status=='exclude':
            stats['krawczyk_excluded']+=1;continue
        w=b[1]-b[0];h=b[3]-b[2]
        if max(w,h)<min_width or stats['split']>max_boxes:
            unresolved.append(b)
            if len(unresolved)>=10 or stats['split']>max_boxes: break
            continue
        if w>=h:
            mid=(b[0]+b[1])/2
            queue.append((b[0],mid,b[2],b[3]));queue.append((mid,b[1],b[2],b[3]))
        else:
            mid=(b[2]+b[3])/2
            queue.append((b[0],b[1],b[2],mid));queue.append((b[0],b[1],mid,b[3]))
        stats['split']+=1
    stats['unresolved']=len(unresolved)+len(queue)
    return stats,unresolved

def certify(pot,domain,verbose=True):
    t0=time.monotonic()
    candidates=root_candidates(pot,domain)
    rboxes=[rootbox_solve(pot,x,domain) for x in candidates]
    passed=[r for r in rboxes if r['status']=='unique']
    disjoint=all(not overlap(a['box'],b['box']) for i,a in enumerate(passed) for b in passed[i+1:])
    for r in passed:
        r['type']=classify(pot,r['box'])
        center=midpoint(r['box'])
        refined=None
        for radius in [0.005,0.001,0.0002,0.00005]:
            b=init_box(center,domain,radius)
            status,K,_=krawczyk(pot,b)
            if status=='unique' and all(b[2*i]>=r['box'][2*i] and b[2*i+1]<=r['box'][2*i+1] for i in range(2)):
                refined=[[float(K[i].a),float(K[i].b)] for i in range(2)]
        r['refined_enclosure']=refined

    if len(passed)!=len(candidates) or not disjoint:
        st={'unresolved':1, 'reason':'candidate boxes failed inclusion or overlap'}
    else:
        st,_=exclude_complement(pot,domain,[r['box'] for r in passed])
    return {'candidate_count':len(candidates),'root_boxes':rboxes,'disjoint':disjoint,'global':st,'certified':len(passed)==len(candidates) and disjoint and st.get('unresolved',1)==0 and all(r.get('type')!='undetermined' for r in passed),'elapsed_s':round(time.monotonic()-t0,3)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',default='two_wells',choices=['two_wells','near_bifurcation','three_wells','bifurcation_flip'])
    ap.add_argument('--out',default='.')
    ap.add_argument('--only',choices=['reference','compressed','homotopy','both','all'],default='both')
    ap.add_argument('--precision',type=int,default=45,help='Decimal digits in mpmath.iv')
    args=ap.parse_args();iv.dps=args.precision;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sigma,domain,ex,approx=case_data(args.case)
    result={'case':args.case,'sigma':sigma,'domain':domain,'anchors':len(ex),'landmarks':len(approx)}
    for name,anchors in [('reference',ex),('compressed',approx)]:
        if args.only not in ('both','all') and name!=args.only: continue
        pot=Potential(anchors,sigma)
        result[name]=certify(pot,domain)
        print(name,result[name]['certified'],result[name]['candidate_count'],result[name]['global'],result[name]['elapsed_s'],flush=True)
        for r in result[name]['root_boxes']:print(' ',r,flush=True)
    if args.only in ('homotopy','all'):
        result['homotopy']=certify(HomotopyPotential(Potential(ex,sigma),Potential(approx,sigma)),domain)
        print('homotopy',result['homotopy']['certified'],result['homotopy']['candidate_count'],result['homotopy']['global'],result['homotopy']['elapsed_s'],flush=True)
    with open(out/f'{args.case}.json','w') as f:json.dump(result,f,indent=2)
    print('Saved',out/f'{args.case}.json')

if __name__=='__main__':main()
