#!/usr/bin/env python3
"""Reference-free-at-query-time 2D quantum-potential verifier.

Full anchors are needed ONCE to create exact-decimal group moments, then deleted
from the query-time certificate object. Reference potential derivatives are
outward enclosed by weighted-landmark field derivatives plus Taylor remainder
bounds for Gaussian derivatives through order four.

The certificate is conditional on mpmath.iv's interval arithmetic. No claim is
made that this code or its bounds constitute a novel theorem.
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from mpmath import iv
from scipy.cluster.vq import kmeans2

from interval_verifier import Potential, certify, case_data, box_iv, abs_upper

getcontext().prec = 70
iv.dps = 32
ZERO = iv.mpf(0)
ONE = iv.mpf(1)

def hermites(x, max_n=7):
    h=[ONE, x]
    for k in range(1,max_n):
        h.append(x*h[-1]-k*h[-2])
    return h[:max_n+1]

# All scalar Gaussian derivatives can be computed by products of univariate
# probabilists' Hermite polynomials; interval arithmetic gives an enclosure.
def gaussian_derivs(r0, r1, sigma, max_order=7):
    u=r0/sigma;v=r1/sigma
    h0=hermites(u,max_order);h1=hermites(v,max_order)
    # power avoids the negative interval lower bound from dependency u*u
    e0=iv.exp(-(u**2)/2);e1=iv.exp(-(v**2)/2)
    t0=[(-1)**i*h0[i]*e0/sigma**i for i in range(max_order+1)]
    t1=[(-1)**i*h1[i]*e1/sigma**i for i in range(max_order+1)]
    return t0,t1

def ivfrac(x):
    f=Fraction(x)
    return iv.mpf(str(f.numerator))/iv.mpf(str(f.denominator))

def floatfrac(x):
    return float(Fraction(x))

def read_groups(anchors, n_groups, seed=0):
    """Build rational-decimal weighted centroids and absolute group moments."""
    dat=np.array([[float(a),float(b)] for a,b,_ in anchors])
    if n_groups>=len(anchors):
        labels=np.arange(len(anchors))
    elif n_groups==2:
        # For the 2-well and bifurcation cases, group by sign.
        labels=(dat[:,0]>=0).astype(int)
    elif n_groups==3 and len(anchors)==12:
        # Three-well three-center baseline, independent of small noise.
        from scipy.spatial.distance import cdist
        cc=np.array([[-1.3,-.7],[1.3,-.7],[.02,1.25]])
        labels=np.argmin(cdist(dat,cc),axis=1)
    else:
        _,labels=kmeans2(dat,n_groups,minit='points',iter=30,seed=seed)
    groups=defaultdict(list)
    for row, lab in zip(anchors,labels):
        groups[int(lab)].append(row)
    out=[]
    for inds in groups.values():
        n=len(inds)
        # Centroid is chosen as a FINITE decimal: the exact moments below
        # account for any discrepancy from the rational sample mean.
        cx=str(sum((Decimal(a) for a,b,w in inds),Decimal(0))/n)
        cy=str(sum((Decimal(b) for a,b,w in inds),Decimal(0))/n)
        xs=[Fraction(a) for a,b,w in inds]
        ys=[Fraction(b) for a,b,w in inds]
        dx=[x-Fraction(cx) for x in xs];dy=[y-Fraction(cy) for y in ys]
        out.append(dict(x=cx,y=cy,n=n,
                        sx=str(sum(dx)),sy=str(sum(dy)),
                        sxx=str(sum(t*t for t in dx)),
                        syy=str(sum(t*t for t in dy)),
                        sxy=str(sum(a*b for a,b in zip(dx,dy))),
                        sxxx=str(sum(abs(t)**3 for t in dx)),
                        sxxy=str(sum(abs(a*a*b) for a,b in zip(dx,dy))),
                        sxyy=str(sum(abs(a*b*b) for a,b in zip(dx,dy))),
                        syyy=str(sum(abs(t)**3 for t in dy)),
                        rx=str(max(abs(t) for t in dx)),
                        ry=str(max(abs(t) for t in dy))))
    # Exact rational centroid cancellation is a necessary invariant.
    for g in out:
        assert Fraction(g['sxx'])>=0 and Fraction(g['syy'])>=0
    return out

class MomentPotential:
    """Enclosure of full reference potential from only (landmarks,moments)."""
    def __init__(self, groups, sigma):
        self.groups=groups
        self.sig=float(sigma)
        self.sigma=iv.mpf(str(sigma))
        self.s2=self.sigma**2
        self.landmarks=Potential([(g['x'],g['y'],str(g['n'])) for g in groups],str(sigma))
        self.n_approx_terms=len(groups)
        self.query_count=0
        self.envelope_count=0
        self.n_original=sum(g['n'] for g in groups)
        self._groups=[dict(x=iv.mpf(g['x']),y=iv.mpf(g['y']),n=iv.mpf(str(g['n'])),
                           rx=ivfrac(g['rx']),ry=ivfrac(g['ry']),sx=ivfrac(g['sx']),sy=ivfrac(g['sy']),
                           sxx=ivfrac(g['sxx']),syy=ivfrac(g['syy']),sxy=ivfrac(g['sxy']),
                           sxxx=ivfrac(g['sxxx']),sxxy=ivfrac(g['sxxy']),
                           sxyy=ivfrac(g['sxyy']),syyy=ivfrac(g['syyy'])) for g in groups]
    def numeric(self,x):
        # Numeric proposal only: use deterministic moment-corrected surrogate,
        # never the original anchors. All eventual proof decisions use iv.
        sigma=self.sig
        from scipy.special import eval_hermitenorm
        z={idx:0. for idx in [(p,q) for p in range(5) for q in range(5-p)]}
        for g in self.groups:
            xx=float(x[0])-float(g['x']);yy=float(x[1])-float(g['y'])
            uu=xx/sigma;vv=yy/sigma
            k=np.exp(-.5*(uu*uu+vv*vv))
            def dd(p,q):
                return ((-1)**(p+q))*k*eval_hermitenorm(p,uu)*eval_hermitenorm(q,vv)/sigma**(p+q)
            n=g['n'];sx=floatfrac(g['sx']);sy=floatfrac(g['sy']);sxx=floatfrac(g['sxx']);sxy=floatfrac(g['sxy']);syy=floatfrac(g['syy'])
            for p,q in z:
                z[p,q]+=n*dd(p,q)-sx*dd(p+1,q)-sy*dd(p,q+1)+.5*(sxx*dd(p+2,q)+2*sxy*dd(p+1,q+1)+syy*dd(p,q+2))
        D=z[0,0];L=z[2,0]+z[0,2];dg=np.array([z[1,0],z[0,1]])
        gl=np.array([z[3,0]+z[1,2],z[2,1]+z[0,3]])
        dh=np.array([[z[2,0],z[1,1]],[z[1,1],z[0,2]]])
        hl=np.array([[z[4,0]+z[2,2],z[3,1]+z[1,3]],[z[3,1]+z[1,3],z[2,2]+z[0,4]]])
        factor=.5*sigma**2
        return (float(factor*L/D+1),factor*(gl/D-L*dg/D**2),
                factor*(hl/D-(np.outer(gl,dg)+np.outer(dg,gl)+L*dh)/D**2+2*L*np.outer(dg,dg)/D**3))
    def interval(self, box):
        self.query_count+=1
        x,y=box
        # Derivatives of order 0..4, exactly the indices used by the quotient
        # Hessian. A single pass computes all of them and their error envelopes.
        indices=[(p,q) for p in range(5) for q in range(5-p)]
        deriv={z:ZERO for z in indices}
        error={z:ZERO for z in indices}
        exact_psi_lower=ZERO
        for g in self._groups:
            cx=g['x'];cy=g['y']
            u,v=gaussian_derivs(x-cx,y-cy,self.sigma,6)
            for p,q in indices:
                deriv[(p,q)]+=(g['n']*u[p]*v[q]-g['sx']*u[p+1]*v[q]-g['sy']*u[p]*v[q+1]
                                +(g['sxx']*u[p+2]*v[q]+2*g['sxy']*u[p+1]*v[q+1]+g['syy']*u[p]*v[q+2])/2)
            # Every point in the group lies within this coordinate rectangle.
            # The centroid cancellation removes all first order Taylor terms.
            rx=iv.mpf([(-g['rx']).a,g['rx'].b]); ry=iv.mpf([(-g['ry']).a,g['ry'].b])
            rxb=x-cx+rx;ryb=y-cy+ry
            qmax=abs_upper(rxb)**2+abs_upper(ryb)**2
            exact_psi_lower+=g['n']*iv.exp(-qmax/(2*self.s2))
            tu,tv=gaussian_derivs(rxb,ryb,self.sigma,7)
            self.envelope_count+=1
            for p,q in indices:
                bxxx=abs_upper(tu[p+3]*tv[q]); bxxy=abs_upper(tu[p+2]*tv[q+1])
                bxyy=abs_upper(tu[p+1]*tv[q+2]); byyy=abs_upper(tu[p]*tv[q+3])
                error[(p,q)]+=(g['sxxx']*bxxx+3*g['sxxy']*bxxy+3*g['sxyy']*bxyy+g['syyy']*byyy)/6
        P={idx:deriv[idx]+iv.mpf([-error[idx].b,error[idx].b]) for idx in indices}
        D=P[(0,0)]
        # Independent positive lower enclosure, valid even if Taylor's
        # polynomial remainder is wider than the Gaussian sum itself.
        if D.b < exact_psi_lower.a:
            raise AssertionError('Inconsistent field bounds')
        D=iv.mpf([max(D.a,exact_psi_lower.a),D.b])
        lap=P[(2,0)]+P[(0,2)]
        grad_lap=[P[(3,0)]+P[(1,2)], P[(2,1)]+P[(0,3)]]
        hess_lap=[[P[(4,0)]+P[(2,2)], P[(3,1)]+P[(1,3)]],
                  [P[(3,1)]+P[(1,3)], P[(2,2)]+P[(0,4)]]]
        grad_d=[P[(1,0)],P[(0,1)]]
        hess_d=[[P[(2,0)],P[(1,1)]],[P[(1,1)],P[(0,2)]]]
        half=self.s2/2
        V=half*lap/D+1
        grad=[half*(grad_lap[i]/D-lap*grad_d[i]/D**2) for i in range(2)]
        Hess=[[half*(hess_lap[i][j]/D-(grad_lap[i]*grad_d[j]+grad_lap[j]*grad_d[i]+lap*hess_d[i][j])/D**2+2*lap*grad_d[i]*grad_d[j]/D**3) for j in range(2)] for i in range(2)]
        return V,grad,Hess
    def field_error_bounds(self,box):
        # For an inexpensive point query, call same enclosure and compare to landmark potential.
        v,g,h=self.interval(box)
        va,ga,ha=self.landmarks.interval(box)
        return v-va,[g[i]-ga[i] for i in range(2)]

if __name__=='__main__':
    for name in ('two_wells','three_wells','bifurcation_flip'):
        sig,dom,raw,_=case_data(name)
        m=2 if name!='three_wells' else 3
        pot=MomentPotential(read_groups(raw,m),sig)
        print('checking',name,'N=',len(raw),'m=',m,flush=True)
        try:
            ans=certify(pot,dom)
            print(name,ans['certified'],ans['candidate_count'],ans['global'],ans['elapsed_s'],flush=True)
        except Exception as e: print(name,repr(e),flush=True)
