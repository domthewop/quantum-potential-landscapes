#!/usr/bin/env python3
"""Independent MPFR-256 point checks for mpmath.iv quantum potential enclosures.
This checks values at points, NOT an independent box/subdivision proof.
"""
import ctypes as C, json, sys, csv
from pathlib import Path
import numpy as np
from mpmath import iv
ROOT=Path(__file__).resolve().parents[1]
from moment_enclosures import MomentPotential,read_groups
from interval_verifier import box_iv
from geometry_stress import sample_case,SIGMA
class MPFRStruct(C.Structure):
    _fields_=[('_mpfr_prec',C.c_long),('_mpfr_sign',C.c_int),('_mpfr_exp',C.c_long),('_mpfr_d',C.POINTER(C.c_ulong))]
lib=C.CDLL('libmpfr.so.6')
p=C.POINTER(MPFRStruct)
lib.mpfr_init2.argtypes=[p,C.c_long];lib.mpfr_clear.argtypes=[p]
lib.mpfr_set_str.argtypes=[p,C.c_char_p,C.c_int,C.c_int];lib.mpfr_set_str.restype=C.c_int
for op in ('add','sub','mul','div'):
    fn=getattr(lib,'mpfr_'+op);fn.argtypes=[p,p,p,C.c_int];fn.restype=C.c_int
lib.mpfr_exp.argtypes=[p,p,C.c_int]
lib.mpfr_get_d.argtypes=[p,C.c_int];lib.mpfr_get_d.restype=C.c_double
class M:
    def __init__(self,x=0):
        self.q=MPFRStruct();lib.mpfr_init2(C.byref(self.q),256)
        if isinstance(x,M):lib.mpfr_set_str(C.byref(self.q),str(x.asfloat()).encode(),10,0)
        else:
            r=lib.mpfr_set_str(C.byref(self.q),str(x).encode(),10,0)
            if r!=0:raise ValueError(x)
    def __del__(self):
        try:lib.mpfr_clear(C.byref(self.q))
        except Exception:pass
    def __op(self,o,kind):
        if not isinstance(o,M):o=M(o)
        z=M(0);getattr(lib,'mpfr_'+kind)(C.byref(z.q),C.byref(self.q),C.byref(o.q),0);return z
    def __add__(self,o):return self.__op(o,'add')
    def __radd__(self,o):return self.__add__(o)
    def __sub__(self,o):return self.__op(o,'sub')
    def __rsub__(self,o):return M(o).__sub__(self)
    def __mul__(self,o):return self.__op(o,'mul')
    def __rmul__(self,o):return self.__mul__(o)
    def __truediv__(self,o):return self.__op(o,'div')
    def __rtruediv__(self,o):return M(o).__truediv__(self)
    def __neg__(self):return M(0)-self
    def __pow__(self,n):
        assert n>=0 and isinstance(n,int)
        z=M(1)
        for _ in range(n):z=z*self
        return z
    def exp(self):
        z=M(0);lib.mpfr_exp(C.byref(z.q),C.byref(self.q),0);return z
    def asfloat(self):return lib.mpfr_get_d(C.byref(self.q),0)

def qpot_mpfr(raw,sigma,xy):
    s=M(str(sigma))**2
    D=A=M(0)
    Dg=[M(0),M(0)];Ag=[M(0),M(0)]
    Dh=[[M(0),M(0)] for _ in range(2)]
    Ah=[[M(0),M(0)] for _ in range(2)]
    for xx,yy,weight in raw:
        r=[M(str(xy[0]))-M(xx),M(str(xy[1]))-M(yy)]
        q=r[0]**2+r[1]**2
        w=M(weight)*(-q/(2*s)).exp()
        D=D+w;A=A+w*q
        for j in range(2):
            Dg[j]=Dg[j]-w*r[j]/s
            Ag[j]=Ag[j]+w*r[j]*(2-q/s)
            for k in range(2):
                delta=int(j==k)
                Dh[j][k]=Dh[j][k]+w*(r[j]*r[k]/s**2-M(delta)/s)
                Ah[j][k]=Ah[j][k]+w*(M(delta)*(2-q/s)+r[j]*r[k]*(q/s**2-4/s))
    V=A/(2*s*D)
    grad=[(Ag[j]/D-A*Dg[j]/D**2)/(2*s) for j in range(2)]
    hess=[[ (Ah[j][k]/D-(Ag[j]*Dg[k]+Dg[j]*Ag[k]+A*Dh[j][k])/D**2+2*A*Dg[j]*Dg[k]/D**3)/(2*s) for k in range(2)] for j in range(2)]
    return float(V.asfloat()),np.array([x.asfloat() for x in grad]),np.array([[z.asfloat() for z in row] for row in hess])

def contained(ivalue, value):
    # MPFR rounded double only used for a sampled point check, with 10^-12 allowance.
    # Avoid false confidence in rounding exact inclusion.
    return (float(ivalue.a)-1e-12 <= value <= float(ivalue.b)+1e-12)

def main():
    rng=np.random.default_rng(2323);rows=[]
    for case in ['compact','overlap','three','near_bifurcation','anisotropic','null']:
        X,_,dom=sample_case(case,18,7)
        raw=[(format(a,'.17g'),format(b,'.17g'),'1') for a,b in X]
        sig=SIGMA[case];m=3 if case=='three' else 4
        pot=MomentPotential(read_groups(raw,m,seed=7),sig)
        for k in range(12):
            pt=np.array([rng.uniform(dom[0],dom[1]),rng.uniform(dom[2],dom[3])])
            dx=rng.uniform(.001,.025)
            box=box_iv((pt[0]-dx,pt[0]+dx,pt[1]-dx,pt[1]+dx))
            try:
                vb,gb,hb=pot.interval(box)
                # Two random interior points per interval box.
                ok=True; margin=float('inf'); max_abs=0
                for _ in range(2):
                    xy=pt+rng.uniform(-dx,dx,size=2)
                    v,g,h=qpot_mpfr(raw,sig,xy)
                    bvals=[vb,gb[0],gb[1],hb[0][0],hb[0][1],hb[1][0],hb[1][1]]
                    vals=[v,g[0],g[1],h[0,0],h[0,1],h[1,0],h[1,1]]
                    ok &= all(contained(z,q) for z,q in zip(bvals,vals))
                    max_abs=max(max_abs, max(abs(a) for a in vals))
                rows.append(dict(case=case,probe=k,pass_point_containment=bool(ok),error='',box_half_width=dx))
            except Exception as e:
                rows.append(dict(case=case,probe=k,pass_point_containment=False,error=repr(e),box_half_width=dx))
        print(case, sum(x['pass_point_containment'] for x in rows if x['case']==case),'/12',flush=True)
    with open(Path(__file__).with_name('mpfr_point_audit.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('PASS',sum(x['pass_point_containment'] for x in rows),'/',len(rows))

if __name__=='__main__':main()
