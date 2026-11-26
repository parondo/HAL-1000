"""HFP-17 / empirical optical adapter audit, 2026-11-26.

HUMAN: predict periods and lateral shift from published instrument settings,
then compare with published measurements; no fitted parameters.
AI: external optical adapter in O only. Trigonometric bounds translate the
source's angle labels, not a primitive or active statistical tool of Nyx.
All numerical calculations use fractions; printed decimals are display only.
Source values are reported decimals, not exact physical observations.
"""
from fractions import Fraction as Q
from itertools import product
from math import factorial
from collections import Counter
from pathlib import Path
import json
import platform

# SOURCE PROVENANCE: Lowell et al., Micromachines 7, 128 (2016), §§4.1–4.3;
# https://doi.org/10.3390/mi7070128 ; Lin et al., Nat Commun 17, 7175 (2026),
# Fig. 5a, https://doi.org/10.1038/s41467-026-73975-4.
# Gao et al., Microsystems & Nanoengineering 11, 41 (2025), Fig. 6;
# https://doi.org/10.1038/s41378-025-00889-4.


def arctan_bounds(x, n=20):
    """Alternating rational series; adjacent partial sums enclose arctan."""
    s = sum(((-1)**k*x**(2*k+1)/ (2*k+1) for k in range(n)), Q(0))
    t = s + (-1)**n*x**(2*n+1)/(2*n+1)
    return min(s,t), max(s,t)


def pi_bounds():
    # Machin identity, solely for reading the instrument's degree convention.
    a,b = arctan_bounds(Q(1,5)); c,d = arctan_bounds(Q(1,239))
    return 16*a-4*d,16*b-4*c


def outward(lo, hi, digits=12):
    scale=10**digits
    return Q(lo*scale//1,scale), Q(-((-hi*scale)//1),scale)


def trig_at(x, cosine=False, n=12):
    """Enclose sin or cos on 0 <= x <= 1 by adjacent alternating sums."""
    if not 0 <= x <= 1:
        raise ValueError('STOP: angle outside adapter bound')
    shift=0 if cosine else 1
    s=sum(((-1)**k*x**(2*k+shift)/factorial(2*k+shift)
           for k in range(n)),Q(0))
    t=s+(-1)**n*x**(2*n+shift)/factorial(2*n+shift)
    return min(s,t),max(s,t)


def trig_deg(deg):
    p0,p1=pi_bounds()
    # Compact, outward-rounded rational interval before the Taylor step.
    x0,x1=outward(p0*Q(deg,180),p1*Q(deg,180),14)
    sin0,_=trig_at(x0); _,sin1=trig_at(x1)
    cos0,_=trig_at(x1,True); _,cos1=trig_at(x0,True)
    return outward(sin0,sin1),outward(cos0,cos1)


def predict(kind,wavelength,angle,travel=None):
    """Instrument adapter: a relative phase turn closes after one period.
    Convention of the source; no extra refraction correction is applied.
    Axial periods use lambda/(1-cos theta); lateral use lambda/sin theta.
    """
    (s0,s1),(c0,c1)=trig_deg(angle)
    if kind=='axial':
        lo,hi=wavelength/(1-c0),wavelength/(1-c1)
    elif kind=='lateral':
        lo,hi=wavelength/s1,wavelength/s0
    elif kind=='tilt':
        lo,hi=travel*s0/c1,travel*s1/c0
    elif kind=='radial5':
        # 54 degrees is within the Taylor interval; cos(3*pi/10).
        _,(d0,d1)=trig_deg(54)
        lo,hi=wavelength/(2*s1*d1),wavelength/(2*s0*d0)
    else:
        raise ValueError('STOP: unknown measurement type')
    return outward(lo,hi,6)


def row(name,source,kind,wavelength,angle,observed,unit,travel=None,note=''):
    lo,hi=predict(kind,Q(wavelength),angle,Q(travel) if travel else None)
    o=Q(observed)
    # Relative signed discrepancy, no aggregation and no fitted threshold.
    e0,e1=outward((o/hi-1)*100,(o/lo-1)*100,6)
    return dict(name=name,source=source,kind=kind,wavelength=str(wavelength),
                angle_deg=angle,travel=str(travel) if travel else None,
                observed=str(o),unit=unit,predicted_interval=[str(lo),str(hi)],
                signed_error_interval=[str(o-hi),str(o-lo)],
                signed_relative_percent_interval=[str(e0),str(e1)],note=note)


def run():
    # Freeze all source inputs; do not choose angles by minimizing discrepancies.
    # 4+1 asymmetric: both component references retained, no average constructed.
    rows=[row('Lin 2026 / optical tilt','Lin 2026 Fig. 5a','tilt','405',10,80,'um',450,
              'Measured optical pattern in air; approximate 80 um; not a cured depth.'),
          row('Lowell 2016 / 4+1 transverse y','Lowell 2016 §4.1 Fig. 2','lateral','514.5',46,668,'nm',note='AFM period.'),
          row('Lowell 2016 / 4+1 axial, 43-degree mode','Lowell 2016 §4.1 Fig. 2','axial','514.5',43,1967,'nm',note='Published mean axial measurement; component reference, not full asymmetric lattice period.'),
          row('Lowell 2016 / 4+1 axial, 46-degree mode','Lowell 2016 §4.1 Fig. 2','axial','514.5',46,1967,'nm',note='Same measurement, other mode; not an independent sample.'),
          row('Lowell 2016 / 4+1 transverse x, 43-degree mode','Lowell 2016 §4.1 Fig. 2','lateral','514.5',43,690,'nm',note='Published mean x period, asymmetric modes; no averaging.'),
          row('Lowell 2016 / 4+1 transverse x, 46-degree mode','Lowell 2016 §4.1 Fig. 2','lateral','514.5',46,690,'nm',note='Same measurement; second component reference.'),
          row('Lowell 2016 / 5+1 axial','Lowell 2016 §4.2 Fig. 3','axial','514.5',46,1723,'nm',note='SU-8 SEM; equal side angles.'),
          row('Lowell 2016 / 6+1 axial','Lowell 2016 §4.3 Fig. 4','axial','514.5',46,1741,'nm',note='SU-8 SEM; equal side angles.'),
          row('Lowell 2016 / 5+1 radial','Lowell 2016 §4.2 Fig. 3','radial5','514.5',46,689,'nm',note='DPHPA AFM; different material, separate observable.')]
    # Full phase register audit of the fifth-direction witness, with threshold 2.
    # Cosine sign follows directly from the turn sector (strict quarter-turn).
    # This is not a continuous volume or a counting capacity on the product.
    positive={j for j in range(17) if 4*min(j,17-j)<17}
    fibres=Counter((x+y-z)%17 for x,y,z in product(range(17),repeat=3))
    assert len(fibres)==17 and set(fibres.values())=={289}
    support=[(x,y,z) for x,y,z in product(range(17),repeat=3) if (x+y-z)%17 in positive]
    cube={(x,y,z) for x,y,z in product((0,8),repeat=3)}
    target={p for p in cube if sum(t==8 for t in p)%2==0}
    assert set(support)&cube==target
    assert len(support)==2601
    # HFP17 previous four-mode bound: absolute pair phase error in turns.
    # |delta I| <= 4*pi*sum_(i<j) weights*epsilon; use pi < 22/7.
    # This is a conservative sufficient budget under phase errors alone.
    margin=Q(297,1616); weight_sum=Q(600,101)
    eps=margin/(4*Q(22,7)*weight_sum)
    assert 4*Q(22,7)*weight_sum*eps==margin
    # Gao: maximum of region-average errors; NOT a local pointwise bound.
    gao={'aligned_phase_turns':'61/1000','unaligned_phase_turns':'361/1000',
         'aligned_in_Q17_steps':str(Q(61,1000)*17),
         'unaligned_in_Q17_steps':str(Q(361,1000)*17),
         'source':'Gao et al. 2025 Fig. 6; region-average phase difference; not HFP-17.'}
    return {'python':platform.python_version(),'source_kind':'published numerical observations, no raw arrays',
            'rows':rows,'fifth_mode':{'positive_residues':sorted(positive),'fibres':dict(sorted(fibres.items())),
            'active':len(support),'atlas_size':4913,'target_cube':sorted(target),
            'outside_cube_active':len(support)-len(target),'fraction_active':str(Q(len(support),4913))},
            'phase_budget':{'absolute_pair_error_turns_strict_upper':str(eps),
                            'in_Q17_steps':str(17*eps),'optical_path_nm_at_405':str(405*eps),
                            'conditional_on':'four equal amplitude plane waves, phase errors only; sufficient, not necessary'},
            'published_metrology':gao,'fit_parameters':0,'raw_arrays_imported':False}


if __name__=='__main__':
    print(json.dumps(run(),indent=2,ensure_ascii=False))
