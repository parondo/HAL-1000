"""Finite Q17 optical adapter; not a full wave/photochemistry solver."""
from fractions import Fraction as Q
from itertools import product
from math import comb
import json
import platform

P = 17
# PHYSICAL INPUT: Lin, Baker, Menon, Nat Commun 17, 7175 (2026).
# Reported nominal wavelength and index at 405 nm; not HFP17 measurements.
WAVELENGTH_NM = Q(405)
INDEX = Q(1613, 1000)
STEP_NM = WAVELENGTH_NM / INDEX / P
# MODEL CHOICE: four coherent, equal-amplitude plane waves in a homogeneous
# isotropic medium; vector polarizations are transverse to their directions.
# Coordinates are local optical adapter only, not primitives of Nyx.
DIRECTIONS = ((0,0,101),(20,0,99),(-20,0,99),(0,20,99))
POLARIZATIONS = ((0,101,0),(0,101,0),(0,101,0),(0,99,-20))
WEIGHTS = tuple(tuple(Q(sum(a*b for a,b in zip(u,v)),101**2)
                      for v in POLARIZATIONS) for u in POLARIZATIONS)
MAXIMUM = sum(sum(row) for row in WEIGHTS)
GAP = Q(3,8)*Q(99,101)
GATE = MAXIMUM-GAP/2


def physical_point(j):
    """Local position in units of wavelength in the medium, from phase ratios."""
    u,v,w = (Q(a,P) for a in j)
    x = Q(101,40)*(u-v)
    z = -Q(101,4)*(u+v)
    y = (101*w+2*z)/20
    return x,y,z


def relative_phase(r):
    x,y,z = r
    return ((20*x-2*z)/101,(-20*x-2*z)/101,(20*y-2*z)/101)


def histogram(q):
    return tuple(sum((a-b)%P == k for a in q for b in q) for k in range(P))


def vector_histogram(q):
    h = [Q(0)]*P
    for i,a in enumerate(q):
        for k,b in enumerate(q):
            h[(a-b)%P] += WEIGHTS[i][k]
    return tuple(h)


def reduce_coeff(h):
    return tuple(x-h[-1] for x in h[:-1])


def recapture(a):
    """17 calibrated phase steps on a two-channel interferometer.
    Return exact coefficients of sum I_k z^(-k)/17 in K17.
    """
    out = [0]*P
    for k in range(P):
        for i,c in enumerate(histogram((0,a+k))):
            out[(i-k)%P] += c
    return tuple(Q(c,P) for c in reduce_coeff(out))


def bound_certificate():
    # p(t)=1+sum_{k=1}^8(z^k+z^-k), t=z+z^-1.
    # coefficients in increasing order; substitute t = s + 15/8.
    p = (1,-4,-10,10,15,-6,-7,1,1)
    shifted = [sum(Q(p[i])*comb(i,k)*Q(15,8)**(i-k)
                   for i in range(k,9)) for k in range(9)]
    assert all(c>0 for c in shifted)
    return shifted


def residue(r):
    if r.denominator%P == 0:
        return {'status':'STOP','reason':'noninvertible denominator'}
    return r.numerator*pow(r.denominator,-1,P)%P


def run():
    cert = bound_certificate()
    assert all(sum(x*x for x in d)==101**2 for d in DIRECTIONS)
    assert all(sum(x*x for x in d)==101**2 for d in POLARIZATIONS)
    assert all(sum(x*y for x,y in zip(d,e))==0
               for d,e in zip(DIRECTIONS,POLARIZATIONS))
    assert min(min(row) for row in WEIGHTS)==Q(99,101)
    # At least 3 unlike unordered pairs for any nonconstant quadruple.
    # Each contributes a loss strictly > 1/8; off-target I < 125/8.
    n = 0
    for j in product(range(P), repeat=3):
        assert relative_phase(physical_point(j)) == tuple(Q(a,P) for a in j)
        q = (0,*j)
        h = histogram(q)
        assert sum(h)==16
        assert (h[0]==16)==(j==(0,0,0))
        if j != (0,0,0):
            assert 16-h[0]>=6
        vh = vector_histogram(q)
        assert sum(vh)==MAXIMUM
        assert (vh[0]==MAXIMUM)==(j==(0,0,0))
        if j != (0,0,0):
            assert MAXIMUM-vh[0]>=6*Q(99,101)
        n += 1
    for a in range(P):
        expected = reduce_coeff(tuple(int(i==a) for i in range(P)))
        assert recapture(a)==expected
    # Published observations: section Results, Fig. 5a, Fig. 6.
    # These are reported values, not raw arrays or HFP17 predictions.
    observed = {'lateral_shift_um':80,'propagation_air_um':450,
                'structure_depth_um':720,'wall_width_um':6,
                'capillary_depth_um':[[1,169],[10,233],[20,243]],
                'effective_modulus_GPa':'5.7'}
    ratios = {'tilt_shift_over_travel':Q(80,450),
              'wall_over_depth':Q(6,720),
              'capillary_end_over_depth':Q(243,720)}
    return {'python':platform.python_version(),
            'status':'conditional optical adapter; HFP17 not physically tested',
            'source':'https://doi.org/10.1038/s41467-026-73975-4',
            'published_observations':observed,
            'imported_ratios':{k:{'exact':str(v),'mod17':residue(v)} for k,v in ratios.items()},
            'optical_path_step_nm':str(WAVELENGTH_NM/P),
            'material_path_step_nm':str(STEP_NM),
            'address_checks':n,'recaptured_phase_states':P,
            'positive_shifted_coefficients':[str(c) for c in cert],
            'target_intensity':'16','off_target_upper_bound':'125/8',
            'synthetic_gate':'253/16','strict_uniform_error_bound':'3/16',
            'vector_maximum':str(MAXIMUM),
            'vector_off_target_upper_bound':str(MAXIMUM-GAP),
            'vector_gate':str(GATE),'vector_error_bound':str(GAP/2),
            'raw_arrays_imported':False}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
