"""Exact local mode audit; finite phase chart, not a resin simulation.
Human: can global phase programming select a prescribed four-point support?
AI: source P1a; no new Nyx primitive or physical H_perp bridge.
"""
from itertools import product
import json
from pathlib import Path

P = 17
D = ((0,0,101),(20,0,99),(-20,0,99),(0,20,99),(0,-20,99))
E = ((0,101,0),(0,101,0),(0,101,0),(0,99,-20),(0,99,20))
CUBE = tuple(product((0,8), repeat=3))

def character(j):
    x,y,z = j
    return (0,x,y,z,(x+y-z)%P)

def coeff(j, offset, channels):
    """Integer coefficients of 101^2 |E|^2 in Q[z]/Phi_17."""
    q = character(j)
    h = [0]*P
    for a in channels:
        for b in channels:
            h[(q[a]+offset[a]-q[b]-offset[b])%P] += sum(x*y for x,y in zip(E[a],E[b]))
    return tuple(c-h[-1] for c in h[:-1])

def sign(j):
    return (-1)**sum(x==8 for x in j)

def run():
    assert all(sum(x*x for x in d)==101**2 for d in D+E)
    assert all(sum(x*y for x,y in zip(d,e))==0 for d,e in zip(D,E))
    assert tuple(D[1][k]+D[2][k]-D[3][k] for k in range(3))==D[4]
    checks=0
    for q in product(range(P), repeat=3):
        offset=(0,*q,0)
        total=[0]*16
        for j in CUBE:
            for k,c in enumerate(coeff(j,offset,range(4))):
                total[k]+=sign(j)*c
        assert total==[0]*16
        checks+=1
    rows=[]
    for j in CUBE:
        a=character(j)[4]
        # Real optical embedding: positive cosine for residues 0,16;
        # negative for 8,9. Certified by position in a quarter-turn,
        # without numerical cosines or rounding.
        assert (a in (0,16) if sign(j)==1 else a in (8,9))
        rows.append({'address':j,'target':sign(j)==1,'new_exponent':a,
                     'intensity_coefficients_scaled_by_10201':coeff(j,(0,)*5,(0,4)),
                     'relative_to_threshold_2':'above' if sign(j)==1 else 'below'})
    return {'scope':'eight prescribed addresses; no outside-cube or cure claim',
            'four_mode_offset_checks':checks,'cube_samples':8,
            'five_mode_witness_active_channels':[0,4],
            'new_direction_numerator':D[4],'new_polarization_numerator':E[4],
            'direction_denominator':101,'normalized_threshold':2,'rows':rows}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
