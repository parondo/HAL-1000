"""FEUILLET III — NATIVE MATERIAL LAW / FINITE RATIONAL CERTIFICATES.

SOURCE: Morcillo, Composition morphologique, 18 September 2026;
Traité des matériaux de jauge, sections 2 and 5.6.
No external statistics. Published rounded entries are not raw observations.
"""
from fractions import Fraction as Q
import json
from pathlib import Path


def root_interval(power, degree, scale=10**9):
    """Finite rational bracket for 0 <= power <= 1, no root instantiated."""
    if not 0 <= power <= 1 or degree <= 0:
        raise ValueError('STOP: outside material envelope')
    lo, hi = 0, scale
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Q(mid, scale)**degree <= power:
            lo = mid
        else:
            hi = mid
    if Q(hi, scale)**degree == power:
        lo = hi
    elif Q(lo, scale)**degree == power:
        hi = lo
    a, b = Q(lo, scale), Q(hi, scale)
    assert a**degree <= power <= b**degree
    return a, b


def modulus_interval(E1, E2, f):
    if not E1 > E2 >= 0 or not 0 <= f <= 1:
        raise ValueError('STOP: invalid constituent or fraction')
    a, b = root_interval(f**5, 8)
    return E2 + (E1-E2)*a, E2 + (E1-E2)*b


def composition_interval(E1, E2, target):
    if not E1 > E2 >= 0 or not E2 <= target <= E1:
        raise ValueError('STOP: target outside constituent envelope')
    r = (target-E2)/(E1-E2)
    return root_interval(r**8, 5)


def run():
    E1, E2, f = Q('73.1'), Q('3.45'), Q(1,2)
    elo,ehi = modulus_interval(E1,E2,f)
    assert Q(45) <= elo <= ehi <= Q(50)
    # Native inverse example: target 45 GPa, not an instrument setting.
    a,b = composition_interval(E1,E2,Q(45))
    r = (Q(45)-E2)/(E1-E2)
    assert a**5 <= r**8 <= b**5
    # Finite sequence of exact rational witnesses from the power law.
    witnesses=[]
    for k in range(17):
        t=Q(k,16)
        f0,r0=t**8,t**5
        assert f0**5 == r0**8
        witnesses.append({'t':str(t),'f':str(f0),'r':str(r0)})
    # Source table values, not independently acquired RUS measurements.
    rows=[('i-Al-Pd-Mn','0.2489','0.0015',Q(1,4)),
          ('i-Al-Cu-Fe','0.2478','0.0020',Q(1,4)),
          ('Cr3Si','0.2842','0.0030',Q(13,46)),
          ('Fe alpha','0.2835','0.0025',Q(13,46)),
          ('W','0.2800','0.0020',Q(13,46)),
          ('Vitreloy 4','0.3510','0.0040',Q('0.3506')),
          ('Pd40Ni40P20','0.3980','0.0050',Q('0.3962'))]
    poisson=[]
    for name,value,spread,pred in rows:
        o,s=Q(value),Q(spread)
        poisson.append({'name':name,'reported':value,'reported_plusminus':spread,
                        'source_prediction':str(pred),'signed_defect':str(o-pred),
                        'relative_percent':str(100*(o-pred)/pred),
                        'in_reported_interval':o-s<=pred<=o+s})
    # External holdout: Seleznev et al. 2023, Fig. 3. Bulk flexural
    # modulus and effective volume fractions are NOT local f(x).
    # This is deliberately a test of the naive bulk substitution, not a
    # calibrated local constitutive test. Constituent values are references.
    bulk=[]
    for fraction, observed in [('0.36','89'),('0.48','129'),('0.60','147'),('0.35','142')]:
        lo,hi=modulus_interval(Q(370),Q(71),Q(fraction))
        o=Q(observed)
        bulk.append({'f_global':fraction,'observed_GPa':observed,
                     'predicted_interval_GPa':[str(lo),str(hi)],
                     'signed_residual_interval_GPa':[str(o-hi),str(o-lo)],
                     'signed_relative_percent_interval':[str(100*(o/hi-1)),str(100*(o/lo-1))],
                     'scope':'bulk substitution; distinct local/mesoscale geometry'})
        assert o<lo
    return {'scope':'source-table arithmetic and external bulk negative control; no raw material arrays',
            'bulk_negative_control':{'source':'https://doi.org/10.1038/s41598-023-43514-y','E1_ref_GPa':370,'E2_ref_GPa':71,'rows':bulk},
            'source':'https://www.fdaa.fr/2026/09/18/composition-morphologique/',
            'glass_epoxy':{'E1_GPa':str(E1),'E2_GPa':str(E2),'f':str(f),
                'predicted_interval_GPa':[str(elo),str(ehi)],
                'source_reported_range_GPa':['45','50'],
                'source_rounded_prediction_GPa':'48.7'},
            'inverse_target_45_GPa':{'r':str(r),'f_interval':[str(a),str(b)]},
            'exact_witnesses':witnesses,'poisson':poisson,
            'notes':['Glass/epoxy range and Poisson observations are reported in the corpus; primary specimen data not recovered.',
                     'Metallic-glass predictions use a calibrated beta_v; not an independent parameter-free test.',
                     'Distinct Lame regimes lambda=mu and lambda/mu=13/10 must remain typed.',
                     'A field command to independently measured f, delta, chi_c is required; optical dose is not f.']}

if __name__=='__main__':
    print(json.dumps(run(),indent=2,ensure_ascii=False))
