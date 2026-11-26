"""FEUILLET III — NATIVE GAUGE CONTRACT / AUDIT ONLY.
HUMAN: preserve the numerical witnesses and enforce their type boundary.
AI: no physical calibration or active material model is created here.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).parent
ARCHIVE = ROOT / 'empirical_archive'


def run():
    checks = {}
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = 'PASS'

    optical = json.loads((ARCHIVE / 'results.json').read_text())
    material = json.loads((ARCHIVE / 'material_results.json').read_text())
    # Corpus: Frustration Controlee, corrected addendum, not the old mean.
    resonance = Q(5, 8) * Q(13, 16)
    check('corrected_resonance_square', resonance == Q(65, 128))
    check('retired_mean_not_same', Q(23, 32)**2 - resonance == Q(9, 1024))
    check('distinct_poisson_regimes', Q(1, 2*(1+1)) == Q(1, 4)
          and Q(13, 2*(13+10)) == Q(13, 46))
    # Exact power relation from the composite article, NOT a command law.
    r = Q(material['inverse_target_45_GPa']['r'])
    lo, hi = map(Q, material['inverse_target_45_GPa']['f_interval'])
    check('local_composition_power_bracket', lo**5 <= r**8 <= hi**5)
    for i, w in enumerate(material['exact_witnesses']):
        check(f'composition_witness_{i:02}', Q(w['f'])**5 == Q(w['r'])**8)
    # Reproduce source-table arithmetic only. Not a Morcillien or an independent QEC.
    for i, row in enumerate(material['poisson']):
        obs, pred, pm = map(Q, (row['reported'], row['source_prediction'],
                               row['reported_plusminus']))
        check(f'poisson_source_row_{i+1}', obs-pred == Q(row['signed_defect'])
              and (obs-pm <= pred <= obs+pm) == row['in_reported_interval'])
    check('fifth_mode_support', optical['fifth_mode']['active'] == 9*17**2)
    check('fifth_mode_excess', optical['fifth_mode']['outside_cube_active'] == 9*17**2-4)
    check('no_raw_arrays_claim', not optical['raw_arrays_imported'])
    # A finite admissible J17 witness, NOT the capacity of the optical atlas.
    # PV49 at lambda=1; do not extend to J17^3 or infer physical measure.
    def mu(s):
        if not set(s) <= set(range(17)):
            raise ValueError('STOP: outside J17')
        k = len(set(s))
        return Q(k*(34-k), 289)
    a, b = {0, 1}, {0}
    check('finite_union_budget', mu(a | b) == Q(64, 289) < Q(1, 4))
    overlap = mu(a & b) / mu(a | b)
    check('finite_overlap', overlap == Q(33, 64))
    check('finite_defects_distinct', 1-overlap == Q(31, 64)
          and mu(a ^ b) == Q(33, 289))
    # A physical transport certificate cannot be manufactured by a JSON flag.
    stopped = {
        'global_bulk_fraction_to_local_f': 'STOP: transport not established',
        'optical_address_count_to_capacity': 'STOP: no calibrated capacity on J17^3',
        'optical_dose_to_material_state': 'STOP: process transfer not calibrated',
        'instrument_distance_to_gauge_distance': 'STOP: typed recapture missing',
        'full_morcillien': 'STOP: calibrated centers, distance and capacity required',
        'hardware_execution': 'STOP: outside scope of this audit'
    }
    return {'scope':'source arithmetic and finite typed witness only',
            'checks': checks, 'passed':len(checks), 'stops':stopped,
            'corrected_resonance_square':str(resonance),
            'finite_J17_witness':{'capacity_union':str(mu(a|b)),
                                  'overlap':str(overlap),
                                  'normalized_defect':str(1-overlap),
                                  'symmetric_difference_capacity':str(mu(a^b))},
            'archive_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(ARCHIVE.iterdir()) if p.is_file()}}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
