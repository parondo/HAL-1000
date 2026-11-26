"""Stream a detailed JSONL audit journal; exact values, no sampling."""
from itertools import product
import json
import bridge

for j in product(range(17),repeat=3):
    q=(0,*j)
    r=bridge.physical_point(j)
    h=bridge.vector_histogram(q)
    print(json.dumps({'address':j,'position_in_medium_wavelengths':[str(x) for x in r],
                      'relative_turns':[str(x) for x in bridge.relative_phase(r)],
                      'quadratic_coefficients':[str(x) for x in bridge.reduce_coeff(h)],
                      'maximal':h[0]==bridge.MAXIMUM}))
