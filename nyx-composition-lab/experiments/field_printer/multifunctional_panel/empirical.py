"""Exact arithmetic on published test settings, not specimen-level measurements.
Crespo-Miguel et al., Nature Communications 16, 1359 (2025).
Gauge thermal relation: Thermodynamique de Jauge v2.2, theorem 3.
"""
from fractions import Fraction as Q
import json

def run():
    # The paper's Methods give 125,187.5,250; Results give 150,187.5,250.
    # Preserve both; do not silently fix the raw-source record.
    length=Q(40,1000)
    voltages=[Q(5),Q(15,2),Q(10)]
    derived=[v/length for v in voltages]
    assert derived==[Q(125),Q(375,2),Q(250)]
    # Ecell/kB = (E0+D*xi^3)/kB = Dx*T under the source constitutive law.
    # Kelvin conversion at the instrument boundary, no xi or E0 invented.
    rows=[]
    for c in (25,35,45):
        t=Q(c)+Q(27315,100); u=Q(17,8)*t
        rows.append({'celsius':c,'kelvin':str(t),'Ecell_over_kB_K':str(u)})
    assert Q(rows[1]['Ecell_over_kB_K'])-Q(rows[0]['Ecell_over_kB_K'])==Q(85,4)
    return {'doi':'10.1038/s41467-025-56707-y','scope':'published settings and source arithmetic, not predictive validation',
            'electrode_spacing_m':str(length),'voltage_V':[str(v) for v in voltages],
            'field_from_methods_V_per_m':[str(e) for e in derived],
            'field_in_results_V_per_m':['150','375/2','250'],
            'first_setting_discrepancy_V_per_m':'25','thermal_settings':rows,
            'coupled_test':{'field_V_per_m':'375/2','preheat_s':300,'strain_rate_per_s':'1/1000'},
            'reported_sequence':['electrothermal stabilization','tensile loading','localized loss of conductive pathways','cooling'],
            'raw_workbook_acquired':False,
            'STOP':['first field setting requires original records','sample-specific pressure, contact conductance and fracture calibration missing','observed temperature is not native activation density without xi and E0']}
if __name__=='__main__': print(json.dumps(run(),indent=2))
