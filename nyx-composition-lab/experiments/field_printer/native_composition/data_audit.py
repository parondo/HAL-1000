"""Published-value arithmetic only; never modify a source observation.
No calibration, fitting or conversion of O measurements into Nyx capacity.
"""
from fractions import Fraction as Q
import json

def run():
    beta=Q(5,4); a=Q(1,4); b=Q(3,92)
    rows=[('Vitreloy 4','0.3510','0.0040','0.3506','0.08'),
          ('Pd40Ni40P20','0.3980','0.0050','0.3962','0.12')]
    glasses=[]
    for name,o,s,p,v in rows:
        o,s,p,v=map(Q,(o,s,p,v))
        # Conditional interval of free capacity if delta/delta_ico is in [0,1].
        lo,hi=(p-a-b)/beta,(p-a)/beta
        glasses.append({'material':name,'prediction_in_reported_interval':o-s<=p<=o+s,
                        'reported_prediction':str(p),'signed_defect':str(o-p),
                        'phi_required_interval_given_exact_beta_5_4':[str(lo),str(hi)],
                        'approximate_phi_as_exact_compatible':lo<=v<=hi})
    other=[('i-Al-Cu-Fe',Q('0.2478'),Q('0.0020'),Q(1,4)),
           ('W',Q('0.2800'),Q('0.0020'),Q(13,46))]
    outside=[{'material':n,'prediction_minus_interval_upper':str(p-o-s)} for n,o,s,p in other]
    composites=[]
    for infill,rounded,modulus in [('0.52','0.31',None),('0.60','0.36',89),('0.80','0.48',129),('1','0.60',147)]:
        composites.append({'nominal_infill':infill,'local_nominal_ceramic':'3/5',
                           'apparent_fraction_product':str(Q(infill)*Q(3,5)),
                           'reported_apparent_fraction':rounded,'reported_flexural_GPa':modulus})
    return {'scope':'source-table arithmetic, no raw specimen arrays, no fitting',
            'glass_rows':glasses,'other_interval_excess':outside,
            'free_volume_reduced_as_quenched_Pd':'117/10000',
            'nominal_gauge_to_reduced_value_ratio':str(Q('0.12')/Q('0.0117')),
            'alumina_aluminum':composites,'equal_infill_fracture_work_ratio':'8/5 (reported approximate)',
            'dual_color':{'UV_nm':365,'blue_nm':455,'before_postprocessing_kPa':{'UV':542,'blue':159},
                          'after_postprocessing_MPa':{'UV':201,'blue':127},
                          'before_UV_blue_ratio':str(Q(542,159)),
                          'after_UV_blue_ratio':str(Q(201,127)),
                          'UV_post_pre_ratio':str(Q(201000,542)),
                          'blue_post_pre_ratio':str(Q(127000,159))},
            'stops':['volume free normalization not established','effective nominal fraction is not local activation profile',
                     'two interpenetrating polymer networks are not proven complementary fractions',
                     'raw workbook download failed; Seleznev data available on request']}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
