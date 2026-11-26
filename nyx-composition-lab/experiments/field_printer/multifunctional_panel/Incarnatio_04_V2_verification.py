"""FEUILLET IV — CONTROLES EXACTS, SANS AJUSTEMENT.
Rejoue le témoin PANEL-01, le contrôle du volume libre et les entrées publiées.
Aucun calcul de résistance physique, aucune donnée instrumentale inventée.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import json

checks = {}
def check(name, condition):
    if not condition: raise AssertionError(name)
    checks[name] = 'PASS'

# 1. Composition et adjunction sur un sous-domaine fini à trois adresses.
# Les témoins sont synthétiques, et ne sont pas un échantillonnage statistique.
N = 17
def comp(f,g):
    return tuple(max(min(f[u],g[(z-u)%N]) for u in range(N)) for z in range(N))
def res(k,h):
    return tuple(min(1 if k[v]<=h[(u+v)%N] else h[(u+v)%N] for v in range(N)) for u in range(N))
def prof(vals): return tuple(map(Q,vals))+(Q(0),)*(N-len(vals))
def le(f,g): return all(x<=y for x,y in zip(f,g))
f,g,gp=prof([1,Q(3,4)]),prof([1,Q(1,2)]),prof([1,0,Q(1,2)])
h=comp(f,g)
check('composition_admise',h==prof([1,Q(3,4),Q(1,2)]))
check('composition_rejetee',comp(f,gp)==prof([1,Q(3,4),Q(1,2),Q(1,2)]))
check('residu_large',res(f,h)==g)
check('residu_etroit',res(f,f)==prof([1]))
# Test fini indépendant de l'argument symbolique : valeurs 0,1/2,1 sur deux positions.
p=[prof(x) for x in product((0,Q(1,2),1),repeat=2)]
for i,(a,b,c) in enumerate(product(p,repeat=3)):
    check(f'adjonction_{i}',le(comp(a,b),c)==le(b,res(a,c)))
    check(f'associativite_{i}',comp(comp(a,b),c)==comp(a,comp(b,c)))

# 2. Capacité locale (paramètre de témoin déclaré, pas ajusté à des mesures).
def mu(a,lam=Q(1,2)):
    n=len(a); return (1-lam)*Q(n,17)+lam*Q(n*(34-n),289)
V=set(range(6)); patches=[set(range(i,i+3)) for i in range(4)]
check('capacite_locale',all(mu(a)==Q(72,289)<Q(1,4) for a in patches))
check('borne_lambda',mu(patches[0],Q(85,168))==Q(1,4))
check('capacite_non_globale',mu(V)>Q(1,4))
check('defaut_site',mu({0,2})/mu({0,1,2})==Q(49,72))
check('defaut_symetrique',mu({1})==Q(25,289))

# 3. Pannes de contacts partagés ; chaque canal reçoit le même défaut.
def edges(s): return set(combinations(sorted(s),2))
E=set().union(*(edges(a) for a in patches))
railp=edges({0,1,2}); railm=edges({3,4,5})
heat={(1,2),(2,4),(1,3),(3,4)}
models={'minimal':(E,{(0,1),(1,2)},{(3,4),(4,5)},{(1,2),(2,4)}),'redondant':(E,railp,railm,heat)}
def reach(es,a,alive=V):
    seen={a}&alive
    while True:
        nxt=seen|{b for x,y in es for u,b in [(x,y),(y,x)] if u in seen and b in alive}
        if nxt==seen:return seen
        seen=nxt
def evaluate(m,fault=(),alive=V,leak=(),thermal_leak=()):
    s,p,n,t=({e for e in es if e not in fault and set(e)<=alive} for es in m)
    el=p|n|set(leak);t|=set(thermal_leak)
    return [alive==V and reach(s,0,alive)==V,2 in reach(el,0,alive),5 in reach(el,3,alive),not(reach(el,0,alive)&{3,4,5}) and not(reach(el,3,alive)&{0,1,2}),4 in reach(t,1,alive),not(reach(t,1,alive)&{0,5})]
crashes={}
for name,m in models.items():
    crashes[name]={}
    for k in (1,2):
        rows=[{'removed':r,'requirements':evaluate(m,r),'pass':all(evaluate(m,r))} for r in combinations(sorted(E),k)]
        crashes[name][str(k)]={'pass':sum(r['pass'] for r in rows),'total':len(rows),'rows':rows}
for name,a,b in [('minimal',4,6),('redondant',9,28)]:
    check('crash_'+name,crashes[name]['1']['pass']==a and crashes[name]['2']['pass']==b)
check('site_deletions',all(not all(evaluate(models['redondant'],alive=V-{i})) for i in V))
check('electrical_leaks',all(not evaluate(models['redondant'],leak={e})[3] for e in E-railp-railm))
check('thermal_leaks',all(not evaluate(models['redondant'],thermal_leak={e})[5] for e in E if set(e)&{0,5} and set(e)&{1,2,3,4}))
S={1,2,3,4,5,8,9,12,13,14,15,16}
for a,b in product((1,4,13,16),range(17)):
    m=lambda x:(a*x+b)%17
    check(f'transport_{a}_{b}', all(mu({m(x) for x in p})==mu(p) for p in patches) and all((m(x)-m(y))%17 in S for x,y in E))

# 4. Audit des verres métalliques : toutes les entrées sont les valeurs imprimées.
base=Q(1,4); gap=Q(3,92); beta=Q(5,4)
rows=[]
for name,v,p,phi in [('Vitreloy 4',Q('0.3510'),Q('0.3506'),Q('0.08')),('Pd40Ni40P20',Q('0.3980'),Q('0.3962'),Q('0.12'))]:
    lo,hi=(p-base-gap)/beta,(p-base)/beta
    # r=delta/delta_ico ; diagnostic conditionnel à 0<=r<=1 et aux nombres exacts.
    r=1-(p-base-beta*phi)/gap
    rows.append({'material':name,'observed':str(v),'published_prediction':str(p),'nominal_phi':str(phi),'phi_interval_for_prediction':[str(lo),str(hi)],'implied_r':str(r),'range_at_nominal_phi':[str(base+beta*phi),str(base+gap+beta*phi)]})
check('vit_nominal_compatible',0<=Q(rows[0]['implied_r'])<=1)
check('pd_nominal_incompatible',Q(rows[1]['implied_r'])>1)
check('reduced_volume_ratio',Q('0.12')/Q('0.0117')==Q(400,39))
# External metrology identity, not imported into Nyx: isotropic K/G -> nu.
def nu(r): return (3*r-2)/(2*(3*r+1))
check('pd_acoustic_1986',nu(Q('4.56'))==Q(292,734))
# 5. Instrument boundary: voltage/spacing and conditional source-law thermal values.
fields=[v/Q(1,25) for v in (Q(5),Q(15,2),Q(10))]
check('field_conflict',fields==[125,Q(375,2),250] and 150-fields[0]==25)
thermal=[Q(17,8)*(Q(t)+Q('273.15')) for t in (25,35,45)]
check('thermal_difference',thermal[1]-thermal[0]==Q(85,4))
check('native_resonance',Q(5,8)*Q(13,16)==Q(65,128))
check('retired_mean',Q(23,32)**2-Q(65,128)==Q(9,1024))
check('local_composite_witnesses',all((Q(i,16)**8)**5==(Q(i,16)**5)**8 for i in range(17)))
# 6. FEUILLET IV V2 — AUDIT DOCUMENTAIRE, GRANDEURS EXTERNES DISTINCTES.
# Les chiffres arrondis sont traités comme rationnels pour les seuls contrôles.
# Aucune assimilation x_reduit = Phi_libre ; aucune recalibration.
span=Q('0.403')-Q('0.360')
relative=abs(Q('0.400')-Q('0.3962'))/Q('0.400')
pressure_dx=-Q('0.0025')*Q('0.75')
check('v2_published_span',span==Q(43,1000) and span/Q('0.360')==Q(43,360))
check('v2_relative_gap_1986',relative==Q(19,2000))
check('v2_pressure_reduced_volume',pressure_dx==-Q(3,1600))
check('v2_acoustic_inside_reported_interval',Q('0.393')<=nu(Q('4.56'))<=Q('0.403'))
report={'scope':'finite algebra and source arithmetic, not physical predictive validation','checks':checks,'passed':len(checks),'crashes':crashes,'glass':rows,'external_acoustic_nu':str(nu(Q('4.56'))),'fields':list(map(str,fields)),'thermal':list(map(str,thermal)),'stops':['volume libre native vs reduced free volume: no typed bridge','physical material interfaces and local measured fields missing','raw specimen arrays missing','no pressure or electrical constitutive transport independently recovered','physical printer not fabricated; no physical crash test']}
report['glass_documentary_v2']={'published_span':str(span),'relative_span_to_0.360':str(span/Q('0.360')),'relative_gap_to_0.400':str(relative),'pressure_delta_x_reduced':str(pressure_dx),'status':'source arithmetic only; no same-specimen pairing; no bridge to native Phi'}
if __name__=='__main__': print(json.dumps(report,indent=2,ensure_ascii=False))
