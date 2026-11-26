"""Finite max-min composition witness, not calibrated material data.
J17 additive chart, explicit envelopes, native PV49 capacity family.
All synthetic choices are declared; no fitted coefficient or physical bridge.
"""
from fractions import Fraction as Q
from itertools import product
import json
N = 17
S = {1,2,3,4,5,8,9,12,13,14,15,16}
ZERO = (Q(0),)*N

def profile(entries):
    return tuple(Q(entries.get(i,0)) for i in range(N))

def support(f):
    return {i for i,v in enumerate(f) if v>0}

def active(f,t):
    return {i for i,v in enumerate(f) if v>=t}

def compose(f,k):
    return tuple(max(min(f[u],k[(z-u)%N]) for u in range(N)) for z in range(N))

def residual(k,g):
    return tuple(min(Q(1) if k[v]<=g[(u+v)%N] else g[(u+v)%N]
                     for v in range(N)) for u in range(N))

def le(f,g):
    return all(a<=b for a,b in zip(f,g))

def capacity(A,lam):
    assert A<=set(range(N)) and 0<=lam<=1
    n=len(A)
    return (1-lam)*Q(n,N)+lam*Q(n*(34-n),289)

def evaluate(f,k,Vf,Vk,lam,t=Q(1,2),budget=Q(1,4)):
    h=compose(f,k); V=Vf|Vk; out=support(h)
    reasons=[]
    if not support(f)<=Vf or not support(k)<=Vk: reasons.append('input envelope')
    if not out<=V: reasons.append('containment')
    if not active(f,t)|active(k,t)<=active(h,t): reasons.append('activation preservation')
    if capacity(V,lam)>budget: reasons.append('union budget')
    if any((a-b)%N not in S for a in out for b in out if a!=b): reasons.append('witness clique contract')
    return {'status':'STOP' if reasons else 'PASS','reasons':reasons,
            'profile':{str(i):str(h[i]) for i in sorted(out)},
            'active':sorted(active(h,t)), 'input_union':sorted(V),
            'capacity_union':str(capacity(V,lam)), 'outside_union':sorted(out-V)}

def run():
    A=profile({0:1,1:Q(3,4)}); B=profile({0:1,1:Q(1,2)})
    B2=profile({0:1,2:Q(1,2)}); V={0,1,2}; lam=Q(1,2)
    h=compose(A,B); tight=residual(A,A); wide=residual(A,h)
    assert compose(tight,A)==A and compose(wide,A)==h and le(B,wide)
    assert support(tight)=={0} and tight[0]==1
    good=evaluate(A,B,V,{0,1},lam)
    budget_fail=evaluate(A,B,V,{0,1},Q(1))
    displaced=evaluate(A,B2,V,{0,2},lam)
    assert good['status']=='PASS' and budget_fail['reasons']==['union budget']
    assert displaced['status']=='STOP' and displaced['outside_union']==[3]
    assert capacity({0,1},lam)==capacity({0,2},lam)
    # Restricted finite exhaustive check; proof in README covers full domain.
    candidates=[profile({0:a,1:b}) for a,b in product((Q(0),Q(1,2),Q(1)),repeat=2)]
    count=0
    for f,k,g in product(candidates,repeat=3):
        assert le(compose(f,k),g)==le(f,residual(k,g))
        assert compose(compose(f,k),g)==compose(f,compose(k,g))
        count+=1
    target={0,1}; predicted=support(h); union=target|predicted
    c=capacity(target&predicted,lam)/capacity(union,lam)
    assert c==Q(49,72) and capacity(target^predicted,lam)==Q(25,289)
    return {'scope':'synthetic finite profile algebra; no physical material calibration',
            'threshold':'1/2 (witness only, not Sigma*)','lambda':'1/2 (declared scenario, not fitted)',
            'accepted':good,'budget_rejection':budget_fail,'arrangement_rejection':displaced,
            'residual_tight':{str(i):str(tight[i]) for i in sorted(support(tight))},
            'residual_wide':{str(i):str(wide[i]) for i in sorted(support(wide))},
            'adjunction_cases':count,'associativity_cases':count,
            'native_comparison':{'overlap':str(c),'defect':str(1-c),'symmetric_difference_capacity':str(capacity(target^predicted,lam))},
            'stops':['physical density and phase transport not supplied','full Morcillien requires calibrated centers and angle','no inferred material modulus from synthetic profiles']}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
