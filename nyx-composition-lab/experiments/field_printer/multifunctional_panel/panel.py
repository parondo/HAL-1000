"""PANEL-01 — finite Nyx composition and contact crash-test witness.
Synthetic local roles; no constitutive conductivity, strength or heat-flow prediction.
Standard library. All capacities are local, never summed across the panel.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path

N=17
S={1,2,3,4,5,8,9,12,13,14,15,16}
V=set(range(6))
PATCHES=[{i,i+1,i+2} for i in range(4)]
LAM=Q(1,2)  # predeclared synthetic witness, not calibrated material
BUDGET=Q(1,4)
PORTS={'supply':(0,2),'return':(3,5),'thermal':(1,4)}

def edge(a,b):
    return tuple(sorted((a,b)))

def edges(a):
    return set(combinations(sorted(a),2))

def capacity(a):
    n=len(a)
    assert set(a)<=set(range(N))
    return (1-LAM)*Q(n,N)+LAM*Q(n*(34-n),N*N)

def compose(f,g):
    return tuple(max(min(f[u],g[(z-u)%N]) for u in range(N)) for z in range(N))

def reachable(es,a,alive):
    seen={a} if a in alive else set()
    while True:
        nxt=seen|{b for x,y in es for u,b in [(x,y),(y,x)] if u in seen and b in alive}
        if nxt==seen: return seen
        seen=nxt

def satisfied(ch,alive=V):
    e=ch['supply']|ch['return']|ch.get('leak',set())
    return {'structure':alive==V and reachable(ch['structure'],0,alive)==V,
            'supply':2 in reachable(e,0,alive),
            'return':5 in reachable(e,3,alive),
            'electrical_isolation':not (reachable(e,0,alive)&{3,4,5}) and not (reachable(e,3,alive)&{0,1,2}),
            'thermal_route':4 in reachable(ch['thermal'],1,alive),
            'thermal_keepout':not (reachable(ch['thermal'],1,alive)&{0,5})}

def remove(ch,es=(),nodes=()):
    return {k:{e for e in v if e not in es and not(set(e)&set(nodes))} for k,v in ch.items()}

def crash(ch,struct):
    out={}
    for n in (1,2):
        rows=[]
        for fault in combinations(sorted(struct),n):
            state=satisfied(remove(ch,fault))
            rows.append({'removed':fault,'checks':state,'pass':all(state.values())})
        out[str(n)]={'total':len(rows),'pass':sum(r['pass'] for r in rows),'rows':rows}
    return out

def minimal_robust(candidates,a,b):
    # Only declared contacts in the prescribed channel are candidates.
    for k in range(len(candidates)+1):
        good=[]
        for es in combinations(sorted(candidates),k):
            e=set(es)
            if b in reachable(e,a,V) and all(b in reachable(e-{x},a,V) for x in e):
                good.append(es)
        if good: return {'minimum_edges':k,'solutions':good}
    raise ValueError('STOP: no robust route')

def run():
    # Local composition of {0,1} with itself; transport the resulting
    # unit-valued profiles by a translation into four overlapping envelopes.
    f=tuple(Q(int(i in {0,1})) for i in range(N))
    h=compose(f,f)
    assert {i for i,x in enumerate(h) if x}=={0,1,2}
    for p in PATCHES:
        assert capacity(p)==Q(72,289)<BUDGET
        assert all((a-b)%N in S for a,b in edges(p))
    # Shared addresses carry identical structural profile values (=1).
    # Separate channel roles are declared, not inferred from scalar density.
    struct=set().union(*(edges(p) for p in PATCHES))
    plus=edges({0,1,2}); minus=edges({3,4,5})
    heat={(1,2),(2,4),(1,3),(3,4)}
    redundant={'structure':struct,'supply':plus,'return':minus,'thermal':heat}
    sparse={'structure':struct,'supply':{(0,1),(1,2)},'return':{(3,4),(4,5)},'thermal':{(1,2),(2,4)}}
    assert all(v<=struct for v in redundant.values())
    assert all(satisfied(redundant).values()) and all(satisfied(sparse).values())
    cases={k:crash(ch,struct) for k,ch in [('sparse',sparse),('redundant',redundant)]}
    assert cases['redundant']['1']['pass']==len(struct)
    assert cases['sparse']['1']['pass']<len(struct)
    # Shared failure is removed in every channel, not only in the rail.
    vertices=[{'removed':i,'checks':satisfied(remove(redundant,nodes={i}),V-{i})} for i in sorted(V)]
    leaks=[]
    for e in sorted(struct-plus-minus):
        ch={**redundant,'leak':{e}}
        leaks.append({'added':e,'checks':satisfied(ch)})
    assert all(not x['checks']['electrical_isolation'] for x in leaks)
    thermal_leaks=[]
    for e in sorted(e for e in struct if set(e)&{0,5} and set(e)&{1,2,3,4}):
        ch={**redundant,'thermal':heat|{e}}
        thermal_leaks.append({'added':e,'checks':satisfied(ch)})
    assert all(not x['checks']['thermal_keepout'] for x in thermal_leaks)
    # Check all native affine contact-preserving changes of chart.
    transport=0
    for a,b in product((1,4,13,16),range(17)):
        m=lambda x:(a*x+b)%N
        mv={m(i) for i in V}
        for p in PATCHES:
            pp={m(i) for i in p}
            assert capacity(pp)==capacity(p)
            assert all((x-y)%N in S for x,y in edges(pp))
        for ch in (sparse,redundant):
            e={edge(m(x),m(y)) for x,y in ch['structure']}
            assert reachable(e,m(0),mv)==mv
            for role,(u,v) in PORTS.items():
                e={edge(m(x),m(y)) for x,y in ch[role]}
                assert m(v) in reachable(e,m(u),mv)
        transport+=1
    # Local native defects after loss of one site (not global panel measure).
    p=PATCHES[0]; q=p-{1}; c=capacity(q)/capacity(p)
    assert c==Q(49,72)
    return {'scope':'synthetic contact topology; not a physical panel or strength test',
            'local_capacity':str(capacity(PATCHES[0])),'patches':[sorted(p) for p in PATCHES],
            'structural_contacts':sorted(struct),'redundant_channels':{k:sorted(v) for k,v in redundant.items()},
            'crashes':cases,'node_crashes':vertices,'electrical_leaks':leaks,'thermal_leaks':thermal_leaks,
            'minimum_robust':{k:minimal_robust(redundant[k],*PORTS[k]) for k in PORTS},
            'transport_cases':transport,'local_loss':{'overlap':str(c),'defect':str(1-c),'symmetric_difference':str(capacity({1}))},
            'STOP':['pressure and electrical/thermal constitutive transfer not calibrated','physical impact load and failure thresholds unavailable','full Morcillien and hardware commands not inferred']}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
