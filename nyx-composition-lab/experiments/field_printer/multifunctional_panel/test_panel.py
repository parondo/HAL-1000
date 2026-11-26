"""Independent connectivity audit of all crash records."""
import unittest
from itertools import combinations
import panel


def components(es, alive):
    groups=[{a} for a in sorted(alive)]
    for a,b in es:
        x=next((g for g in groups if a in g),set())
        y=next((g for g in groups if b in g),set())
        if x and y and x!=y:
            groups.remove(x); groups.remove(y); groups.append(x|y)
    return groups


class TestPanel(unittest.TestCase):
    def test_all_crash_records(self):
        r=panel.run(); st=set(map(tuple,r['structural_contacts']))
        full={k:set(map(tuple,v)) for k,v in r['redundant_channels'].items()}
        sparse={'structure':st,'supply':{(0,1),(1,2)},'return':{(3,4),(4,5)},'thermal':{(1,2),(2,4)}}
        count=0
        for name,ch in [('redundant',full),('sparse',sparse)]:
            for k,records in r['crashes'][name].items():
                self.assertEqual(len(records['rows']),len(list(combinations(st,int(k)))))
                for row in records['rows']:
                    removed=set(row['removed'])
                    def connected(es,a,b):
                        return any({a,b}<=g for g in components(es-removed,panel.V))
                    expected={'structure':len(components(st-removed,panel.V))==1,
                              'supply':connected(ch['supply'],0,2),'return':connected(ch['return'],3,5),
                              'electrical_isolation':True,'thermal_route':connected(ch['thermal'],1,4),'thermal_keepout':True}
                    self.assertEqual(row['checks'],expected)
                    self.assertEqual(row['pass'],all(expected.values())); count+=1
        self.assertEqual(count,90)

    def test_separate_local_capacity(self):
        self.assertTrue(all(panel.capacity(p)<panel.BUDGET for p in panel.PATCHES))
        self.assertGreater(panel.capacity(panel.V),panel.BUDGET)

if __name__=='__main__': unittest.main()
