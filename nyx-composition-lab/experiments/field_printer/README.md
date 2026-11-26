# HFP-17 — Field composition research prototype

**Status: P0 exact synthetic experiment; external audit requested.**
Research companion to HAL-1000, within the Nyx framework of Patrick Morcillo. Not a physical printer, material model or machine controller.

## Reproduce

Python 3.9+, standard library only. From this directory:

```sh
python HFP17_phase_simulation.py
python -m unittest -v
```

The first command regenerates the checked-in JSON. The original experiment was rerun locally before publication. The additional audit uses an alternative unordered-pair difference histogram. Run it independently; the maintainer has not yet accepted the external audit.

## Results

- 289 finite relational addresses, no metric assigned.
- Two-field maximum selects 17 addresses; three-field maximum selects exactly (4,13).
- 10,115 grouped invariance checks: 4,913 common phase; 289 conjugation; 4,624 Galois maximum predicate; 289 translations (each searches 289 addresses). These are loop counters, not an empirical sample size or a count of all assertions.
- Distinct conjugate phase configurations have identical quadratic traces. All 289 relative phase pairs yield 33 distinct traces: one fibre of size 1, sixteen of size 6, sixteen of size 12. This additional finite calculation was checked locally. It is not a physical chirality result.

## Reading order

1. [Mathematical specification](MATHEMATICS.md)
2. [FieldScript example](example.fieldscript) and [language status](FIELDSCRIPT.md)
3. [Experiment protocol and data sources](PROTOCOL.md)
4. [External audit brief](AUDIT.md)
5. [AI contract](contract.json) and [lab-wide agent rules](../../AGENTS.md)

Do not conflate rational residues, cyclotomic phases, Galois actions, or physical angles. No full Morcillien or physical bridge is claimed. No external dataset is vendored or executed.
