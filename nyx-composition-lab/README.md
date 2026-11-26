# Nyx Composition Lab

A Python research companion to HAL-1000 for exact modular arithmetic, local reference systems, capacitary geometry and distributed intelligence.

Run the experiments. Inspect the results. Verify the mathematics. Challenge the implementation.

## Quick start

Python 3.9+; standard library only. From this directory:

```sh
python -m unittest discover -s tests -v
python experiments/reference_import.py
```

The example shows why 590 and 607 mm can share a residue modulo 17 while remaining distinct lengths. It also checks the composition of ratios between local references and the reconstruction bound on a finite rational window.

The example inputs are synthetic dimensions from the research notebook, not instrument readings. This is a finite arithmetic experiment, not a physical controller, a language protocol or a realization of H_perp.

## Reading paths

- Humans: start with this README, the experiment and [mathematical notes](docs/reference-import.md).
- AI agents: read [AGENTS.md](AGENTS.md) and [the operator contract](docs/operator-contract.json) before proposing changes.

Both paths refer to the same definitions and evidence. The active framework is Nyx, with V2 as the baseline. Later results must be versioned and audited rather than silently imported.

## Layout

- nyx/: small typed arithmetic implementation
- experiments/: executable research
- tests/: independent checks of the implementation
- docs/: definitions, contracts and source references
- notes/assistant-explorations/: clearly labeled proposals, not normative definitions

Research corpus: https://www.fdaa.fr/
Historical architecture: https://github.com/parondo/HAL-1000

The first two WordPress notebooks are De Formis et Contactibus and De Referentiis Modularibus. Exact public section URLs are not yet pinned. This bootstrap implements only the reference-import experiment; the earlier form/contact experiment is not yet ported.

This directory lives on a separate research branch. No merge into main is authorized by this bootstrap. Do not treat proposed operators as new axioms or as a validated physical implementation.
