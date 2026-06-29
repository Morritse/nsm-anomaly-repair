# nsm anomaly repair

`nsm` is an explainable symbolic engine for diagnosing and repairing anomalous
\(U(1)_X\) extensions of the Standard Model.

This public repository is scoped to the CPC submission:

> `nsm`: explainable menu-relative anomaly repair for \(U(1)_X\) extensions of
> the Standard Model

It contains the Paper 1 manuscript, the anomaly/repair code path, tests, and
reproduction scripts only. Other private research artifacts are intentionally
not part of this submission repository.

## Repository layout

```text
nsm/                  Python package: anomaly, derivation, repair, and structural helpers
scripts/repair_map.py repair-census command used by the manuscript
tests/                validation and regression tests for this package
paper/                manuscript source/PDF and submission materials
docs/                 short scope notes and B-L case study
figures/              generated illustrative HTML/SVG outputs
```

## Install and test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

The package uses the convention \(Q_{\rm em}=T_3+Y/2\), so the quark doublet
has \(Y=1/3\).

## Reproduce the repair census

The manuscript census table was generated with:

```bash
.venv/bin/python scripts/repair_map.py \
  --radius 2 --max-added 6 --timeout-ms 8000 \
  --output csv --workers 16
```

Expected CSV SHA-256:

```text
eca2f579baef8d91f11fcbe79273b6e907dd13d001136134c7487e33f9f4f244
```

## Build the manuscript

```bash
make -C paper
```

This rebuilds `paper/paper.pdf` and refreshes the submission-named copy
`paper/nsm-anomaly-repair.pdf`.

## Scope

`nsm` establishes structural and quantum-number consistency. It does not compute
cross sections, decay rates, detector significance, scalar sectors, or physical
observables. Returned repairs are anomaly-cancelling rational witnesses within
the declared matter menu and cost ordering.

## License

MIT License. See [LICENSE](LICENSE).
