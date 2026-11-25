# Reproduction guide

Exploratory runs used Python 3.13, NumPy, SciPy, scikit-learn and mpmath; plotting and CSV analysis use Matplotlib and pandas. The independent point audit uses system `libmpfr.so.6` on Linux. Timing results depend on the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python code/heldout_resolution.py
python code/mpfr_point_audit.py
python code/paired_certification.py --case compact_big --n 32 --m 2 --method moment --out compact_moment.json
python code/paired_certification.py --case compact_big --n 32 --m 2 --method direct --out compact_direct.json
```

The held-out script writes generated CSVs under `code/`. Do not overwrite the archived raw results in `data/` when comparing output. Source-to-source parameter parity should be checked prior to treating the experiments as independently reproduced.

**Interpretation safeguards**

- A failed or timed-out certificate is inconclusive, not a proven change in critical-point topology.
- The MPFR audit covers sampled points inside reported enclosures; it does not constitute an independent formal proof of the full interval subdivision.
- Split-detection and some clustering results are based on numerical grid searches, not certified root counts.
- The largest anchor-count experiment kept the underlying geometry fixed to two compact groups.
- The primitive interval-evaluation speedup should not be interpreted as the total certification speedup.

See [the report](report/Quantum_Potential_Landscape_Study.md) for all methodological details.
