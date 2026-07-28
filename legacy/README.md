# legacy/ — does not run

These 13 scripts were written against an earlier API that no longer exists in
this repository. They import modules (`gai_unified`, `modules.core`,
`legacy.gai_final_boss`, `gai_moe`, `gai`) that are not present, so **every one
of them raises `ImportError` on execution.**

They are kept rather than deleted because several produced figures and claims
that circulated with the project, and it should be possible to see what
generated them. Nothing here is referenced by `README.md`, and no measurement in
`CHANGES.md` comes from this directory.

| file | what it was for |
|---|---|
| `run_benchmark.py`, `compare_moe.py`, `cost_efficiency_benchmark.py` | benchmark harnesses |
| `nas_comparison_mnist.py`, `test_integrated_efficiency.py` | NAS comparison, efficiency tests |
| `run_ecoli_judge.py`, `run_symbolic_judge.py` | biology / symbolic use cases |
| `run_ctz_*.py` | control use case |
| `chaos_benchmark.py`, `double_pendulum_benchmark.py`, `run_lorenz_experiment.py` | physics use cases |

**One of these matters historically.** The old README quoted an E. coli result
of 96.5% against a baseline's 88.2%. The committed artifact
(`results/comparative_benchmark.png`) shows the opposite — XGBoost at ~95%
against this model at ~87%, with zero precision and zero recall, i.e. a collapse
to majority-class prediction. That claim was removed from the README rather than
restated. See the Status section there.

To revive any of these, the missing modules would need restoring and the
numbers re-measured on a held-out split. Do not cite results from this directory.
