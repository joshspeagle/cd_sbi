"""Regenerate the run-dirs the PIT/Jacobian/calibration figures source from.

Runs four CDSBI replication configs into STABLE outputs/figure_data/<id>/ paths
(via the run_dir= override) so the figure manifest can pin them, and regenerates
the catastrophic-folding loss tail for E7.

Usage:  python tools/regen_figure_data.py
(Each CDSBI run is a few minutes on GPU; longer on CPU.)
"""
from __future__ import annotations

import os
import subprocess
import sys

# Make `tools` importable when run as `python tools/regen_figure_data.py`
# (the script dir, not the repo root, is on sys.path by default).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (experiment, extra overrides) -> stable run_dir.
# NOTE: do NOT pass method=cd_sbi as a top-level override — the experiment
# replication configs already set it, and a redundant top-level method= makes
# run.py's _build_flow fall back to the method's default 1D flow (AdditiveFlow1D)
# instead of the experiment's flow override (e.g. doubly_monotone for §8.4).
# Pass the intended flow explicitly per experiment to be safe.
# Pass BOTH flow=<x> and method.flow=<x>: cd_sbi.yaml hardcodes flow:additive_umnn,
# and run.py's _build_flow only fires the intended flow when method.flow agrees
# with the Hydra flow group (the cb1e08e guard). Overriding method.flow makes the
# experiment's flow win for CDSBI (verified: §8.4 -> DoublyMonotoneUMNN, loss ~1.0).
RUNS = [
    ("8_1_replication", ["flow=additive_umnn", "method.flow=additive_umnn"], "outputs/figure_data/8_1_cdsbi"),
    ("8_2_replication", ["flow=triangular_additive", "method.flow=triangular_additive"], "outputs/figure_data/8_2_cdsbi"),
    ("8_3_replication", ["flow=triangular_additive", "method.flow=triangular_additive"], "outputs/figure_data/8_3_cdsbi"),
    ("8_4_replication", ["flow=doubly_monotone", "method.flow=doubly_monotone"], "outputs/figure_data/8_4_cdsbi"),
]


def main():
    for exp, extra, run_dir in RUNS:
        cmd = [sys.executable, "-m", "cdsbi.experiments.run",
               f"experiment={exp}", "seed=0",
               f"run_dir={run_dir}", *extra]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    from tools._folding_regen import regen_folding
    regen_folding("outputs/figure_data/8_4_folding")
    print("done", flush=True)


if __name__ == "__main__":
    main()
