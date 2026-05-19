#!/usr/bin/env python3
"""
verify_results.py
=================

Quick verification script for reviewers of:

    H. Ugail and N. Howard, "Symmetry-Organised Complexity in Quantum Neural
    Networks", MDPI Symmetry, 2026 (to appear).

Loads the precomputed result tables in `Results/` and confirms that every
headline number reported in the manuscript is reproducible from those tables.
Takes a few seconds, needs no GPU, and prints a clean PASS/FAIL summary.

Usage:
    python verify_results.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "Results"
TOL = 5e-3   # absolute tolerance for floating-point comparisons (3-decimal manuscript values)

# -----------------------------------------------------------------------------
# Check infrastructure
# -----------------------------------------------------------------------------

class CheckRunner:
    def __init__(self) -> None:
        self.n_pass = 0
        self.n_fail = 0
        self.failures: list[str] = []

    def check(self, name: str, observed: float, expected: float, tol: float = TOL) -> None:
        ok = math.isfinite(observed) and abs(observed - expected) <= tol
        if ok:
            self.n_pass += 1
            status = "PASS"
        else:
            self.n_fail += 1
            self.failures.append(name)
            status = "FAIL"
        delta = observed - expected
        print(f"  [{status}] {name:<70s}  observed={observed:.4f}  expected={expected:.4f}  delta={delta:+.4f}")

    def check_eq(self, name: str, observed, expected) -> None:
        ok = observed == expected
        if ok:
            self.n_pass += 1
            status = "PASS"
        else:
            self.n_fail += 1
            self.failures.append(name)
            status = "FAIL"
        print(f"  [{status}] {name:<70s}  observed={observed!r}  expected={expected!r}")

    def summary(self) -> int:
        total = self.n_pass + self.n_fail
        print()
        print("=" * 80)
        if self.n_fail == 0:
            print(f"ALL {total} CHECKS PASSED.")
            return 0
        print(f"{self.n_pass} of {total} checks passed; {self.n_fail} failed:")
        for f in self.failures:
            print(f"  - {f}")
        return 1


# -----------------------------------------------------------------------------
# Result-table loaders
# -----------------------------------------------------------------------------

def _require(path: Path) -> Path:
    if not path.exists():
        print(f"ERROR: required file not found: {path}", file=sys.stderr)
        sys.exit(2)
    return path


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "toy":         pd.read_csv(_require(RESULTS_DIR / "toy_examples_summary.csv")),
        "trained":     pd.read_csv(_require(RESULTS_DIR / "trained_task_summary.csv")),
        "per_seed":    pd.read_csv(_require(RESULTS_DIR / "trained_task_per_seed.csv")),
        "sensitivity": pd.read_csv(_require(RESULTS_DIR / "sensitivity_summary.csv")),
    }
    return tables


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_toy_examples(c: CheckRunner, toy: pd.DataFrame) -> None:
    """Check headline composite-index values for the six toy configurations."""
    print("\n[1/4] Toy examples (Section 6, Table 2, Figures 1-4)")
    expected = {
        ("U(1)",  "collapsed"):            ("Psi_G", 0.000),
        ("U(1)",  "symmetry-organised"):   ("Psi_G", 0.400),
        ("U(1)",  "symmetry-breaking"):    ("Psi_G", 0.361),
        ("SU(2)", "collapsed"):            ("Psi_G", 0.000),
        ("SU(2)", "symmetry-organised"):   ("Psi_G", 0.209),
        ("SU(2)", "symmetry-breaking"):    ("Psi_G", 0.145),
    }
    for (sym, model), (col, val) in expected.items():
        row = toy[(toy["Symmetry"] == sym) & (toy["Model"] == model)]
        if row.empty:
            c.check_eq(f"toy[{sym}, {model}] present", False, True)
            continue
        c.check(f"toy[{sym:5s}, {model:20s}].{col}", float(row[col].iloc[0]), val)

    # Compliance penalty on U(1) breaking trajectory (resolves the previous near-tie)
    row = toy[(toy["Symmetry"] == "U(1)") & (toy["Model"] == "symmetry-breaking")]
    c.check("toy[U(1),  symmetry-breaking   ].C_G",     float(row["C_G"].iloc[0]),     0.732)
    c.check("toy[U(1),  symmetry-breaking   ].Delta_G", float(row["Delta_G"].iloc[0]), 0.104)

    # Organised-to-breaking gaps
    u1_o = float(toy[(toy["Symmetry"] == "U(1)")  & (toy["Model"] == "symmetry-organised")]["Psi_G"].iloc[0])
    u1_b = float(toy[(toy["Symmetry"] == "U(1)")  & (toy["Model"] == "symmetry-breaking")] ["Psi_G"].iloc[0])
    su_o = float(toy[(toy["Symmetry"] == "SU(2)") & (toy["Model"] == "symmetry-organised")]["Psi_G"].iloc[0])
    su_b = float(toy[(toy["Symmetry"] == "SU(2)") & (toy["Model"] == "symmetry-breaking")] ["Psi_G"].iloc[0])
    c.check("organised - breaking gap, U(1)",  u1_o - u1_b, 0.039)
    c.check("organised - breaking gap, SU(2)", su_o - su_b, 0.064)


def check_trained_task(c: CheckRunner, trained: pd.DataFrame) -> None:
    """Check trained-task test accuracy and composite-index ordering across the three ansatze."""
    print("\n[2/4] Trained U(1)-compatible classification task (Section 7, Figures 5-7)")
    expected_acc = {
        "trained equivariant":     ("Test_acc_mean", 0.880),
        "trained hybrid":          ("Test_acc_mean", 0.755),
        "trained non-equivariant": ("Test_acc_mean", 0.425),
    }
    for model, (col, val) in expected_acc.items():
        row = trained[trained["Model"] == model]
        if row.empty:
            c.check_eq(f"trained[{model}] present", False, True)
            continue
        c.check(f"trained[{model:25s}].{col}", float(row[col].iloc[0]), val)

    expected_psi = {
        "trained equivariant":     0.527,
        "trained hybrid":          0.485,
        "trained non-equivariant": 0.425,
    }
    for model, val in expected_psi.items():
        row = trained[trained["Model"] == model]
        if row.empty: continue
        c.check(f"trained[{model:25s}].Psi_G_mean", float(row["Psi_G_mean"].iloc[0]), val)

    # Ranking: equivariant > hybrid > non-equivariant on both test accuracy and Psi_G
    acc = {m: float(trained[trained["Model"] == m]["Test_acc_mean"].iloc[0])
           for m in ["trained equivariant", "trained hybrid", "trained non-equivariant"]}
    psi = {m: float(trained[trained["Model"] == m]["Psi_G_mean"].iloc[0])
           for m in ["trained equivariant", "trained hybrid", "trained non-equivariant"]}
    c.check_eq("test accuracy ordering equivariant > hybrid > non-equivariant",
               acc["trained equivariant"] > acc["trained hybrid"] > acc["trained non-equivariant"], True)
    c.check_eq("Psi_G          ordering equivariant > hybrid > non-equivariant",
               psi["trained equivariant"] > psi["trained hybrid"] > psi["trained non-equivariant"], True)


def check_invariance(c: CheckRunner, per_seed: pd.DataFrame) -> None:
    """Check parameter-independence of Psi_G for the exactly equivariant ansatz (Remark 2)."""
    print("\n[3/4] Parameter-independence of Psi_G on the equivariant ansatz (Remark 2)")
    eq = per_seed[per_seed["Kind"] == "equivariant"]
    psi_vals = eq["Psi_G"].astype(float).tolist()
    n_seeds = len(psi_vals)
    c.check_eq("equivariant ansatz: number of seeds in per-seed table", n_seeds, 5)
    if n_seeds >= 2:
        spread = max(psi_vals) - min(psi_vals)
        c.check("equivariant Psi_G max - min across seeds (expected exactly 0)", spread, 0.0, tol=1e-9)

    # Hybrid and non-equivariant should have NON-zero seed dispersion in Psi_G
    for kind in ("hybrid", "non_equivariant"):
        psi_kind = per_seed[per_seed["Kind"] == kind]["Psi_G"].astype(float).tolist()
        if len(psi_kind) >= 2:
            spread = max(psi_kind) - min(psi_kind)
            ok = spread > 1e-6
            status = "PASS" if ok else "FAIL"
            if ok:
                c.n_pass += 1
            else:
                c.n_fail += 1
                c.failures.append(f"{kind} Psi_G shows non-zero seed dispersion")
            print(f"  [{status}] {kind:<70s}  Psi_G max - min across seeds = {spread:.4f}  (expected > 1e-6)")


def check_sensitivity(c: CheckRunner, sens: pd.DataFrame) -> None:
    """Check sensitivity headline values from Section 7.3, Table 3."""
    print("\n[4/4] Weight-and-sharpness sensitivity (Section 7.3, Table 3, Figure 8)")
    expected_pct = {
        # (case, gamma) -> expected percentage of weight settings on the simplex
        ("U(1) toy", 0):  27.27,
        ("U(1) toy", 1):  38.53,
        ("U(1) toy", 2):  55.41,
        ("U(1) toy", 3):  80.52,
        ("U(1) toy", 5): 100.00,
        ("SU(2) toy", 0):  96.97,
        ("SU(2) toy", 1):  99.57,
        ("SU(2) toy", 2): 100.00,
        ("SU(2) toy", 3): 100.00,
        ("SU(2) toy", 5): 100.00,
        ("trained U(1) task", 0):  83.12,
        ("trained U(1) task", 1): 100.00,
        ("trained U(1) task", 3): 100.00,
        ("trained U(1) task", 5): 100.00,
    }
    for (case, gamma), expected in expected_pct.items():
        row = sens[(sens["Case"] == case) & (sens["gamma"] == gamma)]
        if row.empty:
            c.check_eq(f"sensitivity[{case}, gamma={gamma}] present", False, True)
            continue
        observed = float(row["pct_organised_gt_breaking"].iloc[0])
        c.check(f"sensitivity[{case:18s}, gamma={gamma}].pct", observed, expected, tol=0.05)

    # Grid size: every row should use 231 weight settings (simplex grid of spacing 0.05).
    grid_sizes = sens["n_weight_settings"].unique().tolist()
    c.check_eq("sensitivity grid size: all rows use 231 simplex points",
               grid_sizes, [231])


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 80)
    print("Verification of headline numerical results")
    print("Symmetry-Organised Complexity in Quantum Neural Networks")
    print("=" * 80)

    tables = load_tables()
    c = CheckRunner()

    check_toy_examples (c, tables["toy"])
    check_trained_task (c, tables["trained"])
    check_invariance   (c, tables["per_seed"])
    check_sensitivity  (c, tables["sensitivity"])

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
