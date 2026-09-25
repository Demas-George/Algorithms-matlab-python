"""
Master Runner: Execute All 35 Metaheuristic Optimization Algorithms Sequentially
================================================================================
Runs every metaheuristic algorithm in the repository one after another on the
configured benchmark problem (defined in `problem.py`).

For each algorithm:
- Executes n_runs independent optimization runs
- Generates publication-ready figures in results/figures/
    (*_Main.png, *_Convergence.png, *_Boxplot.png)
- Generates 3-sheet styled Excel workbook in results/excel/*_Results.xlsx
- Reports console statistics and execution time

Upon completion of all algorithms:
- Automatically runs compare_algorithms.py to generate:
    * Console comparison leaderboard
    * results/excel/Problem_Results.xlsx
    * results/excel/Problem_Results.csv
    * results/figures/Comparison_Convergence_All.png (overlaid convergence curves)

Usage:
    python run_all_algorithms.py
    python run_all_algorithms.py --agents 30 --iter 500 --runs 30
    python run_all_algorithms.py --smoke   # Fast sanity check (agents=10, iter=25, runs=2)
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure current working directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from problem import run_experiment
from compare_algorithms import compare_algorithms

# ═════════════════════════════════════════════════════════════════════════════
# Import Optimizer Modules (All 35 Algorithms)
# ═════════════════════════════════════════════════════════════════════════════

# Base 22 Algorithms
import apo
import aro
import bto
import cpo
import do
import doa
import fpa
import gwo
import hho
import ho
import ivy
import koa
import mpa
import msfoa
import po
import pso
import rsa
import run as run_mod
import sma
import so
import woa
import zoa

# 13 New Algorithms
import ao
import eooa
import esc
import gto
import hba
import info
import jsa
import kma
import ooa
import parrot
import sca
import swo
import wso


# =============================================================================
# Adapters for Algorithms with Non-Standard Signatures or Return Orders
# =============================================================================

def apo_adapter(agents, iters, lb, ub, dim, fobj):
    pos, score, curve = apo.apo(agents, iters, lb, ub, dim, fobj)
    return score, pos, curve


def aro_adapter(agents, iters, lb, ub, dim, fobj):
    pos, score, curve = aro.aro(agents, iters, lb, ub, dim, fobj)
    return score, pos, curve


def cpo_adapter(agents, iters, lb, ub, dim, fobj):
    # CPO accepts (ub, lb) inverted in parameter order
    return cpo.cpo(agents, iters, ub, lb, dim, fobj)


def fpa_adapter(agents, iters, lb, ub, dim, fobj):
    pos, score, curve = fpa.fpa(agents, iters, lb, ub, dim, fobj)
    return score, pos, curve


def koa_adapter(agents, iters, lb, ub, dim, fobj):
    # KOA accepts (ub, lb) inverted in parameter order
    return koa.koa(agents, iters, ub, lb, dim, fobj)


def msfoa_adapter(agents, iters, lb, ub, dim, fobj):
    pos, score, curve = msfoa.msfoa(agents, iters, lb, ub, dim, fobj)
    return score, pos, curve


def po_adapter(agents, iters, lb, ub, dim, fobj):
    pos, score, curve = po.puma(agents, iters, lb, ub, dim, fobj)
    return score, pos, curve


def so_adapter(agents, iters, lb, ub, dim, fobj):
    # Snake Optimizer signature: (N, T, fobj, dim, lb, ub)
    pos, score, curve = so.so(agents, iters, fobj, dim, lb, ub)
    return score, pos, curve


# =============================================================================
# Algorithm Registry (Complete 35 Algorithms)
# =============================================================================
# Format: (algo_name, callable_func, full_display_name)
ALGORITHMS = [
    # ── Original 22 Algorithms ──────────────────────────────────
    ("GWO",    gwo.gwo,              "Grey Wolf Optimizer (GWO)"),
    ("APO",    apo_adapter,          "Artificial Protozoa Optimizer (APO)"),
    ("ARO",    aro_adapter,          "Artificial Rabbits Optimization (ARO)"),
    ("BTO",    bto.bto,              "Barrel Tumbleweed Optimizer (BTO)"),
    ("CPO",    cpo_adapter,          "Crested Porcupine Optimizer (CPO)"),
    ("DO",     do.do,                "Dandelion Optimizer (DO)"),
    ("DOA",    doa.doa,              "Dream Optimization Algorithm (DOA)"),
    ("FPA",    fpa_adapter,          "Flower Pollination Algorithm (FPA)"),
    ("HHO",    hho.hho,              "Harris Hawks Optimization (HHO)"),
    ("HO",     ho.ho,                "Hippopotamus Optimization (HO)"),
    ("IVY",    ivy.ivy,              "Ivy Algorithm (IVYA)"),
    ("KOA",    koa_adapter,          "Kepler Optimization Algorithm (KOA)"),
    ("MPA",    mpa.mpa,              "Marine Predators Algorithm (MPA)"),
    ("MSFOA",  msfoa_adapter,        "Modified Starfish Optimization Algorithm (MSFOA)"),
    ("PO",     po_adapter,           "Puma Optimizer (PO)"),
    ("PSO",    pso.pso,              "Particle Swarm Optimization (PSO)"),
    ("RSA",    rsa.rsa,              "Reptile Search Algorithm (RSA)"),
    ("RUN",    run_mod.run_opt,      "Runge Kutta Optimization (RUN)"),
    ("SMA",    sma.sma,              "Slime Mould Algorithm (SMA)"),
    ("SO",     so_adapter,           "Snake Optimizer (SO)"),
    ("WOA",    woa.woa,              "Whale Optimization Algorithm (WOA)"),
    ("ZOA",    zoa.zoa,              "Zebra Optimization Algorithm (ZOA)"),

    # ── 13 New Algorithms ───────────────────────────────────────
    ("AO",     ao.ao,                "Artemisinin Optimizer (AO)"),
    ("EOOA",   eooa.eooa,            "Enhanced Osprey Optimization Algorithm (EOOA)"),
    ("ESC",    esc.esc,              "Escape Optimization Algorithm (ESC)"),
    ("GTO",    gto.gto,              "Gorilla Troops Optimizer (GTO)"),
    ("HBA",    hba.hba,              "Honey Badger Algorithm (HBA)"),
    ("INFO",   info.info,            "Weighted Mean of Vectors (INFO)"),
    ("JSA",    jsa.jsa,              "Jellyfish Search Algorithm (JSA)"),
    ("KMA",    kma.kma,              "Komodo Mlipir Algorithm (KMA)"),
    ("OOA",    ooa.ooa,              "Osprey Optimization Algorithm (OOA)"),
    ("PARROT", parrot.parrot,        "Parrot Optimizer (PO)"),
    ("SCA",    sca.sca,              "Sine Cosine Algorithm (SCA)"),
    ("SWO",    swo.swo,              "Spider Wasp Optimizer (SWO)"),
    ("WSO",    wso.wso,              "White Shark Optimizer (WSO)"),
]


def run_all(n_agents=30, max_iter=1000, n_runs=5, figures_dir="results/figures", excel_dir="results/excel"):
    """
    Executes all 35 registered algorithms sequentially.
    """
    total_start = time.perf_counter()
    n_algos = len(ALGORITHMS)

    print("=" * 80)
    print("  SEQUENTIAL 35-ALGORITHM OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"  Total Algorithms : {n_algos}")
    print(f"  Search Agents    : {n_agents}")
    print(f"  Max Iterations   : {max_iter}")
    print(f"  Independent Runs : {n_runs}")
    print(f"  Figures Directory: {figures_dir}/")
    print(f"  Excel Directory  : {excel_dir}/")
    print("=" * 80 + "\n")

    completed = []
    failed = []

    for idx, (name, func, full_name) in enumerate(ALGORITHMS, 1):
        banner_title = f"[{idx:02d}/{n_algos:02d}] {name} - {full_name}"
        print("\n" + "#" * 80)
        print(f"  {banner_title}")
        print("#" * 80)

        algo_t0 = time.perf_counter()
        try:
            res = run_experiment(
                algo_name=name,
                algo_func=func,
                full_name=full_name,
                n_agents=n_agents,
                max_iter=max_iter,
                n_runs=n_runs,
                figures_dir=figures_dir,
                excel_dir=excel_dir,
            )
            elapsed_algo = time.perf_counter() - algo_t0
            completed.append((name, res['min_score'], elapsed_algo))
            print(f"[OK] {name} completed in {elapsed_algo:.1f}s - Best Fitness: {res['min_score']:.6e}")
        except Exception as e:
            elapsed_algo = time.perf_counter() - algo_t0
            import traceback
            print(f"[FAIL] {name} FAILED after {elapsed_algo:.1f}s: {e}")
            traceback.print_exc()
            failed.append((name, str(e)))

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("  ALL 35 ALGORITHMS EXECUTION COMPLETED")
    print("=" * 80)
    print(f"  Successfully Completed : {len(completed)}/{n_algos}")
    if failed:
        print(f"  Failed                 : {len(failed)} ({', '.join([f[0] for f in failed])})")
    print(f"  Total Benchmark Time   : {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print("=" * 80 + "\n")

    # Automatically aggregate comparison leaderboard and plot
    print("Aggregating comparison metrics and plotting overlaid curves...\n")
    compare_algorithms(excel_dir=excel_dir, figures_dir=figures_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run all 35 optimization algorithms sequentially on the benchmark."
    )
    parser.add_argument("--agents", type=int, default=30, help="Number of search agents (default: 30)")
    parser.add_argument("--iter", type=int, default=500, help="Maximum iterations per run (default: 500)")
    parser.add_argument("--runs", type=int, default=4, help="Number of independent runs (default: 4)")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test (agents=10, iter=25, runs=2)")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for figures")
    parser.add_argument("--excel-dir", type=str, default="results/excel", help="Directory for Excel workbooks")

    args = parser.parse_args()

    if args.smoke:
        print("[!] Running in fast smoke-test mode...")
        run_all(
            n_agents=10,
            max_iter=25,
            n_runs=2,
            figures_dir=args.figures_dir,
            excel_dir=args.excel_dir,
        )
    else:
        run_all(
            n_agents=args.agents,
            max_iter=args.iter,
            n_runs=args.runs,
            figures_dir=args.figures_dir,
            excel_dir=args.excel_dir,
        )


if __name__ == "__main__":
    main()
