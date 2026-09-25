"""
Benchmark Suite: 35 Metaheuristic Optimization Algorithms (Standalone Python Implementations)

Algorithms:
    - apo: Artificial Protozoa Optimizer (apo)
    - aro: Artificial Rabbits Optimization (aro)
    - bto: Barrel Theory-Based Optimizer (bto)
    - cpo: Crested Porcupine Optimizer (cpo)
    - do: Dandelion Optimizer (do)
    - doa: Dream Optimization Algorithm (doa)
    - fpa: Flower Pollination Algorithm (fpa)
    - gwo: Grey Wolf Optimizer (gwo)
    - hho: Harris Hawks Optimization (hho)
    - ho: Hippopotamus Optimization (ho)
    - ivy: Ivy Algorithm (ivy)
    - koa: Kepler Optimization Algorithm (koa)
    - mosfoa: Multi-Objective Starfish Optimization Algorithm (mosfoa)
    - msfoa: Modified Starfish Optimization Algorithm (msfoa)
    - mpa: Marine Predators Algorithm (mpa)
    - po: Puma Optimizer (puma)
    - pso: Particle Swarm Optimization (pso)
    - rsa: Reptile Search Algorithm (rsa)
    - run: Runge Kutta Optimization (run_opt)
    - sma: Slime Mould Algorithm (sma)
    - so: Snake Optimizer (so)
    - woa: Whale Optimization Algorithm (woa)
    - zoa: Zebra Optimization Algorithm (zoa)
    - ao: Artemisinin Optimizer (ao)
    - eooa: Enhanced Osprey Optimization Algorithm (eooa)
    - esc: Escape Optimization Algorithm (esc)
    - gto: Gorilla Troops Optimizer (gto)
    - hba: Honey Badger Algorithm (hba)
    - info: Weighted Mean of Vectors (info)
    - jsa: Jellyfish Search Algorithm (jsa)
    - kma: Komodo Mlipir Algorithm (kma)
    - ooa: Osprey Optimization Algorithm (ooa)
    - parrot: Parrot Optimizer (parrot)
    - sca: Sine Cosine Algorithm (sca)
    - swo: Spider Wasp Optimizer (swo)
    - wso: White Shark Optimizer (wso)
"""

from .apo import apo
from .aro import aro
from .bto import bto
from .cpo import cpo
from .do import do
from .doa import doa
from .fpa import fpa
from .gwo import gwo
from .hho import hho
from .ho import ho
from .ivy import ivy
from .koa import koa
from .mosfoa import mosfoa
from .msfoa import msfoa
from .mpa import mpa
from .po import puma
from .pso import pso
from .rsa import rsa
from .run import run_opt
from .sma import sma
from .so import so
from .woa import woa
from .zoa import zoa

# New Algorithms
from .ao import ao
from .eooa import eooa
from .esc import esc
from .gto import gto
from .hba import hba
from .info import info
from .jsa import jsa
from .kma import kma
from .ooa import ooa
from .parrot import parrot
from .sca import sca
from .swo import swo
from .wso import wso

__all__ = [
    "apo",
    "aro",
    "bto",
    "cpo",
    "do",
    "doa",
    "fpa",
    "gwo",
    "hho",
    "ho",
    "ivy",
    "koa",
    "mosfoa",
    "msfoa",
    "mpa",
    "puma",
    "pso",
    "rsa",
    "run_opt",
    "sma",
    "so",
    "woa",
    "zoa",
    "ao",
    "eooa",
    "esc",
    "gto",
    "hba",
    "info",
    "jsa",
    "kma",
    "ooa",
    "parrot",
    "sca",
    "swo",
    "wso",
]
