"""
Cross-checks a subset of the NeuralFoil-optimized Pareto front against XFoil.

NeuralFoil is a fast surrogate; it can be locally wrong even at high reported
analysis_confidence for shapes that push the edges of its training
distribution. Before trusting a design, this module re-runs each selected
front member through actual XFoil (Re/Mach-matched, same CL target or fixed
alpha as used during optimization) and reports the discrepancy, so the user
can see which candidates are trustworthy and which aren't.

XFoil calls are farmed out to a thread pool since each is a slow (~1-3s)
blocking subprocess call that mostly waits on I/O.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from . import objectives as obj
from . import xfoil_driver as xd
from .moo import OptimizationResult


@dataclass
class VerificationRow:
    front_index: int
    design_point_label: str
    neuralfoil_alpha_deg: float
    neuralfoil_CL: float
    neuralfoil_CD: float
    neuralfoil_CM: float
    xfoil_converged: bool
    xfoil_alpha_deg: float | None = None
    xfoil_CL: float | None = None
    xfoil_CD: float | None = None
    xfoil_CM: float | None = None
    cd_relative_error: float | None = None  # (xfoil_CD - neuralfoil_CD) / xfoil_CD


def select_representative_indices(F_pareto: np.ndarray, max_points: int) -> list[int]:
    """Evenly spread `max_points` indices across the front, sorted by the first
    objective, so the verified subset spans the whole trade-off rather than
    clustering."""
    n = F_pareto.shape[0]
    if n <= max_points:
        return list(range(n))
    order = np.argsort(F_pareto[:, 0])
    pick = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [int(order[i]) for i in pick]


def _verify_one(
    front_idx: int,
    dp: obj.DesignPoint,
    coords: np.ndarray,
    nf_result: obj.DesignPointResult,
    n_panels: int,
    timeout: float,
) -> VerificationRow:
    try:
        if dp.cl_target is not None:
            point = xd.polar_at_target_cl(
                coords, cl_target=dp.cl_target, re=dp.re, mach=dp.mach,
                n_crit=dp.n_crit, xtr_upper=dp.xtr_upper, xtr_lower=dp.xtr_lower,
                n_panels=n_panels, timeout=timeout,
            )
        else:
            point = xd.polar_at_alpha(
                coords, alpha_deg=dp.alpha_fixed_deg, re=dp.re, mach=dp.mach,
                n_crit=dp.n_crit, xtr_upper=dp.xtr_upper, xtr_lower=dp.xtr_lower,
                n_panels=n_panels, timeout=timeout,
            )
    except xd.XFoilError:
        point = None

    cd_rel_err = None
    if point is not None and point.CD:
        cd_rel_err = (point.CD - float(nf_result.CD[0])) / point.CD

    return VerificationRow(
        front_index=front_idx,
        design_point_label=dp.label,
        neuralfoil_alpha_deg=float(nf_result.alpha_deg[0]),
        neuralfoil_CL=float(nf_result.CL[0]),
        neuralfoil_CD=float(nf_result.CD[0]),
        neuralfoil_CM=float(nf_result.CM[0]),
        xfoil_converged=point is not None,
        xfoil_alpha_deg=point.alpha_deg if point else None,
        xfoil_CL=point.CL if point else None,
        xfoil_CD=point.CD if point else None,
        xfoil_CM=point.CM if point else None,
        cd_relative_error=cd_rel_err,
    )


def verify_front_with_xfoil(
    result: OptimizationResult,
    max_points: int = 15,
    n_panels: int = 200,
    timeout: float = 45.0,
    max_workers: int | None = None,
    progress_callback=None,
) -> list[VerificationRow]:
    indices = select_representative_indices(result.F_pareto, max_points)
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() or 4))

    jobs = []
    for idx in indices:
        kulfan = geo.design_vector_to_kulfan_airfoil(result.X_pareto[idx], name=f"front_{idx}")
        coords = np.array(kulfan.to_airfoil(n_coordinates_per_side=200).coordinates)
        for dp in result.config.design_points:
            jobs.append((idx, dp, coords, result.per_point_results[idx][dp.label]))

    rows: list[VerificationRow] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_verify_one, idx, dp, coords, nf_result, n_panels, timeout): (idx, dp.label)
            for idx, dp, coords, nf_result in jobs
        }
        n_done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            n_done += 1
            if progress_callback is not None:
                progress_callback(n_done, len(jobs))

    rows.sort(key=lambda r: (r.front_index, r.design_point_label))
    return rows


def baseline_verification(
    result: OptimizationResult, n_panels: int = 200, timeout: float = 45.0
) -> list[VerificationRow]:
    """Same cross-check, but for the starting airfoil (useful as a sanity
    reference alongside the optimized front)."""
    coords = np.array(result.config.baseline.airfoil.coordinates)
    rows = []
    for dp in result.config.design_points:
        nf_result = result.baseline_per_point[dp.label]
        rows.append(_verify_one(-1, dp, coords, nf_result, n_panels, timeout))
    return rows
