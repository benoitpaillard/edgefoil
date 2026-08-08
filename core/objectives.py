"""
Design points and batched NeuralFoil evaluation for the whole population at once.

A DesignPoint is either:
  - a target-CL point (cl_target set): we don't know alpha in advance, so we
    sweep a batched alpha grid per generation, interpolate per-individual to
    find the alpha that hits the target CL, then re-evaluate once more at
    that alpha to get CD/CM/confidence.
  - a fixed-alpha point (alpha_fixed_deg set): evaluated directly, no search
    needed. Typically used for a stall-margin / high-alpha-CL objective.

Every DesignPoint contributes one scalar to minimize (`objective` field
picks which: drag, |moment|, or negative lift) plus an analysis-confidence
constraint, so the resulting pymoo Problem is naturally multi-objective as
soon as more than one DesignPoint (or the geometry itself) creates competing
pressure on the design vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import neuralfoil as nf

from . import geometry as geo

VALID_OBJECTIVE_KINDS = ("minimize_cd", "minimize_abs_cm", "maximize_cl")


@dataclass(frozen=True)
class DesignPoint:
    label: str
    re: float
    objective: str = "minimize_cd"
    mach: float = 0.0
    n_crit: float = 9.0
    xtr_upper: float = 1.0
    xtr_lower: float = 1.0
    cl_target: float | None = None
    alpha_fixed_deg: float | None = None
    alpha_search_range_deg: tuple[float, float] = (-8.0, 18.0)
    alpha_search_n: int = 25
    min_confidence: float = 0.85
    max_cl_error: float = 0.03

    def __post_init__(self):
        if self.objective not in VALID_OBJECTIVE_KINDS:
            raise ValueError(f"objective must be one of {VALID_OBJECTIVE_KINDS}, got {self.objective!r}.")
        target_set = self.cl_target is not None
        alpha_set = self.alpha_fixed_deg is not None
        if target_set == alpha_set:
            raise ValueError(f"DesignPoint {self.label!r} must set exactly one of cl_target / alpha_fixed_deg.")
        if self.objective == "maximize_cl" and target_set:
            raise ValueError(f"DesignPoint {self.label!r}: maximize_cl requires alpha_fixed_deg, not cl_target.")
        if self.re <= 0:
            raise ValueError(f"DesignPoint {self.label!r}: re must be positive.")


@dataclass
class DesignPointResult:
    label: str
    alpha_deg: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CM: np.ndarray
    confidence: np.ndarray
    cl_error: np.ndarray  # 0 for fixed-alpha points


def _neuralfoil_batch(params: dict, alpha: np.ndarray, re: np.ndarray, dp: DesignPoint, model_size: str) -> dict:
    return nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=params,
        alpha=alpha,
        Re=re,
        n_crit=dp.n_crit,
        xtr_upper=dp.xtr_upper,
        xtr_lower=dp.xtr_lower,
        model_size=model_size,
    )


def _resolve_alpha_for_target_cl(params: dict, dp: DesignPoint, n_pop: int, model_size: str) -> tuple[np.ndarray, np.ndarray]:
    """Batched alpha search: returns (alpha_resolved, cl_at_resolved_alpha) shape (n_pop,)."""
    a_lo, a_hi = dp.alpha_search_range_deg
    alpha_grid = np.linspace(a_lo, a_hi, dp.alpha_search_n)  # (n_grid,)
    n_grid = alpha_grid.size

    grid_params = {
        "upper_weights": np.repeat(params["upper_weights"], n_grid, axis=1),
        "lower_weights": np.repeat(params["lower_weights"], n_grid, axis=1),
        "leading_edge_weight": np.repeat(params["leading_edge_weight"], n_grid),
        "TE_thickness": np.repeat(params["TE_thickness"], n_grid),
    }
    alpha_batched = np.tile(alpha_grid, n_pop)
    re_batched = np.full(n_pop * n_grid, dp.re)

    res = _neuralfoil_batch(grid_params, alpha_batched, re_batched, dp, model_size)
    cl_grid = res["CL"].reshape(n_pop, n_grid)

    alpha_resolved = np.empty(n_pop)
    cl_resolved = np.empty(n_pop)
    for i in range(n_pop):
        cl_row = cl_grid[i]
        order = np.argsort(cl_row)
        cl_sorted = cl_row[order]
        alpha_sorted = alpha_grid[order]
        alpha_resolved[i] = np.interp(dp.cl_target, cl_sorted, alpha_sorted)
        cl_resolved[i] = np.interp(alpha_resolved[i], alpha_grid, cl_row)

    return alpha_resolved, cl_resolved


def evaluate_design_point(X: np.ndarray, dp: DesignPoint, model_size: str = "medium") -> DesignPointResult:
    """X: (n_pop, 18) design matrix. Returns one DesignPointResult with n_pop-length arrays."""
    X = np.atleast_2d(X)
    n_pop = X.shape[0]
    params = geo.unpack_design_matrix(X)

    if dp.cl_target is not None:
        alpha_resolved, cl_at_alpha = _resolve_alpha_for_target_cl(params, dp, n_pop, model_size)
        final = _neuralfoil_batch(params, alpha_resolved, np.full(n_pop, dp.re), dp, model_size)
        cl_error = np.abs(final["CL"] - dp.cl_target)
    else:
        alpha_resolved = np.full(n_pop, dp.alpha_fixed_deg)
        final = _neuralfoil_batch(params, alpha_resolved, np.full(n_pop, dp.re), dp, model_size)
        cl_error = np.zeros(n_pop)

    return DesignPointResult(
        label=dp.label,
        alpha_deg=alpha_resolved,
        CL=final["CL"],
        CD=final["CD"],
        CM=final["CM"],
        confidence=final["analysis_confidence"],
        cl_error=cl_error,
    )


def evaluate_all_design_points(
    X: np.ndarray, design_points: list[DesignPoint], model_size: str = "medium"
) -> dict[str, DesignPointResult]:
    return {dp.label: evaluate_design_point(X, dp, model_size=model_size) for dp in design_points}


def sweep_polar_neuralfoil(
    x: np.ndarray,
    re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    alpha_range: tuple[float, float] = (-6.0, 16.0),
    n_alpha: int = 45,
    model_size: str = "medium",
) -> dict:
    """Alpha-sweep polar for a single design vector `x`, e.g. for plotting."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    params = geo.unpack_design_matrix(x)
    alpha_grid = np.linspace(alpha_range[0], alpha_range[1], n_alpha)

    grid_params = {
        "upper_weights": np.repeat(params["upper_weights"], n_alpha, axis=1),
        "lower_weights": np.repeat(params["lower_weights"], n_alpha, axis=1),
        "leading_edge_weight": np.repeat(params["leading_edge_weight"], n_alpha),
        "TE_thickness": np.repeat(params["TE_thickness"], n_alpha),
    }
    res = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=grid_params,
        alpha=alpha_grid,
        Re=np.full(n_alpha, re),
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
    )
    return dict(alpha_deg=alpha_grid, CL=res["CL"], CD=res["CD"], CM=res["CM"], confidence=res["analysis_confidence"])


def objective_value(result: DesignPointResult, dp: DesignPoint) -> np.ndarray:
    """The scalar-to-minimize array (n_pop,) pymoo will see for this design point."""
    if dp.objective == "minimize_cd":
        return result.CD
    if dp.objective == "minimize_abs_cm":
        return np.abs(result.CM)
    if dp.objective == "maximize_cl":
        return -result.CL
    raise ValueError(dp.objective)  # unreachable given __post_init__ validation
