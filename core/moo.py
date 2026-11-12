"""
NSGA-II multi-objective airfoil shape optimization on top of the batched
NeuralFoil objective/constraint evaluators in objectives.py and geometry.py.

Why NSGA-II (a gradient-free, population-based genetic algorithm) rather than
the gradient-based CasADi/IPOPT approach used elsewhere in this user's other
airfoil scripts: gradient-based solvers need a single scalar objective, so
multi-point drag minimization there is done via weighted-sum scalarization
(one solve per weight combination), which can miss non-convex regions of the
true Pareto front and requires picking weights ahead of time. NSGA-II
optimizes all objectives simultaneously and returns the actual Pareto front
in one run. This is only practical because NeuralFoil is cheap enough to
batch-evaluate an entire population (and its alpha-search grid) in
milliseconds -- see objectives.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from . import geometry as geo
from . import objectives as obj

THICKNESS_STATIONS = np.array([0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 0.90])


@dataclass(frozen=True)
class GeometryConstraints:
    thickness_retention: float = 0.85  # local thickness >= this fraction of baseline, at each station
    min_max_thickness_ratio: float | None = None  # absolute floor on max t/c, e.g. 0.06; None to disable
    le_radius_retention: float = 0.4  # LE radius >= this fraction of baseline
    min_te_angle_deg: float = 3.0
    min_ixx_retention: float | None = None  # bending stiffness (flapwise) floor, as a fraction of baseline Ixx
    min_j_retention: float | None = None  # polar moment of area (Ixx+Iyy) floor, as a fraction of baseline J
    # NOTE: `J` here is the polar SECOND MOMENT OF AREA about the centroid (Ixx + Iyy),
    # not a Saint-Venant torsional constant -- see core/geometry.py:batched_polygon_inertia
    # for why it does not match XFoil's BEND "J" (a true torsional rigidity, much smaller
    # for thin sections). Useful as a fast bending/preliminary-sizing proxy, not a
    # torsional-stiffness guarantee.


@dataclass(frozen=True)
class ModelSizeSchedule:
    """Two-phase multi-fidelity schedule across NeuralFoil's own model sizes.

    Benchmarked against real XFoil on perturbed NACA 2412 variants: model size
    spans a real ~8x speed range and ~10x mean-CD-error range (e.g. "xsmall" ~0.23s/
    generation at pop=150 with ~7% mean error vs XFoil; "xlarge" ~0.52s/generation
    with ~2% error) -- and unlike an XFoil-based multi-fidelity scheme, every size
    is still a batched neural net call, so there's no subprocess-per-individual cost
    that would make a "many small evaluations, few large ones" split worthwhile.

    Running most generations at a small/fast size for broad exploration, then
    switching to a large/accurate size for the last `late_phase_fraction` of
    generations to refine and rank the converging front, can end up both faster
    overall than a single flat mid-size model *and* more accurate on the returned
    front -- not a tradeoff you have to pick a side of.
    """

    early_model_size: str = "small"
    late_model_size: str = "xlarge"
    late_phase_fraction: float = 0.20  # fraction of total generations spent in the late (accurate) phase

    def __post_init__(self):
        if not (0.0 < self.late_phase_fraction < 1.0):
            raise ValueError("late_phase_fraction must be strictly between 0 and 1.")


@dataclass(frozen=True)
class OptimizationConfig:
    baseline: geo.KulfanBaseline
    design_points: list[obj.DesignPoint]
    max_weight_step: float = 0.30
    geometry: GeometryConstraints = field(default_factory=GeometryConstraints)
    model_size: str = "medium"  # used when model_size_schedule is None, and for final front/baseline re-evaluation
    model_size_schedule: ModelSizeSchedule | None = None
    population_size: int = 80
    n_generations: int = 120
    seed: int | None = 0

    def __post_init__(self):
        if len(self.design_points) < 1:
            raise ValueError("At least one design point is required.")
        obj.validate_design_points(self.design_points)

    @property
    def final_model_size(self) -> str:
        """The model size used to re-evaluate/display the final front -- the late
        (accurate) phase's size when multi-fidelity is enabled, else the flat size."""
        return self.model_size_schedule.late_model_size if self.model_size_schedule is not None else self.model_size


@dataclass
class OptimizationResult:
    config: OptimizationConfig
    X_pareto: np.ndarray  # (n_front, 18) design vectors on the non-dominated front
    F_pareto: np.ndarray  # (n_front, n_obj) objective values (already sign-corrected for display)
    objective_labels: list[str]
    per_point_results: list[dict[str, obj.DesignPointResult]]  # one dict per front member
    n_evaluations: int
    baseline_per_point: dict[str, obj.DesignPointResult]
    baseline_F: np.ndarray  # (1, n_obj), same objective definitions/order as F_pareto


class AirfoilMOOProblem(Problem):
    def __init__(self, config: OptimizationConfig):
        self.config = config
        baseline_vec = geo.baseline_design_vector(config.baseline)
        xl, xu = geo.design_vector_bounds(baseline_vec, max_weight_step=config.max_weight_step)

        self.baseline_vec = baseline_vec
        self.baseline_max_thickness = float(
            geo.batched_max_thickness(*_unpack_for_geom(baseline_vec[None, :]))[0]
        )
        self.baseline_thickness = geo.batched_local_thickness(
            *_unpack_for_geom(baseline_vec[None, :]), THICKNESS_STATIONS
        )[0]
        self.baseline_le_radius = float(geo.batched_le_radius(
            geo.unpack_design_matrix(baseline_vec[None, :])["upper_weights"],
            geo.unpack_design_matrix(baseline_vec[None, :])["lower_weights"],
        )[0])

        baseline_structural = geo.design_matrix_structural_summary(baseline_vec[None, :])
        self.baseline_Ixx = float(baseline_structural["Ixx"][0])
        self.baseline_J = float(baseline_structural["J"][0])

        n_obj = len(config.design_points)
        # constraints: per design point (confidence margin [+ cl-error margin if target-CL]),
        # plus geometry: thickness stations, LE radius, TE angle, max-thickness floor (optional)
        n_constr = 0
        for dp in config.design_points:
            n_constr += 1  # confidence
            if dp.cl_target is not None:
                n_constr += 1  # cl error
        n_constr += len(THICKNESS_STATIONS)  # local thickness retention
        n_constr += 1  # LE radius retention
        n_constr += 1  # TE angle
        if config.geometry.min_max_thickness_ratio is not None:
            n_constr += 1
        if config.geometry.min_ixx_retention is not None:
            n_constr += 1
        if config.geometry.min_j_retention is not None:
            n_constr += 1

        super().__init__(n_var=geo.N_DESIGN_VARS, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        cfg = self.config
        results = obj.evaluate_all_design_points(X, cfg.design_points, model_size=cfg.model_size)

        F = np.column_stack([obj.objective_value(results[dp.label], dp) for dp in cfg.design_points])

        g_list = []
        for dp in cfg.design_points:
            r = results[dp.label]
            g_list.append(dp.min_confidence - r.confidence)  # <= 0 means OK
            if dp.cl_target is not None:
                g_list.append(r.cl_error - dp.max_cl_error)

        upper_w, lower_w, le_w, te_t = _unpack_for_geom(X)
        thickness = geo.batched_local_thickness(upper_w, lower_w, le_w, te_t, THICKNESS_STATIONS)
        min_allowed_thickness = cfg.geometry.thickness_retention * self.baseline_thickness
        for j in range(len(THICKNESS_STATIONS)):
            g_list.append(min_allowed_thickness[j] - thickness[:, j])

        le_radius = geo.batched_le_radius(upper_w, lower_w)
        g_list.append(cfg.geometry.le_radius_retention * self.baseline_le_radius - le_radius)

        te_angle = geo.batched_te_angle_deg(upper_w, lower_w, te_t)
        g_list.append(cfg.geometry.min_te_angle_deg - te_angle)

        if cfg.geometry.min_max_thickness_ratio is not None:
            max_t = geo.batched_max_thickness(upper_w, lower_w, le_w, te_t)
            g_list.append(cfg.geometry.min_max_thickness_ratio - max_t)

        if cfg.geometry.min_ixx_retention is not None or cfg.geometry.min_j_retention is not None:
            structural = geo.design_matrix_structural_summary(X)
            if cfg.geometry.min_ixx_retention is not None:
                g_list.append(cfg.geometry.min_ixx_retention * self.baseline_Ixx - structural["Ixx"])
            if cfg.geometry.min_j_retention is not None:
                g_list.append(cfg.geometry.min_j_retention * self.baseline_J - structural["J"])

        out["F"] = F
        out["G"] = np.column_stack(g_list)


def _unpack_for_geom(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = geo.unpack_design_matrix(X)
    return params["upper_weights"], params["lower_weights"], params["leading_edge_weight"], params["TE_thickness"]


class BaselineSeededSampling(Sampling):
    """Seeds the initial population around the baseline airfoil instead of pure
    uniform-random over the whole bounded box. Row 0 is the exact baseline
    (so the optimizer can never end up worse than the starting airfoil, since
    NSGA-II's survival selection is elitist); the rest are Gaussian
    perturbations of the baseline, clipped to bounds, plus a handful of fully
    random individuals for diversity. This converges much faster than random
    init in the ~18-dimensional CST space, especially at small population/
    generation budgets."""

    def __init__(
        self,
        baseline_vec: np.ndarray,
        spread_fraction: float = 0.35,
        random_fraction: float = 0.15,
        seed: int | None = None,
    ):
        super().__init__()
        self.baseline_vec = baseline_vec
        self.spread_fraction = spread_fraction
        self.random_fraction = random_fraction
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        half_range = (xu - xl) / 2.0
        rng = np.random.default_rng(self.seed)

        n_random = max(1, int(round(n_samples * self.random_fraction)))
        n_perturbed = n_samples - n_random - 1  # -1 for the exact baseline row

        rows = [self.baseline_vec.copy()]

        if n_perturbed > 0:
            noise = rng.normal(loc=0.0, scale=self.spread_fraction * half_range, size=(n_perturbed, len(xl)))
            perturbed = np.clip(self.baseline_vec[None, :] + noise, xl, xu)
            rows.append(perturbed)

        random_rows = rng.uniform(low=xl, high=xu, size=(n_random, len(xl)))
        rows.append(random_rows)

        return np.vstack(rows)[:n_samples]


class PopulationSeededSampling(Sampling):
    """Seeds the initial population from a prior population -- used to hand off
    between multi-fidelity phases, continuing the search from wherever a fast,
    approximate phase left off, now evaluated at a slower, more accurate model
    size. Unlike BaselineSeededSampling (which spreads out from a single known-good
    point to explore), this stays close to the given population, since the point is
    to refine an already-converged search, not re-explore. If the given population
    is smaller than the requested sample count, the shortfall is filled with small
    perturbations of randomly-chosen seed members rather than fresh random draws."""

    def __init__(self, seed_population: np.ndarray, spread_fraction: float = 0.05, seed: int | None = None):
        super().__init__()
        self.seed_population = seed_population
        self.spread_fraction = spread_fraction
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        half_range = (xu - xl) / 2.0
        rng = np.random.default_rng(self.seed)

        n_seed = self.seed_population.shape[0]
        if n_seed >= n_samples:
            idx = rng.choice(n_seed, size=n_samples, replace=False)
            return np.clip(self.seed_population[idx], xl, xu)

        n_fill = n_samples - n_seed
        fill_base_idx = rng.integers(0, n_seed, size=n_fill)
        noise = rng.normal(loc=0.0, scale=self.spread_fraction * half_range, size=(n_fill, len(xl)))
        filled = np.clip(self.seed_population[fill_base_idx] + noise, xl, xu)

        return np.vstack([self.seed_population, filled])[:n_samples]


def _run_phase(
    config: OptimizationConfig,
    model_size: str,
    n_generations: int,
    sampling: Sampling,
    verbose: bool,
):
    """Runs NSGA-II for one multi-fidelity phase at a fixed NeuralFoil model size,
    returning the raw pymoo result."""
    phase_config = replace(config, model_size=model_size)
    problem = AirfoilMOOProblem(phase_config)

    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", n_generations)

    return pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        verbose=verbose,
        save_history=False,
    )


def run_optimization(config: OptimizationConfig, verbose: bool = False) -> OptimizationResult:
    baseline_vec = geo.baseline_design_vector(config.baseline)

    if config.model_size_schedule is None:
        result = _run_phase(
            config, config.model_size, config.n_generations,
            sampling=BaselineSeededSampling(baseline_vec, seed=config.seed), verbose=verbose,
        )
        n_evaluations = int(result.algorithm.evaluator.n_eval)
    else:
        sched = config.model_size_schedule
        n_late = max(1, round(config.n_generations * sched.late_phase_fraction))
        n_early = max(1, config.n_generations - n_late)

        if verbose:
            print(f"Multi-fidelity phase 1/2: {n_early} generations @ model_size={sched.early_model_size!r}")
        phase1 = _run_phase(
            config, sched.early_model_size, n_early,
            sampling=BaselineSeededSampling(baseline_vec, seed=config.seed), verbose=verbose,
        )

        if verbose:
            print(f"Multi-fidelity phase 2/2: {n_late} generations @ model_size={sched.late_model_size!r}")
        phase2 = _run_phase(
            config, sched.late_model_size, n_late,
            sampling=PopulationSeededSampling(
                phase1.pop.get("X"),
                seed=None if config.seed is None else config.seed + 1,
            ), verbose=verbose,
        )

        result = phase2
        n_evaluations = int(phase1.algorithm.evaluator.n_eval) + int(phase2.algorithm.evaluator.n_eval)

    if result.X is None or len(result.X) == 0:
        raise RuntimeError(
            "NSGA-II found no feasible designs. Loosen the geometry constraints (thickness "
            "retention, LE radius retention) or the max weight step, and try again."
        )

    X_front = np.atleast_2d(result.X)
    F_front = np.atleast_2d(result.F)

    per_point_results = _evaluate_front_per_point(X_front, config)

    baseline_per_point_batched = obj.evaluate_all_design_points(
        baseline_vec[None, :], config.design_points, model_size=config.final_model_size
    )

    objective_labels = [_objective_label(dp) for dp in config.design_points]
    baseline_F = np.array(
        [[obj.objective_value(baseline_per_point_batched[dp.label], dp)[0] for dp in config.design_points]]
    )

    return OptimizationResult(
        config=config,
        X_pareto=X_front,
        F_pareto=F_front,
        objective_labels=objective_labels,
        per_point_results=per_point_results,
        n_evaluations=n_evaluations,
        baseline_per_point=baseline_per_point_batched,
        baseline_F=baseline_F,
    )


def _evaluate_front_per_point(X_front: np.ndarray, config: OptimizationConfig) -> list[dict[str, obj.DesignPointResult]]:
    """Re-evaluate every front member individually so downstream code (plots, tables,
    XFoil verification) can address `per_point_results[i][label]` per design."""
    n_front = X_front.shape[0]
    batched = obj.evaluate_all_design_points(X_front, config.design_points, model_size=config.final_model_size)
    per_individual = []
    for i in range(n_front):
        entry = {}
        for label, r in batched.items():
            entry[label] = obj.DesignPointResult(
                label=label,
                alpha_deg=r.alpha_deg[i : i + 1],
                CL=r.CL[i : i + 1],
                CD=r.CD[i : i + 1],
                CM=r.CM[i : i + 1],
                confidence=r.confidence[i : i + 1],
                cl_error=r.cl_error[i : i + 1],
            )
        per_individual.append(entry)
    return per_individual


def knee_point_index(F: np.ndarray) -> int:
    """Index of the front member closest (in per-objective-normalized space) to the
    ideal/utopia point -- a reasonable default "balanced" pick across objectives."""
    f_min, f_max = F.min(axis=0), F.max(axis=0)
    span = np.where(f_max - f_min == 0, 1.0, f_max - f_min)
    norm = (F - f_min) / span
    return int(np.argmin(np.linalg.norm(norm, axis=1)))


def _objective_label(dp: obj.DesignPoint) -> str:
    kind_text = {
        "minimize_cd": "CD",
        "minimize_abs_cm": "|CM|",
        "maximize_cl": "CL",
    }[dp.objective]
    return f"{kind_text} @ {dp.label}"
