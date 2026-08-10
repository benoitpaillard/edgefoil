#!/usr/bin/env python3
"""
Command-line entry point for the airfoil multi-objective optimizer.

Quick example (two competing drag objectives — a genuine Pareto trade-off):

    python3 cli.py --airfoil naca2412 \\
        --point "label=cruise,re=1e6,cl=0.3,objective=cd" \\
        --point "label=climb,re=6e5,cl=0.9,objective=cd" \\
        --out outputs/run1

Or drive it from a JSON config file (see --config) for anything more
elaborate (>2 objectives, custom geometry/analysis settings). Run with
--help for the full option list.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import geometry as geo
from core import io_utils as io_u
from core import moo
from core import objectives as obj
from core import verify

MODEL_SIZES = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]

OBJECTIVE_SHORTHAND = {
    "cd": "minimize_cd",
    "drag": "minimize_cd",
    "cm": "minimize_abs_cm",
    "moment": "minimize_abs_cm",
    "maxcl": "maximize_cl",
    "cl": "maximize_cl",
}


def parse_point_spec(spec: str, defaults: dict) -> obj.DesignPoint:
    fields = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad --point field {chunk!r} in {spec!r} (expected key=value).")
        key, value = chunk.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    if "label" not in fields:
        raise ValueError(f"--point {spec!r} is missing required field 'label'.")
    if "re" not in fields:
        raise ValueError(f"--point {spec!r} is missing required field 're'.")

    objective_key = fields.get("objective", "cd").lower()
    if objective_key not in OBJECTIVE_SHORTHAND:
        raise ValueError(f"--point {spec!r}: unknown objective {objective_key!r}. Use one of {sorted(OBJECTIVE_SHORTHAND)}.")

    kwargs = dict(defaults)
    kwargs["label"] = fields["label"]
    kwargs["re"] = float(fields["re"])
    kwargs["objective"] = OBJECTIVE_SHORTHAND[objective_key]
    if "mach" in fields:
        kwargs["mach"] = float(fields["mach"])
    if "cl" in fields:
        kwargs["cl_target"] = float(fields["cl"])
    if "alpha" in fields:
        kwargs["alpha_fixed_deg"] = float(fields["alpha"])

    return obj.DesignPoint(**kwargs)


def load_airfoil(spec: str):
    path = Path(spec)
    if path.exists():
        return geo.airfoil_from_dat_text(path.read_text(), name=path.stem)

    clean = spec.strip().lower().replace("naca", "").replace(" ", "")
    if clean.isdigit() and len(clean) in (4, 5):
        return geo.airfoil_from_naca(clean)

    for preset_key in geo.PRESET_AIRFOILS:
        if spec.strip().lower() in preset_key.lower():
            return geo.airfoil_from_preset(preset_key)

    raise ValueError(
        f"Could not interpret --airfoil {spec!r} as an existing .dat file path, a 4/5-digit NACA code, "
        f"or a preset name. Presets: {list(geo.PRESET_AIRFOILS.keys())}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-objective airfoil optimizer (NeuralFoil + XFoil).", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--airfoil", required=True, help="NACA code (e.g. naca2412), path to a .dat file, or a preset name.")
    p.add_argument("--point", action="append", default=[], help="Design point spec: 'label=...,re=...,cl=...|alpha=...,objective=cd|cm|maxcl[,mach=...]'. Repeat for each objective (need >= 2).")
    p.add_argument("--config", type=Path, default=None, help="JSON config file. See README for schema; overrides/supplements --point.")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: outputs/<timestamp>).")

    p.add_argument("--population", type=int, default=80)
    p.add_argument("--generations", type=int, default=120)
    p.add_argument(
        "--model-size", choices=MODEL_SIZES, default="large",
        help="Flat NeuralFoil model size for every generation. Ignored if --early-model-size/--late-model-size are given. "
        "'large' is the best time/quality trade found in testing -- see README.",
    )
    p.add_argument(
        "--early-model-size", choices=MODEL_SIZES, default=None,
        help="Enable multi-fidelity scheduling: fast/small model size for the early (exploration) generations. "
        "Must be given together with --late-model-size. NOT recommended -- a rigorous study found this ties "
        "flat 'medium' and loses to flat 'large'; see README. Left available for experimentation.",
    )
    p.add_argument(
        "--late-model-size", choices=MODEL_SIZES, default=None,
        help="Multi-fidelity: accurate/large model size for the late (refinement) generations. "
        "Must be given together with --early-model-size.",
    )
    p.add_argument(
        "--late-phase-fraction", type=float, default=0.20,
        help="Multi-fidelity: fraction of total generations spent in the late (accurate) phase.",
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-weight-step", type=float, default=0.30)
    p.add_argument("--thickness-retention", type=float, default=0.85)
    p.add_argument("--le-radius-retention", type=float, default=0.40)
    p.add_argument("--min-te-angle", type=float, default=3.0)
    p.add_argument("--min-thickness-ratio", type=float, default=None)
    p.add_argument("--min-ixx-retention", type=float, default=None, help="Minimum bending stiffness (Ixx) as a fraction of the baseline's, e.g. 0.5 for >=50%%.")
    p.add_argument("--min-j-retention", type=float, default=None, help="Minimum polar moment of area (J=Ixx+Iyy, NOT a torsional constant -- see README) as a fraction of baseline.")

    p.add_argument("--n-crit", type=float, default=9.0)
    p.add_argument("--xtr-upper", type=float, default=1.0)
    p.add_argument("--xtr-lower", type=float, default=1.0)
    p.add_argument("--min-confidence", type=float, default=0.85)
    p.add_argument("--max-cl-error", type=float, default=0.03)
    p.add_argument("--alpha-search-range", type=float, nargs=2, default=(-8.0, 18.0), metavar=("LO", "HI"))

    p.add_argument("--verify", action="store_true", help="Cross-check a subset of the Pareto front against real XFoil after optimizing.")
    p.add_argument("--verify-points", type=int, default=10)

    p.add_argument("--max-export", type=int, default=None, help="Cap how many front designs to write .dat files for (default: all).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    point_defaults = dict(
        n_crit=args.n_crit, xtr_upper=args.xtr_upper, xtr_lower=args.xtr_lower,
        min_confidence=args.min_confidence, max_cl_error=args.max_cl_error,
        alpha_search_range_deg=tuple(args.alpha_search_range),
    )

    design_points: list[obj.DesignPoint] = []
    if args.config is not None:
        cfg_json = json.loads(args.config.read_text())
        for entry in cfg_json.get("design_points", []):
            merged = dict(point_defaults)
            merged.update(entry)
            design_points.append(obj.DesignPoint(**merged))
    for spec in args.point:
        design_points.append(parse_point_spec(spec, point_defaults))

    if len(design_points) < 2:
        print("error: need at least 2 design points (via --point, repeated, or --config) for multi-objective optimization.", file=sys.stderr)
        sys.exit(2)

    try:
        airfoil = load_airfoil(args.airfoil)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if (args.early_model_size is None) != (args.late_model_size is None):
        print("error: --early-model-size and --late-model-size must be given together.", file=sys.stderr)
        sys.exit(2)

    model_size_schedule = None
    if args.early_model_size is not None:
        model_size_schedule = moo.ModelSizeSchedule(
            early_model_size=args.early_model_size,
            late_model_size=args.late_model_size,
            late_phase_fraction=args.late_phase_fraction,
        )

    baseline = geo.to_kulfan_baseline(airfoil)
    geometry_constraints = moo.GeometryConstraints(
        thickness_retention=args.thickness_retention,
        min_max_thickness_ratio=args.min_thickness_ratio,
        le_radius_retention=args.le_radius_retention,
        min_te_angle_deg=args.min_te_angle,
        min_ixx_retention=args.min_ixx_retention,
        min_j_retention=args.min_j_retention,
    )
    config = moo.OptimizationConfig(
        baseline=baseline,
        design_points=design_points,
        max_weight_step=args.max_weight_step,
        geometry=geometry_constraints,
        model_size=args.model_size,
        model_size_schedule=model_size_schedule,
        population_size=args.population,
        n_generations=args.generations,
        seed=args.seed,
    )

    out_dir = args.out or Path("outputs") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optimizing '{airfoil.name}' with {len(design_points)} objectives: {[dp.label for dp in design_points]}")
    if model_size_schedule is not None:
        print(
            f"Population {config.population_size} x {config.n_generations} generations, "
            f"multi-fidelity {model_size_schedule.early_model_size} -> {model_size_schedule.late_model_size} "
            f"(last {model_size_schedule.late_phase_fraction:.0%} of generations)"
        )
    else:
        print(f"Population {config.population_size} x {config.n_generations} generations, model_size={config.model_size}")
    t0 = time.time()
    result = moo.run_optimization(config, verbose=True)
    print(f"Done in {time.time() - t0:.1f}s — {result.X_pareto.shape[0]} non-dominated designs, {result.n_evaluations} evaluations.")

    (out_dir / "pareto_front_summary.csv").write_text(io_u.pareto_front_csv(result))

    export_indices = list(range(result.X_pareto.shape[0]))
    if args.max_export is not None:
        export_indices = export_indices[: args.max_export]
    zip_bytes = io_u.pareto_front_zip(result, indices=export_indices)
    (out_dir / "pareto_front.zip").write_bytes(zip_bytes)

    knee = moo.knee_point_index(result.F_pareto)
    base_coords = np.array(airfoil.coordinates)
    knee_coords = np.array(geo.design_vector_to_kulfan_airfoil(result.X_pareto[knee]).to_airfoil(n_coordinates_per_side=150).coordinates)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(base_coords[:, 0], base_coords[:, 1], "--", color="0.6", label="Baseline")
    ax.plot(knee_coords[:, 0], knee_coords[:, 1], color="#54A24B", label=f"Knee-point design #{knee}")
    ax.axis("equal")
    ax.legend()
    ax.set_title("Baseline vs. balanced ('knee point') optimized design")
    fig.tight_layout()
    fig.savefig(out_dir / "knee_point_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Wrote: {out_dir}/pareto_front_summary.csv, pareto_front.zip, knee_point_comparison.png")
    print(f"Balanced ('knee point') design: #{knee} — " + ", ".join(f"{label}={val:.5f}" for label, val in zip(result.objective_labels, result.F_pareto[knee])))

    struct_base = geo.design_matrix_structural_summary(geo.baseline_design_vector(baseline)[None, :])
    struct_knee = geo.design_matrix_structural_summary(result.X_pareto[knee][None, :])
    print(
        f"Structural (knee point vs. baseline): Ixx={float(struct_knee['Ixx'][0]):.4e} "
        f"({float(struct_knee['Ixx'][0]) / float(struct_base['Ixx'][0]) - 1:+.1%}), "
        f"J={float(struct_knee['J'][0]):.4e} ({float(struct_knee['J'][0]) / float(struct_base['J'][0]) - 1:+.1%}) "
        f"-- J is polar moment of area (Ixx+Iyy), not an XFoil-equivalent torsional constant."
    )

    if args.verify:
        print(f"Verifying {args.verify_points} front designs with XFoil...")
        rows = verify.verify_front_with_xfoil(result, max_points=args.verify_points)
        (out_dir / "xfoil_verification.csv").write_text(io_u.verification_csv(rows))
        n_failed = sum(1 for r in rows if not r.xfoil_converged)
        print(f"Wrote {out_dir}/xfoil_verification.csv ({len(rows)} points, {n_failed} did not converge).")


if __name__ == "__main__":
    main()
