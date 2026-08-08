"""Export helpers: .dat coordinate files, summary CSVs, and a bundled .zip."""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np

from . import geometry as geo
from .moo import OptimizationResult
from .verify import VerificationRow


def airfoil_dat_text(x: np.ndarray, name: str) -> str:
    kulfan = geo.design_vector_to_kulfan_airfoil(x, name=name)
    airfoil = kulfan.to_airfoil(n_coordinates_per_side=100)
    coords = np.array(airfoil.coordinates)
    lines = [name] + [f"{px:.6f} {py:.6f}" for px, py in coords]
    return "\n".join(lines) + "\n"


def pareto_front_csv(result: OptimizationResult) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)

    dp_labels = [dp.label for dp in result.config.design_points]
    structural = geo.design_matrix_structural_summary(result.X_pareto)
    header = ["front_index"] + result.objective_labels
    for label in dp_labels:
        header += [f"{label}_alpha_deg", f"{label}_CL", f"{label}_CD", f"{label}_CM", f"{label}_confidence"]
    header += ["Ixx", "Iyy", "Ixy", "J_polar_moment_of_area", "area"]
    writer.writerow(header)

    for i in range(result.X_pareto.shape[0]):
        row = [i] + list(result.F_pareto[i])
        for label in dp_labels:
            r = result.per_point_results[i][label]
            row += [float(r.alpha_deg[0]), float(r.CL[0]), float(r.CD[0]), float(r.CM[0]), float(r.confidence[0])]
        row += [
            float(structural["Ixx"][i]), float(structural["Iyy"][i]), float(structural["Ixy"][i]),
            float(structural["J"][i]), float(structural["area"][i]),
        ]
        writer.writerow(row)

    return buf.getvalue()


def verification_csv(rows: list[VerificationRow]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "front_index", "design_point", "neuralfoil_alpha_deg", "neuralfoil_CL", "neuralfoil_CD", "neuralfoil_CM",
            "xfoil_converged", "xfoil_alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM", "cd_relative_error",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.front_index, r.design_point_label, r.neuralfoil_alpha_deg, r.neuralfoil_CL, r.neuralfoil_CD, r.neuralfoil_CM,
                r.xfoil_converged, r.xfoil_alpha_deg, r.xfoil_CL, r.xfoil_CD, r.xfoil_CM, r.cd_relative_error,
            ]
        )
    return buf.getvalue()


def pareto_front_zip(result: OptimizationResult, indices: list[int] | None = None) -> bytes:
    """Bundles .dat files for the given front indices (default: all) plus baseline.dat
    and summary.csv into a single zip archive, returned as bytes."""
    if indices is None:
        indices = list(range(result.X_pareto.shape[0]))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        baseline_vec = geo.baseline_design_vector(result.config.baseline)
        zf.writestr("baseline.dat", airfoil_dat_text(baseline_vec, "baseline"))

        for i in indices:
            name = f"pareto_{i:03d}"
            zf.writestr(f"{name}.dat", airfoil_dat_text(result.X_pareto[i], name))

        zf.writestr("pareto_front_summary.csv", pareto_front_csv(result))

    return buf.getvalue()
