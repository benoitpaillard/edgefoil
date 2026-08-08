"""
Minimal, robust subprocess driver for the `xfoil` binary.

Note: aerosandbox.XFoil (the built-in wrapper) crashes with a SIGFPE on this
machine's xfoil 6.99 build partway through its PANE/PCOP paneling sequence
(reproduced independently of this project). Driving xfoil directly with the
sequence below (LOAD -> PANE -> OPER -> VISC -> PACC -> ASEQ/ALFA/CL -> QUIT)
was verified to work reliably, so this module talks to the xfoil binary
directly instead of going through that wrapper.

Polars are captured via XFoil's PACC file-accumulation feature and parsed
from the resulting fixed-format text file, which is far more robust than
scraping the interactive stdout log.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class XFoilError(RuntimeError):
    pass


@dataclass
class XFoilPolarPoint:
    alpha_deg: float
    CL: float
    CD: float
    CDp: float
    CM: float
    converged: bool


def _find_xfoil(xfoil_command: str = "xfoil") -> str:
    path = shutil.which(xfoil_command)
    if path is None:
        raise XFoilError(
            f"Could not find the '{xfoil_command}' executable on PATH. "
            "Install XFoil (e.g. `apt install xfoil` on Debian/Ubuntu, or build from "
            "https://web.mit.edu/drela/Public/web/xfoil/) and make sure it's on PATH."
        )
    return path


def _write_dat(coordinates: np.ndarray, path: Path, name: str = "airfoil") -> None:
    lines = [name]
    for x, y in coordinates:
        lines.append(f"{x:.6f} {y:.6f}")
    path.write_text("\n".join(lines) + "\n")


def _parse_polar_file(path: Path) -> list[XFoilPolarPoint]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("---") or line.strip().startswith("------"):
            data_start = i + 1
            break
    if data_start is None:
        return []

    points = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            alpha, cl, cd, cdp, cm = (float(v) for v in parts[:5])
        except ValueError:
            continue
        points.append(XFoilPolarPoint(alpha_deg=alpha, CL=cl, CD=cd, CDp=cdp, CM=cm, converged=True))
    return points


def run_polar(
    coordinates: np.ndarray,
    re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    alpha_sweep: tuple[float, float, float] | None = None,
    alpha_list: list[float] | None = None,
    cl_list: list[float] | None = None,
    iterations: int = 150,
    n_panels: int = 200,
    timeout: float = 45.0,
    xfoil_command: str = "xfoil",
) -> list[XFoilPolarPoint]:
    """Runs xfoil on a coordinate airfoil and returns converged polar points.

    Exactly one of alpha_sweep=(a0, a1, da), alpha_list, or cl_list must be given.
    Raises XFoilError (with a diagnostic message, including a tail of xfoil's own
    output) if the binary can't be found, the run times out, xfoil exits nonzero,
    or it completes but produces no converged polar points -- never silently
    returns an empty/ambiguous result.
    """
    xfoil_path = _find_xfoil(xfoil_command)
    modes_given = sum(x is not None for x in (alpha_sweep, alpha_list, cl_list))
    if modes_given != 1:
        raise ValueError("Provide exactly one of alpha_sweep, alpha_list, or cl_list.")

    case_desc = f"Re={re:.4g}, mach={mach:.3g}" + (
        f", CL target(s)={cl_list}"
        if cl_list is not None
        else f", alpha(s)={alpha_list}"
        if alpha_list is not None
        else f", alpha sweep={alpha_sweep}"
    )

    with tempfile.TemporaryDirectory(prefix="xfoil_") as tmpdir:
        tmp = Path(tmpdir)
        dat_path = tmp / "airfoil.dat"
        polar_path = tmp / "polar.txt"
        _write_dat(coordinates, dat_path)

        cmds = [
            f"LOAD {dat_path.name}",
            f"PPAR",
            f"N {n_panels}",
            "",
            "",
            "PANE",
            "OPER",
            f"VISC {re:.6g}",
        ]
        if mach and mach > 0:
            cmds.append(f"MACH {mach:.6g}")
        cmds += [
            "VPAR",
            f"N {n_crit:.4g}",
            f"XTR {xtr_upper:.4g} {xtr_lower:.4g}",
            "",
            f"ITER {iterations}",
            "PACC",
            polar_path.name,
            "",
        ]
        if alpha_sweep is not None:
            a0, a1, da = alpha_sweep
            cmds.append(f"ASEQ {a0:.4g} {a1:.4g} {da:.4g}")
        elif alpha_list is not None:
            for a in alpha_list:
                cmds.append(f"ALFA {a:.4g}")
        else:
            for cl in cl_list:
                cmds.append(f"CL {cl:.4g}")
        cmds += ["PACC", "", "QUIT"]

        input_text = "\n".join(cmds) + "\n"

        try:
            result = subprocess.run(
                [xfoil_path],
                cwd=tmp,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise XFoilError(
                f"xfoil timed out after {timeout}s on case ({case_desc}) -- it may have stalled on a non-converging point."
            ) from e

        if result.returncode != 0:
            tail = (result.stdout or "")[-600:]
            raise XFoilError(
                f"xfoil exited with code {result.returncode} on case ({case_desc}) -- it likely crashed rather "
                f"than failing to converge. Last output:\n{tail}"
            )

        points = _parse_polar_file(polar_path)
        if not points:
            tail = (result.stdout or "")[-600:]
            raise XFoilError(
                f"xfoil ran (exit 0) but produced no converged polar points for case ({case_desc}). This usually "
                f"means genuine non-convergence (near/past stall, or an unreachable operating point for this "
                f"shape) rather than a setup problem. Last output:\n{tail}"
            )
        return points


def polar_at_target_cl(
    coordinates: np.ndarray,
    cl_target: float,
    re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    **kwargs,
) -> XFoilPolarPoint:
    """Raises XFoilError on any failure -- see run_polar()."""
    points = run_polar(coordinates, re=re, mach=mach, n_crit=n_crit, xtr_upper=xtr_upper, xtr_lower=xtr_lower, cl_list=[cl_target], **kwargs)
    return points[0]


def polar_at_alpha(
    coordinates: np.ndarray,
    alpha_deg: float,
    re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    **kwargs,
) -> XFoilPolarPoint:
    """Raises XFoilError on any failure -- see run_polar()."""
    points = run_polar(coordinates, re=re, mach=mach, n_crit=n_crit, xtr_upper=xtr_upper, xtr_lower=xtr_lower, alpha_list=[alpha_deg], **kwargs)
    return points[0]
