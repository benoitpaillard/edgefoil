"""
Airfoil loading and the Kulfan (CST) parameterization used for optimization.

NeuralFoil's neural network is trained on exactly 8 CST weights per side, so
that count is fixed throughout this project (N_WEIGHTS_PER_SIDE = 8).

All batched_* functions evaluate the Kulfan/CST geometry equations directly
with numpy across an entire population at once (shape (8, N_pop) weights),
rather than constructing one AeroSandbox KulfanAirfoil object per individual.
That keeps per-generation constraint evaluation fast inside the NSGA-II loop.
The formulas mirror aerosandbox.KulfanAirfoil.upper_coordinates /
lower_coordinates / local_thickness / LE_radius / TE_angle exactly (N1=0.5,
N2=1.0, the standard airfoil class function).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import comb

import aerosandbox as asb

N_WEIGHTS_PER_SIDE = 8
N_DESIGN_VARS = 2 * N_WEIGHTS_PER_SIDE + 2  # upper(8) + lower(8) + LE(1) + TE(1)
UPPER_SLICE = slice(0, 8)
LOWER_SLICE = slice(8, 16)
LE_INDEX = 16
TE_INDEX = 17

PRESET_AIRFOILS = {
    "NACA 0012 — symmetric, general purpose": "naca0012",
    "NACA 2412 — light camber, general aviation": "naca2412",
    "NACA 4412 — higher camber, high lift": "naca4412",
    "NACA 6409 — thin, high camber, low Re": "naca6409",
    "NACA 23012 — 5-digit, low pitching moment": "naca23012",
}


# --------------------------------------------------------------------------
# Loading airfoils
# --------------------------------------------------------------------------


def read_airfoil_coordinates(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        clean = line.strip().replace(",", " ")
        if not clean:
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        rows.append([x, y])

    coords = np.asarray(rows, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 5:
        raise ValueError("Could not parse a valid 2-column airfoil (.dat) from the provided text.")
    return coords


def airfoil_from_dat_text(text: str, name: str = "custom") -> asb.Airfoil:
    coords = read_airfoil_coordinates(text)
    return asb.Airfoil(name=name, coordinates=coords).normalize().repanel(n_points_per_side=200)


def airfoil_from_naca(code: str) -> asb.Airfoil:
    clean = code.strip().lower().replace("naca", "").replace(" ", "")
    if not (clean.isdigit() and len(clean) in (4, 5)):
        raise ValueError(f"'{code}' is not a valid 4- or 5-digit NACA code.")
    return asb.Airfoil(name=f"naca{clean}").repanel(n_points_per_side=200)


def airfoil_from_preset(preset_key: str) -> asb.Airfoil:
    if preset_key not in PRESET_AIRFOILS:
        raise ValueError(f"Unknown preset '{preset_key}'.")
    return airfoil_from_naca(PRESET_AIRFOILS[preset_key])


@dataclass(frozen=True)
class KulfanBaseline:
    airfoil: asb.Airfoil
    kulfan: asb.KulfanAirfoil


def to_kulfan_baseline(airfoil: asb.Airfoil) -> KulfanBaseline:
    kulfan = airfoil.to_kulfan_airfoil(n_weights_per_side=N_WEIGHTS_PER_SIDE)
    return KulfanBaseline(airfoil=airfoil, kulfan=kulfan)


# --------------------------------------------------------------------------
# Design-vector encode / decode
#
# A "design vector" is a flat length-18 array:
#   [upper_weights(8), lower_weights(8), leading_edge_weight(1), TE_thickness(1)]
# A population is a matrix of shape (N_pop, 18).
# --------------------------------------------------------------------------


def baseline_design_vector(baseline: KulfanBaseline) -> np.ndarray:
    k = baseline.kulfan
    return np.concatenate(
        [
            np.asarray(k.upper_weights, dtype=float).reshape(-1),
            np.asarray(k.lower_weights, dtype=float).reshape(-1),
            [float(k.leading_edge_weight)],
            [float(k.TE_thickness)],
        ]
    )


def design_vector_bounds(
    baseline_vec: np.ndarray,
    max_weight_step: float = 0.30,
    le_bounds: tuple[float, float] = (-0.5, 1.5),
    te_thickness_bounds: tuple[float, float] = (0.0005, 0.03),
) -> tuple[np.ndarray, np.ndarray]:
    lower = baseline_vec - max_weight_step
    upper = baseline_vec + max_weight_step

    lower[LE_INDEX] = max(lower[LE_INDEX], le_bounds[0])
    upper[LE_INDEX] = min(upper[LE_INDEX], le_bounds[1])

    lower[TE_INDEX] = max(lower[TE_INDEX], te_thickness_bounds[0])
    upper[TE_INDEX] = min(upper[TE_INDEX], te_thickness_bounds[1])
    if lower[TE_INDEX] >= upper[TE_INDEX]:
        lower[TE_INDEX], upper[TE_INDEX] = te_thickness_bounds

    return lower, upper


def unpack_design_matrix(X: np.ndarray) -> dict:
    """X: shape (N_pop, 18) -> NeuralFoil-ready kulfan_parameters dict.

    NeuralFoil expects weights transposed to shape (8, N_pop): each row is one
    CST coefficient's value across the whole batch.
    """
    X = np.atleast_2d(X)
    return dict(
        upper_weights=X[:, UPPER_SLICE].T,
        lower_weights=X[:, LOWER_SLICE].T,
        leading_edge_weight=X[:, LE_INDEX],
        TE_thickness=X[:, TE_INDEX],
    )


def design_vector_to_kulfan_airfoil(x: np.ndarray, name: str = "design") -> asb.KulfanAirfoil:
    x = np.asarray(x, dtype=float).reshape(-1)
    return asb.KulfanAirfoil(
        name=name,
        upper_weights=x[UPPER_SLICE],
        lower_weights=x[LOWER_SLICE],
        leading_edge_weight=x[LE_INDEX],
        TE_thickness=x[TE_INDEX],
    )


# --------------------------------------------------------------------------
# Vectorized Kulfan/CST geometry (batched over a population)
# --------------------------------------------------------------------------


def _bernstein_matrix(x_over_c: np.ndarray, n_weights: int) -> np.ndarray:
    """Returns shape (n_weights, n_x)."""
    N = n_weights - 1
    i = np.arange(n_weights)
    K = comb(N, i)
    x = np.asarray(x_over_c, dtype=float)
    return K[:, None] * (x[None, :] ** i[:, None]) * ((1 - x[None, :]) ** (N - i)[:, None])


def batched_surface_y(
    weights: np.ndarray,
    le_weight: np.ndarray,
    te_thickness: np.ndarray,
    x_over_c: np.ndarray,
    sign: float,
) -> np.ndarray:
    """weights: (8, N_pop); le_weight, te_thickness: (N_pop,); x_over_c: (n_x,).

    Returns y ordinates, shape (N_pop, n_x).
    """
    n_weights = weights.shape[0]
    x = np.asarray(x_over_c, dtype=float)
    B = _bernstein_matrix(x, n_weights)  # (n_weights, n_x)
    C = (x**0.5) * ((1 - x) ** 1.0)  # (n_x,)
    S = weights.T @ B  # (N_pop, n_x)
    y = C[None, :] * S
    y += sign * x[None, :] * te_thickness[:, None] / 2
    y += le_weight[:, None] * x[None, :] * ((1 - x[None, :]) ** (n_weights + 0.5))
    return y


def batched_local_thickness(
    upper_w: np.ndarray,
    lower_w: np.ndarray,
    le_weight: np.ndarray,
    te_thickness: np.ndarray,
    x_stations: np.ndarray,
) -> np.ndarray:
    yu = batched_surface_y(upper_w, le_weight, te_thickness, x_stations, sign=+1.0)
    yl = batched_surface_y(lower_w, le_weight, te_thickness, x_stations, sign=-1.0)
    return yu - yl  # (N_pop, n_stations)


def batched_max_thickness(
    upper_w: np.ndarray,
    lower_w: np.ndarray,
    le_weight: np.ndarray,
    te_thickness: np.ndarray,
    n_probe: int = 81,
) -> np.ndarray:
    x_probe = np.linspace(0.002, 0.998, n_probe)
    t = batched_local_thickness(upper_w, lower_w, le_weight, te_thickness, x_probe)
    return np.max(t, axis=1)


def batched_le_radius(upper_w: np.ndarray, lower_w: np.ndarray) -> np.ndarray:
    u0 = upper_w[0, :]
    l0 = lower_w[0, :]
    ru = np.where(u0 > 0, u0**2, 0.0)
    rl = np.where(l0 < 0, l0**2, 0.0)
    return np.minimum(ru, rl) / 2.0


def batched_te_angle_deg(upper_w: np.ndarray, lower_w: np.ndarray, te_thickness: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(upper_w[-1, :]) - np.arctan(lower_w[-1, :]) + np.arctan(te_thickness))


def batched_closed_polygon_xy(
    upper_w: np.ndarray,
    lower_w: np.ndarray,
    le_weight: np.ndarray,
    te_thickness: np.ndarray,
    n_points_per_side: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Full closed-loop airfoil polygon per individual: upper surface from TE to LE,
    then lower surface from LE to TE (matches AeroSandbox's own coordinate ordering).
    Returns (x, y), each shape (N_pop, 2 * n_points_per_side)."""
    x_upper = np.linspace(1.0, 0.0, n_points_per_side)
    x_lower = np.linspace(0.0, 1.0, n_points_per_side)
    y_upper = batched_surface_y(upper_w, le_weight, te_thickness, x_upper, sign=+1.0)
    y_lower = batched_surface_y(lower_w, le_weight, te_thickness, x_lower, sign=-1.0)
    n_pop = upper_w.shape[1]
    x = np.concatenate([np.tile(x_upper, (n_pop, 1)), np.tile(x_lower, (n_pop, 1))], axis=1)
    y = np.concatenate([y_upper, y_lower], axis=1)
    return x, y


def _polygon_shoelace_terms(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x, y: (N_pop, n_points). Returns (a, A): per-vertex shoelace cross terms and
    signed area, both (N_pop, n_points) and (N_pop,) respectively."""
    x_n = np.roll(x, -1, axis=1)
    y_n = np.roll(y, -1, axis=1)
    a = x * y_n - x_n * y
    A = 0.5 * np.sum(a, axis=1)
    return a, A


def batched_polygon_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _, A = _polygon_shoelace_terms(x, y)
    return A


def batched_polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, A = _polygon_shoelace_terms(x, y)
    x_n = np.roll(x, -1, axis=1)
    y_n = np.roll(y, -1, axis=1)
    x_c = 1.0 / (6.0 * A) * np.sum(a * (x + x_n), axis=1)
    y_c = 1.0 / (6.0 * A) * np.sum(a * (y + y_n), axis=1)
    return x_c, y_c


def batched_polygon_inertia(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Area moments of inertia about the CENTROID (parallel-axis corrected), matching
    aerosandbox.Polygon.Ixx()/Iyy()/Ixy() convention exactly (verified against them).
    Also returns `J` = Ixx + Iyy about the centroid -- the polar SECOND MOMENT OF AREA,
    not a Saint-Venant torsional constant. (aerosandbox.Polygon.J() has a bug where it
    omits the parallel-axis shift, silently returning the origin-referenced value
    instead of the centroid-referenced one its docstring claims; this implementation
    does the shift correctly. Even corrected, this polar-moment-of-area J is a
    different, larger quantity than XFoil's BEND "J", which is a true torsional
    rigidity solved via torsion theory -- for a thin elongated section like an
    airfoil, actual torsional stiffness is dominated by thickness^3 effects and is
    much smaller than Ixx + Iyy suggests. Treat this J as a fast bending/preliminary-
    sizing proxy, not an XFoil-equivalent torsion constant.)"""
    a, A = _polygon_shoelace_terms(x, y)
    x_n = np.roll(x, -1, axis=1)
    y_n = np.roll(y, -1, axis=1)

    x_c, y_c = batched_polygon_centroid(x, y)

    Ixx_origin = 1.0 / 12.0 * np.sum(a * (y**2 + y * y_n + y_n**2), axis=1)
    Iyy_origin = 1.0 / 12.0 * np.sum(a * (x**2 + x * x_n + x_n**2), axis=1)
    Ixy_origin = 1.0 / 24.0 * np.sum(a * (x * y_n + 2 * x * y + 2 * x_n * y_n + x_n * y), axis=1)

    Ixx = Ixx_origin - A * y_c**2
    Iyy = Iyy_origin - A * x_c**2
    Ixy = Ixy_origin - A * x_c * y_c
    J = Ixx + Iyy

    return dict(area=A, x_c=x_c, y_c=y_c, Ixx=Ixx, Iyy=Iyy, Ixy=Ixy, J=J)


def design_matrix_structural_summary(X: np.ndarray, n_points_per_side: int = 100) -> dict:
    """Convenience: area, centroid, Ixx, Iyy, Ixy, J (polar moment of area) for a
    whole population at once, straight from the design matrix."""
    params = unpack_design_matrix(X)
    x, y = batched_closed_polygon_xy(
        params["upper_weights"], params["lower_weights"], params["leading_edge_weight"], params["TE_thickness"],
        n_points_per_side=n_points_per_side,
    )
    return batched_polygon_inertia(x, y)


def design_matrix_geometry_summary(X: np.ndarray, x_stations: np.ndarray) -> dict:
    """Convenience: all batched geometry quantities for a population at once."""
    params = unpack_design_matrix(X)
    upper_w, lower_w = params["upper_weights"], params["lower_weights"]
    le_w, te_t = params["leading_edge_weight"], params["TE_thickness"]
    return dict(
        local_thickness=batched_local_thickness(upper_w, lower_w, le_w, te_t, x_stations),
        max_thickness=batched_max_thickness(upper_w, lower_w, le_w, te_t),
        le_radius=batched_le_radius(upper_w, lower_w),
        te_angle_deg=batched_te_angle_deg(upper_w, lower_w, te_t),
    )
