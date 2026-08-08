"""
EdgeFoil — multi-objective airfoil optimizer, Streamlit app.

Pick a starting airfoil, define 2+ operating-point objectives (a genuine
multi-objective trade-off, e.g. cruise drag vs. climb drag), run NSGA-II
(driven by batched NeuralFoil evaluations), inspect the resulting Pareto
front, optionally cross-check candidates against real XFoil, and download
the optimized airfoil(s).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import geometry as geo
from core import io_utils as io_u
from core import moo
from core import objectives as obj
from core import verify

st.set_page_config(page_title="EdgeFoil", layout="wide", page_icon="✈️")

OBJECTIVE_LABEL_TO_KIND = {
    "Minimize drag (CD)": "minimize_cd",
    "Minimize |pitching moment| (CM)": "minimize_abs_cm",
    "Maximize lift (CL) — stall margin": "maximize_cl",
}

DEFAULT_DESIGN_POINTS_DF = pd.DataFrame(
    [
        {"label": "cruise", "Re": 1.0e6, "Mach": 0.0, "point_type": "Target CL", "value": 0.3,
         "objective": "Minimize drag (CD)"},
        {"label": "climb", "Re": 6.0e5, "Mach": 0.0, "point_type": "Target CL", "value": 0.9,
         "objective": "Minimize drag (CD)"},
    ]
)

for key, default in [("result", None), ("verification", None), ("baseline_verification", None), ("selected_index", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def build_design_points(
    df: pd.DataFrame, n_crit, xtr_upper, xtr_lower, min_confidence, max_cl_error, alpha_search_range
) -> tuple[list[obj.DesignPoint], list[str]]:
    points, errors = [], []
    for i, row in df.iterrows():
        label = str(row.get("label", "")).strip() or f"point_{i + 1}"
        try:
            objective = OBJECTIVE_LABEL_TO_KIND[row["objective"]]
            kwargs = dict(
                label=label, re=float(row["Re"]), mach=float(row["Mach"]), objective=objective,
                n_crit=n_crit, xtr_upper=xtr_upper, xtr_lower=xtr_lower,
                min_confidence=min_confidence, max_cl_error=max_cl_error,
                alpha_search_range_deg=alpha_search_range,
            )
            if row["point_type"] == "Target CL":
                kwargs["cl_target"] = float(row["value"])
            else:
                kwargs["alpha_fixed_deg"] = float(row["value"])
            points.append(obj.DesignPoint(**kwargs))
        except Exception as e:
            errors.append(f"Row {i + 1} ('{label}'): {e}")
    return points, errors


def airfoil_xy(x: np.ndarray, name: str) -> np.ndarray:
    kulfan = geo.design_vector_to_kulfan_airfoil(x, name=name)
    return np.array(kulfan.to_airfoil(n_coordinates_per_side=150).coordinates)


st.title("✈️ EdgeFoil")
st.caption("Multi-objective airfoil optimization that gives you the edge — NeuralFoil-driven NSGA-II shape optimization, cross-checked against XFoil.")

# ---------------------------------------------------------------------
# Sidebar: starting airfoil
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("1. Starting airfoil")
    source_mode = st.radio("Source", ["Preset", "NACA code", "Upload .dat"], index=0)

    baseline_airfoil = None
    load_error = None
    try:
        if source_mode == "Preset":
            preset_key = st.selectbox("Preset", list(geo.PRESET_AIRFOILS.keys()))
            baseline_airfoil = geo.airfoil_from_preset(preset_key)
        elif source_mode == "NACA code":
            code = st.text_input("NACA 4 or 5-digit code", value="2412")
            if code:
                baseline_airfoil = geo.airfoil_from_naca(code)
        else:
            uploaded = st.file_uploader("Upload a .dat airfoil (Selig format, x y per line)", type=["dat", "txt"])
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="ignore")
                baseline_airfoil = geo.airfoil_from_dat_text(text, name=Path(uploaded.name).stem)
    except Exception as e:
        load_error = str(e)

    if load_error:
        st.error(load_error)
    elif baseline_airfoil is not None:
        st.success(f"Loaded **{baseline_airfoil.name}** ({len(baseline_airfoil.coordinates)} pts)")
        preview_coords = np.array(baseline_airfoil.coordinates)
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(x=preview_coords[:, 0], y=preview_coords[:, 1], mode="lines", line=dict(color="#4C78A8")))
        fig_preview.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_preview.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_preview, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.header("3. Optimizer settings")
    preset = st.radio(
        "Speed / quality",
        ["Fast draft (~15s)", "Balanced (~1 min)", "Thorough (~3-5 min)", "Custom"],
        index=1,
    )
    preset_map = {"Fast draft (~15s)": (40, 40), "Balanced (~1 min)": (80, 120), "Thorough (~3-5 min)": (150, 250)}
    if preset == "Custom":
        pop_size = st.number_input("Population size", 10, 500, 80, step=10)
        n_gen = st.number_input("Generations", 5, 600, 120, step=10)
    else:
        pop_size, n_gen = preset_map[preset]
        st.caption(f"Population {pop_size} · Generations {n_gen} · ~{pop_size * n_gen} evaluations")

    model_size_options = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
    use_multi_fidelity = st.checkbox(
        "Multi-fidelity scheduling (recommended)", value=True,
        help="Run most generations on a small, fast NeuralFoil model for broad exploration, then switch to "
        "a larger, more accurate model for the final generations to refine and rank the converging front. "
        "Benchmarked against real XFoil on perturbed shapes: roughly break-even to ~5% faster overall than a "
        "flat 'medium' run, with meaningfully better accuracy on the final result (~2% mean CD error at "
        "'xlarge' vs ~5% at 'medium'). The speed gain is modest, not dramatic -- accuracy is the main benefit.",
    )
    if use_multi_fidelity:
        c1, c2 = st.columns(2)
        early_model_size = c1.selectbox("Early-phase (fast) size", model_size_options, index=model_size_options.index("small"))
        late_model_size = c2.selectbox("Late-phase (accurate) size", model_size_options, index=model_size_options.index("xlarge"))
        late_phase_fraction = st.slider(
            "Late-phase fraction", 0.05, 0.50, 0.20, 0.05,
            help="Fraction of total generations spent in the accurate late phase, refining the front found "
            "during the fast early phase.",
        )
        model_size = late_model_size  # used for final display (polars, etc.) before a result exists
    else:
        model_size = st.selectbox(
            "NeuralFoil model size", model_size_options, index=model_size_options.index("medium"),
            help="Bigger = more accurate, slower. Applied flat across every generation.",
        )

# ---------------------------------------------------------------------
# Main: design points (multi-objective, mandatory >= 2)
# ---------------------------------------------------------------------
st.header("2. Design points — this is where the trade-off comes from")
st.caption(
    "Each row becomes one objective. At least 2 are required so the optimizer explores a genuine "
    "Pareto trade-off (e.g. low-speed drag vs. high-lift drag) instead of a single-point optimum."
)
design_points_df = st.data_editor(
    DEFAULT_DESIGN_POINTS_DF,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "label": st.column_config.TextColumn("Label", required=True),
        "Re": st.column_config.NumberColumn("Reynolds #", min_value=1e3, format="%.0f", required=True),
        "Mach": st.column_config.NumberColumn("Mach", min_value=0.0, max_value=0.9, step=0.01, required=True),
        "point_type": st.column_config.SelectboxColumn("Point type", options=["Target CL", "Fixed alpha (deg)"], required=True),
        "value": st.column_config.NumberColumn("CL target / alpha (deg)", format="%.3f", required=True),
        "objective": st.column_config.SelectboxColumn("Objective", options=list(OBJECTIVE_LABEL_TO_KIND.keys()), required=True),
    },
    key="design_points_editor",
)

with st.expander("4. Geometry constraints (advanced)"):
    c1, c2 = st.columns(2)
    max_weight_step = c1.slider(
        "Max shape deviation from baseline", 0.05, 0.60, 0.30, 0.01,
        help="Bound on each Kulfan (CST) weight's change from the baseline. Larger = more design freedom, "
        "but further from NeuralFoil's best-validated shape range.",
    )
    thickness_retention = c1.slider("Minimum local thickness (% of baseline)", 50, 100, 85) / 100.0
    le_radius_retention = c1.slider("Minimum leading-edge radius (% of baseline)", 10, 100, 40) / 100.0
    min_te_angle = c2.number_input("Minimum trailing-edge angle (deg)", 0.0, 20.0, 3.0)
    enable_min_thickness = c2.checkbox("Enforce an absolute minimum max-thickness ratio (t/c)", value=False)
    min_thickness_ratio = c2.number_input("Minimum t/c", 0.02, 0.30, 0.08) if enable_min_thickness else None

    st.caption(
        "Structural section properties (Ixx, J) are computed directly from the airfoil's solid cross-section "
        "shape — the same kind of calculation XFoil's `BEND` command does. **J here is the polar second moment "
        "of area (Ixx + Iyy) about the centroid, not a Saint-Venant torsional constant** — XFoil's BEND reports "
        "true torsional rigidity, which for a thin section like an airfoil is much smaller. Use J as a fast "
        "bending/preliminary-sizing proxy, not an XFoil-equivalent torsion limit."
    )
    enable_min_ixx = c1.checkbox("Enforce a minimum bending stiffness (Ixx)", value=False)
    min_ixx_retention = (c1.slider("Minimum Ixx (% of baseline)", 5, 100, 50) / 100.0) if enable_min_ixx else None
    enable_min_j = c2.checkbox("Enforce a minimum polar moment of area (J = Ixx+Iyy)", value=False)
    min_j_retention = (c2.slider("Minimum J (% of baseline)", 5, 100, 50) / 100.0) if enable_min_j else None

with st.expander("5. Analysis settings (advanced)"):
    c1, c2, c3 = st.columns(3)
    n_crit = c1.number_input("Ncrit (transition)", 1.0, 18.0, 9.0, help="9 = avg wind tunnel · 11-13 = motorglider · 12-14 = sailplane")
    xtr_upper = c1.slider("Forced transition x/c, upper", 0.0, 1.0, 1.0)
    xtr_lower = c1.slider("Forced transition x/c, lower", 0.0, 1.0, 1.0)
    min_confidence = c2.slider("Min. NeuralFoil confidence required", 0.5, 1.0, 0.85)
    max_cl_error = c2.number_input("Max allowed CL tracking error", 0.005, 0.20, 0.03)
    alpha_range = c3.slider("Alpha search range for CL targets (deg)", -20, 30, (-8, 18))

st.divider()
run_clicked = st.button("🚀 Run optimization", type="primary", disabled=(baseline_airfoil is None), use_container_width=True)

if run_clicked:
    design_points, dp_errors = build_design_points(
        design_points_df, n_crit, xtr_upper, xtr_lower, min_confidence, max_cl_error, tuple(alpha_range)
    )
    if dp_errors:
        for e in dp_errors:
            st.error(e)
    elif len(design_points) < 2:
        st.error("Add at least 2 design points — multi-objective optimization needs a genuine trade-off.")
    else:
        baseline = geo.to_kulfan_baseline(baseline_airfoil)
        geometry_constraints = moo.GeometryConstraints(
            thickness_retention=thickness_retention,
            min_max_thickness_ratio=min_thickness_ratio,
            le_radius_retention=le_radius_retention,
            min_te_angle_deg=min_te_angle,
            min_ixx_retention=min_ixx_retention,
            min_j_retention=min_j_retention,
        )
        model_size_schedule = (
            moo.ModelSizeSchedule(
                early_model_size=early_model_size, late_model_size=late_model_size, late_phase_fraction=late_phase_fraction
            )
            if use_multi_fidelity
            else None
        )
        config = moo.OptimizationConfig(
            baseline=baseline,
            design_points=design_points,
            max_weight_step=max_weight_step,
            geometry=geometry_constraints,
            model_size=model_size,
            model_size_schedule=model_size_schedule,
            population_size=int(pop_size),
            n_generations=int(n_gen),
            seed=0,
        )
        if use_multi_fidelity:
            spinner_text = (
                f"Running NSGA-II — {pop_size} population × {n_gen} generations, {len(design_points)} objectives "
                f"({early_model_size} → {late_model_size} multi-fidelity)..."
            )
        else:
            spinner_text = f"Running NSGA-II — {pop_size} population × {n_gen} generations, {len(design_points)} objectives..."
        with st.spinner(spinner_text):
            try:
                result = moo.run_optimization(config)
                st.session_state.result = result
                st.session_state.verification = None
                st.session_state.baseline_verification = None
                st.session_state.selected_index = None
            except Exception as e:
                st.error(f"Optimization failed: {e}")

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
result: moo.OptimizationResult | None = st.session_state.result
if result is not None:
    st.header("Results")
    F = result.F_pareto
    labels = result.objective_labels
    n_front = F.shape[0]
    st.write(f"**{n_front}** non-dominated designs found after **{result.n_evaluations}** NeuralFoil evaluations.")

    col_x, col_y = st.columns(2)
    x_obj = col_x.selectbox("X axis", labels, index=0)
    y_obj = col_y.selectbox("Y axis", labels, index=min(1, len(labels) - 1))
    xi, yi = labels.index(x_obj), labels.index(y_obj)

    if st.session_state.selected_index is None or st.session_state.selected_index >= n_front:
        st.session_state.selected_index = moo.knee_point_index(F)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=F[:, xi], y=F[:, yi], mode="markers", name="Pareto front",
            marker=dict(size=10, color="#4C78A8"),
            hovertemplate=f"{x_obj}: %{{x:.5f}}<br>{y_obj}: %{{y:.5f}}<br>index: %{{text}}<extra></extra>",
            text=[str(i) for i in range(n_front)],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.baseline_F[0, xi]], y=[result.baseline_F[0, yi]], mode="markers", name="Baseline",
            marker=dict(size=16, color="#E45756", symbol="star"),
        )
    )
    sel = st.session_state.selected_index
    fig.add_trace(
        go.Scatter(
            x=[F[sel, xi]], y=[F[sel, yi]], mode="markers", name="Selected",
            marker=dict(size=14, color="#54A24B", symbol="circle-open", line=dict(width=3)),
        )
    )
    fig.update_layout(xaxis_title=x_obj, yaxis_title=y_obj, height=450, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    front_table = pd.DataFrame({"index": range(n_front), **{labels[j]: F[:, j] for j in range(len(labels))}})
    st.session_state.selected_index = st.selectbox(
        "Inspect design # (defaults to the balanced 'knee point')",
        options=list(range(n_front)),
        index=int(st.session_state.selected_index),
        format_func=lambda i: f"{i} — " + ", ".join(f"{labels[j]}={F[i, j]:.5f}" for j in range(len(labels))),
    )
    sel = st.session_state.selected_index

    st.subheader(f"Design #{sel}")
    col_shape, col_table = st.columns([1, 1])

    with col_shape:
        base_coords = np.array(result.config.baseline.airfoil.coordinates)
        sel_coords = airfoil_xy(result.X_pareto[sel], f"design_{sel}")
        fig_shape = go.Figure()
        fig_shape.add_trace(go.Scatter(x=base_coords[:, 0], y=base_coords[:, 1], mode="lines", name="Baseline", line=dict(color="#B0B0B0", dash="dash")))
        fig_shape.add_trace(go.Scatter(x=sel_coords[:, 0], y=sel_coords[:, 1], mode="lines", name=f"Design #{sel}", line=dict(color="#54A24B", width=2)))
        fig_shape.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_shape.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_shape, use_container_width=True)

        dat_text = io_u.airfoil_dat_text(result.X_pareto[sel], f"design_{sel}")
        st.download_button("⬇️ Download this airfoil (.dat)", dat_text, file_name=f"design_{sel}.dat", use_container_width=True)

    with col_table:
        rows = []
        for dp in result.config.design_points:
            r_opt = result.per_point_results[sel][dp.label]
            r_base = result.baseline_per_point[dp.label]
            rows.append(
                {
                    "point": dp.label,
                    "alpha [deg]": f"{float(r_opt.alpha_deg[0]):.2f}",
                    "CL": f"{float(r_opt.CL[0]):.4f}",
                    "CD": f"{float(r_opt.CD[0]):.5f}",
                    "CD baseline": f"{float(r_base.CD[0]):.5f}",
                    "Δ CD": f"{float(r_opt.CD[0] - r_base.CD[0]):+.5f}",
                    "CM": f"{float(r_opt.CM[0]):.4f}",
                    "confidence": f"{float(r_opt.confidence[0]):.3f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        struct_base = geo.design_matrix_structural_summary(geo.baseline_design_vector(result.config.baseline)[None, :])
        struct_sel = geo.design_matrix_structural_summary(result.X_pareto[sel][None, :])
        st.caption("Structural section properties (nondimensionalized by chord; J = Ixx + Iyy, polar moment of area — see note above, not XFoil's torsional constant)")
        m1, m2 = st.columns(2)
        m1.metric("Ixx", f"{float(struct_sel['Ixx'][0]):.3e}", f"{(float(struct_sel['Ixx'][0]) / float(struct_base['Ixx'][0]) - 1):+.1%} vs baseline")
        m2.metric("J (Ixx+Iyy)", f"{float(struct_sel['J'][0]):.3e}", f"{(float(struct_sel['J'][0]) / float(struct_base['J'][0]) - 1):+.1%} vs baseline")

    st.subheader("Drag polar (NeuralFoil sweep, baseline vs. selected)")
    re_for_sweep = result.config.design_points[0].re
    sweep_base = obj.sweep_polar_neuralfoil(geo.baseline_design_vector(result.config.baseline), re=re_for_sweep, model_size=result.config.final_model_size)
    sweep_sel = obj.sweep_polar_neuralfoil(result.X_pareto[sel], re=re_for_sweep, model_size=result.config.final_model_size)
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatter(x=sweep_base["CD"], y=sweep_base["CL"], mode="lines", name="Baseline", line=dict(color="#B0B0B0", dash="dash")))
    fig_polar.add_trace(go.Scatter(x=sweep_sel["CD"], y=sweep_sel["CL"], mode="lines", name=f"Design #{sel}", line=dict(color="#54A24B")))
    fig_polar.update_layout(xaxis_title="CD", yaxis_title="CL", height=380, legend=dict(orientation="h", y=1.1))
    st.caption(f"Swept at Re={re_for_sweep:.3g} (first design point's Reynolds number).")
    st.plotly_chart(fig_polar, use_container_width=True)

    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    zip_bytes = io_u.pareto_front_zip(result)
    col_dl1.download_button("⬇️ Download full Pareto front (.zip: all .dat + CSV)", zip_bytes, file_name="pareto_front.zip", mime="application/zip", use_container_width=True)
    csv_text = io_u.pareto_front_csv(result)
    col_dl2.download_button("⬇️ Download Pareto front summary (.csv)", csv_text, file_name="pareto_front_summary.csv", mime="text/csv", use_container_width=True)

    # -------------------------------------------------------------
    # XFoil verification
    # -------------------------------------------------------------
    st.divider()
    st.header("XFoil verification")
    st.caption(
        "NeuralFoil is a fast neural-network surrogate; it can be locally wrong even at high reported "
        "confidence. This re-runs real XFoil on a representative subset of the front (same Re/Mach/CL "
        "targets) so you can see how much to trust each design before building it."
    )
    max_verify_points = st.slider("Number of front designs to verify", 2, min(30, n_front), min(10, n_front))
    if st.button("🔬 Verify with XFoil"):
        progress_bar = st.progress(0.0, text="Running XFoil...")

        def _progress(done, total):
            progress_bar.progress(done / total, text=f"Running XFoil... {done}/{total}")

        try:
            rows_v = verify.verify_front_with_xfoil(result, max_points=max_verify_points, progress_callback=_progress)
            base_rows_v = verify.baseline_verification(result)
            st.session_state.verification = rows_v
            st.session_state.baseline_verification = base_rows_v
        except Exception as e:
            st.error(f"XFoil verification failed: {e}")
        finally:
            progress_bar.empty()

    if st.session_state.verification is not None:
        def _rows_to_df(rows_v):
            return pd.DataFrame(
                [
                    {
                        "front_index": r.front_index,
                        "point": r.design_point_label,
                        "NF alpha": round(r.neuralfoil_alpha_deg, 3),
                        "NF CL": round(r.neuralfoil_CL, 4),
                        "NF CD": round(r.neuralfoil_CD, 5),
                        "XFoil converged": r.xfoil_converged,
                        "XFoil alpha": round(r.xfoil_alpha_deg, 3) if r.xfoil_alpha_deg is not None else None,
                        "XFoil CL": r.xfoil_CL,
                        "XFoil CD": r.xfoil_CD,
                        "CD rel. error": f"{r.cd_relative_error:+.1%}" if r.cd_relative_error is not None else "—",
                        "Error": (r.error_message or "").splitlines()[0][:100] if r.error_message else "—",
                    }
                    for r in rows_v
                ]
            )

        st.caption("Baseline airfoil, for reference:")
        st.dataframe(_rows_to_df(st.session_state.baseline_verification), use_container_width=True, hide_index=True)
        st.caption("Verified Pareto-front designs:")
        st.dataframe(_rows_to_df(st.session_state.verification), use_container_width=True, hide_index=True)

        all_rows = list(st.session_state.baseline_verification) + list(st.session_state.verification)
        failed_rows = [r for r in all_rows if not r.xfoil_converged]
        if failed_rows:
            n_failed, n_total = len(failed_rows), len(all_rows)
            if n_failed == n_total:
                st.error(
                    f"All {n_total} XFoil runs failed — this is a setup/environment problem, not real "
                    f"aerodynamic non-convergence (real non-convergence is selective, not 100%). See the "
                    f"error detail below."
                )
            else:
                st.warning(f"{n_failed} of {n_total} XFoil run(s) did not converge — treat those designs as unverified/suspect.")
            with st.expander(f"First failure detail ({failed_rows[0].design_point_label}, front_index={failed_rows[0].front_index})"):
                st.code(failed_rows[0].error_message or "(no error message captured)")

        verify_csv = io_u.verification_csv(st.session_state.verification)
        st.download_button("⬇️ Download verification results (.csv)", verify_csv, file_name="xfoil_verification.csv", mime="text/csv")
else:
    st.info("Configure a starting airfoil and at least 2 design points, then click **Run optimization**.")
