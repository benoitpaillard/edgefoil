# Airfoil Multi-Objective Optimizer

A shape optimizer for 2D airfoils that combines **NeuralFoil** (a fast neural-network
aerodynamics surrogate) with **XFoil** (the classic panel-method solver) behind a
web UI anyone can use — no Python required to run an optimization.

- **NeuralFoil drives the optimization loop.** It evaluates an entire population of
  candidate airfoils (and the alpha-sweep needed to hit each CL target) in milliseconds,
  which is what makes a real genetic multi-objective search (NSGA-II, via `pymoo`)
  practical here instead of a single-point gradient optimization.
- **XFoil verifies the result.** NeuralFoil is a surrogate — it can be locally wrong
  even at high reported confidence. After optimizing, a representative subset of the
  Pareto front is re-analyzed with actual XFoil at the same Re/Mach/CL targets, so you
  can see the agreement before trusting a design.
- **Multi-objective is the point, not an option.** You define 2+ "design points"
  (e.g. minimize drag at a low-lift cruise condition *and* at a high-lift climb
  condition); the optimizer returns the actual Pareto front of that trade-off using
  NSGA-II, rather than a single blended answer.

## Quickstart (web app)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the printed URL (defaults to `http://localhost:8501`). Pick a starting airfoil,
edit the design-point table (2+ rows required), optionally tweak the geometry/analysis
constraints, and click **Run optimization**.

### Sharing it with others

- **Same network:** `streamlit run app.py --server.address 0.0.0.0` and share
  `http://<your-machine-ip>:8501` with anyone on the same LAN.
- **Public link, zero server maintenance:** push this folder to a GitHub repo and
  deploy it for free on [Streamlit Community Cloud](https://streamlit.io/cloud) —
  point it at `app.py`. (Not done automatically here since it needs your own GitHub/
  Streamlit account.)

## Quickstart (command line)

```bash
python3 cli.py --airfoil naca2412 \
  --point "label=cruise,re=1e6,cl=0.3,objective=cd" \
  --point "label=climb,re=6e5,cl=0.9,objective=cd" \
  --verify --out outputs/my_run
```

`--airfoil` accepts a 4/5-digit NACA code, a path to a Selig-format `.dat` file, or a
preset name (`naca0012`, `naca2412`, `naca4412`, `naca6409`, `naca23012`). Repeat
`--point` for each objective (at least 2 required). Each point spec is
`key=value` pairs separated by commas:

| Key | Meaning |
|---|---|
| `label` | Name for this design point (required) |
| `re` | Reynolds number (required) |
| `mach` | Mach number (default 0) |
| `cl` | Target CL — the optimizer solves for the alpha that hits it | 
| `alpha` | Fixed angle of attack (deg) — use instead of `cl`, e.g. for a stall-margin check |
| `objective` | `cd` (minimize drag), `cm` (minimize \|pitching moment\|), or `maxcl` (maximize CL — needs `alpha`, not `cl`) |

For more than a couple of objectives or custom analysis settings, use `--config
some.json` instead (or alongside) `--point`:

```json
{
  "design_points": [
    {"label": "cruise", "re": 1000000, "cl_target": 0.3, "objective": "minimize_cd"},
    {"label": "climb", "re": 600000, "cl_target": 0.9, "objective": "minimize_cd"},
    {"label": "stall_margin", "re": 600000, "alpha_fixed_deg": 14, "objective": "maximize_cl"}
  ]
}
```

Run `python3 cli.py --help` for every geometry/analysis/optimizer flag (population
size, generations, model size, thickness/LE-radius/TE-angle constraints, etc.).

CLI output directory contains: `pareto_front_summary.csv`, `pareto_front.zip` (every
front airfoil as a `.dat` + the summary CSV), `knee_point_comparison.png` (baseline vs.
the balanced trade-off pick), and — with `--verify` — `xfoil_verification.csv`.

## Structural constraints (Ixx, J)

Besides the aerodynamic objectives/constraints, you can require a minimum bending
stiffness (`Ixx`) and/or polar moment of area (`J`), each expressed as a percentage
of the baseline airfoil's value — in the UI's "Geometry constraints" expander, or via
`--min-ixx-retention` / `--min-j-retention` (0-1 fraction) on the CLI. Both are computed
directly from the solid cross-section shape (same kind of calculation as XFoil's
`BEND` command), nondimensionalized by chord, and are exact — verified against
AeroSandbox's own `Ixx()`/`Iyy()` to within polygon-discretization noise.

**Important caveat on `J`:** this is the polar *second moment of area* about the
centroid (`Ixx + Iyy`), not a Saint-Venant torsional constant. XFoil's `BEND` command
also reports a "J", but that one is a true torsional rigidity solved via torsion
theory — for a thin elongated section like an airfoil, actual torsional stiffness is
dominated by thickness³ effects and comes out much smaller than `Ixx + Iyy` suggests
(checked on a NACA 2412: XFoil's `BEND` gives `J ≈ 2.72e-4`, while the correct
centroid-referenced `Ixx + Iyy` is `≈ 4.61e-3` — about 17× larger). A true torsion
constant would need a membrane-analogy/FEM-style solve per candidate shape, which
isn't practical to run thousands of times per NSGA-II generation. Treat this `J` as a
fast bending/preliminary-sizing proxy, not an XFoil-equivalent torsion limit.

Separately: AeroSandbox's own `Polygon.J()` method has a bug (as of `aerosandbox`
4.2.9) — it omits the parallel-axis correction its docstring claims, so it silently
returns the *origin*-referenced `Ixx + Iyy` instead of the *centroid*-referenced one.
This project's `core/geometry.py:batched_polygon_inertia()` implements the correction
correctly from scratch (needed anyway for fast whole-population batching) and is
verified against AeroSandbox's `Ixx()`/`Iyy()`/`Ixy()` (which do not have this bug).

## How it works

```
core/geometry.py     Airfoil loading (NACA / .dat / presets) + Kulfan (CST, 8 weights/
                      side — what NeuralFoil is trained on) parameterization. All
                      geometry constraint math (thickness, LE radius, TE angle) is
                      implemented as plain vectorized numpy so it stays fast across an
                      entire optimizer population, cross-checked against AeroSandbox's
                      own KulfanAirfoil to machine precision.
core/objectives.py   "Design points" (Re/Mach/CL-or-alpha/objective) and batched
                      NeuralFoil evaluation, including alpha-search to hit a target CL
                      across a whole population in one call.
core/moo.py           pymoo NSGA-II problem: objectives + constraints (geometry,
                      NeuralFoil confidence, CL-tracking error), seeded around the
                      baseline airfoil so the optimizer never does worse than the
                      starting point and converges fast in the ~18-dimensional CST
                      design space.
core/xfoil_driver.py  Minimal subprocess driver that talks to the `xfoil` binary
                      directly via its PACC polar-accumulation feature (see note
                      below on why this isn't AeroSandbox's built-in XFoil wrapper).
core/verify.py        Re-checks a representative (evenly spread) subset of the Pareto
                      front against real XFoil, in parallel (thread pool — XFoil calls
                      are blocking subprocesses).
core/io_utils.py       .dat / CSV / zip export.
app.py                 Streamlit UI.
cli.py                  Command-line entry point (same core, headless).
```

### Why NSGA-II instead of a gradient-based solver

Gradient-based NLP solvers (CasADi/IPOPT, used in this user's other airfoil scripts)
need a single scalar objective, so multi-point drag minimization there is done via
weighted-sum scalarization — one solve per weight combination, which can miss
non-convex regions of the true Pareto front and requires picking weights ahead of
time. NSGA-II (via `pymoo`) optimizes all objectives simultaneously and returns the
actual Pareto front in one run. This is only practical here because NeuralFoil is
cheap enough to batch-evaluate an entire population in milliseconds.

### Note on XFoil integration

`aerosandbox.XFoil` (the built-in wrapper) reproducibly crashes with a SIGFPE partway
through its paneling sequence on this machine's XFoil 6.99 build. `core/xfoil_driver.py`
drives the `xfoil` binary directly instead, using a sequence
(`LOAD → PANE → OPER → VISC → PACC → ASEQ/ALFA/CL → QUIT`) verified to work reliably,
with polars parsed from XFoil's own `PACC` output file rather than scraped from
interactive stdout.

## Requirements

- Python packages: see `requirements.txt`.
- The `xfoil` binary on `PATH` (only needed for the verification step — optimization
  itself only needs NeuralFoil/AeroSandbox). On Debian/Ubuntu: `apt install xfoil`, or
  build from [the MIT source](https://web.mit.edu/drela/Public/web/xfoil/).

## Limitations

- NeuralFoil's Kulfan parameterization is fixed at 8 CST weights per side (a network
  training constraint, not a choice made here).
- The optimizer stays within a bounded neighborhood of the starting airfoil (control
  via "max shape deviation" in the UI / `--max-weight-step` in the CLI) — this keeps
  designs inside NeuralFoil's well-validated range and avoids degenerate geometry. It
  is a *shape refiner*, not a from-scratch airfoil designer.
- 2D only. No 3D wing/planform effects, structural sizing, or off-design robustness
  beyond whatever design points you explicitly add.
