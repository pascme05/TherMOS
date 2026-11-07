# Plan: TherMOS GUI

Status: 07.11.2026, draft for review. Nothing is implemented yet.

## 1. Goal

A local Streamlit GUI for the Python port of TherMOS (`src/py`), built like the OPTELIA (RAISE) and PyPowerSim GUIs with
the `toolkit-gui` scaffold. The toolkit stays unchanged and runnable from `start*.py` / `setup/**/*.py`; the GUI only
wraps `src.py.main.main(setup, path)`.

1. **One mode per model order reduction / model family** (RC, SS, SF, PO, PS, ML, DL, PINN).
2. **Theory & Tools** section as in OPTELIA: power losses and thermal stack taken over from RAISE, plus new pages on
   thermal networks, MOR principles and numerics.
3. **Board designer** (2D POD): place chips on a board, select board and chip technology, simulate the 2D field, look
   at the POD live and export the result as a TherMOS data set that the PO/PS modes can train on.

## 2. Survey results (what the code dictates)

| Item | Finding | Consequence for the GUI |
|---|---|---|
| Entry point | `main(setup, path)` returns `{'data', 'pred', 'grt', 'results', 'mdls', 'para'}` | results are read directly from the return value, `parity.py` can compare scalars |
| Config loader | `loadPara(setup['config'], path)` is called **inside** `main` (line 125) | `FEED = 'xlsx'`: the GUI writes `config/_gui_run.xlsx`, zero toolkit changes |
| Config workbook | sheets `Exp` (6), `Dat` (9), `Mdl` (53, categories gen/ss/rc/pod/dl/nn/knn/dt/svm/gpr/en), `Par` (12); columns Name / Category / Description / Variable / **Type** / Value / Unit | nested `para[sheet][cat][var]`; the extra `Type` column (`integer+`, `float+`, `string`) is used for the widget type. Parameters the Python code reads but the workbook lacks (`pod.protocol`, `pinn.*`, `featRoll`, `EWMA`, `EWMS`) are added through `adapter.OPTIONS`/extra rows, not by editing the user's workbooks |
| Run settings | 35 `setup[...]` keys in `start.py` (files/sheets/IDs/ratios, `inp`/`out`/`ref`, `sel*`, `datDim`, `legacy`, …) | `RUN_FIELDS` with file pickers on `data/**` (`.mat`/`.xlsx`), list widgets for `inp`/`out`/`ref` |
| Model selection | eight `sel*` flags, several may be active at once | the mode page sets its own flag (and sub-mode value) and all others to 0; a separate **Compare** page runs several |
| Ready-made configs | `config/tutorial/tutorial_<X>.xlsx` for every model, `default*`, `journal/*`, `lecture/*`, `other/*` | each mode loads its tutorial config + start setup as default; the others are offered as `EXTRA_CONFIGS` |
| 2D/3D data | `.mat` with `geo, k, rho, Cp, hc, Ta, fl, X (Nt×Nn), y, t, Ts, Lx, Ly, dx, dy` | the board designer writes exactly this format |
| Figures | `main` now saves PNGs via `saveFigs` and closes them | the runner must capture figures **before** `saveFigs` closes them, or read the PNGs from `results/<name>/` (chosen: read the PNGs, no toolkit change) |
| POD projection | `calcPodQuad` ignores `hc` ("WARN: Convective boundary conditions … not included") | must be fixed for the board designer, see WP 7 |
| Environment | the active Python has no `streamlit`/`plotly` | `gui/requirements.txt`; install only after your OK |

## 3. Navigation

```
Project            Setup (name, user, data split, inputs/outputs, legacy switch), file groups
Modes              RC · SS · SF · PO · PS · ML · DL · PINN      (each: Workflow | Parameters | Inputs | Theory | Run | Results)
Compare            run several models on the same data, one table + overlay chart
Theory & Tools     Power losses · Thermal stack · Thermal networks · MOR & POD · Numerics · Board designer
Results            browser of results/<name>/ (PNGs, gui_run.pkl, key figures)
Help               Tutorials (setup/tutorial/*), Documentation (docu/GUI.md)
```

## 4. Modes

| Mode | Flag (sub-mode) | Default config / data | Results view (Plotly) |
|---|---|---|---|
| RC Foster/Cauer | `selRC = 1` | `tutorial_RC` | tiles MAE/RMSE/MAX/FIT; T(t) measured vs predicted per output; table of Rth/τ pairs; Zth(t) curve |
| SS state space | `selSS = 1` | `tutorial_SS` / `default` | tiles; T(t); poles in the s-plane, step response, model order vs. fit |
| SF structure function | `selSF = 1` | `defaultRth` | tiles; cumulative/differential structure function, Zth |
| PO POD-Galerkin | `selPO = 1` | `tutorial_PO` / `default2D` | tiles; singular-value energy, mode shapes φ₁…φ_K, field measured/predicted/error with time slider, node response |
| PS POD state space | `selPS = 1` (2D) / `2` (3D) | `tutorial_PS` / `FR4` | as PO, plus the z-slice selector for 3D |
| ML | `selML = 1…` (KNN/RF/SVR/…) | `tutorial_ML` | tiles; T(t); feature importance / ranking |
| DL | `selDL = 1` | `tutorial_DL` | tiles; T(t); training/validation loss curve; warning: hours on CPU |
| PINN | `selPI = 1/2/3` (CNN/LSTM/CNN+LSTM) | `tutorial_PI` | as DL plus the physics-loss share |

Per mode (OPTELIA phase-2 items): `help.py` texts for every parameter (from the implementation, not the workbook
labels), `BASIC` set, `ACTIVE` rules (e.g. `pod.protocol` only in PO/PS, `ss.*` only in SS), run summary with
warnings (long runs, `train = 0` needs an existing `mdl/mdl_<tag>_<name>.pkl`, 1D data with PO/PS), workflow chart
(load → resample → normalise → train → test → metrics), Theory view with the model equations (Foster/Cauer, state space,
Galerkin projection, …).

## 5. Theory & Tools

| Page | Source | Content |
|---|---|---|
| Power losses | RAISE `src/tools/lossCalc.py`, `toolData.py`, `gui/views/tools/losses.py`, `topo_art.py` (copied, header Title → TherMOS) | loss calculator (conduction/switching per topology), technology comparison Si/SiC/GaN; button "use as P(t) in board designer" |
| Thermal stack | RAISE `src/tools/thermStack.py`, `gui/views/tools/thermal.py`, `pkg_art.py` | stack designer, Zth inside/outside/total, transient Tj, die attach & voids |
| Thermal networks | new, calls TherMOS `model/RC`, `model/SF` | Foster ↔ Cauer conversion, Zth(t) fit from a step response, structure function explorer |
| MOR & POD | new, calls `model/PO` | 1D rod / 2D plate example: snapshots → SVD → energy criterion → Galerkin ROM; error and speed-up vs. number of modes K; projection vs. calibrated protocols (2/3 vs. 5/6, F-P15/F-P16 explained) |
| Numerics | new, calls `numeric/*` | ODE solvers (Euler fwd/back, trapezoid, exact FOH/expm), stability regions, stiffness of the thermal system, effect of Ts; least squares / truncated SVD (why rcond matters) |
| Board designer | new, see section 6 | |

RAISE files are copied into `src/py/tools/` and `gui/views/tools/` (TherMOS stays self-contained, no import from a
sibling repo); the material, board, cooling and device libraries of `toolData.py` are shared by all pages.

## 6. Board designer (2D POD)

**Physics model (top view, x-y plane).** A board of size W × L with a technology (FR4 1/2/4-layer Cu, FR4 with
thermal vias, IMS Al, DBC Al₂O₃/AlN/Si₃N₄, Cu baseplate — from `toolData.BOARDS`) is reduced to effective in-plane
properties k_eff = Σ kᵢ tᵢ / Σ tᵢ, (ρc)_eff, and a vertical heat transfer coefficient to the coolant
h_eff = 1 / (R_vertical + 1/h_cooling) (W/m²K) from the cooling library (`toolData.COOLING`). Chips are rectangles
(x, y, w, l) with a technology (Si IGBT/diode, Si MOSFET, SiC MOSFET, GaN HEMT from `toolData.DEVICES`) that sets the
die properties and the default loss; the chip region gets the die's k/ρc in series with the board. Heat equation

    (ρc)_eff · t · ∂T/∂t = ∇·(k_eff · t · ∇T) − h_eff · (T − T_a) + q''(x, y, t)

solved with finite volumes on a structured grid (dx = dy, user-selectable), sparse matrices, exact/implicit time
stepping. Loss profiles per chip: step, pulse train, mission profile, or from the Power losses page.

**GUI.** Left: board canvas (Plotly) with chips drawn to scale; chips are added by clicking on the board (Plotly
`on_select`) or in a table (`st.data_editor`: name, technology, x, y, w, l, P(t)) with snapping and an overlap check.
Right: board/chip/cooling technology, grid and time settings. Tabs:

1. Layout & properties: k, ρc and h maps, total area, power density.
2. Simulation: field at a time slider, Tj per chip over time, steady-state check (Σ P = ∫ h (T − T_a) dA).
3. POD live: singular values / energy, first modes, ROM with K modes vs. full model (error, speed-up), protocol 2 vs. 5.
4. Export: writes `data/board/<name>.mat` (TherMOS format, **hc stored, F-P16 avoided**) for training and testing
   profiles plus `config/board/<name>.xlsx` from `tutorial_PO`, then opens the PO or PS mode with these files.

**Toolkit additions** (new files in Pascal's style, `legacy` untouched):

- `src/py/board/genBoard.py` (layout → property maps), `solveBoard2D.py` (FV solver), `saveBoard.py` (export),
  `boardLib.py` (technology tables from `toolData`).
- `model/PO/calcPodQuad.py`: add the Robin/sink term G += ∫ h φᵢ φⱼ dA and the source h·T_a to q for legacy = 0
  (today: WARN and ignored). Without it the pure projection has no heat sink on any board (F-P16 again).

**Verification:** 1D slab against the analytic solution (steady state and time constant), energy balance, grid
refinement (dx, dx/2), PO protocol 2/5 on an exported board (MAE target < 0.5 K), unit tests in `tests/`.

## 7. Work packages (one commit each)

| WP | Content | Check |
|---|---|---|
| 1 | Scaffold (`toolkit-gui`), adapter (sections, run settings, `FEED = 'xlsx'`, figure capture from PNGs), schema from `templateConfig.xlsx` + missing Python parameters | `build_schema.py` counts = workbook rows; `apptest.py` renders all pages |
| 2 | Eight modes: flags/sub-modes, default configs, `help.py` (HELP, BASIC, ACTIVE), run summaries/warnings | `apptest.py <mode>` + `parity.py <mode>` for RC, SS, SF, PO, PS, ML (DL/PINN: 1 epoch) |
| 3 | Results views per mode + Compare page | real runs of every tutorial config |
| 4 | Inputs: 1D time-series viewer, 2D/3D field viewer (time slider, z-slice, material maps) via `DataLoader` | all `data/` sets that exist |
| 5 | Workflow charts and mode Theory views | |
| 6 | Theory & Tools: port Power losses + Thermal stack from RAISE; new Thermal networks, MOR & POD, Numerics | pages render, explorers call toolkit functions |
| 7 | Board designer: `src/py/board/*`, hc term in `calcPodQuad`, GUI page, export → PO/PS | analytic + energy + grid checks, unit tests, PO on exported board |
| 8 | Tutorials page, `docu/GUI.md`, README section, `.gitignore` (`config/_gui_run.xlsx`, `results/**/gui_run.*`) | `python start_gui.py` started and looked at |

Estimated size: ≈ 8–10 k lines, most of it in WP 3, 6 and 7. WP 1–2 give a usable GUI on their own.

## 8. Decisions (07.11.2026)

1. **Board model:** both views: top view (x-y, lumped thickness) and cross-section (x-z, layers resolved like the
   journal data).
2. **Chip placement:** positions in a table (`st.data_editor`), the canvas only shows the layout.
3. **RAISE code:** only the thermal stack and the loss model are copied into TherMOS (`src/py/tools/`,
   `gui/views/tools/`).
4. **Environment:** `streamlit`, `plotly`, `openpyxl` installed into the TherMOS Python.
5. **Workbooks:** missing Python parameters are added as rows to `config/template/templateConfig.xlsx` and all config
   workbooks, so start scripts and GUI both see them.
