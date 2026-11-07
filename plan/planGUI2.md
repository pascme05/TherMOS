# Plan: TherMOS GUI, phase 2

Status: 07.11.2026, decisions taken (section 6), WP A–G implemented. Builds on `plan/planGUI.md` (WP 1–7 done, WP 8 open).

## 1. Requests

| # | Request | Where |
|---|---|---|
| 1 | Train/test switches below the "Plot model output" and "Numerics" selectors | `gui/adapter.py` `RUN_FIELDS`, `gui/views/project.py` |
| 2 | A Theory view for every operating mode | `gui/core/theory.py` (+ `gui/core/theory_*.py`) |
| 3 | At least one tutorial per mode, plus grid search and optimisation | `gui/core/tutorials.py`, new Study page |
| 4 | Architecture of DL and PINN, especially the physics (PDE) loss | Theory views DL/PI, results views DL/PI |
| 5 | Sketches of the problem (the rod etc.) on Thermal networks, MOR & POD, Numerics | `gui/views/tools/sketch.py` (new) |
| 6 | Board designer: add chips, edit the stack-up, 6-layer 35/70 µm boards, (averaged) thermal vias | `src/py/board/*`, `gui/views/tools/board.py` |

## 2. Survey findings that shape the plan

- **Project page.** It lays out `RUN_FIELDS` in 4 columns, in their order: name, user, train, test / save, plotMdl,
  plotOut, legacy. The fix is to reorder them to name, user, plotOut, legacy / save, plotMdl, train, test, so that
  train/test sit directly below plotOut/legacy.
- **No grid search or optimisation in the toolkit.**
  - `setup/opti/opti_DL_Feat.py` and `opti_DL_NN.py` are single runs. They differ only in the input features (8 vs. 4
    channels), which is a manual feature study.
  - A sweep therefore has to be added. The GUI will loop over `main(setup, path)` and change the config between runs.
    The toolkit itself stays unchanged.
- **DL networks** (`src/py/model/DL/dlNet.py`):
  - CNN: Conv1d(F→64, k5) → BN → ReLU → Conv1d(64→32, k3) → ReLU → Linear(32→32) → Linear(32→M).
  - LSTM: LSTM(F→64) → Linear(64→64) → Linear(64→M).
  - CNN-LSTM: Conv1d(F→32, k3) → LSTM(32→64) → Linear(64→32) → Linear(32→M).
  - The diagrams can be generated from the live `nn.Module` (layer, shape, parameter count), so they always match the
    code.
- **PINN loss** (`src/py/model/PINN/pinnFit.py`):
  - `L = α·mean((T̂ − T)²) + β·mean(r²)`, with the residual `r = Cth·dT̂/dt − ((T_amb − T̂)/Rth + 3·Rs·Is²)`.
  - The physical model behind it is a lumped single-node stator: the ODE `Cth·dT/dt = (T_amb − T)/Rth + 3·Rs·Is²`.
  - The derivative uses forward differences over the window.
  - β = 0 by default, as in MATLAB. In that case the PINN trains on the data only, and the GUI must say so.
- **Board library.**
  - `BOARDS` is a fixed dict of layer tuples (name, mat, t, kxy, kz, rho, cp).
  - The chip table is already a `data_editor` with dynamic rows, but chip technologies are fixed (`CHIPS`).
  - Vias exist only as one board-wide "FR4 with vias" layer, with `kz = via_k(1.0, 0.3, 25)`.
  - `genBoard` uses one stack-up for the whole board, so it has no local kz.
- **Existing sketch style.** `pkg_art.py` and `topo_art.py` (copied from RAISE) draw inline SVG with `_r`, `_t` and
  `_arrow` helpers. The new sketches reuse this style.

## 3. Work packages (one commit each)

### WP A: Setup layout (request 1)

- Reorder `RUN_FIELDS` to name, user, plotOut, legacy, save, plotMdl, train, test. The selectors form the first row
  and the switches sit below them.
- Alternative: two explicit rows "Output & numerics" and "Run". I would choose the reorder because it is smaller.
- Check: `apptest.py`, and visually.

### WP B: Problem sketches (request 5)

New `gui/views/tools/sketch.py` with inline-SVG drawings in the `pkg_art` style, theme-aware. Each sketch has
labelled quantities and units:

| Page | Sketch |
|---|---|
| Numerics | Rod of length L: heat P in at x = 0 (arrow), convection h, T_a at x = L, insulated sides. N nodes with control volumes (half volumes at the ends), Δx, node i with C_i and the conductance G between nodes. Below it, the equivalent RC ladder. |
| MOR & POD | Same rod, plus a strip "snapshots θ(t₁…t_n) → SVD → modes φ₁, φ₂, φ₃", plus the projection θ ≈ Φa (N states → K states). |
| Thermal networks | Heat path junction → die attach → substrate → TIM → heat sink, with the Foster chain (parallel RC pairs, nodes without physical meaning) and the Cauer ladder (grounded C, physical nodes) side by side. The Cauer elements are mapped to the layers. |

The sketches use the page's current values (N, L, material, number of RC pairs), so they change with the inputs.

### WP C: Board designer extensions (request 6)

1. **Stack-up editor.**
   - The selected technology is a starting point: "Board technology" fills an editable `data_editor`.
   - Columns: layer, material (selectbox from `MATERIALS`), t (µm), Cu coverage (%), kxy, kz.
   - kxy and kz are computed from material and coverage, and can be overridden.
   - Rows can be added, deleted and reordered. "Reset to technology" restores the preset.
   - A dielectric's kz can be raised by vias (see 3).
   - `genBoard(layout)` takes `layout['layers']` (a list of dicts) instead of looking up `BOARDS`. The `BOARDS` names
     remain the presets.
   - Cu coverage: kxy = c·k_Cu + (1−c)·k_FR4 (parallel) and kz = 1/(c/k_Cu + (1−c)/k_FR4) (series). Signal layers use
     about 30–50 % coverage, planes 90 %.
2. **New presets.**
   - FR4 6-layer, 35 µm and 70 µm Cu, 1.6 mm. Stack: Cu / prepreg 0.2 / Cu / core 0.36 / Cu / prepreg 0.2 / Cu /
     core 0.36 / Cu / prepreg 0.2 / Cu, with inner planes at 90 % coverage and signal layers at 40 %.
   - Also FR4 4-layer at 70 µm.
   - Thicknesses are rescaled so that the total is 1.6 mm.
3. **Thermal vias.**
   - New "Vias" container with the controls:
     - mode: none / under chips / everywhere;
     - drill d, plating thickness, pitch, fill (air or plugged, for example Cu paste);
     - margin around the chip footprint (mm).
   - The vias are averaged, not modelled explicitly:
     - Cu area fraction f = π·((d/2)² − (d/2 − t_pl)²)/pitch², or π·(d/2)²/pitch² for filled vias.
     - The dielectric gets kz,eff = f·k_Cu + (1−f)·kz. kxy stays the same (vias are poor in-plane).
   - In the model:
     - Top view: kz,eff changes the local through-plane resistance, so `R_z(x, y)` and `h_eff(x, y)` become maps. The
       sink `hv` is already per node.
     - Cross-section: per-cell kz in the columns under the via region.
   - `via_k` from `toolData` is reused or extended for this.
4. **Add chips.**
   - An "Add chip" row above the table: technology, x, y, then "Add".
   - A chip-technology editor (expander) where custom chips can be defined (material, t, w, l, attach, t_att,
     Tjmax). They are stored in the session and in the exported layout `<name>.json`.
   - The board views show the via regions (hatched) and highlight a selected chip.
5. **Tests** (`tests/test_board.py`):
   - Coverage 100 % equals solid Cu.
   - Vias with f = 0 give the previous results.
   - Energy balance with local via maps.
   - A 6-layer board with uniform load matches the analytic result.
   - Vias under a chip lower T_j monotonically with f.

### WP D: Theory views of the modes (requests 2 and 4)

`THEORY[mode]` for all 8 modes. Each view follows the same structure: question → equations with all symbols and units
→ an explorer that calls the toolkit functions with the current config → interpretation. One file per family keeps
`theory.py` small: `theory_rc.py`, `theory_pod.py`, `theory_ml.py`.

| Mode | Equations | Explorer (toolkit functions) |
|---|---|---|
| RC | Foster `Z_th(t) = Σ R_i(1 − e^{−t/τ_i})`, convolution `T = Z_th' * P + T_ref`, the fit objective | Number of pairs K vs. fit on a synthetic Z_th; `rcSim` pulse response (links to Thermal networks) |
| SS | `ẋ = Ax + Bu`, `T = Cx + Du`, discretisation, N4SID/MOESP + PEM | Order n vs. singular values of the Hankel matrix (`fitN4sid` on generated data), poles in the s-plane, step response |
| SF | Nodal network `C·Ṫ = P − G·T`, Rth matrix (self- and cross-heating) | 2–3 node example: coupling vs. Rth matrix, cross-heating curves |
| PO | Heat equation, weak form, `Φᵀ C Φ ȧ = −Φᵀ G Φ a + Φᵀ B P`, Gauss-Legendre quadrature, sink `hv`, anisotropy `ky` | Snapshot energy vs. K, the first modes on the current data set (`DataLoader` + POD), link to the board designer |
| PS | POD modes + identified modal state space `ȧ = A_r a + B_r P` | As PO, plus the modal step response |
| ML | Feature vector with lags/EWMA/EWMS, LR, RF, SVR, GPR, EN, NN (per `selML`), metrics | Feature construction on a profile (rolled features), ranking |
| DL | Windowing (W, stride, seq2seq/seq2one), normalisation, MSE, Adam, early stopping | **Architecture diagram** (Graphviz) from the live `dlNet(sel, F, M, …)`: layers, tensor shapes (B×W×F → …), parameter counts. Window sketch over the time series |
| PI | `L = α·L_data + β·L_phys`, residual `r = Cth·dT̂/dt − ((T_amb − T̂)/Rth + 3·Rs·Is²)`, finite-difference dT/dt | **Architecture + loss graph:** network → T̂ → {data loss vs. T, ∂/∂t → residual with the physical model} → weighted sum. Explorer: the lumped ODE alone, with Rth/Cth/Rs from the config, against the measured profile, residual over time, effect of β. Note: β = 0 by default |

- The results views for DL and PI also get a loss-curve tab. The PI view splits the curve into data and physics loss,
  which needs `info` to store both parts. That is a small change in `pinnFit`, made only with legacy = 0.
- The DL/PI diagram is also shown in a "Model" tab of the results view.

### WP E: Grid search and optimisation (request 3, prerequisite for its tutorials)

New page **Study** (navigation group "Modes", after Compare), GUI-side only:

- **Grid search.**
  - Choose a mode, then 1–3 parameters from that mode's parameter set (e.g. `rc.K`, `ss.order`, `pod.K`, `dl.W`,
    `dl.epoch`, `ml.*`), with a list of values each.
  - It runs `main` for each combination: `train = 1`, `test = 1`, `save = 0`, plots off.
  - Output:
    - a table of MAE, RMSE, MAX, FIT and run time per combination;
    - a line plot for 1 parameter and a heat map for 2;
    - the best combination, with a "Load into config" button.
- **Optimisation.**
  - Same parameter choice, with bounds and type (int/float/log).
  - The method is sequential random search plus a local refinement (coordinate search around the best point), with
    budget N runs and a fixed seed of 42.
  - No new dependency. Optuna would be the alternative.
  - Objective: MAE or RMSE on the validation data where the mode uses it, otherwise on the test data. Optimising on
    test data overfits, and the page will say so when there is no validation data.
- **Feature selection (opti_DL_Feat).** A "Channels" parameter type that sweeps subsets of `inp` (forward selection),
  so the study in `setup/opti` becomes a one-click run.
- The run loop reuses `runner.run_mode`, with a progress bar and cancel between runs. Results are stored in
  `results/<name>/study.json`.

### WP F: Tutorials (request 3)

`TUTORIALS`: one or more per mode, plus the studies. Each tutorial has a story, goals, config changes, steps, and a
"what to look at" text taken from real runs, with an exercise.

| id | Mode | Base | Topic |
|---|---|---|---|
| rc_basics | RC | tutorial_RC | Fit Foster pairs, read Rth/τ, K = 2 vs. 4 |
| ss_order | SS | tutorial_SS | Model order and cross-coupling, poles |
| sf_nodes | SF | startRth | Self- vs. cross-heating, Rth matrix |
| po_fem | PO | tutorial_PO | POD of FEM fields, K vs. error, energy |
| po_board | PO | board export | Design a board → export → train PO (connects the board designer) |
| ps_3d | PS | FR4 / IMS3D | Modal state space, z-slices |
| ml_features | ML | tutorial_ML | LR vs. RF, rolled features |
| dl_window | DL | tutorial_DL | CNN vs. LSTM, window length (few epochs) |
| pi_physics | PI | tutorial_PI | β = 0 vs. β > 0, residual, data efficiency |
| study_grid | Study | tutorial_RC / SS | Grid search over K (RC) or order (SS) |
| study_opti | Study | tutorial_DL / ML | Hyper-parameter optimisation; feature selection (opti_DL_Feat) |

DL and PI tutorials use reduced epochs, so each takes at most about 5 min on CPU, and the page states the run time. The
configs are written to `config/tutorials/*.json` via `python -m gui.core.tutorials`.

### WP G: Documentation and finish (old WP 8)

- `docu/GUI.md`: modes, Theory & Tools, board designer (stack-up, vias, export), Study, tutorials.
- README GUI section.
- Requirements.
- `apptest.py` for all pages.
- `python start_gui.py`, and a look at it in the browser.

## 4. Order and size

A → B → C → D → E → F → G. Tutorials (F) come after the Study page (E) and the board extensions (C), because they
use both.

Estimate: about 4–5 k lines. Most of it is in D (about 1.5 k), C (about 0.8 k) and E (about 0.6 k).

## 5. Open points (my default in brackets)

1. **Optimisation method.** Built-in random search + local refinement, or add Optuna as a dependency? [built-in]
2. **Vias in the top view.** Local kz (proposed), or also a local in-plane kxy increase? [kz only]
3. **PINN loss logging.** May `pinnFit` store data and physics loss separately (legacy = 0 only)? [yes]
4. **Custom chips and stack-ups.** Persist them only in the exported layout JSON, or also in a user library file
   `config/gui/board_lib.json`? [layout JSON + library file]

## 6. Decisions (07.11.2026)

1. **Optimisation:** both — built-in random search + local refinement (default, no dependency) and Optuna (TPE) when
   installed (`pip install optuna`, optional in `gui/requirements.txt`).
2. **Vias:** through-plane conductivity only (kz,eff of the dielectric in the via region), kxy unchanged.
3. **PINN loss logging:** `pinnFit` stores data and physics loss separately (legacy = 0).
4. **Custom chips and stack-ups:** stored in the exported layout JSON and in the library `config/gui/board_lib.json`.
