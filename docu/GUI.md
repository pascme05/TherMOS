# TherMOS GUI — User Guide

The TherMOS GUI is a local web application built with Streamlit. It wraps the Python version of TherMOS (`src/py`)
without changing it: every run calls `src.py.main.main(setup, path)`, exactly as the start scripts do.

- **Where parameters live.** They are edited in the browser and stored as JSON under `config/gui/`. For a run, the GUI
  writes them into `config/_gui_run.xlsx`, and the toolkit loads that workbook as usual.
- **Where it runs.** On your own machine only (`http://localhost:8501`). No data leaves it.

## 1. Installation and start

```
pip install -r requirements.txt          # toolkit (numpy, scipy, torch, scikit-learn, ...)
pip install -r gui/requirements.txt      # streamlit, plotly, openpyxl
pip install optuna                       # optional: Optuna optimiser on the Study page
python start_gui.py                      # opens http://localhost:8501
```

- Stop the GUI with `Ctrl+C` in the terminal.
- Server settings (port, localhost only, theme) are in `.streamlit/config.toml`.
- After changing code under `gui/` or `src/`, restart `start_gui.py`.

## 2. Layout

| Area | Content |
|---|---|
| **Sidebar** | Expert switch, configuration manager (load / save / download / import / reset), the runs of this session |
| **Project › Setup** | Ready-made setups (tutorials, journal, lecture), project settings (name, output, numerics, train/test switches), data (format, dimension, split, files, channels) |
| **Project › Input data** | 1D time series and 2D/3D fields of the data sets, read with the toolkit loader (`DataLoader`) |
| **Operating modes** | One page per model family (RC, SS, SF, PO, PS, ML, DL, PI) plus the **Study** page |
| **Theory & Tools** | Power losses, Thermal stack, Thermal networks, MOR & POD, Numerics, Board designer |
| **Results** | Browser of all results folders, including runs started from the start scripts |
| **Help** | Tutorials and this guide |

**Project settings**
- First row: *Plot model output* and *Numerics*.
  - Numerics 0 is the corrected Python port.
  - Numerics 1 reproduces the behaviour of the MATLAB version (`docu/pythonPort.md`).
- Second row: the switches *Save*, *Plot model structure*, *Train* and *Test*.
  - With *Train* off, the model `mdl/mdl_<model>_<name>.pkl` (or `.pt`) of an earlier run is loaded.

## 3. Operating modes

Every page has the views below:
- **Workflow:** flow chart showing the current settings; skipped steps are dashed.
- **Parameters:** model parameters with help. Expert parameters appear with the switch in the sidebar.
- **Inputs:** the training and test data of the mode.
- **Theory:** equations, symbols, an interactive explorer and an interpretation.
- **Run:** summary, blocking checks, streamed console output.
- **Results:** key figures and interactive charts.

| Mode | Model | Variants | Data | Theory explorer |
|---|---|---|---|---|
| RC | Foster network with K RC pairs, output-error fit | – | 1D | Fit of K = 1…6 pairs to a known 4-pair heat path; Z_th, error over K |
| SS | Linear state space (N4SID/MOESP, optional output-error refinement) | – | 1D | Identification of a 3-node chain; singular values, poles, step test |
| SF | Coupled Cauer network: one ladder per heat source, couplings at every stage, fitted to any profiles (sub-mode 1: nodal Foster from single-source experiments) | – | 1D | Full bridge with 4 chips; self- and cross-heating, R_th matrix |
| PO | POD-Galerkin, 6 calibration protocols | – | 2D | DBC board with two chips; energy, modes, error over K, effect of the cooling term h_v |
| PS | POD + identified modal state space | 2D / 3D | 2D/3D | Same board; identified modal dynamics vs. Galerkin |
| ML | Static regression on (rolled) features | LR, tree, SVR, GPR, ensemble, MLP | 1D | Linear regression with and without EWMA features |
| DL | CNN / LSTM / CNN + LSTM on windows | 3 networks | 1D | **Architecture diagram from the network code** (layers, shapes, parameters), windows over a profile |
| PI | CNN + LSTM with a physics loss | – | 1D | **Network and loss graph** (data loss + residual of the lumped stator model), residual explorer |

**Notes on DL and PI**
- The results view shows:
  - the training history;
  - for PI, the data loss and the physics loss separately;
  - the architecture of the trained network.
- The PINN adds the residual of a lumped model of one output (`Mdl › pinn › node`):
  `C_th·dT/dt = (T_ref − T)/R_th + P_v`.
  - P_v is an input feature (`PvSrc = k`) or an equation (`PvSrc = 0`, `PvEq`) of the inputs `x1, x2, …` (raw
    units), the predicted temperature `T` and the constants `Rth, Cth, Rs` and those of `const` (`a=1; b=2`).
    Allowed: `+ − * / **`, `abs sqrt exp log sin cos tan tanh minimum maximum where`. Example with the
    temperature-dependent copper resistance: `3*Rs*(1+0.00393*(T-20))*x4**2`.
  - T_ref is an input feature (`TrefSrc = k`) or the constant `Tref` (`TrefSrc = 0`).
  - The equation is checked before training; the PI theory page evaluates it at an operating point.
  - The network follows the sub-mode (CNN, LSTM, CNN + LSTM).
- The physics weight β (`Mdl › pinn › beta`) is 0 by default, as in MATLAB. The network then trains on the data
  only.
- DL and PI training on the CPU takes minutes to hours. `Mdl › gen › dev = 1` trains on the GPU (CUDA). This needs
  a CUDA build of PyTorch, e.g. `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132`; otherwise a
  warning is printed and the CPU is used. The ML models (scikit-learn) always run on the CPU.

- PO/PS results and the 2D/3D data viewer have a z slider (3D), line profiles T(x), T(y) through any point and
  probes over time (hot spot, coolest node, mean, any point) with the error at the probes.

## 4. Study (grid search, optimisation, feature selection)

The Study page runs the selected mode repeatedly on top of the current configuration. Each run trains and tests the
model once, saves nothing and draws no figures. TherMOS itself has no optimiser.

| Study | What it does |
|---|---|
| Grid search | Every combination of 1–3 parameters. Values are given as a list `1, 2, 4`, an integer range `1:8`, or `a:b:n` (n values, log spaced if b/a ≥ 10). Result: a line chart (1 parameter) or a heat map (2) |
| Optimisation | 1–4 parameters with bounds and a log option, and a budget of N runs. **Built-in:** random search over half the budget, then a coordinate search around the best point (seed 42). **Optuna:** TPE sampler, if installed |
| Feature selection | Greedy forward selection of the input channels (as in `setup/opti/opti_DL_Feat`): start with none, add the channel that improves the objective most, stop when nothing helps |

- **Objective.** MAE, RMSE or MAX (minimised), or FIT (maximised).
- **Objective data.** By default the objective is computed on the **validation data**: the testing set is replaced by
  the first validation file, sheet or the validation IDs. The test data therefore stay unseen. After the study,
  *Load best into the config* and run the mode normally to confirm the result on the test data.
- **Stored results.** `results/study_<mode>_<name>/study.json`.
- **Stopping.** Changing a widget stops a running study; the finished runs are kept.

## 5. Theory & Tools

Self-contained calculators. They do not change the configuration.

| Page | Content |
|---|---|
| Power losses | Conduction and switching losses of power semiconductors (from OPTELIA) |
| Thermal stack | Junction-to-coolant stack with spreading, Foster/Cauer networks (from OPTELIA) |
| Thermal networks | Sketch of a heat path next to its Foster and Cauer networks; Foster → Cauer conversion, Z_th, structure function, pulse response (rcSim) |
| MOR & POD | Sketch of the POD pipeline; POD-Galerkin vs. modal truncation on a 1D rod |
| Numerics | Sketch of the rod with control volumes and its RC ladder; explicit/implicit/Crank-Nicolson, stability, stiffness, grid convergence; quadrature weights W, relative element size J (legacy) and the calibration of C_th/G_th (rcond, prior λ, SF bounds) with explorers |
| Board designer | 2D board model, POD reduced model, export to the PO/PS modes (section 6) |

## 6. Board designer

The board designer builds a finite-volume model of a board with chips. The model is in `src/py/board`.

**Board and cooling**
- Board size, ambient temperature, cooling of the bottom side (preset or h), natural convection on top, grid spacing.

**Stack-up**
- A technology preset (FR4 2/4/6-layer with 35 or 70 µm Cu, IMS, DBC Al₂O₃/AlN, AMB Si₃N₄, DBC on a Cu baseplate)
  fills an **editable table** (top → bottom): layer, material, thickness, copper coverage, optional k overrides.
- Rows can be added and deleted; *Reset to preset* restores the preset.
- Patterned Cu layers (coverage c) are homogenised:
  - in-plane: `k_xy = c·k_Cu + (1 − c)·k_FR4,xy`;
  - through-plane: `k_z = 1/(c/k_Cu + (1 − c)/k_FR4,z)`.
- The 6-layer presets use signal layers with 40 % and planes with 90 % coverage.
- *Save to library* stores the stack-up in `config/gui/board_lib.json`. Saved stack-ups appear in the preset list
  marked ★.

**Below the board**
- A preset (none, gap filler 2 mm + Al heat sink 5 mm, gap pad + Al housing, grease + Al heat sink, insulating foil
  + Cu cold plate) fills an editable table (top → bottom): layer, material, thickness in mm, optional k.
- The cooling h then acts on the bottom of the last layer.
- The layers are resolved in the simulation like the board layers (select them as *Layer of the top view*). The
  lumped export model treats them as a second plane (heat sink T₂) coupled through R₁₂ and cooled through
  R₂ₐ = t/(2k) + 1/h.

**Thermal vias (averaged)**
- Region: none, under the chips plus a margin, or the whole board.
- Array: drill, plating, pitch, fill.
- The FR4 layers in the via region get `k_z,eff = φ_Cu·k_Cu + φ_hole·k_fill + (1 − φ_Cu − φ_hole)·k_z,FR4`, where φ
  are the area fractions of the copper barrels and the holes. The in-plane conductivity is not changed.
- The simulation raises k_z of the FR4 cells in the via region; the lumped export model lowers the local
  through-plane resistance instead.

**Chips**
- *Add a chip* (technology, position, loss) or edit the table directly: position, size, loss, profile (step, pulse,
  sine, random), period.
- **Load:** *Chip profiles* (profile column of the table) or *Mission profile*: a sequence of operating segments
  (`segment, duration [s], ramp [s], scale [-], <chip> [W] …`). The loss of a chip in a segment is its own column in
  W, or scale × P of the chip table if that column is empty; ramps are linear. The sequence can be repeated, edited
  and saved. Template and example (inverter drive cycle):
  `data/board/mission/template_mission.csv`, `data/board/mission/example_driveCycle.csv`.
- Own chip technologies are defined in *Chip technologies › Own technologies* (die, thickness, size, die attach,
  T_j,max). They can be saved to the library.

**Simulation**
- One **layer-resolved 3D finite-volume model** of the whole board: the x-y grid in every row of the stack (layers
  below, board layers, die attach and die over the chip footprints), h on the bottom of the lowest row, h_top on top.
- **Top view (x-y)** and **cross-section (x-z) at y = y_cut** are two views of this one result: the top view shows a
  selectable layer (dies, top copper, inner layers, heat sink), the cross-section the slice through all rows.
- The former 2D cross-section (no spreading along y) over-estimated T_j several times under thick interface
  layers; the lumped top view under-estimates multilayer FR4 boards (inner planes assumed perfectly coupled).
- Results:
  - junction temperatures with their margin to T_j,max;
  - steady-state and transient fields;
  - line profiles through any point and probes over time (hot spot, mean, any x/y);
  - a POD reduced model with energy, modes and error.
- **Expected duration:** before a run the page shows the expected time (implicit steps × nodes × the speed of the
  last run); while it runs a progress bar shows the elapsed and remaining time.
- **Electro-thermal coupling** (switch): the losses of every chip rise linearly with its junction temperature,
  P = P₀·(1 + α·(T_j − 25 °C)), with α per chip (column *α [%/K]*, empty = value of the technology). The transient
  uses the T_j of the previous sub-step, the steady state is solved exactly; a thermal runaway is reported. The
  tab *Losses* shows the nominal and the effective losses.

**Thermal characterisation**
- **R_th matrix** (exact steady state of the 3D model): R_th,ij = rise of T_j of chip i per W in chip j, so
  T_j = T_a + Σ_j R_th,ij·P_j. Diagonal = self-heating R_th,ja, off-diagonal = coupling between the chips. The table
  adds the coupling factor R_th,ij/R_th,ii and the heat split to the coolant below and to the top.
- **Step responses Z_th,ij(t)** (*Run step responses*): one 1 W step per chip on log-spaced times up to 5× the
  slowest time constant.
- **R/C network of the chips:** the coupled Cauer network of the SF mode (1–4 RC stages per chip from the junction
  to the cooling, couplings between the chips at every stage) fitted to all step responses: R and C per stage
  (C_th), coupling resistances per stage, fit error. Usable as a compact model in a circuit simulator.

**POD reduced model**
- *Training snapshots*: **step responses** (train on the 1 W steps, test on the loads of this run, e.g. the mission
  profile — a real test of the reduced model) or **this run** (train and test on the same data). The electro-thermal
  coupling is applied in the reduced model as well. Example (gap filler + Al heat sink, drive cycle, K = 20): max.
  error 0.1 K.

**Export to TherMOS**
1. *Export* writes:
   - `data/board/<name>_train.mat` (random loss levels);
   - `data/board/<name>_test.mat` (the designed profiles or the mission profile);
   - the layout `<name>.json`, including the stack-up, vias and own chips;
   - `config/gui/board_<name>.json`, a PO config with protocol 2.
2. *Load config* makes that config the active one. Train the PO (or PS) mode on it.
3. The data contain the distributed heat sink h_v and the anisotropy k_y. The PO projection uses both (legacy = 0).
4. The PO mode needs one temperature per node of a plane, so the export uses the lumped top-view model; the page
   shows how far its junction temperatures are from the layer-resolved model.

## 7. Tutorials

*Help › Tutorials* lists guided scenarios: at least one per mode, a board → PO workflow, a grid search, an
optimisation and a feature selection. *Load* sets the configuration of the tutorial (or the settings of the Study or
board page) and opens the page. *What you should see* contains numbers from real runs of the tutorial
configurations. The configs are in `config/gui/tutorials/`; regenerate them with `python -m gui.core.tutorials`.

## 8. Configurations and results

**Configurations**
- A configuration is the nested dictionary `setup` of the toolkit.
  - *Load* reads `config/gui/<name>.json`; *Save* writes it.
  - *Import* reads a `.json` file or a config workbook `.xlsx`.
- The ready-made setups are generated from the start scripts (`start*.py`, `setup/**/*.py`).
- If parameters are added to the config workbook, regenerate the schema with `python gui/build_schema.py`.

**Results**
- Every run is stored as `gui_run.pkl` / `gui_run.json` in its results folder.
- The results browser also lists runs that were started from the start scripts.

## 9. Troubleshooting and known limits

| Problem | Remedy |
|---|---|
| Port 8501 in use | Stop the other instance or change `server.port` in `.streamlit/config.toml` |
| Changes in the code are not visible | Restart `start_gui.py` |
| PO/PS: "need 2D or 3D field data" | Setup › Data › Dimension 2D/3D and Parameters › Dat › Data Reduction ≥ 2D |
| Training off, but no stored model | Train once with the same simulation name |
| DL/PI runs take very long | Reduce `Mdl › dl › epoch` or the window `W`; the tutorials use 3 epochs |
| A config from an older version misses parameters | Missing parameters are filled with the defaults when it is loaded |

**Known limits**
- **SF mode:** sources driven always together (e.g. diagonal chips of a full bridge) cannot be separated by the
  network fit; the R_th matrix entries of such pairs are not identifiable (see SF › Theory).
- **PO on exported board data:**
  - The board designer's own POD model (discrete Galerkin on the finite-volume matrices) reaches errors well below
    0.1 K.
  - The PO mode on the same data reaches about 1 K (tutorial `po_board`).
  - The difference comes from the PO pipeline: modes centred on the mean field, regridding, quadrature and
    gradients by finite differences.
