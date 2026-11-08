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
| **Theory & Tools** | Power losses, Thermal stack, Hand calculator, Thermal networks, MOR & POD, Numerics, Neural networks, Board designer |
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

**Network view of the DL mode**
- Shows the network that will be trained: parameters, receptive field, output shape, a layer table (output shape and
  parameters per layer) and the architecture diagram.
- *User-defined architecture* (switch) replaces the built-in network of the model variant with a layer table
  (`setup['archDL']`, saved with the configuration): conv (units, kernel, dilation), lstm / gru (units,
  bidirectional), transformer encoder (units, heads), dense; normalisation, activation and dropout per layer; the
  pooling of seq2point and the hidden dense layers of the head.
- Presets to start from: CNN, LSTM, CNN-LSTM (identical to the built-in networks), GRU, stacked LSTM,
  bidirectional LSTM, TCN (dilated CNN), transformer. Own architectures are saved to `config/gui/dl_arch_lib.json`
  and appear marked ★.

**Notes on DL and PI**
- The results view shows:
  - the training history;
  - for PI, the data loss and the physics loss separately;
  - the architecture of the trained network.
- **PINN (PI mode).** The network predicts the states x of a linear thermal network,
  `C·dx/dt = E·P − G·x`, and the outputs are `T̂ = T_ref + O·x`. The loss is
  `L = α·L_data + β·L_phys + γ·L_init`, with every part relative to the temperature scale ΔT (the 99th percentile of
  |T − T_ref|), so β is dimensionless (typically 0.01 … 1):
  - `L_data = mean(((T̂ − T)/ΔT)²)`;
  - `L_phys = mean(((C·dx/dt + G·x − E·P)/(diag(G)·ΔT))²)`;
  - `L_init` holds T̂ at the measured start temperature during about τ_min.

  The first `warm` samples of every window are left out.
  - **Where the settings are.** All PINN settings are in the view *Physics*: physics model, loss weights α, β, γ,
    warm-up, initial state, prediction, collocation, reference temperature, constants, loss sources and the
    structure. *Parameters* only holds the general settings and the training of the network (`dl`: W, stride,
    epochs, learning rate and its schedule, batch, early stopping).
  - **Physics** (`Mdl › pinn › phys`, view *Physics*, stored as `setup['physPI']`). Three choices:
    - 0 = none;
    - 1 = Foster terms (one state per term, τ·dθ/dt = R·P_j − θ; single or multiple outputs, coupling chains
      allowed);
    - 2 = Cauer / nodal network (node and resistance tables, hidden nodes only enter L_phys).

    Without physPI a default single Foster term is used; edit it in the structure table.
  - **Constants** (view *Physics* › section 3): a general list `name = value [unit] meaning`, e.g.
    `Rs = 0.0141 [Ohm] stator resistance; k = 0.00393 [1/K] temperature coefficient`, stored in `Mdl › pinn › const`.
    The V1.x parameters `Rth`, `Cth`, `Rs`, `PvSrc`, `PvEq` are no longer shown in the GUI; the command line still
    reads them, and an `Rs` used in an equation is moved into the list.
  - **Loss sources P_j** are equations of the inputs `x1, x2, …` (raw units), the predicted outputs `T1, T2, …`
    (`T` = output `node`) and the constants. Allowed: `+ − * / **`, `abs sqrt exp log sin cos tan tanh minimum
    maximum where`. Example: `3*Rs*(1+k*(T-20))*x4**2`.
  - **T_ref** (`TrefSrc`): 0 = constant `Tref`, k = feature k, −1 = the reference channels of the data.
  - **Import** (view *Physics* › section 6). Five sources:
    - fit on the training data with the RC mode (Foster chain per output) or the SF mode (coupled Cauer network);
    - trained `mdl/mdl_rc_*.pkl` / `mdl_sf_*.pkl`;
    - the network exported by the board designer (`data/board/<name>_pinn.json` or its CSV);
    - exact Cauer → Foster conversion (modal);
    - Foster → Cauer conversion (continued fraction per self-heating chain).

    The step responses Z_th(t), the network diagram, the matrices and a simulation of the physics alone on the
    training data show how good the physics is.
  - **Training details:**
    - `x0 = 1` gives the network the measured rise at the start of a window as an extra input. Training: every
      window (one per `stride`) gets its own measured start. Validation and testing: only the first sample of every
      profile is measured.
    - `pred = 1` predicts every test profile in one pass from its measured initial temperature; `pred = 2` chains
      windows, each starting from the predicted end of the previous one.
    - The validation runs exactly like the testing (whole profiles from their first sample), so early stopping and
      the kept network refer to the test procedure. The log also prints the error with measured window starts; the
      gap between the two is the error that builds up over a profile.
    - Learning rate (`Mdl › dl › lrAdapt`, DL and PI): 0 = factor `lrDropFa` every `lrDropPr` epochs; 1 = adaptive,
      factor `lrDropFa` after `lrDropPr` epochs without a better validation error (minimum 1e-3·initLr). The rate
      is printed every epoch and drawn in the convergence chart of the results.
    - Profiles are cut at every session ID of the data.
    - `col` (collocation) evaluates L_phys also on the inputs of the validation (1), testing (2) or both (3) data
      sets. Their temperatures are never used.
  - **Network** (view *Network*): any architecture as in the DL mode (`setup['archPI']`), with outputs = the states
    of the physics.
  - **Explanations:**
    - *Physics* › *How the parameters enter the PINN* shows the equations with the current value of every
      parameter.
    - *Theory* › *Example: variation of α, β, γ* trains a small PINN on synthetic data, with sliders for noise,
      physics mismatch, labelled and unlabelled profiles, and a β sweep.
  - **Results:** the temperature chart groups its legend per output: a click hides or shows the measured and
    predicted temperature and the error of that output together.
  - **Tutorials** (all settings are in the tutorial, *Settings of the tutorial* lists them and loads every run of
    the result table):
    - `pi_physics` (motor, fitted two-term Foster physics, β = 0.01): test MAE 1.66 K vs 2.00 K without physics weight
    - `pi_board` (three chips of the board designer, coupled Cauer network with 6 nodes, β = 1, col = 3): mean MAE
      0.82 K vs 1.47 K without physics. The data are generated with `python setup/tutorial/genBoardPINN.py`, or with
      *Export for the PI mode* in the board designer.
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
| Hand calculator | Closed-form estimates, one tab per question: layer stack with contact resistances → effective h (R'' per element, temperature profile, unit converter); contact resistance of rough metal joints over the pressure (Mikic/Yovanovich, gap gas/fill) vs. a TIM, measured values; spreading resistance (Lee et al.) with the optimum plate thickness; convection h (forced plate, natural convection + radiation, liquid channels with Δp — can be handed to the layer stack); radiation → equivalent h_rad (small body, parallel plates, view factor; h_tot = h_conv + h_rad, can be handed to the layer stack); plate-fin heat sink (fin efficiency, R_sa over the fin count); thermal vias (k_eff, R_pad); coolant energy balance; short pulses (semi-infinite chip, diffusion time), pulse trains on a Foster network (exact peak per pulse, periodic peak/valley, datasheet approximation) and lumped τ = R·C; skin & eddy currents (skin depth, R_ac/R_dc of round wires (Bessel) and bars, Dowell proximity, self-heating with ρ(T); eddy loss induced in an IMS/baseplate below a trace with or without adjacent return). Every tab has a sketch driven by its inputs (`gui/views/tools/hand_art.py`). Calculations in `src/py/tools/handCalc.py` |
| Thermal networks | Sketch of a heat path next to its Foster and Cauer networks; Foster → Cauer conversion, Z_th, structure function, pulse response (rcSim) |
| MOR & POD | Sketch of the POD pipeline; POD-Galerkin vs. modal truncation on a 1D rod |
| Numerics | Sketch of the rod with control volumes and its RC ladder; explicit/implicit/Crank-Nicolson, stability, stiffness, grid convergence; quadrature weights W, relative element size J (legacy) and the calibration of C_th/G_th (rcond, prior λ, SF bounds) with explorers |
| Neural networks | DNN, CNN, LSTM, transformer (TRA) and PINN on thermal examples: sketch, live demo (Z_th fit, receptive field as truncated impulse response, forget gate = exp(−h/τ), attention as exponential memory, physics loss fixing the extrapolation), equations with symbols; data & initial state (why a constant load is ambiguous in a transient — best possible error from the chosen inputs: window length, start temperature θ₀, last temperature, time since start; coverage of the test operating points by the training profile; checklist for a training data set); training (learning rate, setup parameters) |
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

**Additional losses in the board** (switch)
- Losses that are not in the chips: I²R of traces and planes, shunts, gate drivers, passives.
- One row per region: layer of the stack-up (0 = top), centre x, y, size w × l (empty = whole board), loss P in W.
- The loss is constant over the run and spread over the control volumes of the region in that layer (term `q_i` of
  the model). The finite-volume model and the POD model include it (with step-response training the steady field of
  the board losses is added as one snapshot); the Cauer network adds its steady junction temperature rise as an
  offset. The step responses, the R_th matrix and the export to the PO mode describe the chips only.
- The regions are drawn dashed red in the top view and in the cross-section.

**Chips**
- *Add a chip* (technology, position, loss) or edit the table directly: position, size, loss, profile (step, pulse,
  sine, random), period.
- **Load:** *Chip profiles* (profile column of the table) or *Mission profile*: a sequence of operating segments
  (`segment, duration [s], ramp [s], scale [-], <chip> [W] …`). The loss of a chip in a segment is its own column in
  W, or scale × P of the chip table if that column is empty; ramps are linear. The sequence can be repeated, edited
  and saved. Template and example (inverter drive cycle):
  `data/board/mission/template_mission.csv`, `data/board/mission/example_driveCycle.csv`.
- The **layout** (top view and cross-section side by side, same height; cut position y above them) is always shown
  below the chip table.
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
- **Junction temperature:** *die average* (virtual junction temperature T_vj, default) or *hottest cell*. The die
  average is linear in the losses, so the R_th matrix and the network superpose exactly. The hot spot moves with
  the load distribution (max of a sum ≠ sum of the maxima), so superposing it over-estimates T_j when several
  chips are loaded. In the drive-cycle/highway example this gives up to 13 K too much on the diode.
- **R_th matrix** (exact steady state of the 3D model): R_th,ij = rise of T_j of chip i per W in chip j, so
  T_j = T_a + Σ_j R_th,ij·P_j. Diagonal = self-heating R_th,ja, off-diagonal = coupling between the chips. The table
  adds the coupling factor R_th,ij/R_th,ii and the heat split to the coolant below and to the top.
- **Step responses Z_th,ij(t)** (*Run step responses*): one 1 W step per chip on log-spaced times up to 5× the
  slowest time constant.
- **R/C network of the chips:** the coupled Cauer network of the SF mode (1–4 RC stages per chip from the junction
  to the cooling, couplings between the chips at every stage) fitted to all step responses: R and C per stage
  (C_th), coupling resistances per stage, fit error. Usable as a compact model in a circuit simulator.
- **Matrix form:** C_th·dT/dt = E·P − G_th·T with the conductance matrix G_th (heatmap, W/K), the diagonal
  capacitance matrix C_th (Ws/K) and the input matrix E. The table adds the conductance of every node to the
  cooling. The R_th matrix of the network, G_th⁻¹ at the junctions, is compared with the exact one. *Download
  G_th, C_th, E (CSV)* exports the network.

**POD reduced model**
- *Training snapshots*: **step responses** (train on the 1 W steps, test on the loads of this run, e.g. the mission
  profile — a real test of the reduced model) or **this run** (train and test on the same data). The electro-thermal
  coupling is applied in the reduced model as well. Example (gap filler + Al heat sink, drive cycle, K = 20): max.
  error 0.1 K.

**Prediction with the reduced models**
- Runs the POD model (section 5) and/or the Cauer network (section 4) on another mission profile, chosen from
  `data/board/mission` or uploaded, with its own repeat count and T_s. The finite-volume model is not run, so there
  is no reference: check the errors of the reduced models in sections 4 and 5 first.
- The electro-thermal coupling follows the switch in section 3. In the network the coupling uses the die average,
  like the board model.
- Output: T_j(t) of every chip per model, the losses, a table of T_j,max (time, mean, swing, margin to the limit),
  run times and a CSV download.
- Example: trained on the step responses, predicting `example_highway` with coupling on (K = 10, 3 stages, die
  average). The error against the finite-volume model, which is not shown in the GUI, is 0.05 K for POD and 2.8 K
  for the network, on a rise of 128 K. The finite-volume model would take about 18 s; the POD model takes 0.08 s
  and the network 0.01 s.

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
