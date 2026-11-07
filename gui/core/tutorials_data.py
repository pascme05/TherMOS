#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         tutorials_data.py
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Tutorials of the GUI (Help › Tutorials): at least one per operating mode, the board designer → PO workflow and the
studies (grid search, optimisation, feature selection). The numbers under 'look' come from runs of exactly these
configurations (07.11.2026, Python version, corrected numerics unless stated). Extra keys: 'page' (page the tutorial
opens: 'study' or a tool url), 'runMode' (operating mode of the config if 'mode' is not one), 'preset' (widget values
of that page).
"""

#######################################################################################################################
# Tutorials
#######################################################################################################################
TUTORIALS = [
    # ------------------------------------------
    # RC
    # ------------------------------------------
    dict(id='rc_basics', mode='RC', level='Beginner', minutes=10, base='tutorial_RC',
         title="Foster network of a transformer winding",
         story="A transformer winding (FE simulation) is heated with a loss step. You fit a Foster network with three "
               "RC pairs on operating point OP1 and test it on OP2.",
         goals=["read R_th and τ of a fitted Foster network", "see why an RC model is only valid at the operating "
                "point it was fitted on"],
         changes={},
         steps=["Load the tutorial and open *Theory*: the explorer shows how K pairs fit a known heat path.",
                "Open *Inputs*: training sheet OP1, test sheet OP2, input P_Sec, output T_Sec.",
                "*Run* the mode and open *Results*.",
                "Compare measured and predicted T_Sec and read the Foster table."],
         look="- Fitted network: R_th = 0.356 / 1.602 / 2.403 K/W, τ = 8.6 / 96 / 314 s (ΣR_th ≈ 4.36 K/W).\n"
              "- Test on OP2 (20 W): MAE **17.3 K**, MAX 22.7 K, FIT −63 %. The shape of the heating curve is "
              "right, but the predicted rise is 87 K instead of the measured 64 K.\n"
              "- Reason: the thermal resistance at OP2 is lower (≈ 3.2 K/W) than at OP1 — the RC model has no "
              "dependence on the operating point. With K = 1 the error is the same (MAE 17.4 K): the problem is "
              "the operating point, not the order.",
         exercise=["Set Setup › Data › Sheets › Training to OP2 and run again — how small does the error get?",
                   "Train on OP1 and OP2 together: which compromise does the fit find?"]),

    # ------------------------------------------
    # SS
    # ------------------------------------------
    dict(id='ss_order', mode='SS', level='Beginner', minutes=10, base='tutorial_SS',
         title="State-space model with three heat sources",
         story="The same transformer with three loss inputs (primary, secondary, core) and three temperatures. One "
               "MIMO state-space model describes self- and cross-heating.",
         goals=["choose the model order n", "read poles as time constants"],
         changes={},
         steps=["Open *Theory*: identify the 3-node chain and watch the singular-value gap.",
                "*Run* with the default order n = 2, then with n = 1 and n = 4 (Parameters › ss › Model Order)."],
         look="| n | MAE T_Pri / T_Sec / T_Cor (K) | FIT (%) | poles |z| |\n|---|---|---|---|\n"
              "| 1 | 9.88 / 9.36 / 0.74 | 35 / 16 / 79 | 0.9945 |\n"
              "| 2 | 1.86 / 0.92 / 2.87 | 87 / 86 / 32 | 0.9929, 0.9983 |\n"
              "| 4 | 0.60 / 0.55 / 3.68 | 96 / 95 / 9 | 0.927 … 0.9985 |\n\n"
              "One state cannot describe three temperatures; n = 4 fits the windings best, the core temperature "
              "(small, slow rise) is the hardest output. All poles lie just below 1: slow thermal modes "
              "(τ = −T_s/ln|z| ≈ 140–670 s).",
         exercise=["Run the grid search tutorial (study_grid) to see the error over n = 1 … 6.",
                   "Switch Parameters › ss › Modeling approach to subspace only (1): what changes?"]),

    # ------------------------------------------
    # SF
    # ------------------------------------------
    dict(id='sf_nodes', mode='SF', level='Intermediate', minutes=10, base='startRth',
         title="Coupled heat sources and the R_th matrix",
         story="Four heat sources in a 2 × 2 arrangement on a common substrate (like the four switches of a full "
               "bridge), every source heated alone once (one profile per source). The SF mode fits a coupled "
               "Cauer network: one RC ladder per source plus coupling resistances between the sources.",
         goals=["interpret self- and cross-heating resistances", "see why a coupled network beats single-input "
                                                                 "models here"],
         changes={'Run.selSF': 2, 'Mdl.rc.K': 3},
         steps=["Open *Theory*: look at the full-bridge sketch and fit the explorer with single-chip steps and with "
                "full-bridge operation.", "*Run* the mode (fit ≈ 5 s).",
                "Open *Results*: R_th matrix, fitted ladders and couplings."],
         look="- R_th matrix: diagonal ≈ **6.10 K/W** (self-heating), neighbours **1.82 K/W**, diagonal partner "
              "**1.62 K/W** — a symmetric 2 × 2 arrangement.\n"
              "- Error on the test profile: **MAE ≈ 0.77 K**, MAX ≈ 1.7 K, FIT ≈ 89 % (K = 3 stages; K = 2: "
              "MAE 0.80 K).\n"
              "- For comparison on the same data: the nodal Foster method of the MATLAB version (sub-mode 1) "
              "MAE 4.93 K, MAX 40 K; the RC mode (one input) 23 K.",
         exercise=["Which source heats source 1 most, and by how much at 10 W?",
                   "Switch the sub-mode to *Nodal Foster* and compare the transient error (Trans column).",
                   "Reduce Mdl › sf › bnd to 10: which parameters stop at their bounds (warning in the log)?"]),

    # ------------------------------------------
    # PO
    # ------------------------------------------
    dict(id='po_fem', mode='PO', level='Intermediate', minutes=10, base='tutorial_PO',
         title="POD-Galerkin model of a 2D field",
         story="2D temperature fields of a plate with three heat sources (MATLAB PDE). The POD modes are learnt from "
               "the snapshots and the heat equation is projected onto them.",
         goals=["relate the number of modes K to the error", "compare the pure Galerkin projection with the "
                "data-calibrated protocol"],
         changes={},
         steps=["Open *Theory*: energy, modes and the effect of the cooling term on a board.",
                "*Run* with K = 3 (protocol 5), then K = 8, then K = 3 with protocol 2 (Parameters › pod).",
                "Open *Results* and move the time slider of the field view."],
         look="| K | protocol | mean MAE over the nodes (K) | MAX (K) |\n|---|---|---|---|\n"
              "| 3 | 5 (calibrated) | 0.197 | 1.38 |\n| 8 | 5 | 0.025 | 0.73 |\n"
              "| 3 | 2 (pure Galerkin) | 0.895 | 2.45 |\n\n"
              "Three modes already capture 99.9999 % of the snapshot energy. Calibrating the reduced matrices with "
              "the data (protocol 5) reduces the error of the pure projection by a factor of 4.5. The training and "
              "the test data come from the same file (exp11) in this tutorial.",
         exercise=["Find the smallest K with MAE < 0.05 K.", "Which nodes have the largest error, and why?"]),
    dict(id='po_board', mode='PO', page='tools-board', level='Advanced', minutes=15, base='tutorial_PO',
         title="Design a board and train the PO model on it",
         story="You design a DBC board with an IGBT, a diode and a SiC MOSFET, simulate it, export the data and "
               "train the PO mode on the exported fields.",
         goals=["use the board designer", "export a data set and train a TherMOS model on it"],
         changes={},
         preset={'tl|board|name': 'tutorial'},
         steps=["The tutorial opens the board designer with the default layout (DBC Al₂O₃, liquid cold plate).",
                "Press *Simulate* (top view) and read the junction temperatures.",
                "Under *Export to TherMOS* keep the name 'tutorial', press *Export*, then *Load config "
                "'board_tutorial' in the GUI*.",
                "Open *PO* › *Run*, then *Results*."],
         look="- Board: T_j,max 64.9 °C, board steady-state maximum 63.4 °C at 104 W total loss; the board's own "
              "POD model (5 modes) reproduces the field with 0.07 K maximum error.\n"
              "- PO mode on the exported data: protocol 2 MAE **1.06 K**, FIT 42.7 %; protocol 5 MAE 1.03 K, FIT "
              "49.3 %.\n"
              "- The PO pipeline is less accurate than the discrete projection of the board model (regridding, "
              "mean-centred modes, finite-difference gradients) — see docu/GUI.md, known limits.",
         exercise=["Switch on vias under the chips on an FR4 6-layer board: how much lower is T_j?",
                   "Export the cross-section view and train PO on it."]),

    # ------------------------------------------
    # PS
    # ------------------------------------------
    dict(id='ps_3d', mode='PS', level='Advanced', minutes=10, base='start3D',
         title="3D POD state space of an FR4 board",
         story="3D FEM fields of an FR4 board: the model is trained at 5 W and tested at 10 W. PS identifies the "
               "dynamics of the POD coordinates from data.",
         goals=["train a 3D PS model", "check the extrapolation to a higher load"],
         changes={},
         steps=["Open *Theory*: identified modal dynamics vs. Galerkin on a board.",
                "*Run* (about 75 s) and open *Results*; choose z-slices in the field view."],
         look="- Captured energy 99.986 % with K = 3 modes.\n"
              "- Test at 10 W: mean MAE over the nodes **0.11 K** (worst node 0.55 K), MAX 2.65 K, FIT 96.9 % — the "
              "linear model scales to twice the training load.",
         exercise=["Increase the number of states (Parameters › ss › K): does the error drop?"]),

    # ------------------------------------------
    # ML
    # ------------------------------------------
    dict(id='ml_features', mode='ML', level='Beginner', minutes=10, base='tutorial_ML',
         title="Machine learning with rolled features",
         story="Electric motor measurements (Kaggle): predict the stator winding temperature from ambient, coolant, "
               "voltage and current. A static regression needs the history of the inputs as features.",
         goals=["see why static regression fails on thermal data", "use EWMA/EWMS features"],
         changes={'Run.featRoll': 1, 'Run.EWMA': [60, 600, 3000], 'Run.EWMS': [600]},
         steps=["Open *Theory*: linear regression with and without EWMA features.",
                "*Run* the regression tree (variant 2) with rolled features (EWMA 60/600/3000, EWMS 600 samples).",
                "Switch off Setup › Rolling features and run again; then choose Linear regression (variant 1)."],
         look="| model | features | MAE (K) | MAX (K) | FIT (%) |\n|---|---|---|---|---|\n"
              "| regression tree | raw | 9.76 | 48.6 | 48.8 |\n| regression tree | rolled | 6.21 | 19.6 | 66.7 |\n"
              "| linear regression | rolled | **5.67** | 15.0 | 71.2 |\n\n"
              "The rolled features carry the thermal history; with them a linear model beats the tree "
              "(test profile ID 62).",
         exercise=["Which span matters most? Remove one EWMA span at a time.",
                   "Run the feature selection tutorial (study_feat)."]),

    # ------------------------------------------
    # DL
    # ------------------------------------------
    dict(id='dl_window', mode='DL', level='Intermediate', minutes=25, base='tutorial_DL',
         title="CNN + LSTM on motor data (short training)",
         story="Four motor temperatures from a window of 600 samples of four inputs. The training is cut to "
               "3 epochs so that it runs in minutes on a CPU.",
         goals=["read the network architecture", "compare CNN and CNN + LSTM"],
         changes={'Mdl.dl.epoch': 3},
         steps=["Open *Theory*: the architecture diagram shows every layer with its output shape and parameters.",
                "*Run* the CNN + LSTM (variant 3), then the CNN (variant 1).",
                "Open *Results*: training history and the architecture of the trained network."],
         look="| network | MAE T_sw / T_st / T_so / T_rm (K) | FIT T_sw (%) | time |\n|---|---|---|---|\n"
              "| CNN (1) | 18.0 / 12.3 / 12.1 / 10.9 | 20 | 79 s |\n"
              "| CNN + LSTM (3) | 60.0 / 52.6 / 47.7 / 47.6 | −159 | 254 s |\n\n"
              "After 3 epochs the CNN is already usable, the CNN + LSTM is not: the LSTM needs many more epochs to "
              "learn the slow dynamics (test profile ID 60). Three epochs only show the workflow — the published "
              "results use 100 epochs (hours on a CPU).",
         exercise=["Increase the number of epochs to 20: how much does the error drop, and when does early stopping "
                   "end the training?", "Halve the window W: what happens to the slow temperatures?"]),

    # ------------------------------------------
    # PI
    # ------------------------------------------
    dict(id='pi_physics', mode='PI', level='Advanced', minutes=25, base='tutorial_PI',
         title="Physics-informed training: data loss and physics loss",
         story="The CNN + LSTM is trained on the stator winding temperature with an additional loss: the residual "
               "of a lumped stator model C_th·dT/dt = (T_amb − T)/R_th + P_v with the copper losses P_v = 3·R_s·I_s² "
               "(Mdl › pinn › PvEq = 3*Rs*x4**2, x4 = I_s). 3 epochs.",
         goals=["understand the PINN loss", "balance the data and the physics loss with β"],
         changes={'Mdl.dl.epoch': 3},
         steps=["Open *Theory*: network and loss graph, and the residual explorer.",
                "*Run* with β = 0 (default, data only), then with β = 1e-6 (Parameters › pinn › beta).",
                "Open *Results*: the chart *Data and physics loss* shows both terms per epoch."],
         look="| β | MAE T_sw (K) | MAX (K) | FIT (%) | data loss per epoch | physics loss per epoch |\n"
              "|---|---|---|---|---|---|\n"
              "| 0 | 14.6 | 62.7 | 26.5 | 3618 → 982 → 625 | not evaluated |\n"
              "| 1e-6 | **7.35** | 72.4 | 58.3 | 3630 → 716 → 167 | 3.4e7 → 5.4e7 → 3.5e7 |\n\n"
              "With β = 0 the residual is not computed (as in MATLAB). With β = 1e-6 the weighted physics term "
              "(≈ 35) is of the order of the data loss and halves the error after 3 epochs (about 9 min each "
              "on a CPU). The physics loss stays large: the lumped stator model is only a rough description of "
              "the motor, which limits how large β can be.",
         exercise=["Try β = 1e-5 and 1e-7: where is the best compromise?",
                   "Change R_th by ±50 %: how sensitive is the result to wrong physics?",
                   "Use the temperature-dependent copper resistance: PvEq = 3*Rs*(1+0.00393*(T-20))*x4**2."]),

    # ------------------------------------------
    # Studies
    # ------------------------------------------
    dict(id='study_grid', mode='Study', runMode='SS', page='study', level='Intermediate', minutes=10,
         base='tutorial_SS', title="Grid search: the order of the state-space model",
         story="Instead of trying orders by hand, the Study page runs the SS mode for n = 1 … 6 and plots the error "
               "on the validation sheet.",
         goals=["run a grid search", "pick the order from the error curve"],
         changes={},
         preset={'st|mode': 'SS', 'st|kind': 'grid', 'st|metric': 'MAE', 'st|val': True, 'st|gk': ['Mdl.ss.K'],
                 'st|gv|Mdl.ss.K': '1:6'},
         steps=["The tutorial opens the Study page with SS, grid search, Model Order SS = 1:6.",
                "Press *Start study* (about 1.5 min).", "Read the chart and press *Load best into the config*."],
         look="MAE on the validation sheet OP2 (mean of the three outputs):\n\n"
              "| n | 1 | 2 | 3 | 4 | 5 | 6 |\n|---|---|---|---|---|---|---|\n"
              "| MAE (K) | 6.66 | 1.88 | 1.22 | 1.61 | 1.32 | **1.05** |\n"
              "| time (s) | 1.6 | 1.6 | 3.1 | 16 | 13 | 44 |\n\n"
              "The big step is from n = 1 to 3; beyond that the error scatters (the identification is not monotone "
              "in n) while the time grows. n = 3 is a good compromise. Note: in this config the validation sheet "
              "is also the test sheet.",
         exercise=["Add Modeling approach (ss.dis = 1, 2) as a second parameter: heat map."]),
    dict(id='study_opti', mode='Study', runMode='ML', page='study', level='Intermediate', minutes=10,
         base='tutorial_ML', title="Optimisation of a regression tree",
         story="The minimum leaf size and the minimum split size of the regression tree are tuned with the built-in "
               "optimiser (random search, then local search) on the validation profiles.",
         goals=["define a search space with bounds and log scale", "use the validation data as objective"],
         changes={},
         preset={'st|mode': 'ML', 'st|kind': 'opti', 'st|metric': 'MAE', 'st|val': True, 'st|meth': 'built-in',
                 'st|budget': 12, 'st|ok': ['Mdl.dt.leaf', 'Mdl.dt.split'],
                 'st|lo|Mdl.dt.leaf': 1.0, 'st|hi|Mdl.dt.leaf': 200.0, 'st|log|Mdl.dt.leaf': True,
                 'st|lo|Mdl.dt.split': 2.0, 'st|hi|Mdl.dt.split': 400.0, 'st|log|Mdl.dt.split': True},
         steps=["The tutorial opens the Study page with ML, optimisation, leaf 1–200 and split 2–400 (log), budget 12.",
                "Press *Start study*.", "Compare the best setting with the default (leaf 10, split 50)."],
         look="Objective MAE on the validation profiles (IDs 10, 48, 63):\n\n"
              "- Default (leaf 10, split 50): 12.43 K.\n"
              "- Best of 12 runs: **leaf 200, split 80 → 11.71 K** (found in the local-search phase, run 8; "
              "the random phase was best at leaf 95, split 80 with 11.91 K).\n"
              "- Loaded into the config and run on the test profile (ID 62): MAE 8.15 K, FIT 52.8 % "
              "vs. 9.76 K, 48.8 % with the default.\n\n"
              "Large leaves smooth the tree; the optimum lies at the upper bound — widen the bound to 1000.",
         exercise=["Install Optuna and repeat with the TPE method.", "Optimise with rolled features switched on."]),
    dict(id='study_feat', mode='Study', runMode='ML', page='study', level='Beginner', minutes=5,
         base='tutorial_ML', title="Forward feature selection",
         story="Which of the four motor signals does the regression tree need? The forward selection adds one "
               "channel at a time (as setup/opti/opti_DL_Feat does by hand).",
         goals=["run a feature selection", "interpret the selected channels"],
         changes={},
         preset={'st|mode': 'ML', 'st|kind': 'feat', 'st|metric': 'MAE', 'st|val': True, 'st|fbudget': 10},
         steps=["The tutorial opens the Study page with ML and feature selection over Ta, Tc, Us, Is.",
                "Press *Start study* (about 30 s)."],
         look="MAE on the validation profiles (IDs 10, 48, 63), regression tree:\n\n"
              "- Round 1: Ta 17.2 K, Tc 17.8 K, Is 20.8 K, Us 21.0 K → **Ta**.\n"
              "- Round 2: Ta + Is **14.97 K**, Ta + Tc 16.99 K, Ta + Us 17.79 K → **Ta, Is**.\n\n"
              "Ambient temperature and current (copper losses ∝ I²) carry most of the information; the voltage "
              "adds little.",
         exercise=["Repeat with rolled features switched on — do the selected channels change?"]),
]
