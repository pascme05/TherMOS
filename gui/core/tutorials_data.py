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
    dict(id='pi_physics', mode='PI', level='Advanced', minutes=20, base='tutorial_PI',
         title="Physics-informed training: data loss and physics loss",
         story="An LSTM predicts the stator winding temperature T_sw of the motor data. The physics is a two-term "
               "Foster network fitted on the training sessions with the RC mode (R = 0.029, 0.009 K/W, τ = 187, "
               "1251 s), heated by the copper losses P = 3·R_s·I_s² (view *Physics*, loss source 3*Rs*x4**2 with "
               "the constant Rs = 0.0141 Ω) on top of the coolant temperature. The loss is α·L_data + β·L_phys; both "
               "parts are relative to ΔT, so β is dimensionless. 30 epochs with an adaptive learning rate (about "
               "1 min on a GPU, 10 min on a CPU).",
         goals=["understand the PINN loss and where α, β enter", "see why the physics must be fitted to the data",
                "balance the data and the physics loss with β"],
         changes={'Run.selPI': 2,
                  'Run.physPI': {'type': 'foster', 'src': ['3*Rs*x4**2'],
                                 'foster': [{'out': 1, 'src': 1, 'R': [0.028626181657719574, 0.00918259051226378],
                                             'tau': [187.22138958971937, 1250.6272288997054]}],
                                 'origin': 'RC mode fit (K = 2) on the training sessions'},
                  'Mdl.gen.dev': 1,
                  'Mdl.dl.epoch': 30,
                  'Mdl.dl.W': 1800,
                  'Mdl.dl.stride': 120,
                  'Mdl.dl.batch': 32,
                  'Mdl.dl.initLr': 0.002,
                  'Mdl.dl.lrAdapt': 1,
                  'Mdl.dl.lrDropPr': 3,
                  'Mdl.dl.lrDropFa': 0.5,
                  'Mdl.dl.valPat': 8,
                  'Mdl.dl.gradTh': 1,
                  'Mdl.dl.shu': 3,
                  'Mdl.pinn.phys': 1,
                  'Mdl.pinn.alpha': 1,
                  'Mdl.pinn.beta': 0.01,
                  'Mdl.pinn.gamma': 0,
                  'Mdl.pinn.warm': 0,
                  'Mdl.pinn.x0': 1,
                  'Mdl.pinn.pred': 1,
                  'Mdl.pinn.col': 0,
                  'Mdl.pinn.TrefSrc': 2,
                  'Mdl.pinn.Tref': 25,
                  'Mdl.pinn.node': 1,
                  'Mdl.pinn.const': 'Rs = 0.0141 [Ohm] stator resistance per phase'},
         runs=[dict(label="no physics weight (β = 0)", changes={'Mdl.pinn.beta': 0}),
               dict(label="fitted Foster, β = 0.01 (tutorial configuration)", changes={}),
               dict(label="fitted Foster, β = 0.1", changes={'Mdl.pinn.beta': 0.1}),
               dict(label="default single Foster term, β = 0.1", changes={'Mdl.pinn.beta': 0.1, 'Run.physPI': None})],
         steps=["Open *Physics*: the panel *How the parameters enter the PINN* shows the equations and the current "
                "value of every parameter; section 2 holds α, β, γ, the initial state and the prediction, section 3 "
                "the constant Rs.",
                "Physics › section 5 and 7: the Foster terms and their step response; section 8 simulates the "
                "physics alone on the training data (≈ 12 K RMSE — a usable but rough prior).",
                "*Run* the tutorial configuration (β = 0.01). *Settings of the tutorial* › *Load this run* loads "
                "the other rows of the table.",
                "Open *Results*: the convergence chart shows the losses and the learning rate, the chart *Data and "
                "physics loss* both loss terms per epoch."],
         look="| run | MAE T_sw (K) | RMSE (K) | MAX (K) | FIT (%) |\n|---|---|---|---|---|\n"
              "| β = 0 | 2.00 | 2.58 | 14.7 | 89.7 |\n"
              "| **fitted Foster, β = 0.01** | **1.66** | **2.32** | 16.4 | **90.7** |\n"
              "| fitted Foster, β = 0.1 | 2.55 | 3.61 | 25.9 | 85.6 |\n"
              "| default single Foster term, β = 0.1 | 6.78 | 8.61 | 29.7 | 65.6 |\n\n"
              "A small weight of a fitted physics lowers the mean error by 17 % and the RMSE by 10 %: the physics "
              "keeps the prediction plausible between the load situations of the training sessions. The largest "
              "single deviation stays at 15 … 16 K (a fast transient the two-term network cannot follow). A larger "
              "β pulls the prediction towards the rough two-term network (12 K RMSE on its own) and the gain is "
              "lost. A wrong physics — the default term of the workbook (R = 0.0265 K/W, τ = 551 s) heated by "
              "feature x1, the ambient temperature instead of a loss — makes the PINN more than three times worse "
              "than the network alone. The test profile is predicted in one pass from its measured initial "
              "temperature (pred = 1, x0 = 1); the validation is computed the same way (best 3.5 K RMSE). The "
              "learning rate halves three times (0.002 → 0.00025) when the validation error stalls for 3 epochs. "
              "GPU runs differ by a few hundredths of a kelvin.",
         exercise=["Try β = 0.003 and 0.03: where is the best compromise?",
                   "Physics › section 6: fit the Foster network with K = 3 and run again.",
                   "Theory › *Example: variation of α, β, γ*: run the β sweep with 0 % and 50 % physics mismatch.",
                   "Add the constant k = 0.00393 [1/K] in section 3 and use the temperature-dependent copper "
                   "resistance: source 3*Rs*(1+k*(T-20))*x4**2."]),
    dict(id='pi_board', mode='PI', level='Advanced', minutes=15, base='tutorial_PI_board',
         title="Multi-output PINN with a Cauer network from the board designer",
         story="Three chips (IGBT T1, diode D1, SiC MOSFET M1) on an IMS board: the board designer simulated their "
               "junction temperatures (finite-volume model) for four random training profiles, one validation "
               "and one test profile (setup/tutorial/genBoardPINN.py, or *Export for the PI mode* in the board "
               "designer). The physics is the coupled Cauer network of the chips fitted to their step responses: "
               "6 nodes (two per chip), the junctions are measured, the second stages are hidden and only enter "
               "the physics loss. LSTM, 60 epochs, adaptive learning rate; about 25 s on a GPU.",
         goals=["train one PINN for several outputs", "use a Cauer network with hidden nodes as physics",
                "move a network between the board designer, SF/RC modes and the PI mode"],
         changes={'Run.selPI': 2,
                  'Mdl.gen.dev': 1,
                  'Mdl.dl.epoch': 60,
                  'Mdl.dl.W': 50,
                  'Mdl.dl.stride': 5,
                  'Mdl.dl.batch': 32,
                  'Mdl.dl.initLr': 0.002,
                  'Mdl.dl.lrAdapt': 1,
                  'Mdl.dl.lrDropPr': 4,
                  'Mdl.dl.lrDropFa': 0.5,
                  'Mdl.dl.valPat': 12,
                  'Mdl.dl.gradTh': 1,
                  'Mdl.dl.shu': 3,
                  'Mdl.pinn.phys': 2,
                  'Mdl.pinn.alpha': 1,
                  'Mdl.pinn.beta': 1,
                  'Mdl.pinn.gamma': 0,
                  'Mdl.pinn.warm': 2,
                  'Mdl.pinn.x0': 1,
                  'Mdl.pinn.pred': 1,
                  'Mdl.pinn.col': 3,
                  'Mdl.pinn.TrefSrc': 4,
                  'Mdl.pinn.Tref': 25,
                  'Mdl.pinn.node': 1,
                  'Mdl.pinn.const': ''},
         runs=[dict(label="β = 0 (data only)", changes={'Mdl.pinn.beta': 0, 'Mdl.pinn.col': 0}),
               dict(label="β = 0.1", changes={'Mdl.pinn.beta': 0.1, 'Mdl.pinn.col': 0}),
               dict(label="β = 1", changes={'Mdl.pinn.beta': 1, 'Mdl.pinn.col': 0}),
               dict(label="β = 1, col = 3 (tutorial configuration)", changes={})],
         steps=["Open *Physics*: settings (section 2), node and resistance tables, network diagram and step "
                "responses Z_th(t) (solid: self-heating, dotted: coupling between the chips).",
                "Section 6 › Convert Foster ↔ Cauer: convert to Foster and compare the step responses, then back.",
                "*Run* the tutorial configuration (β = 1, col = 3). *Settings of the tutorial* › *Load this run* "
                "loads the other rows of the table (col = 0, β = 0, β = 0.1).",
                "Open *Results*: convergence with the learning rate, junction temperatures of all three chips (a "
                "click on a legend entry hides the temperature and the error of that chip)."],
         look="| run | MAE T1 / D1 / M1 (K) | FIT T1 / D1 / M1 (%) | mean MAE (K) |\n|---|---|---|---|\n"
              "| β = 0 | 1.66 / 1.38 / 1.37 | 84.4 / 78.0 / 74.7 | 1.47 |\n"
              "| β = 0.1 | 1.13 / 0.90 / 1.28 | 89.2 / 84.4 / 76.2 | 1.10 |\n"
              "| β = 1 | 0.97 / 1.08 / 1.27 | 91.0 / 82.9 / 76.2 | 1.11 |\n"
              "| **β = 1, col = 3** | **0.65 / 0.92 / 0.88** | **93.2 / 85.1 / 82.9** | **0.82** |\n\n"
              "With only four 40-s training profiles the physics lowers the mean error by 25 % (β = 0.1 or 1): the "
              "Cauer network describes the simulated board well. The two coupled chips T1 and D1 gain most (their "
              "heating of each other is in the network); M1 sits apart and changes little. Evaluating the physics "
              "also on the inputs of the validation and test profiles (col = 3, their temperatures are not used) "
              "lowers the mean error by another quarter, to 44 % below the data-only network. The validation "
              "error printed in the log is computed like the test (best 0.91 K RMSE for the tutorial "
              "configuration); with the measured temperature at every window start it would be 0.44 K — the "
              "difference is the error that builds up over a profile. The learning rate drops from 0.002 to 0.001 "
              "after the validation error stalls for 4 epochs. GPU runs differ by a few hundredths of a kelvin.",
         exercise=["Reduce the training profiles to one sheet (Setup › trSheet = train1): how much does the physics "
                   "help now?", "Set the output of the hidden node of T1 to 1 in the node table: what changes?",
                   "Set lrAdapt = 0 (fixed steps every 4 epochs): compare the convergence chart."]),

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
