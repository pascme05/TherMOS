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
COORD = {'Run.selPF': 2, 'Mdl.dl.epoch': 150, 'Mdl.dl.valPat': 40, 'Mdl.dl.lrDropPr': 15}     # coordinate network

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
    # PF
    # ------------------------------------------
    dict(id='pf_board', mode='PF', level='Advanced', minutes=20, base='tutorial_PF',
         title="Field PINN of a board: 3D temperature field from data and the heat equation",
         story="Three chips (IGBT T1, diode D1, SiC MOSFET M1) on a 30 × 20 mm IMS board. The board designer "
               "simulated the layer-resolved 3D temperature field (finite-volume model, 3453 nodes) for four random "
               "training profiles of 20 s, one validation profile and a designed pulse/sine test profile of 40 s "
               "with a maximum rise of 107 K (setup/tutorial/genBoardPF.py, or *Export for the PF mode* in the "
               "board designer). The operator network (variant 1) learns the field of any load profile; the "
               "coordinate network (variant 2) reconstructs the field of the test profile from 50 sensors. About "
               "3 min on a GPU for the tutorial configuration.",
         goals=["see why a network trained on temperatures alone fails on a new load and how the heat equation "
                "repairs it", "choose the physics loss (finite-volume operator or automatic differentiation)",
                "reconstruct a field from few sensors"],
         changes={},
         runs=[dict(label="physics 0 (data only)", changes={'Mdl.pinf.phys': 0}),
               dict(label="physics 2 (tutorial configuration)", changes={}),
               dict(label="physics 2, col = 0", changes={'Mdl.pinf.col': 0}),
               dict(label="physics 2, α = 0 (no temperatures)", changes={'Mdl.pinf.alpha': 0}),
               dict(label="physics 1, β = 0.01", changes={'Mdl.pinf.phys': 1, 'Mdl.pinf.beta': 0.01}),
               dict(label="coordinate network, 50 sensors, physics 0",
                    changes={'Run.selPF': 2, 'Mdl.pinf.phys': 0, 'Mdl.pinf.nSens': 50,
                             'Mdl.dl.epoch': 150, 'Mdl.dl.valPat': 40, 'Mdl.dl.lrDropPr': 15}),
               dict(label="coordinate network, 50 sensors, physics 2",
                    changes={'Run.selPF': 2, 'Mdl.pinf.phys': 2, 'Mdl.pinf.nSens': 50,
                             'Mdl.dl.epoch': 150, 'Mdl.dl.valPat': 40, 'Mdl.dl.lrDropPr': 15})],
         steps=["Generate the data once: `python setup/tutorial/genBoardPF.py` (about 1 min).",
                "Open *Theory*: the two variants, the loss with all symbols, the three source footprints found in "
                "the data and the sensor placement.",
                "*Run* the tutorial configuration (operator network, physics 2). *Settings of the tutorial* › *Load "
                "this run* loads the other rows of the table.",
                "Open *Results*: verdict, loss history (data and physics separately), field against the reference "
                "with the time slider, error map and the learned basis functions of the trunk."],
         look="| run | RMSE (K) | MAE (K) | MAX (K) | training (s) |\n|---|---|---|---|---|\n"
              "| operator, physics 0 (data only) | 8.94 | 8.46 | 18.9 | 22 |\n"
              "| **operator, physics 2 (tutorial)** | **0.81** | **0.58** | 19.8 | 159 |\n"
              "| operator, physics 2, col = 0 | 2.32 | 2.11 | 12.8 | 108 |\n"
              "| operator, physics 2, α = 0 | 1.98 | 1.20 | 29.2 | 82 |\n"
              "| operator, physics 1, β = 0.01 | 31.9 | 30.8 | 66.3 | 66 |\n"
              "| coordinate, 50 sensors, physics 0 | 5.64 | 3.12 | 44.9 | 42 |\n"
              "| coordinate, 50 sensors, physics 2 | 3.18 | 2.31 | 43.9 | 245 |\n"
              "| for comparison: PS mode, K = 6 | 1.09 | 0.37 | 30.6 | 26 |\n\n"
              "Errors of the test profile over all 3453 nodes and 800 time steps (RTX 3070).\n"
              "- **Data only:** 2.1 K on the validation profile but 8.9 K on the test profile (8 % of the rise). "
              "The test drives all chips at full power at once, which the random training loads rarely do.\n"
              "- **Physics 2** lowers the RMSE by a factor of 11 to 0.81 K (0.8 % of the rise). Without the "
              "collocation profiles (col = 0) 2.3 K remain: the heat balance on the sources of the test profile, "
              "without its temperatures, is what carries the network to the new load. Even without any "
              "temperature (α = 0) it reaches 2.0 K.\n"
              "- **Physics 1** fails here (31.9 K): the board has copper next to a dielectric (conductivity ratio "
              "195) and the residual with automatic differentiation ignores this jump. The log prints a warning. "
              "On the same board as a lumped 2D field it helps (7.1 K → 2.3 K).\n"
              "- **Coordinate network:** with 50 sensors (1.4 % of the nodes) the physics lowers the RMSE from 5.6 "
              "to 3.2 K. The maximum error of 44 K sits in the chips, where the field is steepest and no sensor "
              "is placed.\n"
              "- **PS mode:** as accurate (1.09 K RMSE) and 6 times faster — for a linear board with full field "
              "data a POD model is the better tool. The PF mode is for few sensors and loads without data.",
         exercise=["Generate the lumped 2D field (genBoardPF.main(name='tutorial_pf2d', view='xy') or the board "
                   "designer) and run the coordinate network with 30 sensors and physics 1: 0.30 K RMSE.",
                   "Coordinate network: raise nSens from 50 to 200 — how does the maximum error in the chips "
                   "change?", "Operator network: set K = 4. Which error does the missing basis cause?",
                   "Halve the warm-up (warm = 35): why does the error at the start of the windows grow?"]),
    dict(id='pf_compare', mode='PF', level='Advanced', minutes=30, base='tutorial_PF',
         title="Operator or coordinate network? The same 50 sensors, two ways to use them",
         story="The board of the tutorial *Field PINN of a board* (3D, 3453 nodes, three chips). This time only 50 "
               "nodes carry a temperature sensor (1.4 % of the nodes, the same nodes in every profile). Both networks "
               "see only these sensors and are scored on all nodes of the same test profile. The operator network "
               "learns from the sensors of the four training profiles and has never seen a temperature of the test "
               "profile. The coordinate network learns from the sensors of the test profile itself, but only for "
               "this one profile.",
         goals=["compare a model of the board (operator network) with a reconstruction of one measurement "
                "(coordinate network)", "see how the number of sensors and the physics loss act on each",
                "decide which network fits which task"],
         changes={'Mdl.pinf.nSens': 50, 'Mdl.pinf.phys': 2},
         runs=[dict(label="operator network, 50 sensors, physics 2 (tutorial configuration)", changes={}),
               dict(label="coordinate network, 50 sensors, physics 2", changes={**COORD}),
               dict(label="operator network, 50 sensors, physics 0", changes={'Mdl.pinf.phys': 0}),
               dict(label="coordinate network, 50 sensors, physics 0", changes={**COORD, 'Mdl.pinf.phys': 0}),
               dict(label="operator network, 200 sensors, physics 2", changes={'Mdl.pinf.nSens': 200}),
               dict(label="coordinate network, 200 sensors, physics 2", changes={**COORD, 'Mdl.pinf.nSens': 200})],
         steps=["Generate the data once: `python setup/tutorial/genBoardPF.py`.",
                "Open *Theory* › *Where are the sensors?*: the 50 sensors start at the hottest node and spread over "
                "the board. Both networks use exactly these nodes.",
                "*Run* the tutorial configuration (operator network, 50 sensors, physics 2), then *Load this run* "
                "for the coordinate network with the same sensors and compare the results.",
                "Repeat both without physics and with 200 sensors (rows of the table)."],
         look="| run (50 sensors unless stated) | RMSE (K) | MAE (K) | MAX (K) | training (s) |\n"
              "|---|---|---|---|---|\n"
              "| **operator network, physics 2 (tutorial)** | **0.99** | **0.78** | 19.3 | 144 |\n"
              "| coordinate network, physics 2 | 3.18 | 2.31 | 43.9 | 247 |\n"
              "| operator network, physics 0 | 9.71 | 7.85 | 51.0 | 13 |\n"
              "| coordinate network, physics 0 | 5.64 | 3.12 | 44.9 | 49 |\n"
              "| operator network, 200 sensors, physics 2 | 0.86 | 0.61 | 24.9 | 119 |\n"
              "| coordinate network, 200 sensors, physics 2 | 2.95 | 2.13 | 33.8 | 280 |\n"
              "| for comparison: operator network, all 3453 nodes, physics 2 | 0.81 | 0.58 | 19.8 | 159 |\n\n"
              "Errors of the test profile over all nodes and time steps (maximum rise 107 K, RTX 3070).\n"
              "- **The operator network wins with physics:** 0.99 K from 50 sensors of other profiles, almost as good "
              "as with all nodes (0.81 K). The finite-volume operator fills the field between the sensors; the "
              "sensors only have to anchor it. It predicts any further load profile in 0.07 s.\n"
              "- **The coordinate network sees the test profile itself and is still 3 times worse** (3.18 K). Its "
              "largest error (44 K) sits in the chips, where the field changes fastest in space and time; 200 "
              "sensors barely help (2.95 K). The limit is the network that has to represent the whole field over "
              "x, y, z and t, not the number of sensors. Every new profile needs a new training of 4 min.\n"
              "- **Without physics the order flips:** the operator network has never seen the test load and fails "
              "(9.7 K), the coordinate network interpolates its own sensors (5.6 K).\n"
              "- **Which one?** A model of the board for many loads: operator network with physics 2. A one-off "
              "reconstruction of a measurement for which no training profiles exist: coordinate network.",
         exercise=["Reduce the sensors to 10 for both networks: which one degrades first?",
                   "Coordinate network: raise the epochs to 600. How much of the gap to the operator network is "
                   "training time?", "Operator network: set col = 0. How much do the sources of the test profile "
                   "(without temperatures) contribute when only 50 sensors are measured?"]),
    dict(id='pf_compare2d', mode='PF', level='Advanced', minutes=30, base='tutorial_PF',
         title="Operator or coordinate network on a lumped 2D board: which physics for which network?",
         story="The same three chips, exported as a lumped 2D field (one temperature per point of the top view, 651 "
               "nodes; conductivity and cooling vary smoothly over the board). Both networks see only 30 sensors "
               "(5 % of the nodes) and are scored on all nodes of the test profile. Here both physics losses are "
               "valid: the residual with automatic differentiation (physics 1, needs only the material data) and "
               "the finite-volume operator (physics 2).",
         goals=["compare both networks with both physics losses", "see that the best physics depends on the network"],
         changes={'Run.trFile': ['tutorial_pf2d_train1', 'tutorial_pf2d_train2', 'tutorial_pf2d_train3',
                                 'tutorial_pf2d_train4'],
                  'Run.vlFile': ['tutorial_pf2d_val'], 'Run.teFile': 'tutorial_pf2d_test', 'Run.datDim': 2,
                  'Dat.gen.dOut': 2, 'Mdl.pinf.nSens': 30, 'Mdl.pinf.phys': 1},
         runs=[dict(label="operator network, physics 1 (tutorial configuration)", changes={}),
               dict(label="coordinate network, physics 1", changes={**COORD}),
               dict(label="operator network, physics 2", changes={'Mdl.pinf.phys': 2}),
               dict(label="coordinate network, physics 2", changes={**COORD, 'Mdl.pinf.phys': 2}),
               dict(label="operator network, physics 0", changes={'Mdl.pinf.phys': 0}),
               dict(label="coordinate network, physics 0", changes={**COORD, 'Mdl.pinf.phys': 0})],
         steps=["Generate the data once: `python setup/tutorial/genBoardPF.py tutorial_pf2d 4 20 xy` (or *Export for "
                "the PF mode* of the board designer, top view).",
                "Open *Theory*: the three source footprints and the 30 sensors on the top view.",
                "*Run* the tutorial configuration (operator network, physics 1), then load the other rows."],
         look="| run (30 sensors) | RMSE (K) | MAE (K) | MAX (K) | training (s) |\n|---|---|---|---|---|\n"
              "| operator network, physics 1 (tutorial) | 2.83 | 2.32 | 10.8 | 62 |\n"
              "| **coordinate network, physics 1** | **0.30** | **0.23** | 2.0 | 581 |\n"
              "| operator network, physics 2 | 0.95 | 0.79 | 5.2 | 36 |\n"
              "| coordinate network, physics 2 | 2.08 | 1.60 | 14.9 | 197 |\n"
              "| operator network, physics 0 | 9.63 | 8.59 | 26.0 | 9 |\n"
              "| coordinate network, physics 0 | 3.99 | 2.88 | 20.0 | 28 |\n\n"
              "Errors of the test profile over all 651 nodes and time steps (maximum rise 79 K, RTX 3070).\n"
              "- **The best physics depends on the network.** The coordinate network is best with automatic "
              "differentiation (0.30 K, 0.4 % of the rise): the network is a smooth function of x and t, so its "
              "derivatives are exact everywhere, and the lumped 2D field has no material jumps. The operator network "
              "is best with the finite-volume operator (0.95 K): its time functions are sampled at the time steps "
              "of the data, which is what the discrete operator expects.\n"
              "- **The coordinate network with the finite-volume operator** (2.08 K) gets only one time instant per "
              "step for the residual (the operator acts on all 651 nodes at once), against 1024 random space-time "
              "points with automatic differentiation.\n"
              "- **Without physics** both fail (9.6 K and 4.0 K): 30 sensors do not describe the field.\n"
              "- **Cost:** the best coordinate network trains for almost 10 min for this one profile; the operator "
              "network with physics 2 trains in 36 s and then predicts any load in 0.01 s.",
         exercise=["Coordinate network, physics 2: raise pinf › nCol to 8192 (8 time instants per step). Does the "
                   "gap to physics 1 close?", "Operator network, physics 1: raise nCol to 4096. How much of the gap "
                   "is the number of collocation points?", "Run both networks with 10 sensors: where does the "
                   "physics help more?"]),
    dict(id='pf_design_levers', mode='PF', page='tools-design', level='Beginner', minutes=10, base='tutorial_PF',
         title="Cooling, dielectric or die attach: which lever lowers the junction temperature?",
         story="The IMS board of the parametric PF study (30 × 20 mm, Si IGBT T1 40 W, Si diode D1 15 W, SiC MOSFET "
               "M1 20 W) starts with a weak cold plate (h = 500 W/(m²K)). You change one design parameter at a time "
               "and see where it enters the thermal model and how much it moves T_j.",
         goals=["see which entries of C and G each parameter changes", "rank the levers by their effect on T_j",
                "see that ρc changes only the transient"],
         changes={},
         preset={'tl|de|mu|h': 500.0, 'de|wh|k': 'h', 'de|ma|sw': 'h'},
         steps=["The tutorial opens Theory & Tools › *Design effects* with h = 500 W/(m²K).",
                "Tab *Where does a change act?*: switch between Cooling h, Dielectric k, Die attach k and Base ρc and "
                "read which node rows are outlined.",
                "Tab *Cooling & materials*: read the tiles, then set h back to 1500 and 5000 at the top.",
                "Sub-tab *Sweep of one parameter*: compare h, k_diel and s_att; sub-tab *Which parameter acts most?*."],
         look="Steady junction rise of T1 above ambient (nominal positions, other parameters nominal):\n\n"
              "| parameter | low | nominal | high | slowest τ |\n|---|---|---|---|---|\n"
              "| h (W/(m²K)) 500 / 1500 / 5000 | 296 K | 133 K | 73 K | 8.2 / 2.8 / 0.85 s |\n"
              "| k_diel (W/(mK)) 1 / 2 / 8 | 154 K | 133 K | 112 K | 2.8 s (unchanged) |\n"
              "| s_att 0.5 / 1 / 2 | 134 K | 133 K | 132 K | 2.8 s (unchanged) |\n\n"
              "- **Where:** h changes only the 651 diagonal entries of the bottom row of G; k_diel the series "
              "conductances of the copper, dielectric and top Al rows; the die attach 49–63 nodes under the chips; "
              "ρc only 1953 entries of C (the Al rows) — nothing in G.\n"
              "- **Ranking (+10 % of each, nominal board):** h −8.3 K, k_diel −2.5 K, s_att −0.2 K, ρc 0 K at T1. "
              "The cooling and the dielectric are in series with the whole heat path; the die attach acts only "
              "under each chip.\n"
              "- **Saturation:** at h = 5000 a better dielectric (k 2 → 8) still lowers T1 from 73 K to 52 K, while "
              "at k_diel = 8 the cooling (1500 → 5000) lowers it from 112 K to 52 K: once one resistance is small, "
              "the other one dominates.",
         exercise=["Set s_att = 0.5 (voided solder): how much hotter is T1 in steady state, and in the first 0.1 s of "
                   "the tab *Transients*?", "Which combination of h and k_diel keeps T1 below 100 K at the "
                   "nominal losses?", "Why does ρc not appear in *Which parameter acts most?*?"]),
    dict(id='pf_design_move', mode='PF', page='tools-design', level='Intermediate', minutes=10, base='tutorial_PF',
         title="Moving the IGBT: edges, neighbours and the thermal coupling matrix",
         story="Same board at nominal cooling (h = 1500 W/(m²K)). The IGBT T1 (6 × 6 mm, 40 W) is moved along the "
               "board while the diode and the MOSFET stay in place. A position change is the one design change that "
               "projection-based parametric reduction cannot handle well — the field moves with the chip.",
         goals=["read the junction temperatures over the chip position", "use the R_th coupling matrix",
                "see why a position change is different from a material change"],
         changes={},
         preset={'de|mv|c': 0, 'de|mv|a': 0, 'de|wh|k': 'pos'},
         steps=["The tutorial opens *Design effects*; tab *Where does a change act?* shows the shift of T1 by 2 mm.",
                "Tab *Move a chip*: read the curve over x, then switch the direction to y.",
                "Sub-tab *Coupling between the chips*: read the R_th matrix.",
                "Move T1 at the top to the best position and compare the tiles."],
         look="- **Where:** shifting T1 by 2 mm rewires 28 diagonal and 196 off-diagonal entries of G (the links "
              "between the die/attach cells and the copper) — no entry is scaled, the loss itself moves.\n"
              "- **Over x** (y = 10 mm): T1 is at 143 K at x = 4 mm (1 mm from the edge), 132.3 K at x = 9–10 mm "
              "and 133 K at x = 11 mm (closer to D1). **Over y** (x = 8 mm): 144 K at y = 4 mm, 132.8 K at y = "
              "9–10 mm, 145 K at y = 16 mm. The board edges are adiabatic, so a chip near an edge can spread its heat "
              "to one side only.\n"
              "- **Coupling** (K/W, nominal layout): self-heating 2.55 (T1), 3.17 (D1), 3.26 (M1); T1 gets 1.23 K/W "
              "from D1 and 0.87 K/W from M1, i.e. 18 K + 17 K of its 133 K come from its neighbours.\n"
              "- **Perspective:** the position moves T1 by about 10 K, the cooling (500 → 5000) by 220 K. Positions "
              "matter at the margin, which is why they are optimised last — with gradients of the parametric network "
              "in the PF study.",
         exercise=["Move M1 along y: where is the best position and why is it not in the centre?",
                   "Halve the cooling (h = 750): does the best position of T1 change?",
                   "Use the R_th matrix to predict T_j of D1 when T1 runs at 60 W, then check it at the top."]),
    dict(id='pf_design_transient', mode='PF', page='tools-design', level='Intermediate', minutes=15,
         base='tutorial_PF', title="Time constants of the board and the parametric network",
         story="A 10 s load step of all three chips, then 10 s off. The tab *Transients* solves a block Krylov "
               "model of the 3453-node board (18 states) for the current design and for the nominal condition. The "
               "tab *Parametric network* runs the trained network of the PF study on the same step, for any cooling, "
               "material or position, without retraining.",
         goals=["relate the time constants to the cooling and the heat capacity",
                "compare the parametric network with the exact model inside and outside its training range"],
         changes={},
         preset={'tl|de|mu|h': 500.0},
         steps=["The tutorial opens *Design effects* with h = 500 W/(m²K); open the tab *Transients & time "
                "constants*.", "Set h back to 1500, then raise the base ρc factor to 1.3.",
                "Train the parametric network once (`python setup/tutorial/genBoardPFParam.py`, then "
                "`python setup/tutorial/studyPFParam.py base --out results/pfparam`) and open the tab *Parametric "
                "network*.", "Read *PF › Theory*, section 5, for the structure of the network."],
         look="| design | τ₁ / τ₂ / τ₃ (s) | peak T1 (K) | 63 % after | T1 at t = 20 s |\n|---|---|---|---|---|\n"
              "| nominal (h 1500) | 2.77 / 0.81 / 0.42 | 130 | 1.5 s | 2.3 K |\n"
              "| h = 500 | 8.18 / 1.00 / 0.47 | 223 | 3.9 s | 51 K |\n"
              "| ρc × 1.3 | 3.50 / 1.02 / 0.54 | 128 | 1.8 s | 4.6 K |\n"
              "| s_att = 0.5 | 2.77 / 0.81 / 0.42 | 132 | 1.5 s | 2.3 K |\n\n"
              "- **The slowest mode** is the Al base discharging through the cold plate: lower h stretches it by "
              "3×, more heat capacity by 1.3×. After 10 s the weakly cooled board is still far from its steady "
              "296 K — the peak (223 K) is set by the time constant, not by the steady state.\n"
              "- **The first 0.1 s** belong to the die and the die attach: 41.5 K nominal, 42.9 K with a voided "
              "attach — the attach matters for short pulses, not for long loads.\n"
              "- **Parametric network (study results, K = 32):** about 1.0 K RMSE for new conditions and positions "
              "inside the training range, 1.1 K for unseen chip positions and 0.6 K over a 30 min mission profile, "
              "in 0.02 s; the Krylov model is exact for one known board but must be rebuilt for every design.\n"
              "- **Network on this step test:** 0.77 K junction RMSE at the nominal design, 0.67 K with h = 3000 and "
              "T1 moved to x = 6 mm — but 22 K at h = 250, outside the training range (500–5000): the network "
              "extrapolates poorly in the cooling, the reason for the run *physExt* (physics-only boards there).",
         exercise=["Set h = 250 (outside the training range): how large is the network error compared to h = 1000?",
                   "Raise the simulated time to 60 s at h = 500: when is the steady state reached?",
                   "Which time constant does the die attach change, and why is the slowest one unchanged?"]),

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
