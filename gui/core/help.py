#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         help.py
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
This function holds the extended help of the parameters (shown in the question marks of the forms), the parameters of
the basic (non-expert) mode and the conditions under which a parameter or category is used at all. The texts are
written from the Python implementation (src/py), not from the workbook descriptions.

HELP:            'Section.Category.Variable' -> markdown text (what it does, effect of the value, typical values)
BASIC:           parameters shown when the expert mode is off (empty: all parameters are basic)
ACTIVE:          'Section.Category.Variable' -> function(cfg) that is False if the parameter has no effect with the
                 current settings (hidden in the basic mode, marked as inactive in the expert mode)
RELEVANT:        section -> {category: function(cfg)} False if the category is not used by the selected model
CATEGORY_TITLES: category -> title of its expander
"""

#######################################################################################################################
# Help texts
#######################################################################################################################
_NOT_USED = "\n\n**Not used by the Python version** (kept for the MATLAB version and the workbook layout)."
HELP = {
    # ==================================================================================================================
    # Exp: experiment
    # ==================================================================================================================
    'Exp.gen.init':
        "Offset of the temperatures before training (offRemove): 1) unchanged, 2) the first value of every profile "
        "is subtracted (the model learns the temperature *rise*), 3) the reference temperature (e.g. coolant, "
        "Setup › References) is subtracted. The offset is added back after the prediction. Use 2 for step responses "
        "from a common start temperature, 3 when the coolant temperature varies.",
    'Exp.gen.samp':
        "Interpolation when the data are resampled to Ts: 1) zero-order hold (keeps steps of the losses sharp), "
        "2) linear, 3) spline. ZOH is right for piece-wise constant losses, linear for measured temperatures.",
    'Exp.gen.Ts':
        "Sampling period used inside the toolkit (s). All profiles are resampled to it. Smaller Ts resolves fast "
        "transients (chip, solder) but makes every model slower and the SS/RC fits more ill-conditioned; typical "
        "0.1–1 s for power modules, 1–10 s for heat sinks and motors.",
    'Exp.gen.dx': "Spatial sampling in x (m)." + _NOT_USED,
    'Exp.gen.dy': "Spatial sampling in y (m)." + _NOT_USED,
    'Exp.gen.dz': "Spatial sampling in z (m)." + _NOT_USED,
    'Exp.gen.pad':
        "Number of samples added in front of every profile (padData): inputs with zeros, outputs with their first "
        "value. Gives dynamic models a settled start; 0 = off.",
    # ==================================================================================================================
    # Dat: data
    # ==================================================================================================================
    'Dat.gen.dOut':
        "Dimension used by the model (dimData): 1D = one probe of a 2D/3D field (heat input at (inpX, inpY, inpZ), "
        "temperature at (outX, outY, outZ)) for the 1D models RC, SS, SF, ML, DL, PINN; 2D/3D = whole field for "
        "PO/PS. Must equal the data dimension or be 1 (3D → 2D is not implemented).",
    'Dat.gen.cut': "Axis of the 3D → 2D cut. The reduction 3D → 2D is not implemented (the run stops)." + _NOT_USED,
    'Dat.gen.cutVal': "Position of the 3D → 2D cut (m)." + _NOT_USED,
    'Dat.gen.inpX': "x position of the heat input probe (m) when a field is reduced to 1D, measured from the minimum "
                    "of the geometry (legacy: absolute). Also the marked node of the field plots.",
    'Dat.gen.inpY': "y position of the heat input probe (m), see inpX.",
    'Dat.gen.inpZ': "z position of the heat input probe (m), 3D data only; also the z slice of the field plots.",
    'Dat.gen.outX': "x position of the temperature probe (m) when a field is reduced to 1D (nearest node).",
    'Dat.gen.outY': "y position of the temperature probe (m), see outX.",
    'Dat.gen.outZ': "z position of the temperature probe (m), 3D data only.",
    # ==================================================================================================================
    # Mdl: general
    # ==================================================================================================================
    'Mdl.gen.err':
        "Convergence bound of the temperature-dependent loss iteration in the DL solver (K): the prediction is "
        "repeated with corrected losses until the mean change is below this value (only with Par › loss bw = 1).",
    'Mdl.gen.eps':
        "Lower numerical limit: tolerance of the RC/SF fits and regularisation floor of the POD fits. Leave it at "
        "1e-12 unless a fit stops too early.",
    'Mdl.gen.sol': "ODE solver of the MATLAB version. The Python models use exact discrete solutions "
                   "(matrix exponential, first-order hold)." + _NOT_USED,
    'Mdl.gen.iterMax':
        "Maximum iterations of the output-error fit of SS/PS (fitPem, ss dis = 2). 100–1000 is usually enough; "
        "the default only matters for badly conditioned data.",
    'Mdl.gen.nSub': "Maximum iterations of the loss correction during testing (SS and DL, Par › loss bw = 1).",
    'Mdl.gen.opt':
        "Hyperparameter search: RC and SS try the orders 1…Kmax and keep the one with the smallest validation "
        "error; the ML models 2, 3, 5, 6 use a random search with 3-fold cross validation. Slower, but no "
        "guessing of K.",
    'Mdl.gen.Kmax': "Upper limit of the order search (opt = 1) and of the number of POD modes (pod sel = 2).",
    'Mdl.gen.Emax':
        "Energy criterion of the POD (pod sel = 2): the smallest K with Σσ²(1…K)/Σσ² ≥ Emax is used. 99.9 % "
        "typically needs 3–8 modes for heat spreading; higher values add modes that mostly fit noise.",
    # ==================================================================================================================
    # Mdl: RC
    # ==================================================================================================================
    'Mdl.rc.K':
        "Number of Foster RC pairs (RC), of RC pairs per node (SF Foster) or of Cauer stages per source (SF "
        "coupled network). 3–5 pairs describe chip, solder, substrate and heat sink; more pairs fit noise and give "
        "pairs with nearly equal τ.",
    'Mdl.rc.bc': "Bound of the RC parameters of the MATLAB-style fit (legacy = 1). The corrected fit works on a "
                 "log scale without this bound.",
    # ==================================================================================================================
    # Mdl: SS
    # ==================================================================================================================
    'Mdl.ss.K':
        "Number of states n of the state-space model (SS) or of the POD-SS model (PS). For thermal systems n is "
        "about the number of dominant time constants; start with the number of outputs + 1.",
    'Mdl.ss.dis':
        "Identification method: 1) subspace identification only (N4SID/MOESP, fast), 2) subspace identification "
        "refined by an output-error fit (slower, usually smaller simulation error).",
    'Mdl.ss.noise': "Noise model of the MATLAB identification." + _NOT_USED,
    'Mdl.ss.sol':
        "Simulation of the test data: 1) linear simulation from the initial state, 2) as MATLAB compare (initial "
        "state estimated from the test data, only together with init = 2).",
    'Mdl.ss.init': "Initial state: 1) zero, 2) estimated from the data. Estimating x0 removes start-up errors of "
                   "profiles that do not start in steady state.",
    'Mdl.ss.sta': "1) no constraint, 2) the output-error fit keeps all poles inside the unit circle (stable model, "
                  "recommended for long predictions).",
    # ==================================================================================================================
    # Mdl: POD
    # ==================================================================================================================
    'Mdl.pod.K':
        "Number of POD modes K (pod sel = 1/3). The modes are ordered by energy: the first mode is the mean heating "
        "pattern, the next ones the transient spreading. Too few modes cannot represent local hot spots, too many "
        "make the calibrated fit (protocol 5) ill-conditioned.",
    'Mdl.pod.sel': "How K is chosen: 1) fixed K, 2) smallest K with energy ≥ Emax (≤ Kmax), 3) as 1.",
    'Mdl.pod.protocol':
        "Reduced model C·da/dt = -G·a + α∘q:  \n"
        "2/3) pure Galerkin projection of the heat equation (ρc and k from the data). Physically consistent, but "
        "every heat sink must be in the data (convection hc) — the journal data have hc = 0 (F-P16).  \n"
        "4) projection + steady-state load correction α.  \n"
        "5) C and G fitted to the training trajectories (truncated SVD, rcond 1e-3) + α — default, most accurate "
        "on FEM or measured data.  \n"
        "6) as 5 with a prior that keeps C and G close to the projected matrices (more robust with little data).",
    # ==================================================================================================================
    # Mdl: ML
    # ==================================================================================================================
    'Mdl.knn.K': "Neighbours of a KNN regressor. KNN is not one of the Python ML models (1–6)." + _NOT_USED,
    'Mdl.knn.ns': "KNN search method." + _NOT_USED,
    'Mdl.knn.dist': "KNN distance." + _NOT_USED,
    'Mdl.dt.leaf': "Regression tree: minimum samples per leaf. Larger = smoother prediction, less overfitting.",
    'Mdl.dt.split': "Regression tree: maximum number of splits (leaves = splits + 1).",
    'Mdl.dt.parent': "Regression tree: minimum samples of a node to be split.",
    'Mdl.svm.kernel': "SVR kernel: 1) linear, 2) polynomial, 3) RBF (non-linear). Inputs are scaled internally.",
    'Mdl.svm.C': "SVR box constraint: larger C fits the training data closer (risk of overfitting).",
    'Mdl.svm.eps': "SVR ε-tube (K): errors smaller than ε are not penalised.",
    'Mdl.gpr.basis': "Basis of the GPR mean; approximated by normalising y in scikit-learn." + _NOT_USED,
    'Mdl.gpr.kernel': "GPR kernel: 1) exponential (Matérn ν = 0.5, rough), 2) squared exponential (smooth).",
    'Mdl.gpr.iterMax': "GPR optimiser iterations (MATLAB)." + _NOT_USED,
    'Mdl.en.method': "Ensemble: 1) gradient boosting (LSBoost), 2) bagging of trees.",
    'Mdl.en.cycle': "Number of trees of the ensemble.",
    'Mdl.en.split': "Maximum splits per tree (boosting).",
    'Mdl.nn.iterMax': "Maximum training iterations of the MLP.",
    'Mdl.nn.layer': "Number of hidden layers of the MLP (1–3).",
    'Mdl.nn.nodes': "Neurons per hidden layer of the MLP.",
    'Mdl.nn.act': "Activation of the MLP: 1) ReLU, 2) tanh, 3) sigmoid.",
    # ==================================================================================================================
    # Mdl: DL / PINN
    # ==================================================================================================================
    'Mdl.dl.epoch': "Training epochs (passes through all windows). Early stopping ends earlier when the validation "
                    "loss does not improve for valPat validations. Run time grows linearly.",
    'Mdl.dl.loss': "Training loss: 1) MSE, 2) MAE, 3) Huber, 4) custom.",
    'Mdl.dl.gradTh': "Gradient norm clipping threshold (stabilises LSTM training; 0 or inf = off).",
    'Mdl.dl.initLr': "Initial learning rate of Adam. 1e-3 is a safe start; too large gives a jumping loss.",
    'Mdl.dl.lrDropPr': "The learning rate is multiplied by lrDropFa every lrDropPr epochs.",
    'Mdl.dl.lrDropFa': "Factor of the learning rate drop (0–1).",
    'Mdl.dl.valFreq': "Validation every valFreq iterations (mini-batches).",
    'Mdl.dl.batch': "Mini-batch size (windows per gradient step). Larger = faster per epoch, smoother gradient.",
    'Mdl.dl.shu': "Shuffling of the windows: 1) never, 2) once, 3) every epoch.",
    'Mdl.dl.W':
        "Window length W (samples). The network sees W·Ts seconds of history; it should cover the dominant time "
        "constant of the thermal path. Memory grows with W.",
    'Mdl.dl.stride': "Step between training windows (seq2seq). Larger stride = fewer, less overlapping windows.",
    'Mdl.dl.seq': "0: seq2seq (the network predicts the whole window), x > 0: seq2point (the network predicts "
                  "sample x of the window; x = W means causal, only past inputs).",
    'Mdl.dl.valPat': "Early stopping patience: validations without improvement before training stops.",
    'Mdl.pinn.Rth': "Thermal resistance of the lumped physics model Cth·dT/dt = (T_ref − T)/Rth + Pv (K/W), from "
                    "the node to the reference temperature.",
    'Mdl.pinn.Cth': "Thermal capacitance of the lumped physics model (Ws/K). τ = Rth·Cth is its time constant.",
    'Mdl.pinn.Rs': "Stator resistance (Ω); available as the constant Rs in the loss equation (default 3·Rs·x4²).",
    'Mdl.pinn.alpha': "Weight α of the data loss mean((T̂ − T)²).",
    'Mdl.pinn.beta':
        "Weight β of the physics residual mean(r²). 0 = data only (as MATLAB). The residual is in W, its square is "
        "large (10⁶–10⁸ W²), so useful values are small (1e-8–1e-5): compare β·L_phys with α·L_data in the "
        "results.",
    'Mdl.pinn.PvSrc':
        "Where the losses Pv of the physics residual come from. 0: the equation PvEq; k > 0: input feature k is "
        "already the loss in W (e.g. measured or simulated losses).",
    'Mdl.pinn.PvEq':
        "Loss equation (W) for PvSrc = 0, evaluated on every sample of the window. Symbols: x1, x2, … input "
        "features (raw units), T predicted temperature of the residual node (°C), Rth, Cth, Rs and the constants "
        "of 'Constants'. Operators + − * / ** and abs, sqrt, exp, log, sin, cos, tanh, minimum, maximum. "
        "Examples: 3*Rs*x4**2 (copper losses), 3*Rs*(1+0.00393*(T-20))*x4**2 (with temperature-dependent "
        "resistance), x1*x2 (voltage × current), k*x3**2 + p0 with const 'k=0.01; p0=50'.",
    'Mdl.pinn.const': "Constants of the loss equation, 'name=value' separated by semicolons, e.g. 'k=0.01; p0=50'. "
                      "Empty or 0: none.",
    'Mdl.pinn.TrefSrc': "Reference temperature of the residual (ambient, coolant). 0: the constant Tref; k > 0: "
                        "input feature k in °C (motor data: feature 2 = ambient).",
    'Mdl.pinn.Tref': "Constant reference temperature for TrefSrc = 0 (°C).",
    'Mdl.pinn.node': "Output (1 = first) whose prediction enters the residual; the other outputs are trained on the "
                     "data only.",
    'Mdl.gen.dev':
        "Compute device of the neural networks (DL, PI). 1: GPU via CUDA if PyTorch was installed with CUDA "
        "support, otherwise the CPU is used with a warning. The scikit-learn models of the ML mode always run on "
        "the CPU.",
    'Mdl.sf.bnd':
        "Coupled network: every R and C may move by at most this factor from its initial value during the fit "
        "(bounds of the optimiser). Too small → parameters stick at the bounds (warning in the log); too large → "
        "slower fit, poorly identifiable parameters may drift.",
    'Mdl.sf.Nmax': "Coupled network: the training data are decimated to at most Nmax samples for the fit. More "
                   "samples = more accurate fast dynamics, longer fit.",
    # ==================================================================================================================
    # Par: pre-processing
    # ==================================================================================================================
    'Par.gen.inter': "Interpolation of the MATLAB version." + _NOT_USED,
    'Par.gen.normX': "Normalisation of the inputs: 1) none, 2) 0–1, 3) min–max, 4) mean/std. ML/DL need "
                     "normalised inputs; RC/SS/PO work in physical units (keep 1).",
    'Par.gen.normY': "Normalisation of the outputs, see normX.",
    'Par.gen.err': "Convergence bound of the loss iteration in the SS solver (K).",
    'Par.gen.eps': "Lower numerical limit of the loss correction θ(T).",
    'Par.gen.th':
        "Threshold of the transient/steady-state split of the error metrics: a sample is transient if "
        "|dT| > th·max|dT|.",
    'Par.loss.Rslope': "Temperature coefficient of the on-resistance (1/K): θ(T) = (Rcon + Rohm·(1 + Rslope·(T − "
                       "Tref)))/(Rcon + Rohm). Copper 0.393 %/K.",
    'Par.loss.Rcon': "Temperature-independent part of the resistance (Ω).",
    'Par.loss.Rohm': "Temperature-dependent part of the resistance at Tref (Ω).",
    'Par.loss.Tref': "Temperature at which the losses of the data are given (°C).",
    'Par.loss.fw': "Correct the training losses with θ(T) of the measured temperature (losses given at Tref).",
    'Par.loss.bw': "Iterate the prediction with temperature-dependent losses during testing (SS, DL).",
}

#######################################################################################################################
# Basic parameters (expert mode off)
#######################################################################################################################
BASIC = {
    'Exp.gen.init', 'Exp.gen.Ts', 'Exp.gen.samp',
    'Dat.gen.dOut', 'Dat.gen.inpX', 'Dat.gen.inpY', 'Dat.gen.inpZ', 'Dat.gen.outX', 'Dat.gen.outY', 'Dat.gen.outZ',
    'Mdl.gen.opt', 'Mdl.gen.Kmax', 'Mdl.gen.Emax',
    'Mdl.rc.K', 'Mdl.ss.K', 'Mdl.ss.dis', 'Mdl.ss.init', 'Mdl.ss.sta',
    'Mdl.pod.K', 'Mdl.pod.sel', 'Mdl.pod.protocol',
    'Mdl.dt.leaf', 'Mdl.dt.split', 'Mdl.svm.kernel', 'Mdl.svm.C', 'Mdl.gpr.kernel', 'Mdl.en.method', 'Mdl.en.cycle',
    'Mdl.nn.layer', 'Mdl.nn.nodes', 'Mdl.nn.act',
    'Mdl.dl.epoch', 'Mdl.dl.initLr', 'Mdl.dl.batch', 'Mdl.dl.W', 'Mdl.dl.seq', 'Mdl.dl.loss',
    'Mdl.pinn.alpha', 'Mdl.pinn.beta', 'Mdl.pinn.Rth', 'Mdl.pinn.Cth', 'Mdl.pinn.PvSrc', 'Mdl.pinn.PvEq',
    'Mdl.pinn.const', 'Mdl.pinn.TrefSrc', 'Mdl.pinn.Tref', 'Mdl.gen.dev',
    'Par.gen.normX', 'Par.gen.normY', 'Par.loss.fw', 'Par.loss.bw',
}


#######################################################################################################################
# Activity rules
#######################################################################################################################
def _run(c, k, d=0):
    return c.get('Run', {}).get(k, d)


def _mode(c):
    return _run(c, 'mode', '')


def _ml(c):
    return int(_run(c, 'selML', 1) or 1)


def _probe(c):
    return int(_run(c, 'datDim', 1)) > 1 and int(c['Dat']['gen']['dOut']) == 1


def _loss(c):
    return c['Par']['loss']['bw'] == 1 or c['Par']['loss']['fw'] == 1


_UNUSED = ['Exp.gen.dx', 'Exp.gen.dy', 'Exp.gen.dz', 'Dat.gen.cut', 'Dat.gen.cutVal', 'Mdl.gen.sol', 'Mdl.ss.noise',
           'Mdl.knn.K', 'Mdl.knn.ns', 'Mdl.knn.dist', 'Mdl.gpr.iterMax', 'Mdl.gpr.basis', 'Par.gen.inter']
ACTIVE = {
    **{k: (lambda c: False) for k in _UNUSED},
    'Dat.gen.inpX': _probe, 'Dat.gen.inpY': _probe, 'Dat.gen.outX': _probe, 'Dat.gen.outY': _probe,
    'Dat.gen.inpZ': lambda c: int(_run(c, 'datDim', 1)) == 3,
    'Dat.gen.outZ': lambda c: int(_run(c, 'datDim', 1)) == 3 and int(c['Dat']['gen']['dOut']) == 1,
    'Dat.gen.dOut': lambda c: int(_run(c, 'datDim', 1)) > 1,
    'Mdl.gen.Kmax': lambda c: c['Mdl']['gen']['opt'] == 1 or c['Mdl']['pod']['sel'] == 2,
    'Mdl.gen.Emax': lambda c: c['Mdl']['pod']['sel'] == 2 and _mode(c) in ('PO', 'PS'),
    'Mdl.gen.iterMax': lambda c: _mode(c) in ('SS', 'PS') and c['Mdl']['ss']['dis'] == 2,
    'Mdl.gen.opt': lambda c: _mode(c) in ('RC', 'SS', 'ML'),
    'Mdl.gen.nSub': lambda c: c['Par']['loss']['bw'] == 1,
    'Mdl.gen.err': lambda c: c['Par']['loss']['bw'] == 1 and _mode(c) == 'DL',
    'Mdl.rc.bc': lambda c: int(_run(c, 'legacy', 0)) == 1,
    'Mdl.ss.sol': lambda c: c['Mdl']['ss']['init'] == 2,
    'Mdl.pod.K': lambda c: c['Mdl']['pod']['sel'] != 2,
    'Mdl.pod.protocol': lambda c: _mode(c) == 'PO',
    'Mdl.svm.kernel': lambda c: _ml(c) == 3,
    'Mdl.svm.C': lambda c: _ml(c) == 3,
    'Mdl.svm.eps': lambda c: _ml(c) == 3,
    'Mdl.en.split': lambda c: _ml(c) == 5 and c['Mdl']['en']['method'] == 1,
    'Mdl.dl.seq': lambda c: _mode(c) == 'DL',
    'Mdl.dl.loss': lambda c: _mode(c) == 'DL',
    'Mdl.dl.valFreq': lambda c: _mode(c) == 'DL',
    'Mdl.gen.dev': lambda c: _mode(c) in ('DL', 'PI'),
    'Mdl.pinn.PvEq': lambda c: int(c['Mdl']['pinn'].get('PvSrc', 0) or 0) == 0,
    'Mdl.pinn.const': lambda c: int(c['Mdl']['pinn'].get('PvSrc', 0) or 0) == 0,
    'Mdl.pinn.Rs': lambda c: int(c['Mdl']['pinn'].get('PvSrc', 0) or 0) == 0,
    'Mdl.pinn.Tref': lambda c: int(c['Mdl']['pinn'].get('TrefSrc', 0) or 0) == 0,
    'Mdl.sf.bnd': lambda c: int(_run(c, 'selSF', 2) or 2) == 2,
    'Mdl.sf.Nmax': lambda c: int(_run(c, 'selSF', 2) or 2) == 2,
    'Par.gen.err': lambda c: c['Par']['loss']['bw'] == 1,
    'Par.gen.eps': _loss,
    'Par.loss.Rslope': _loss, 'Par.loss.Rcon': _loss, 'Par.loss.Rohm': _loss, 'Par.loss.Tref': _loss,
}

#######################################################################################################################
# Categories
#######################################################################################################################
RELEVANT = {
    'Mdl': {'rc': lambda c: _mode(c) in ('RC', 'SF'),
            'ss': lambda c: _mode(c) in ('SS', 'PS'),
            'pod': lambda c: _mode(c) in ('PO', 'PS'),
            'knn': lambda c: False,
            'dt': lambda c: _mode(c) == 'ML' and _ml(c) == 2,
            'svm': lambda c: _mode(c) == 'ML' and _ml(c) == 3,
            'gpr': lambda c: _mode(c) == 'ML' and _ml(c) == 4,
            'en': lambda c: _mode(c) == 'ML' and _ml(c) == 5,
            'nn': lambda c: _mode(c) == 'ML' and _ml(c) == 6,
            'dl': lambda c: _mode(c) in ('DL', 'PI'),
            'pinn': lambda c: _mode(c) == 'PI',
            'sf': lambda c: _mode(c) == 'SF'},
}
CATEGORY_TITLES = {'gen': 'General', 'rc': 'RC network', 'ss': 'State space', 'pod': 'POD',
                   'knn': 'K nearest neighbours', 'dt': 'Regression tree', 'svm': 'Support vector regression',
                   'gpr': 'Gaussian process', 'en': 'Ensemble', 'nn': 'Neural network (MLP)', 'dl': 'Deep learning',
                   'pinn': 'Physics (PINN)', 'loss': 'Temperature-dependent losses', 'sf': 'Coupled network (SF)'}


#######################################################################################################################
# Helper
#######################################################################################################################
def key(section, p):
    return f"{section}.{p['cat']}.{p['var']}"


def is_basic(section, p):
    return not BASIC or key(section, p) in BASIC


def is_active(section, p, cfg):
    rule = ACTIVE.get(key(section, p))
    if rule is None:
        return True
    try:
        return bool(rule(cfg))
    except (KeyError, TypeError, ValueError):
        return True
