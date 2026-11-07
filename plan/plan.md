# TherMOS – Python Port and Code Review Plan

Date: 07.11.2026 · Branch: `Python-GUI` · Scope: everything except `gen/`, `dev/`, `temp/` and `src/dev/`

---

## 1. Goal

1. Provide a Python version of TherMOS that runs the same experiments from the same `config/*.xlsx`
   files and data, and writes to the shared `mdl/` and `results/` folders.
2. Keep the MATLAB version working side by side.
3. Use the port to review the code. Wrong numerics, crashes, bad documentation and bad plots are listed in
   Section 6. These need a decision on how to handle them (question Q2).

---

## 2. Target Layout

```
TherMOS/
├── start.m  start2D.m  start3D.m  startRth.m     # MATLAB entry points (unchanged, only path update)
├── start.py start2D.py start3D.py startRth.py    # NEW Python entry points (same options, same names)
├── initPath.m                                    # adds src/mat instead of src
├── requirements.txt                              # NEW
├── config/  data/  mdl/  results/  docu/         # shared by both versions
├── setup/<group>/*.m  +  setup/<group>/*.py      # Python copy next to every MATLAB setup script
├── plan/plan.md                                  # this file
└── src/
    ├── mat/   (current MATLAB code, moved with `git mv` so history is kept)
    │   ├── main.m training.m testing.m data/ general/ model/ numeric/ plotting/
    └── py/    (NEW, same folder tree, one function per camelCase file)
        ├── main.py training.py testing.py
        ├── data/      loadData, DataLoader (class), concatData, dimData, normData, offRemove, offAdd, padData,
        │              resmpData, updLoss
        ├── general/   loadPara, initPath, calcFeat, calcGrad, calcGrad3D, calcError, map2D, map3D, performance,
        │              saving
        ├── model/     RC/ SS/ SF/ PO/ PS/ ML/ DL/ PINN/   (xxFit.py / xxSol.py, PO/optXXX.py)
        ├── numeric/   gaussLegendre, jacobian2D, jacobian3D, sysId (N4SID/PEM/lsim/c2d/d2c), linOde
        └── plotting/  plotMdl, plotResults1D, plotResults2D, plotResults3D, printConsole
```

Conventions (from the `my-coding-style` skill): 119-character `#` headers, H1/H1i/H2/H3 bars, Inputs/Outputs
docstrings with units, `START:/INFO:/WARN:/ERROR:` console prefixes, `setup` and `para` as nested dicts with
the same keys as the MATLAB structs (`para['Mdl']['pod']['K']`). The MATLAB `setup` struct is flat, so the Python
`setup` dict stays flat too. This keeps the two versions easy to compare line by line.

Unused files that will **not** be ported (proposal): `numeric/rk4.m`, `numeric/rk6.m`, `numeric/intStencil.m`,
`numeric/stencilWeight.m`, `numeric/smooth_internal.m` (no caller; `rk6` is not a 6th-order scheme, and
`stencilWeight` case 4 has 12 weights where `intStencil` has 14). Proposal: move them to `src/mat/dev/`.

---

## 3. MATLAB → Python Mapping

| MATLAB | Python | Note |
|---|---|---|
| `readtable` (xlsx config/data) | `pandas.read_excel` (openpyxl) | |
| `load` (*.mat) | `scipy.io.loadmat` | all data files are v5. **1D files hold a MATLAB `table` object that scipy cannot read** (see Q4) |
| `iddata`, `n4sid`, `ssest`, `compare`, `lsim`, `d2c` | own `numeric/sysId.py`: N4SID (subspace, numpy), output-error PEM (`scipy.optimize.least_squares`, starts from N4SID), `scipy.signal.dlsim/lsim`, ZOH/FOH via `scipy.linalg.expm` | no exact copy of MATLAB's `ssest` is possible. Fits will be close, not identical |
| `lsqcurvefit` (LM) | `scipy.optimize.least_squares` (`trf` when bounds are set, `lm` otherwise) | |
| `fminunc`, `lsqnonlin` | `scipy.optimize.minimize(BFGS)` / `least_squares` | `optCthGthUB` is linear least squares → closed form with `np.linalg.lstsq` |
| `ode15s`/`ode23s` with mass matrix | exact discretisation of the linear ODE (`expm`, first-order hold); fallback `solve_ivp(method='BDF')` | the model is linear, so exact discretisation is faster and has no tolerance error |
| `scatteredInterpolant` | `scipy.interpolate.LinearNDInterpolator` / `NearestNDInterpolator` (Delaunay built once) | |
| `interp2`/`interp3` | `RegularGridInterpolator`, assembled once as a sparse matrix | removes the `Nt×K` loop of interpolations |
| `gradient` | `np.gradient` with explicit axis order | |
| `delaunayTriangulation` | `scipy.spatial.Delaunay` | |
| `timeseries/resample` | `np.interp` / `scipy.interpolate.interp1d(kind=previous/linear/cubic)` | the `samp` option is honoured (see F-D5) |
| `fitlm`, `fitrtree`, `fitrsvm`, `fitrgp`, `fitrensemble`, `fitrnet` | scikit-learn `LinearRegression`, `DecisionTreeRegressor`, `SVR`, `GaussianProcessRegressor`, `GradientBoostingRegressor`/`BaggingRegressor`, `MLPRegressor` | Bayesian hyper-parameter optimisation → `RandomizedSearchCV` |
| `relieff` | `sklearn.feature_selection.mutual_info_regression` (or `skrebate.ReliefF`) | |
| `trainnet`, `dlnetwork`, `adamupdate` | PyTorch **or** TensorFlow (see Q3) | |
| `uitable`, `imshow`, `slice`, `pcolor`, `surf`, `pdeplot3D` | matplotlib (`ax.table`, `imshow`, `pcolormesh(shading='nearest')`, 3D scatter/planes) | |
| trained models in `mdl/*.mat` | `mdl/mdl_<xx>_<name>.pkl` (`.pt`/`.keras` for DL) | the MATLAB models contain `idss`/network objects that Python cannot load. Each version trains its own models; the file names keep the same pattern and get a different extension |

Python environment: `C:\venvs\TherMOS` (Python 3.15) has numpy, scipy, pandas and matplotlib. It still needs
`openpyxl` and `scikit-learn`, plus `torch` or `tensorflow`. **Check that the DL framework has a wheel for
Python 3.15.** If not, use a 3.12/3.13 venv.

---

## 4. Work Packages

| # | Package | Content | Check |
|---|---|---|---|
| WP0 | Restructure | `git mv src/* src/mat/`, update `initPath.m`, create `src/py` skeleton, `requirements.txt` | the MATLAB `start.m` still runs |
| WP1 | Data I/O | `loadPara`, `DataLoader`, `loadData`, `resmpData`, `concatData`, MATLAB table export (Q4) | Python and MATLAB load the same arrays (MATLAB R2026a is installed, so I can make reference dumps in batch mode) |
| WP2 | Pre/Post-processing | `offRemove/offAdd`, `dimData`, `normData`, `padData`, `updLoss`, `performance`, `calcError`, `calcGrad(3D)`, `map2D/3D`, `saving` | unit tests against MATLAB dumps |
| WP3 | 1D models | RC, SS (+ `sysId`), SF | `tutorial_RC`, `tutorial_SS`, `startRth` match the MATLAB metrics |
| WP4 | 2D/3D POD | PO (+ opt*), PS, PS3D, `gaussLegendre`, `jacobian2D/3D` | `tutorial_PO`, `start2D`, `start3D`, journal Sec. IV-B (1 s data) |
| WP5 | ML/DL/PINN | `mlFit/mlSol`, `dlFit/dlSol`, `pinnFit/pinnSol`, `calcFeat` | `tutorial_ML`, `tutorial_DL` (statistical match only, since training is random) |
| WP6 | Output | `printConsole`, `plotMdl`, `plotResults1D/2D/3D` with the plot fixes from Section 6.6 | visual check of the figures |
| WP7 | Entry points and docs | `start*.py`, `setup/**/*.py`, README (Python usage, dependencies, folder layout), readme.txt files | every setup script runs end to end |

Order: WP0 → WP1 → WP2 → WP3 → WP4 → WP5 → WP6 → WP7. After each package I commit, and you can review.

---

## 5. Verification

- A MATLAB batch script (`src/mat/...` helper, not shipped in `src/py`) dumps the intermediate results for each tutorial
  setup: data after loading, after offset removal and after normalisation; the model matrices; and the predictions
  and metrics.
- `pytest` tests compare the Python results with these dumps. The tolerances are: exact for data handling,
  1e-8 relative for deterministic numerics, and a "metrics within X %" check for SS, ML and DL fits.
- Where a bug fix changes a result on purpose (Section 6), the test compares against the corrected MATLAB code.
  If the MATLAB code is not fixed, the test records the difference in a list of expected deviations.

---

## 6. Findings from the Review

Severity: **C** = crashes or gives a clearly wrong result · **N** = wrong or questionable numerics ·
**D** = documentation · **P** = plotting · **S** = structure or maintainability.
Line numbers refer to the current MATLAB files in `src/`.

### 6.1 Pipeline (`main`, `training`, `testing`, `saving`)

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-M1 | C | `general/saving.m:63,68` | The file named `result_para_*` holds `setup`, and `result_setup_*` holds `para` (the two names are swapped). |
| F-M2 | C | `training.m`, `testing.m` | With several models selected, `mdl`/`pred` are overwritten and only the last model counts. With none selected, `mdl`/`pred` are undefined and the run crashes. If loading the model fails, the run continues with `mdl=[]` and crashes later. |
| F-M3 | S | `training.m:51ff` | Every training run overwrites the shipped reference models in `mdl/` (for example `mdl_ss_default.mat`). |
| F-M4 | C | `main.m:101` | The 1D/2D sanity check ignores `selPS==2` (3D). |
| F-M5 | S | `main.m:32-34` | `feature_roll`, `EWMA` and `EWMS` are hard-coded in `main` and cannot be set from the config. |
| F-M6 | C | `testing.m:36` | `data.pr.odeTime = 0` is written to a local copy and never reaches the output. |
| F-M7 | S | `testing.m:71-80, 257-266` | RC and ML silently keep only the first output column, without telling the user. |

### 6.2 Data handling

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-D1 | C | `data/updLoss.m:61` | Loss scaling `θ(T) = (Rcon + Rohm·Rslope·(T−Tref)) / (Rcon+Rohm)` drops the `Rohm·1` term. At T = Tref it gives `Rcon/(Rcon+Rohm)` instead of 1. Correct: `(Rcon + Rohm·(1+Rslope·(T−Tref))) / (Rcon+Rohm)`. The same formula is copied into `rcSol`, `ssSol`, `sfSol`, `mlSol`, `dlSol` and `pinnSol`. |
| F-D2 | C | `data/updLoss.m:79,90` | `data.tr.X{1,i}` uses the matrix `X` as a cell array; it should be `X2{1,i}` scaled with `y2{1,i}`, so `fw==1` always crashes. It also assumes input *i* belongs to output *i*, which is not documented. |
| F-D3 | C | `data/normData.m:236` | For Min/Max normalisation (`normY==3`), the structured data use `(y2−min)·(max−min)+avg` instead of `(y2−avg)/(max−min)`. |
| F-D4 | N | `data/normData.m:87-95` | The guard against constant channels only works when *all* channels are constant (vector `if`). A single constant column divides by zero and gives NaN, and `std = 0` is not caught either. `Tref` is normalised with the statistics of the temperature *rise* (after offset removal), which mixes units. |
| F-D5 | N | `data/resmpData.m:79-91,105` | The interpolation option (`samp`: ZOH/linear/spline) is computed but never used, because `timeseries/resample` is always linear. `tRef = 0:Ts:Tend−Ts` drops the last sample and assumes `t(1)=0`. |
| F-D6 | C | `data/offRemove.m:65 vs 71` | With `init==2`, the concatenated `y` uses the first sample of the *first* profile, while `y2` uses each profile's own first sample, so the two data sets differ. For an invalid `init`, `off` is undefined. |
| F-D7 | C | `data/padData.m:48,51,59,61` | `r(1)` and `off(1)` (the first element only) are copied into all columns. With `init==1` the temperature is padded with 0 while the data start at, for example, 25 °C, which creates an artificial step. |
| F-D8 | C | `data/dimData.m:189,216` | After a 2D→1D reduction, `idData` is built from the *unreduced* `data.y2/X2` (thousands of channels). It should use `out.*`. The 3D→2D branch is empty (it returns the data unchanged without a warning). An invalid reduction leaves `out` undefined. 2D→1D multiplies the input by `dx·dy`, but 3D→1D does not multiply by `dx·dy·dz`. |
| F-D9 | N | `data/DataLoader.m:137-148, 214-225` | Ratio split: the test set starts at `tr_idx`, so one sample overlaps, and the validation set is the first `rVl` share of the *training* data, which is data leakage. |
| F-D10 | N | `data/DataLoader.m:131 vs 208` | When the time axis is rebuilt for ID-based data, the xlsx path uses `linspace(t0, Ts·N, N)` (wrong step unless t0 = Ts) and the mat path uses `linspace(0, Ts·(N−1), N)`. |
| F-D11 | C | `data/DataLoader.m:249` | The reference matrix is sized by `setup.out` but filled from `setup.ref`. |
| F-D12 | N | `data/concatData.m:140` vs `plotting/printConsole.m:53` | The probe position `inpX/inpY` is read as an *absolute* coordinate in `concatData` and `dimData`, but *relative to min(geo)* in `printConsole` and the plots. Different nodes are picked whenever the geometry does not start at 0. |
| F-D13 | S | `data/loadData.m` | The same block is repeated 3 × 3 times (tr/te/vl × file/sheet/ID). `Ts` of the last file silently overwrites the others. |

### 6.3 1D models

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-R1 | C | `model/RC/rcFit.m:200-201, 243-244` | `lb`/`ub` are 2×K matrices, but the parameter vector is `[R1..RK, τ1..τK]`. MATLAB reads the bounds column by column as `[R,τ,R,τ,…]`, so the bounds go to the wrong parameters when `bc>0` and K>1. |
| F-R2 | N | `model/RC/rcFit.m:138` | `Zth = ΔT ./ Pv` uses the full loss time series. It is only valid for a constant step in the losses; with `Pv(t)=0` or a varying `Pv` it gives Inf or nonsense. The starting guess `τ = 1 s` for every branch converges poorly; log-spaced starting values (as `sfFit` uses) are better. |
| F-R3 | D | `model/RC/rcSol.m:20` | The documented formula `(Rth·dt)/(τ+dt)·Pv` differs from the implemented and correct Tustin step `Rth·Ts/(2τ+Ts)·(Pv(i)+Pv(i−1))`. |
| F-R4 | N | `rcSol.m:91`, `ssSol`, `mlSol`, … | In the loss feedback, `θ(T + Toff)` adds `r` as the offset even when `init==2` (the offset is then the initial value, not `r`). With normalisation on, normalised temperatures are added to °C references. |
| F-S1 | C | `model/SS/ssSol.m:125` | `mean(abs*(T_old−T_est)…)` is a syntax error, so every run with `Rohm≠0` crashes. The loop also multiplies `Pv2` by θ again in every iteration (it should be `Pv·θ`), and the error is not an absolute value. |
| F-S2 | C | `model/SS/ssSol.m:72` | `opt` is overwritten after the `if`, so the `ss.init` option has no effect. Line 52: `K = length(size(A))` is always 2. |
| F-S3 | N | `model/SS/ssFit.m:69,105-136` | `ssestOptions` is also passed to `n4sid` (which expects `n4sidOptions`). The option `dis` selects both the algorithm (n4sid/ssest) and the continuous conversion, two separate things in one switch. |
| F-F1 | N | `model/SF/sfFit.m`, `sfSol.m` | This is not a thermal *structure function* (cumulative Cauer). It is a Foster fit per node plus a steady-state R-matrix. In the ODE, the branch time constant becomes `C_ik/(w_ik·G_ii)` instead of the fitted `R_ik·C_ik`, so the dynamics do not match the fit. The method needs one single-source experiment per node (K ≥ M), otherwise `Rth` is singular. `θ` and `Toff` are passed to the ODE but not used. |
| F-F2 | C | `model/SF/sfSol.m:70,112` | `T0 = y(1)−Toff` and `out.y = T+Toff` subtract or add the reference although `offRemove` has already done so, and then `offAdd` in `main` adds it a second time when `init==3`. |

### 6.4 POD models (PO / PS / PS3D)

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-P1 | N | `PO/poFit.m:114-119`, `PS/psFit.m:138-143`, `ps3Fit.m:143-148` | The energy criterion uses singular values (`Σσ`), but POD energy is `Σσ²`. The loop also stops one step late (K = n+1). The result is more modes than needed. `sel==3` ("Opti") does the same as `sel==1`. |
| F-P2 | N | `PO/poFit.m:289,292-293` | `Gth` uses `alpha2D` (regular grid) together with fields at the Gauss–Legendre points (`*_i`). Same array size, but different points, so they are spatially misaligned. It should be `alpha2D_i`. |
| F-P3 | N | `PO/poFit.m` (weak form) | The Galerkin projection uses `C = ∫φφ`, `G = ∫α∇φ∇φ` and source `∫Qφ/(ρc)`. That is only correct for homogeneous material. For layered PCB/IMS stacks the consistent form is `C = ∫ρc·φφ`, `G = ∫k·∇φ∇φ`, `q = ∫Qφ`. This is a likely cause of the "numerical discrepancies Gth/Cth" listed in the README. |
| F-P4 | N | `PO/poFit.m:292-295,310-313` | The boundary terms are taken at the first/last Gauss node, which lies *inside* the domain, not on the boundary, and both ends get the same sign (no outward normal). An adiabatic (Neumann) boundary needs no term at all. A convective boundary needs `h∫φiφj ds`. |
| F-P5 | N | `PO/poFit.m:322`, `poSol.m:197,253-255` | `GC = Gth·pinv(Cth)` is in the wrong order (it should be `C⁻¹G`). The non-stiff branch fixes this with `./diag(Cth)`. `mdl.GC` is also computed before the optimisation, so it is stale. |
| F-P6 | N | `PO/optCthGthUB.m` | This is linear least squares solved with quasi-Newton over 2K² variables. A closed-form solution exists. There are no symmetry or positive-definiteness constraints, so unstable ROMs are possible. Optimising `alpha` afterwards with C and G fixed adds no freedom (`alpha≈1`), because the scaling is degenerate. |
| F-P7 | S | `PO/poFit.m:40` | `protocol = 5` is hard-coded, so the method cannot be selected from the config. `Mdl.gen.sol` (ode45/ode15/rk6/euler) in the config is never read: `poSol` hard-codes `is_stiff=1` and `ode23s`. |
| F-P8 | N | `PO/poFit.m:174` vs `psFit.m:192` vs `ps3Fit.m:201` | The Jacobian threshold is 0.001, 0.1 and 0.01 in the three files. `jacobian2D` is a heuristic for FE mesh density (triangle area divided by the median area), not the Jacobian of a coordinate transformation. On a regular quadrature grid, weighting by input mesh density distorts the integral. |
| F-P9 | C | `PS/psSol.m:185`, `ps3Sol.m:189` | `iddata(q,q,Ts)` with `q` of size K×Nt (psFit passes `q'`). This probably gives transposed inputs (to be checked in MATLAB). |
| F-P10 | C | `PS/ps3Fit.m:124-129` | If the SVD fails, the code prints an error and continues, and `S` is undefined. `poFit`/`psFit` first try a *full* SVD (an Nt×Nt U matrix), which can run out of memory; use `econ` directly. |
| F-P11 | S | `PO/poFit.m:381-382` | `matCond` without a semicolon writes debug output to the console. Large blocks of commented-out code. |
| F-P12 | S | PO/PS fit and solve | The `for Nt / for K` loop calls `interp2` on full fields (Nt·K interpolations). A precomputed sparse interpolation matrix is orders of magnitude faster. The fit and solve files repeat about 60 % of the same code. |
| F-P13 | D | `PO/opt*.m` | The headers are placeholders ("Here goes the description …", "Input-1"). |

### 6.5 ML / DL / PINN

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-L1 | C | `ML/mlFit.m:81 vs 178-185` | `Pv = data.X` (all profiles concatenated), but `T` is the *average* over profiles truncated to `min(Nt)`, so the lengths differ as soon as there is more than one profile. |
| F-L2 | C | `ML/mlFit.m:253` | The stray `'MPG'` argument (copied from the MATLAB documentation example) makes the ensemble optimisation crash. The GPR optimisation branch is empty. |
| F-L3 | N | `ML/mlSol.m:72` | The loop starts at 2, so `T(1)=0`. The model maps `Pv(t)→T(t)` with no memory, so it can only learn a static map and no thermal dynamics. |
| F-L4 | D | `start.m:195`, config | `selML` is documented as "1) KNN, 2) RF, 3) SVR", but the code uses 1 = linear regression, 2 = single tree, 3 = SVR, 4 = GPR, 5 = ensemble, 6 = NN. The config has KNN parameters that are never used. |
| F-L5 | C | `DL/dlSol.m:93-99` | Seq2seq: when `Nt` is a multiple of `W`, `padPv` is undefined and the run crashes. Seq2point: training uses the target at window position `seq`, but inference assumes the last position, so the two do not line up unless `seq==W`. |
| F-L6 | C | `DL/dlSol.m:155-180` | The `Rohm≠0` branch uses `Tj_est` before it exists, `num2cell` instead of an array, `continue` instead of `break` (an endless loop), and never increments `iter`. |
| F-L7 | C | `DL/dlFit.m:195-211` | `selDL==1` (CNN) with `seq==0` prints an error but leaves `layers` undefined, so the run crashes. Windows span the boundaries between profiles. `customLoss` adds the data loss twice; the physics term is a placeholder. |
| F-I1 | N | `PINN/pinnFit.m:344` | `beta = 0`: the physics loss is switched off, so the model is a plain CNN-LSTM. The constants (`Rth`, `Cth`, `Rs`, `Ts=1`, feature indices 2 and 4) are hard-coded for the motor dataset and are used on *normalised* data. |
| F-I2 | C | `PINN/pinnFit.m:103,120` | `Y(i:i+W−1,:)'` is put into a W×M slot, which is only correct for M = 1. `adamupdate` gets the epoch number instead of the global iteration count, so the bias correction is wrong. The validation loss is summed over batches instead of averaged. |
| F-G1 | N | `general/calcFeat.m` | EWMA and EWMS run across profile boundaries. The EWMS recursion is not the standard EW variance. |

### 6.6 Metrics, gradients, plots and console

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-E1 | N | `general/calcerror.m:51-55` | "NRMSE" is the MATLAB *fit percentage* `100·(1−‖e‖/‖y−ȳ‖)`, not an NRMSE. When the denominator is 0, the code returns `100·(1−‖e‖)`, which has no meaning. SAE has no definition. |
| F-E2 | N | `general/performance.m:45-46` | The transient/steady split uses `dT > th·max(dT)` without `abs`, so cooling phases always count as "steady". An empty mask is replaced by 0, and `NAE` then becomes 0/0. |
| F-G2 | N | `general/calcGrad.m:75`, `calcGrad3D.m:104,115` | `gradient(T, dy, dx)` returns `[∂/∂x, ∂/∂y]` with spacing (dy, dx). The variable names and the spacings are swapped, which is wrong whenever dx ≠ dy. `qk = −(∂T/∂x + ∂T/∂y)` is the *sum of the components*: neither the magnitude nor a vector. It should be `|∇T|` or the components. The documentation swaps `q` and `qk` (the code returns `q = k·qk`, the flux, and `qk = −∇T`). |
| F-PL1 | P | `plotResults2D.m:474-494`, 3D the same | The panels titled "Gradient (K/m)" show `q` (the flux) and the panels titled "Flux (W/m²)" show `qk` (the gradient). |
| F-PL2 | P | `plotResults3D.m:327-372` | `test.y(selT,selY,:,selY)` and `q(selY,:,selY)` use `selY` as the z index (it should be `selZ`). The title of the Z plot says "x … z" instead of "x … y". The slice planes mix `outX` with `inpY/inpZ`. |
| F-PL3 | P | `plotResults1D.m:189` | The scatter plot normalises the true and the predicted values each by *their own* maximum, which hides bias. `xlim/ylim [0 1]` cuts negative values. The histogram axis says "%" but shows a fraction. |
| F-PL4 | P | `plotResults1D.m`, `plotResults2D.m` | There is no legend for solid = true and dashed = predicted. With `plotOut==2` the legend lists all outputs for a single line. The test/val temperature panels use `zlabel` instead of `ylabel` (`plotResults2D.m:389,403`). The spatial input figure plots `end` but the title states `selT`. Error axes are labelled °C instead of K. |
| F-PL5 | P | `plotMdl.m:227,301` | `surf(Phi)` is plotted against indices but labelled "x (m)/y (m)". The time axis of the temporal modes shows samples but is labelled "t (sec)". The DL legend `['Train','Val']` becomes one entry ("TrainVal"). PS3D and SF have no model plot. |
| F-PL6 | P | `plotResults2D/3D` | `pcolor` drops the last row and column, which matters on coarse grids. `xtick` at every grid point makes the axes unreadable. `pdeplot3D` sits in a silent `try/catch` and plots `Data.y` (raw) against `pr.y`, with the error sign opposite to the other figures. |
| F-PL7 | P | `printConsole.m` | The column widths do not match the header. The "NRMSE (%)" column holds the fit percentage (F-E1). For 2D/3D only one node is printed, labelled with `setup.out(1)`. |

### 6.7 Documentation and config

| ID | Sev | Location | Finding |
|---|---|---|---|
| F-X1 | D | `README.md` | "requirements of the **PyPowerSim** toolkit" (copy-paste). The config is described with a "Unit" column, but the workbooks have "Type". No Python section. |
| F-X2 | D | many headers | Placeholder descriptions ("Here goes the description of the function") in `calcerror`, `performance`, `saving`, `plotMdl` and `opt*`. `offAdd` is described as "removes the offset". `plotResults2D/3D` are described as "1D model". `map3D` is described as "Output 2D". `dimData` reads "adapts the dimension if the input data". `pinnSol` prints "Solving a Deep Learning Model". |
| F-X3 | D | `start*.m` | `selDL` is commented "(Machine Learning)", `selPI` "(tbd)", there is the typo "Strukture", and the `selPS=2` (3D) option is missing in `start.m`/`start2D.m`. |
| F-X4 | D/S | `config/*.xlsx` | Unused parameters: `Exp.gen.dx/dy`, `Par.gen.inter`, `Mdl.gen.sol`, `Mdl.knn.*`, `Mdl.dl.valPat` (the code hard-codes patience 10). `Mdl.gen.err/eps` duplicate `Par.gen.err/eps`. The descriptions of "Input 1D Y/Z" are swapped. `Rslope` has type "percent" but is a value in 1/K. |
| F-X5 | S | `data/templates/templateData2D.mat` | `k`, `rho` and `Cp` are 11×11 matrices while `geo` is 121×3. The real 2D data (for example `exp11`) use vectors of length N with 2 columns. The template does not match the format the loader expects. |

---

## 7. Decisions

Answered on 07.11.2026:

| Q | Decision | Consequence |
|---|---|---|
| Q1 Layout | `src/mat` + `src/py` | `git mv src/* src/mat/`; Python imports `from src.py.<pkg>.<file> import <file>` |
| Q2 Bugs | **Fix in Python only** | MATLAB stays as published. Every fix in Python is listed in `docu/` as a "known deviation from MATLAB" with its finding ID. Parity tests compare against MATLAB and list each fix as an expected difference. Where a fix changes the published numbers (F-P1–F-P5, F-D1, F-E1), Python adds a config switch `legacy` (`0`: corrected, `1`: MATLAB behaviour) so that the journal results can still be reproduced. |
| Q3 DL | PyTorch | `torch` in `requirements.txt`. Check for a Python 3.15 wheel first; if there is none, create a 3.13 venv. |
| Q4 1D data | Convert in place | A one-time MATLAB batch script saves the `table` objects as structs of column vectors (still v5 `.mat`). The MATLAB `DataLoader` gets a small change so it accepts both the table format and the struct format. This is the only MATLAB change besides `initPath`. |

Further defaults (unless you object):

- **System identification**: own N4SID + output-error PEM in `numeric/sysId.py` (no extra dependency, a few
  hundred lines). The alternative is the `SIPPY` package.
- **Model files**: Python writes `mdl_<xx>_<name>.pkl` (DL: `.pt`/`.keras`) next to the MATLAB `.mat` files.
  Training no longer overwrites the shipped reference models unless `setup['save']`/a new flag allows it
  (F-M3).
- **Multiple models**: `training`/`testing` loop over all selected models and keep `mdl`/`pred` per model
  (F-M2). The 1:1 behaviour is not kept here, because the MATLAB behaviour is a bug.
- **Setup scripts**: each `setup/**/x.m` gets a `setup/**/x.py`. Like now, they are copied to the root to run them.
