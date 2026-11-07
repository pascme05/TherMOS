# Python Port of TherMOS

The Python version is in `src/py` and has the same structure as the MATLAB version in `src/mat`. Both use the same
`config/`, `data/`, `mdl/` and `results/` folders. Every MATLAB start script has a Python twin next to it
(`start.m` → `start.py`, `setup/**/x.m` → `setup/**/x.py`) with the same options.

## Legacy switch

The MATLAB code is unchanged. The Python port fixes the problems found during the translation (finding IDs refer
to `plan/plan.md`, section 6). Where a fix changes the published numbers, the MATLAB behaviour can be restored with

```python
setup['legacy'] = 1                                                                                                      # (0): corrected numerics (default), (1): behaviour of the MATLAB version
```

`legacy = 1` reproduces the MATLAB formulas. It does not reproduce MATLAB crashes, endless loops or results that
depend on MATLAB toolbox internals (n4sid/ssest weighting, fminunc iterations, trainnet initialisation).

## Known deviations from MATLAB

| ID | Area | Python behaviour (legacy = 0) | legacy = 1 |
|---|---|---|---|
| F-M1 | saving | `result_setup_*`/`result_para_*` hold setup/para (names were swapped) | – |
| F-M2 | training/testing | all selected models are trained, tested, printed and plotted | – |
| F-M3 | training | Python models are written as `mdl/mdl_<tag>_<name>.pkl` (`.pt` for DL/PINN), the MATLAB reference models are not touched | – |
| F-M5 | main | `featRoll`, `EWMA`, `EWMS` can be set in the setup (defaults 0, [5, 50], [5, 50]) | – |
| F-M8 | main | identification data (`idData`) are rebuilt after normalisation and padding | MATLAB: un-normalised `idData` |
| F-D1 | updLoss | θ(T) = (Rcon + Rohm·(1 + Rslope·(T − Tref)))/(Rcon + Rohm), θ(Tref) = 1 | MATLAB formula |
| F-D2 | updLoss | structured data `X2` scaled with `y2` of the same profile | – |
| F-D3, F-D4 | normData | same formula for `y` and `y2`; constant channels handled per channel | – |
| F-D5 | resmpData | interpolation option `samp` used (ZOH/linear/spline), last sample kept | linear, last sample dropped |
| F-D6 | offRemove | `init = 2`: offset of every profile from its own first sample | first profile for all |
| F-D7 | padData | pad values per channel | first element of channel 1 |
| F-D8 | dimData | `idData` built from the reduced data | – |
| F-D9 | DataLoader | ratio split without overlap between the sets | – |
| F-D12 | findNode | probe position relative to the minimum of the geometry | absolute |
| F-R1, F-R2 | RC | bounds on [R, τ] with log-spaced start values; fit of the simulated network response to the measured losses (any loss profile) | Zth curve fit, MATLAB bounds |
| F-S1, F-S2 | SS | loss iteration works (fixed point); `ss.init` respected | – |
| F-S3 | SS | own subspace (MOESP/N4SID) + output-error fit (`numeric/fitN4sid`, `fitPem`); horizon selected by simulation error; no noise model | – |
| F-F2 | SF | offset removed only once | MATLAB double offset |
| F-F1 | SF | **open**: ODE time constants differ from the fitted Foster pairs, the model is experimental | same |
| F-P1 | PO/PS | POD energy with σ² | σ, one mode more |
| F-P2, F-P3 | PO | C = ∫ρc·φφ, G = ∫k·∇φ∇φ, q = ∫Q·φ (no Jacobian heuristic, no boundary terms at interior Gauss nodes F-P4) | MATLAB weak form incl. Jgrid and boundary terms |
| F-P5 | PO | exact matrix-exponential solution (first-order hold) | same ODE |
| F-P6 | PO | protocols 4–6 solved in closed form; protocol 5 uses the minimum-norm correction of the projected matrices (truncated SVD, rcond = 1e-3) and falls back to them if the result is unstable (F-P15) | – |
| F-P7 | PO | `Mdl.pod.protocol` selects the protocol (default 5) | – |
| F-P9 | PS | projected losses passed as Nt × K | – |
| F-P14 | PO/PS | modal coordinates, derivatives and simulations per profile | concatenated |
| F-L1, F-L2, F-L3 | ML | target aligned with the inputs; ensemble optimisation works; first sample predicted | – |
| F-L5, F-L6, F-L7 | DL | seq2point target at the same window position in training and inference; windows per profile; loss iteration works; CNN seq2seq network | – |
| F-I1, F-I2 | PINN | physics constants from `para['Mdl']['pinn']` (defaults as MATLAB, β = 0); dT/dt with the data sampling time | Ts = 1 sec |
| F-G1 | calcFeat | EWMA/EWMS restart per profile; standard EW variance | MATLAB recursion |
| F-G2 | calcGrad | \|∇T\| and \|q\| = k·\|∇T\| with correct spacings | MATLAB sum of components |
| F-E1, F-E2 | performance | MATLAB "NRMSE" is reported as FIT (%), a real NRMSE is added; transient split with \|dT\| | signed dT |
| F-PL1 … F-PL7 | plotting | units on all axes, legends, common colour limits, symmetric error colour map, correct titles; thin domains plotted with free aspect ratio | – |
| – | plotting | figures are not opened but saved as `results/<name>/fig_<nn>_<title>.png` | – |

## New findings during the port

| ID | Location | Finding |
|---|---|---|
| F-M8 | `main.m` | `idData` (used by SS) is created in `loadData` before normalisation and padding, so SS is fitted on un-normalised data while it is tested on normalised data. |
| F-P14 | `poFit.m`, `psFit.m` | With several training profiles, `theta − g0` uses the first profile's initial state for all profiles and `gradient` differentiates across profile borders. |
| F-P15 | `optCthGthUB.m` | The unbounded C/G fit (protocol 5) is underdetermined for modes with almost no energy; the closed-form solution is unstable on `exp11`. The MATLAB result depends on where fminunc stops. For a single step response (`journal_Sec_IV_B_1sec`) the regressors [da/dt, a] only span the K exponentials plus a constant, so C and G cannot be separated at all; with a tight cut-off (rcond = 1e-8) the fit followed round-off (cond(C) ≈ 1e13, test MAE 2.8 K corrected vs. 0.3 K legacy). With rcond = 1e-3 both modes give 0.11 K. |
| F-P16 | `gen/FEM/Journal/*.m` | The only heat sink of the FEM model is the convection boundary (hc = 1000 W/m²K, edge 1), but the data are saved with `hc = hc * zeros(...)`. The boundary term ∫h·φφ can therefore not be projected and the pure Galerkin model (protocols 2/3) has no heat sink (test MAE 96 K); only the calibrated protocols 5/6 are usable on these data. |

## Verification

| Case | Python (legacy = 0) | Remark |
|---|---|---|
| `start.py` (SS, default) | T1: MAE 0.00 K, FIT 98.7 % | MATLAB: MAE 0.00 K, FIT ≈ 98.5 % |
| `start2D.py` (PO, exp11) | avg MAE 0.20 K | 20–426 °C field |
| `journal_Sec_IV_B_1sec.py` (PO, 5 W → 20 W) | avg MAE 0.11 K | legacy = 1: 0.11 K (before the fix: 2.8 K / 0.31 K) |
| `start3D.py` (PS3D, FR4 5 W → 10 W) | avg MAE 0.11 K | |
| `tests/test_numeric.py` | 7 passed | exact references (analytic ODE, recursion, known system) |

Run the unit tests with `python -m pytest tests`.
