#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfParamFit
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This module trains and evaluates the parametric PF network (pfNet.pfModal) on a board family (pfParam): one network
predicts the temperature rise of boards with other cooling, materials and chip positions without retraining.

    θ̂(x, t)/ΔT = Σ_c Σ_k φ_k(x, z_c, μ)·b_{c,k}(t),      db_{c,k}/dt = λ_k(z_c, μ)·(P_c(t)/P_ref − b_{c,k})

Loss (dimensionless, θ scaled with the 99 % quantile ΔT of the labelled rises):
1) Data (labelled boards, finite-volume temperatures): L_D = mean((θ̂ − θ)²)/ΔT² at nD random nodes plus all die nodes,
   over the whole profile.
2) Physics (labelled and physics-only boards; physics-only boards need only C, G, Bp, no simulation): residual of
   the finite-volume model multiplied by G⁻¹, at nR random nodes of every sample of a random loss profile,

       r = (G⁻¹·C·dθ̂/dt + θ̂ − G⁻¹·Bp·P)/ΔT = Σ_c (M_c·db_c + Φ_c·b_c) − Θ_c·p̂_c·P_ref/ΔT

   with Φ_c (N x K) the space functions of chip c at the nodes, M_c = G⁻¹·C·Φ_c, Θ_c = G⁻¹·Bp_c (steady state per W)
   and db/dt from the lag equation (exact, no finite differences). L_P = mean(r²). Only the rows of G⁻¹ at the
   residual nodes are needed: G⁻¹ is inverted densely on the GPU (Cholesky, float64) for a cache of nCache boards of
   the pool, nNew of them are replaced every epoch.
3) Steady state (all boards of the pool, needs only Θ = G⁻¹·Bp, one sparse LU per board): for a constant loss the
   lags settle at b = p̂, so the space functions of every chip must sum up to its steady-state field,

       r_S,c = Σ_k φ_k(x, z_c, μ) − Θ_c(x)·P_ref/ΔT,      L_S = mean(r_S²)

   It fixes the slow modes, which the short training profiles (and the residual over them) do not determine, and
   removes the drift over long profiles.
   L = α·L_D + β·L_P + γ·L_S; the initial condition θ(0) = 0 holds by construction (b_0 = 0).
Training: Adam, learning rate halved after 5 epochs without improvement of the validation RMSE, early stopping after
'pat' epochs; the network of the best validation RMSE is kept. An epoch is 'nStep' optimiser steps.
Sensors (cfg 'sens' > 0): the labels exist only at the first 'sens' sensor nodes of every board (B['sens'], rises at
every sample in B['ths']), e.g. 4 hot spots or 16 points of a measured module. The data loss and the validation RMSE
then use these nodes only, the physics and steady-state losses still act on the whole board.
Large boards (N > 'nDense'): instead of the dense G⁻¹ only the rows of nR random residual nodes per cached board are
computed (sparse LU, G symmetric: G⁻¹[rows, :] = (G⁻¹·E_rows)ᵀ), which keeps the cache at nR x N per board.
Boards with different numbers of nodes (e.g. a coarse grid on which the die footprints cover a different number of
cells depending on the position) are padded to the largest N: the padded nodes repeat node 0, have C = 0 and zero
columns of G⁻¹, and the residual nodes are drawn per board from its own nodes, so the padding never enters a loss.
Functions:  prepBoard(B, dev), ginv(B, dev), steady(B), predict(net, bt, P, dT, Ts, rows, tIdx), physRes(...),
            ssRes(...), fitParam(train, phys, val, cfg), evalBoard(net, B, dT, ...)

V1.0: 21.11.2026, Pascal Schirmer
- Initial version

V1.1: 21.11.2026, Pascal Schirmer
- Steady-state loss L_S and bound τ_cap of the time constants (drift over long profiles)

V1.2: 21.11.2026, Pascal Schirmer
- Sensor labels (cfg 'sens') and rows of G⁻¹ for large boards (cfg 'nDense')
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfNet import pfModal
from src.py.model.PF.pfParam import P_REF, xHat, chipRel, chipCtx, muHat, randProfile

# ==============================================================================
# External
# ==============================================================================
import copy
import time
import numpy as np
import torch

#######################################################################################################################
# Settings
#######################################################################################################################
CFG = {'K': 32, 'width': 128, 'depth': 4, 'nF': 4, 'rel': True, 'alpha': 1.0, 'beta': 1.0, 'lr': 1e-3,
       'nStep': 100, 'epoch': 200, 'pat': 20, 'nbD': 4, 'nD': 1024, 'nbP': 4, 'nR': 1024, 'tP': 30.0,
       'holds': (0.5, 8.0), 'Ts': 0.05, 'seed': 42, 'clip': 1.0, 'nCache': 32, 'nNew': 8, 'gamma': 1.0, 'nbS': 8,
       'tauCap': 80.0, 'sens': 0, 'nDense': 8000}


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _t(a, dev):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def _pad(x, n, dim, rep0=False):
    """Pads the tensor x along dim to the length n (zeros, or copies of entry 0 if rep0)."""
    m = x.shape[dim]
    if m >= n:
        return x
    if rep0:
        return torch.cat([x, x.narrow(dim, 0, 1).expand(*[n - m if d == dim % x.dim() else -1
                                                          for d in range(x.dim())])], dim=dim)
    shp = list(x.shape)
    shp[dim] = n - m
    return torch.cat([x, torch.zeros(shp, dtype=x.dtype, device=x.device)], dim=dim)


def prepBoard(B, dev):
    """
    Tensors of a board: space inputs xs (nc x N x 5: x̂, ŷ, ẑ, Δx̂_c, Δŷ_c), context ctx (nc x 11: chip descriptor,
    μ̂) and the die nodes die (all chips, concatenated).
    """
    xh = xHat(B['geo'])
    nc = len(B['chips'])
    xs = np.stack([np.column_stack([xh, chipRel(B, c)]) for c in range(0, nc)])
    ctx = np.stack([np.concatenate([chipCtx(B, c), muHat(B['mu'])]) for c in range(0, nc)])
    return {'xs': _t(xs, dev), 'ctx': _t(ctx, dev), 'N': len(B['C']), 'nc': nc,
            'die': torch.as_tensor(np.concatenate(B['die']), dtype=torch.long, device=dev)}


def ginv(B, dev):
    """Dense inverse of the conductance matrix G⁻¹ (N x N, float32, Cholesky in float64), C (N) and Θ = G⁻¹·Bp."""
    G = torch.as_tensor(B['G'].toarray(), dtype=torch.float64, device=dev)
    Gi = torch.cholesky_inverse(torch.linalg.cholesky(G))
    Th = Gi @ torch.as_tensor(B['Bp'], dtype=torch.float64, device=dev)
    return {'Gi': Gi.float(), 'C': _t(B['C'], dev), 'Th': Th.float()}


def ginvRows(B, n, rng, dev):
    """Rows of G⁻¹ at n random nodes (n x N, sparse LU, G symmetric), C (N), the rows of Θ = G⁻¹·Bp and the nodes."""
    from scipy.sparse.linalg import splu
    N = len(B['C'])
    rows = np.sort(rng.choice(N, min(n, N), replace=False))
    lu = splu(B['G'].tocsc(), permc_spec='MMD_AT_PLUS_A')
    E = np.zeros((N, len(rows)))
    E[rows, np.arange(len(rows))] = 1.0
    GiR = lu.solve(E).T
    Th = lu.solve(np.asarray(B['Bp'], dtype=float))[rows]
    return {'Gi': _t(GiR, dev), 'C': _t(B['C'], dev), 'Th': _t(Th, dev),
            'rows': torch.as_tensor(rows, dtype=torch.long, device=dev)}


def steady(B):
    """Steady-state rise per W Θ = G⁻¹·Bp (N x nc) (K/W), sparse LU."""
    from scipy.sparse.linalg import splu
    return splu(B['G'].tocsc()).solve(np.asarray(B['Bp'], dtype=float))


def predict(net, bt, P, dT, Ts, rows=None, tIdx=None, scaled=False):
    """
    Rise θ̂ (L x n) (K, or θ̂/ΔT if scaled) for the losses P (L x nc) (W, tensor or array) at the nodes rows (all if
    None) and the samples tIdx (all if None).
    """
    dev = bt['xs'].device
    p = (P if torch.is_tensor(P) else _t(P, dev)).T / P_REF                    # nc x L
    xs = bt['xs'] if rows is None else bt['xs'][:, rows]
    phi = net.basis(xs, bt['ctx'])                                              # nc x n x K
    b = net.coeff(p, net.rates(bt['ctx']), Ts)                                  # nc x L x K
    if tIdx is not None:
        b = b[:, tIdx]
    out = torch.einsum('clk,cnk->ln', b, phi)
    return out if scaled else out * dT


def physRes(net, xs, ctx, C, GiR, ThR, rows, P, dT, Ts):
    """
    Residual r (B x L x n) of the finite-volume model (G⁻¹ scaling) of a batch of boards at the residual nodes rows
    (n, or B x n per board): xs (B x nc x N x 5), ctx (B x nc x 11), C (B x N), rows of G⁻¹ (B x n x N) and of Θ = G⁻¹·Bp (B x n x nc),
    losses P (B x L x nc) (W).
    """
    p = P.transpose(1, 2) / P_REF                                               # B x nc x L
    p0 = torch.cat([torch.zeros_like(p[..., 0:1]), p[..., 1:]], dim=-1)
    phi = net.basis(xs, ctx)                                                    # B x nc x N x K
    lam = net.rates(ctx)                                                        # B x nc x K
    b = net.coeff(p, lam, Ts)                                                   # B x nc x L x K
    db = lam[..., None, :] * (p0[..., None] - b)
    nb, nc, N, K = phi.shape
    Cphi = (C[:, :, None, None] * phi.permute(0, 2, 1, 3)).reshape(nb, N, nc * K)         # B x N x nc·K
    Mr = torch.bmm(GiR, Cphi).reshape(nb, -1, nc, K).permute(0, 2, 1, 3)                  # B x nc x n x K
    phiR = phi[:, :, rows] if rows.dim() == 1 else torch.gather(phi, 2, rows[:, None, :, None].expand(-1, nc, -1, K))
    return torch.einsum('bclk,bcnk->bln', db, Mr) + torch.einsum('bclk,bcnk->bln', b, phiR) \
        - torch.einsum('bcl,bnc->bln', p0, ThR) * (P_REF / dT)


def ssRes(net, xs, ctx, ThR, dT):
    """
    Steady-state residual r_S (B x nc x n) = Σ_k φ_k − Θ_c·P_ref/ΔT for space inputs xs (B x nc x n x 5), context ctx
    (B x nc x 11) and the rows of Θ = G⁻¹·Bp at the same nodes (B x n x nc).
    """
    return net.basis(xs, ctx).sum(-1) - ThR.transpose(1, 2) * (P_REF / dT)


#######################################################################################################################
# Training
#######################################################################################################################
def fitParam(train, phys, val, cfg=None, dev=None, log=print):
    """
    Trains pfModal. train: labelled boards (with 'P', 'th', same number of chips), phys: boards for the physics loss
    only (may be empty), val: labelled validation boards. Returns {'net', 'dT', 'cfg', 'hist', 'time', 'best'}.
    """
    # ==============================================================================
    # Parameters
    # ==============================================================================
    cfg = dict(CFG, **(cfg or {}))
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    Ts, wD, wP, wS = cfg['Ts'], cfg['alpha'], cfg['beta'], cfg['gamma']
    tP = np.arange(0.0, cfg['tP'] + Ts * 1e-9, Ts)

    # ==============================================================================
    # Data (labelled samples stacked: boards x chips x nodes)
    # ==============================================================================
    nS = int(cfg['sens'])
    lab = (lambda B: [x[:, 0:nS] for x in B['ths']]) if nS else (lambda B: B['th'])        # labels per profile
    dT = float(np.percentile(np.concatenate([np.abs(th).ravel() for B in train for th in lab(B)]), 99))
    bTr = [prepBoard(B, dev) for B in train]
    Nn = [len(B['C']) for B in list(train) + list(phys)]
    N, N0 = max(Nn), min(Nn)                                                               # padded, smallest N
    XS = torch.stack([_pad(b['xs'], N, 1, True) for b in bTr])
    CTX = torch.stack([b['ctx'] for b in bTr])
    DIE = torch.stack([b['die'] for b in bTr]) if len(set(len(b['die']) for b in bTr)) == 1 else None
    SEN = torch.stack([torch.as_tensor(np.asarray(B['sens'][0:nS]), dtype=torch.long, device=dev)
                       for B in train]) if nS else None
    sB = torch.as_tensor([i for i, B in enumerate(train) for _ in B['P']], dtype=torch.long, device=dev)
    sP = torch.stack([_t(P, dev) for B in train for P in B['P']])                          # S x L x nc
    sTh = torch.stack([_pad(_t(th / dT, dev), N if not nS else nS, 1) for B in train for th in lab(B)])
    dense = N <= cfg['nDense']
    usePhys, useSS = wP > 0, wS > 0
    pool = (list(train) + list(phys)) if (usePhys or useSS) else []
    bPo = [prepBoard(B, dev) for B in pool]
    for b in bPo:
        b['xsP'] = _pad(b['xs'], N, 1, True)
    thS = [_t(steady(B), dev) for B in pool] if useSS else []
    nR = min(cfg['nR'], N0)
    bVa = [prepBoard(B, dev) for B in val]
    cache = {}
    log("INFO: %d labelled board(s) with %d profile(s), %d physics-only board(s), %d validation board(s), "
        "ΔT = %.2f K, labels at %s, N = %d (%s G⁻¹)" % (len(train), len(sB), len(phys) if pool else 0, len(val), dT,
                                                       '%d sensors' % nS if nS else 'all nodes', N,
                                                       'dense' if dense else 'rows of'))

    # ==============================================================================
    # Network
    # ==============================================================================
    net = pfModal(5, CTX.shape[-1], cfg['K'], cfg['width'], cfg['depth'], cfg['nF'], rel=cfg['rel'],
                  tauCap=cfg['tauCap']).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    log("INFO: pfModal with K = %d, trunk %d x %d, %d Fourier frequencies, relative coordinates %s, %d parameters" %
        (cfg['K'], cfg['depth'], cfg['width'], cfg['nF'], cfg['rel'], sum(q.numel() for q in net.parameters())))

    def valRmse():
        se, n = 0.0, 0
        with torch.no_grad():
            for B, bt in zip(val, bVa):
                rv = torch.as_tensor(np.asarray(B['sens'][0:nS]), dtype=torch.long, device=dev) if nS else None
                for P, th in zip(B['P'], lab(B)):
                    e = predict(net, bt, P, dT, Ts, rows=rv) - _t(th, dev)
                    se, n = se + float(torch.sum(e ** 2)), n + e.numel()
        return np.sqrt(se / max(n, 1))

    def inv(j):
        out = ginv(pool[j], dev) if dense else ginvRows(pool[j], nR, rng, dev)
        out['C'] = _pad(out['C'], N, 0)
        if not dense:
            out['Gi'] = _pad(out['Gi'], N, 1)
        return out

    def refresh():
        """Replaces nNew boards of the G⁻¹ cache (all at the first call)."""
        if len(pool) <= cfg['nCache']:
            for j in range(0, len(pool)):
                if j not in cache:
                    cache[j] = inv(j)
            return
        if cache:
            for j in rng.choice(list(cache.keys()), min(cfg['nNew'], len(cache)), replace=False):
                del cache[int(j)]
        free = [j for j in range(0, len(pool)) if j not in cache]
        for j in rng.choice(free, cfg['nCache'] - len(cache), replace=False):
            cache[int(j)] = inv(int(j))

    # ==============================================================================
    # Training
    # ==============================================================================
    best, bestNet, wait = np.inf, copy.deepcopy(net.state_dict()), 0
    hist = {'loss': [], 'lossData': [], 'lossPhys': [], 'lossSS': [], 'val': [], 'lr': [], 'time': []}
    t0 = time.time()
    for ep in range(0, cfg['epoch']):
        net.train()
        if usePhys:
            refresh()
            keys = list(cache.keys())
        acc = np.zeros(4)
        for _ in range(0, cfg['nStep']):
            opt.zero_grad()
            # data loss: nbD profiles, shared random nodes plus the die nodes of every board
            j = torch.as_tensor(rng.integers(0, len(sB), cfg['nbD']), dtype=torch.long, device=dev)
            bi = sB[j]
            if nS:                                                              # sensor nodes only
                rows = SEN[bi]
            else:
                rows = torch.cat([torch.randint(0, N0, (cfg['nD'],), device=dev)[None].expand(len(j), -1),
                                  DIE[bi]], dim=1)
            xs = torch.gather(XS[bi], 2, rows[:, None, :, None].expand(-1, XS.shape[1], -1, XS.shape[-1]))
            p = sP[j].transpose(1, 2) / P_REF
            phi = net.basis(xs, CTX[bi])
            b = net.coeff(p, net.rates(CTX[bi]), Ts)
            pred = torch.einsum('bclk,bcnk->bln', b, phi)
            tgt = sTh[j] if nS else torch.gather(sTh[j], 2, rows[:, None, :].expand(-1, sTh.shape[1], -1))
            lD = torch.mean((pred - tgt) ** 2)
            # physics loss: nbP boards of the cache with random loss profiles
            lP = torch.zeros((), device=dev)
            if usePhys:
                ks = [keys[int(i)] for i in rng.integers(0, len(keys), cfg['nbP'])]
                Pp = torch.stack([_t(randProfile(tP, rng, cfg['holds'], pool[k]['chips']), dev) for k in ks])
                if dense:                                                       # random rows per board
                    rr = torch.stack([torch.randperm(bPo[k]['N'], device=dev)[0:nR] for k in ks])
                    GiR = torch.stack([_pad(cache[k]['Gi'][r], N, 1) for k, r in zip(ks, rr)])
                    ThR = torch.stack([cache[k]['Th'][r] for k, r in zip(ks, rr)])
                else:                                                           # fixed rows per cached board
                    rr = torch.stack([cache[k]['rows'] for k in ks])
                    GiR, ThR = torch.stack([cache[k]['Gi'] for k in ks]), torch.stack([cache[k]['Th'] for k in ks])
                r = physRes(net, torch.stack([bPo[k]['xsP'] for k in ks]), torch.stack([bPo[k]['ctx'] for k in ks]),
                             torch.stack([cache[k]['C'] for k in ks]), GiR, ThR, rr, Pp, dT, Ts)
                lP = torch.mean(r ** 2)
            # steady-state loss: nbS boards of the whole pool, random nodes
            lS = torch.zeros((), device=dev)
            if useSS:
                ks = rng.integers(0, len(pool), cfg['nbS'])
                rr = [torch.randperm(bPo[k]['N'], device=dev)[0:nR] for k in ks]
                rS = ssRes(net, torch.stack([bPo[k]['xs'][:, r] for k, r in zip(ks, rr)]),
                           torch.stack([bPo[k]['ctx'] for k in ks]), torch.stack([thS[k][r] for k, r in zip(ks, rr)]), dT)
                lS = torch.mean(rS ** 2)
            loss = wD * lD + wP * lP + wS * lS
            loss.backward()
            if cfg['clip'] > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['clip'])
            opt.step()
            acc += np.array([float(loss.detach()), float(lD.detach()), float(lP.detach()), float(lS.detach())])
        acc /= cfg['nStep']
        net.eval()
        rv = valRmse()
        sch.step(rv)
        for k, v in zip(('loss', 'lossData', 'lossPhys', 'lossSS', 'val', 'lr', 'time'),
                        (acc[0], acc[1], acc[2], acc[3], rv, opt.param_groups[0]['lr'], time.time() - t0)):
            hist[k].append(float(v))
        log("INFO: Epoch %d/%d, lr %.2e, loss %.3e (data %.3e, physics %.3e, steady %.3e), validation RMSE %.3f K, "
            "%.0f s" % (ep + 1, cfg['epoch'], opt.param_groups[0]['lr'], acc[0], acc[1], acc[2], acc[3], rv,
                        time.time() - t0))
        if rv < best:
            best, bestNet, wait = rv, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg['pat']:
                log("INFO: Early stopping (patience %d epochs)" % cfg['pat'])
                break
    net.load_state_dict(bestNet)
    net.eval()
    return {'net': net, 'dT': dT, 'cfg': cfg, 'hist': hist, 'time': time.time() - t0, 'best': best}


#######################################################################################################################
# Evaluation
#######################################################################################################################
def evalBoard(net, B, dT, Ts=0.05, dev=None):
    """
    Errors of the network on a labelled board (all profiles): RMSE, MAE, MAX over all nodes and samples (K), RMSE of
    the junction temperature T_j,c = max over the die nodes of chip c (K, per chip), and the prediction time (s).
    """
    dev = dev or next(net.parameters()).device
    out = {'se': 0.0, 'ae': 0.0, 'mx': 0.0, 'n': 0, 'sej': np.zeros(len(B['chips'])), 'nj': 0, 'time': 0.0}
    with torch.no_grad():
        for P, th in zip(B['P'], B['th']):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            bt = prepBoard(B, dev)
            y = predict(net, bt, P, dT, Ts).cpu().numpy()
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            out['time'] += time.time() - t0
            e = y - th
            out['se'] += float(np.sum(e ** 2))
            out['ae'] += float(np.sum(np.abs(e)))
            out['mx'] = max(out['mx'], float(np.max(np.abs(e))))
            out['n'] += e.size
            for c, d in enumerate(B['die']):
                out['sej'][c] += float(np.sum((y[:, d].max(axis=1) - th[:, d].max(axis=1)) ** 2))
            out['nj'] += len(th)
    return {'rmse': np.sqrt(out['se'] / out['n']), 'mae': out['ae'] / out['n'], 'max': out['mx'],
            'rmseTj': np.sqrt(out['sej'] / out['nj']), 'time': out['time']}


#######################################################################################################################
# References
#######################################################################################################################
# [1] L. Lu et al., Learning nonlinear operators via DeepONet, Nature Machine Intelligence 3, 2021.
# [2] A. C. Antoulas, Approximation of Large-Scale Dynamical Systems, SIAM, 2005.
