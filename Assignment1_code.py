


#!/usr/bin/env python3
"""
MBGD MLP (tanh hidden, linear output)
- Student-ID personalization (seed, alpha_u, c_u, sigma)
- Constant LR, mini-batch
- Per-epoch TRAIN & VAL logging (MSE/RMSE), gradient check


Python 3.x + NumPy + Matplotlib. 
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# ============================ Utilities ============================
@dataclass
class IDParams:
    alpha_u: float
    c_u: float
    sigma: float
    seed: int


def lcg_step(s: np.uint64) -> Tuple[np.uint64, float]:
    s = np.uint64((np.uint64(1664525) * s + np.uint64(1013904223)) % np.uint64(2**32))
    u = float(s) / float(2**32)
    return s, u


def params_from_student_id(student_id: str) -> IDParams:
    """Hash the student ID into reproducible parameters."""
    # Extract digits as Python ints (arbitrary precision)
    d = [ord(ch) - 48 for ch in student_id if '0' <= ch <= '9']
    if not d:
        raise ValueError("ID must contain digits.")

    # Compute s0 in Python-int space with explicit mod 2^64 to mirror uint64 wrap
    MOD64 = 1 << 64
    s0 = 0
    for i, di in enumerate(d):
        s0 = (s0 + di * pow(131, i, MOD64)) % MOD64

    # Seed stream for LCG (uint32 domain), three steps -> u1,u2,u3 in [0,1)
    s = np.uint64(s0 % (1 << 32))
    s, u1 = lcg_step(s)
    s, u2 = lcg_step(s)
    s, u3 = lcg_step(s)

    alpha_u = 0.8 + 0.8 * u1
    c_u     = -0.3 + 0.6 * u2
    sigma   = 0.02 + 0.06 * u3

    # Positive 31-bit-ish seed
    seed = int((s0 % (2**31 - 1)) + 1)
    return IDParams(alpha_u=alpha_u, c_u=c_u, sigma=sigma, seed=seed)


# ====================== Radial Fourier Features =====================
@dataclass
class FFMeta:
    kind: str
    omegas: np.ndarray
    P: IDParams

# Radial Fourier features
def fourier_features_radial(X2d: np.ndarray, K: int, P: IDParams) -> Tuple[np.ndarray, FFMeta]:
    """X2d standardized; K = # radial harmonics; P=Personalized parameters"""
   # Fill in the blank
   # Convert (x, y) to polar system (r, theta)
    r = np.sqrt(X2d[:,0]**2 + X2d[:,1]**2)
    theta = np.arctan2(X2d[:,1], X2d[:,0])
    omega_star = 12.0 * P.alpha_u
    omegas = np.linspace(0.25 * omega_star, 2.5 * omega_star, K) if K > 0 else np.array([])

    Phi_list = []
    Phi_list.append(np.ones_like(r))          # bias
    Phi_list.append(r)                        # low-freq radius
    env1 = 1.0 / (P.alpha_u * (r**2) + (2.0 + P.c_u))
    env2 = env1**2
    Phi_list.extend([env1, env2])

    if K > 0:
        Blk = np.zeros((r.shape[0], 2 * K))
        col = 0
        for w in omegas:
            Blk[:, col] = np.sin(w * r); col += 1
            Blk[:, col] = np.cos(w * r); col += 1
        Phi_list.append(Blk)

    Phi_list.append(np.cos(theta))
    Phi_list.append(np.sin(theta))

    Phi = np.column_stack([c if c.ndim == 1 else c for c in Phi_list])
    meta = FFMeta(kind='radial', omegas=omegas, P=P)
    return Phi, meta


def fourier_features_radial_apply(X2d: np.ndarray, meta: FFMeta) -> np.ndarray: 
   # Fill in the blank
   # Convert (x, y) to polar system (r, theta) 
    r = np.sqrt(X2d[:,0]**2 + X2d[:,1]**2)
    theta = np.arctan2(X2d[:,1], X2d[:,0])
    P = meta.P
    omegas = meta.omegas

    Phi_list = [np.ones_like(r), r]
    env1 = 1.0 / (P.alpha_u * (r**2) + (2.0 + P.c_u))
    env2 = env1**2
    Phi_list.extend([env1, env2])

    K = len(omegas)
    if K > 0:
        Blk = np.zeros((r.shape[0], 2 * K))
        col = 0
        for w in omegas:
            Blk[:, col] = np.sin(w * r); col += 1
            Blk[:, col] = np.cos(w * r); col += 1
        Phi_list.append(Blk)

    Phi_list.append(np.cos(theta))
    Phi_list.append(np.sin(theta))

    Phi = np.column_stack([c if c.ndim == 1 else c for c in Phi_list])
    return Phi


# ============================== Model ===============================

def train_mlp_mbgd(XtrnFF: np.ndarray,
                   ytrn: np.ndarray,
                   H: int,
                   max_epochs: int,
                   eta: float,
                   B: int,
                   XvanFF: np.ndarray,
                   yvanFF: np.ndarray,
                   order_key: np.ndarray,
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    N, D = XtrnFF.shape
    rng = np.random.default_rng(seed)  # reproducible
    W1 = 0.5 * rng.standard_normal((D, H))
    b1 = np.zeros((1, H))
    W2 = 0.5 * rng.standard_normal((H, 1))
    b2 = 0.0

    logT = np.zeros((max_epochs, 7), dtype=float)

    # fixed deterministic sweep based on order_key each epoch
    #p = np.argsort(order_key, kind='mergesort')  # stable

    for e in range(max_epochs):
        
        p = rng.permutation(N) #for batch shuffle

        
        lr = eta
        gW1_last = 0.0
        gW2_last = 0.0


        for s in range(0, N, B):
            t = min(s + B, N)
            idx = p[s:t]
            x = XtrnFF[idx, :]
            tvec = ytrn[idx]
            BT = x.shape[0]

            # forward
            # Fill in the blank
    
            Z1 =  x @ W1 + b1
            A1 = np.tanh(Z1)
            yhat =  A1 @ W2 + b2
            diff = yhat.reshape(-1, 1) - tvec.reshape(-1, 1)

            # backward
            # Fill in the blank
            dY = (2.0 / BT) * diff
            dW2 =  A1.T @ dY
            db2 = float(dY.sum()) # scalar to avoid NumPy deprecation
            dA1 = dY @ W2.T  # [BT,H]
            dZ1 = dA1 * (1.0-A1**2)
            dW1 =  x.T @ dZ1   # [D,H]
            db1 = dZ1.sum(axis=0, keepdims=True)

            # update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

            gW1_last = float(np.linalg.norm(dW1))
            gW2_last = float(np.linalg.norm(dW2))

        # epoch metrics (standardized scale)
        yhat_tr = np.tanh(XtrnFF @ W1 + b1) @ W2 + b2
        diff_tr = yhat_tr.reshape(-1) - ytrn.reshape(-1)
        mse_tr =  float(np.mean(diff_tr**2))       
        rmse_tr = math.sqrt(mse_tr)

        yhat_va = np.tanh(XvanFF @ W1 + b1) @ W2 + b2
        diff_va = yhat_va.reshape(-1) - yvanFF.reshape(-1)
        mse_va = float(np.mean(diff_va**2))
        rmse_va = math.sqrt(mse_va)
        
      
        logT[e, :] = [e + 1, mse_tr, rmse_tr, mse_va, rmse_va, gW1_last, gW2_last]

    return W1, b1, W2, b2, logT


# =========================== Gradient Check ==========================

def loss_fd_ff(X: np.ndarray, y: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: float) -> float:
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    yhat = A1 @ W2 + b2
    return float(np.mean((yhat.reshape(-1) - y.reshape(-1))**2))


def grad_check_small_batch(H: int, XtrnFF: np.ndarray, ytrn: np.ndarray, seed: int) -> None:
    # Select 4 samples randonly
    rng = np.random.default_rng(seed + 1337)
    sel = rng.choice(XtrnFF.shape[0], size=min(4, XtrnFF.shape[0]), replace=False)
    X = XtrnFF[sel, :]
    y = ytrn[sel]
    D = X.shape[1]

    W1 = 0.1 * rng.standard_normal((D, H))
    b1 = np.zeros((1, H))
    W2 = 0.1 * rng.standard_normal((H, 1))
    b2 = 0.0

    #===============================
    #Fill in the blank, the code is the same as
    #in train_mlp_mbgd()

    # Forward
    Z1 =  X @ W1 + b1
    A1 = np.tanh(Z1)
    yhat =  A1 @ W2 + b2
    B = X.shape[0]

    # Backward
    dY = (2.0 / B) * (yhat - y.reshape(-1, 1))
    dW2 = A1.T @ dY
    dA1 = dY @ W2.T
    dZ1 = dA1 * (1.0-A1**2)
    dW1 = X.T @ dZ1
    db2 = float(dY.sum())
    db1 = dZ1.sum(axis=0, keepdims=True)
    #===============================
    
    h = 1e-5

    def rel(an, fd):
        return abs(an - fd) / max(1.0, abs(an), abs(fd))

    def print_table(title, rows):
        print(f"\n{title}")
        print("{:<12s} {:>14s} {:>14s} {:>14s}".format("Param","Analytic","FD","RelErr"))
        max_r = 0.0
        for name, an, fdv, rr in rows:
            max_r = max(max_r, rr)
            print(f"{name:<12s} {an:14.6e} {fdv:14.6e} {rr:14.6e}")
        print(f"Max RelErr ({title}) = {max_r:.3e}")
        return max_r

    overall_max = 0.0

    
    rows = []
    K = min(3, dW1.size)
    fi = np.unravel_index(np.arange(K), dW1.shape)
    for k in range(K):
        i, j = fi[0][k], fi[1][k]
        M = np.zeros_like(W1); M[i, j] = 1.0
        f1 = loss_fd_ff(X, y, W1 + h*M, b1, W2, b2)
        f2 = loss_fd_ff(X, y, W1 - h*M, b1, W2, b2)
        fdv = (f1 - f2) / (2.0*h)
        an  = dW1[i, j]
        rows.append((f"W1[{i},{j}]", an, fdv, rel(an, fdv)))
    overall_max = max(overall_max, print_table("W1 (first 3 entries)", rows))

    
    rows = []
    K_w2 = min(3, dW2.size)
    for k in range(K_w2):
        M = np.zeros_like(W2); M[k, 0] = 1.0
        f1 = loss_fd_ff(X, y, W1, b1, W2 + h*M, b2)
        f2 = loss_fd_ff(X, y, W1, b1, W2 - h*M, b2)
        fdv = (f1 - f2) / (2.0*h)
        an  = dW2[k, 0]
        rows.append((f"W2[{k},0]", an, fdv, rel(an, fdv)))
    overall_max = max(overall_max, print_table("W2 (first 3 entries)", rows))

    
    rows = []
    K_b1 = min(3, db1.size)
    for k in range(K_b1):
        M = np.zeros_like(b1); M[0, k] = 1.0
        f1 = loss_fd_ff(X, y, W1, b1 + h*M, W2, b2)
        f2 = loss_fd_ff(X, y, W1, b1 - h*M, W2, b2)
        fdv = (f1 - f2) / (2.0*h)
        an  = db1[0, k]
        rows.append((f"b1[{k}]", an, fdv, rel(an, fdv)))
    overall_max = max(overall_max, print_table("b1 (first 3 entries)", rows))

    
    rows = []
    f1 = loss_fd_ff(X, y, W1, b1, W2, b2 + h)
    f2 = loss_fd_ff(X, y, W1, b1, W2, b2 - h)
    fdv = (f1 - f2) / (2.0*h)
    an  = db2
    rows.append(("b2", an, fdv, rel(an, fdv)))
    overall_max = max(overall_max, print_table("b2 (scalar)", rows))

    print(f"\nOVERALL Max RelErr = {overall_max:.3e} (threshold < 1e-4)")
 


# =============================== Main ===============================

def main():
    # ---------- Hyperparams ----------
    H_list = [5, 10, 50, 100, 150]    # List of hidden neurons number to run
    max_epochs = 100
    eta = 5e-3
    batch_size = 175000
    H_star = 150
    cases = [
        {"B":15000, "eta":1e-3},
        {"B": 30000, "eta": 3e-3},
        {"B":50000, "eta": 1e-2}
    ]

    Bmax = batch_size
    eta_baseline = eta

    # Radial Fourier features ((for Qs3(k)) only)
    K_radial = 8

    # >>>>>>>>>> SET YOUR STUDENT ID HERE <<<<<<<<<<
    student_id =  '2021142103'

    # ---------- ID-derived params ----------
    P = params_from_student_id(student_id)
    print(f"ID={student_id} | seed={P.seed} | alpha={P.alpha_u:.6f} | c={P.c_u:.6f} | sigma={P.sigma:.6f} | B={batch_size}")

    # Global seeding for reproducibility
    np.random.seed(P.seed)

    # ------------------ Data (grid) ------------------
    nPerAxis = 500; xmin=-2; xmax=2; ymin=-2; ymax=2
    xs = np.linspace(xmin, xmax, nPerAxis)
    ys = np.linspace(ymin, ymax, nPerAxis)
    XX, YY = np.meshgrid(xs, ys)
    X = np.column_stack([XX.ravel(), YY.ravel()])  # [N x 2]

  # Target Function
    def f(x, y):
        r = np.sqrt(x**2 + y**2)
        num = -(1.0 + np.cos(12.0 * P.alpha_u * r))
        den = (P.alpha_u * (x**2 + y**2) + (2.0 + P.c_u))
        return num / den

    Z_clean = f(X[:, 0], X[:, 1])

    # Repro checksum surrogate
    raw = np.column_stack([X, Z_clean]).astype(np.float64, copy=False)
    raw_bytes = raw.ravel(order='C').view(np.uint8)
    checksum = float(raw_bytes.astype(np.float64).sum())
    print(f"Checksum = {checksum:.10f}")

    # Split 70/30
    N = X.shape[0]
    idx = np.random.permutation(N)
    Ntr = max(200, int(round(0.7 * N)))
    tr = idx[:Ntr]
    va = idx[Ntr:]

    Xtr = X[tr, :]; Xva = X[va, :]
    ytr_clean = Z_clean[tr]
    yva_clean = Z_clean[va]

    # Add noise to train and val set
    sigma_train = 0.5 * P.sigma
    sigma_val = 2.0 * P.sigma
    ytr = ytr_clean + sigma_train * np.random.randn(Ntr)
    yva = yva_clean + sigma_val * np.random.randn(len(va))

    #Add additional noise and outliers to validation set
    outlier_frac_val = 0.01
    kv = max(1, int(round(outlier_frac_val * len(va))))
    oi_val = np.random.permutation(len(va))[:kv]
    yva[oi_val] = yva[oi_val] + 3.0 * np.std(yva) * np.sign(np.random.randn(kv))

    # Standardize base coordinates
    muX = Xtr.mean(axis=0)
    sX = Xtr.std(axis=0) + 1e-12
    muy = ytr.mean()
    sy = ytr.std() + 1e-12

    Xtrn = (Xtr - muX) / sX
    Xvan = (Xva - muX) / sX

    ytrn = (ytr - muy) / sy
    yvan = (yva - muy) / sy

    # Radial Fourier features (for Qs3(k) only)
    Xtrn, FF_meta = fourier_features_radial(Xtrn, K_radial, P)
    Xvan = fourier_features_radial_apply(Xvan, FF_meta)

    # ------------------ Sweep H (MBGD, constant LR) ------------------
    best: Dict[str, Any] = dict(H=0, W1=None, b1=None, W2=None, b2=None,
                                rmse_tr=float('inf'), rmse_va=float('inf'))
    results = np.zeros((len(H_list), 3), dtype=float)

    for i, H in enumerate(H_list):
        order_key = Xtrn[:, 0]
        W1, b1, W2, b2, logT = train_mlp_mbgd(
            Xtrn, ytrn, H, max_epochs, eta, batch_size,
            Xvan, yvan, order_key, seed=P.seed + 1000 + i
        )

        # eval on original scale
        yhat_tr = (np.tanh(Xtrn @ W1 + b1) @ W2 + b2).reshape(-1) * sy + muy
        yhat_va = (np.tanh(Xvan @ W1 + b1) @ W2 + b2).reshape(-1) * sy + muy
        rmse_tr = math.sqrt(np.mean((yhat_tr - ytr) ** 2))
        rmse_va = math.sqrt(np.mean((yhat_va - yva) ** 2))
        results[i, :] = [H, rmse_tr, rmse_va]

        if rmse_va < best['rmse_va']:
            best = dict(H=H, W1=W1, b1=b1, W2=W2, b2=b2,
                        rmse_tr=rmse_tr, rmse_va=rmse_va)
        # Save log 
        np.savetxt(f"logs_H{H:03d}_B{batch_size:03d}_mbgd.csv", logT, delimiter=",",
                   header="epoch,mse_tr,rmse_tr,mse_va,rmse_va,gW1_last,gW2_last", comments="")

    print("#Hidden  RMSE_train  RMSE_val")
    for row in results:
        print(f"{int(row[0])}   {row[1]:.6f}    {row[2]:.6f}")

    # ------------------ Plots: Ground truth vs Best Approx ------------------
    Xall = np.column_stack([XX.ravel(), YY.ravel()])
    Xalln = (Xall - muX) / sX
    
    #Only for Qs3(k)
    Xalln = fourier_features_radial_apply(Xalln, FF_meta)
    
    #  Target Function
    Zgt = f(Xall[:, 0], Xall[:, 1])

    #   Estimated Function (Inverse Normalization) 
    Zph = (np.tanh(Xalln @ best['W1'] + best['b1']) @ best['W2'] + best['b2']).reshape(-1)
    Zhat = Zph * sy + muy

    Zgt_grid = Zgt.reshape(XX.shape)
    Zhat_grid = Zhat.reshape(XX.shape)
    stride = max(1, nPerAxis // 200)  # ~<=200x200 points

    FrD = np.linalg.norm(Zgt_grid - Zhat_grid, 'fro')
    print(f"Frobenius distance (FrD) = {FrD:.6f}")
    
   # Plot for target Function
   # Fill in the blank
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(XX,YY,Zgt_grid, cmap='viridis',rstride=stride, cstride=stride)
    
    try:
        ax1.set_box_aspect((1, 1, 0.6))
    except Exception:
        pass
 
    #Plot for Estimated Function (MLP)
    # Fill in the blank
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection = '3d')
    ax2.plot_surface(XX,YY, Zhat_grid, cmap = 'viridis', rstride = stride, cstride = stride)
    try:
        ax2.set_box_aspect((1, 1, 0.6))
    except Exception:
        pass

    plt.tight_layout()

  
    # Plot for MSE curve
    # Fill in the blank
    plt.figure()
    plt.plot(logT[:,0], logT[:,1], label = "MSE_train")
    plt.plot(logT[:,0], logT[:,3], label='MSE_val')
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE")

    # Plot for RMSE curve
    # Fill in the blank
    plt.figure()
    plt.plot(logT[:,0], logT[:,2], label = "RMSE_train")
    plt.plot(logT[:,0], logT[:,4], label = "RMSE_val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("RMSE")

    # Gradient check
    grad_check_small_batch(best['H'], Xtrn, ytrn, seed=P.seed)

    plt.show(block=False)

# Baseline 
    order_key = Xtrn[:, 0]

    i_star = H_list.index(H_star)
    seed_c = P.seed + 1000 + i_star
    W1b, b1b, W2b, b2b, logT_base = train_mlp_mbgd(
        Xtrn, ytrn, H_star, max_epochs, eta_baseline, Bmax,
        Xvan, yvan, order_key, seed=seed_c
    )

    yhat_tr = (np.tanh(Xtrn @ W1b + b1b) @ W2b + b2b).reshape(-1) * sy + muy
    yhat_va = (np.tanh(Xvan @ W1b + b1b) @ W2b + b2b).reshape(-1) * sy + muy
    rmse_tr_base = float(np.sqrt(np.mean((yhat_tr - ytr)**2)))
    rmse_va_base = float(np.sqrt(np.mean((yhat_va - yva)**2)))

    print("\n# (g) Baseline (Bmax, eta=5e-3)")
    print(f"B={Bmax:6d} | eta={eta_baseline:.1e} | RMSE_train={rmse_tr_base:.6f} | RMSE_val={rmse_va_base:.6f}")

    #Three cases
    results = []
    best = {"idx": -1, "rmse_va": float('inf')}

    for i, cfg in enumerate(cases, 1):
        B = cfg["B"]; eta = cfg["eta"]
        assert B < Bmax, "B must be smaller than Bmax for (g)."

        W1, b1, W2, b2, logT = train_mlp_mbgd(
            Xtrn, ytrn, H_star, max_epochs, eta, B,
            Xvan, yvan, order_key, seed=P.seed + 3000 + i
        )
        yhat_tr = (np.tanh(Xtrn @ W1 + b1) @ W2 + b2).reshape(-1) * sy + muy
        yhat_va = (np.tanh(Xvan @ W1 + b1) @ W2 + b2).reshape(-1) * sy + muy
        rmse_tr = float(np.sqrt(np.mean((yhat_tr - ytr)**2)))
        rmse_va = float(np.sqrt(np.mean((yhat_va - yva)**2)))

        results.append({"B": B, "eta": eta, "rmse_tr": rmse_tr, "rmse_va": rmse_va, "logT": logT})
        print(f"Case {i}: B={B:6d} | eta={eta:.1e} | RMSE_train={rmse_tr:.6f} | RMSE_val={rmse_va:.6f}")

        if rmse_va < best["rmse_va"]:
            best = {"idx": i-1, "rmse_va": rmse_va}


    
    best_kind = "baseline"
    best_rmse = rmse_va_base
    best_logT = logT_base
    best_B    = Bmax
    best_eta  = eta_baseline

    for i, r in enumerate(results, 1):
        if r["rmse_va"] < best_rmse:
            best_rmse = r["rmse_va"]
            best_logT = r["logT"]
            best_kind = f"case{i}"
            best_B    = r["B"]
            best_eta  = r["eta"]

    print(f"\nOverall best (g): {best_kind} | B={best_B} | eta={best_eta:.1e} | RMSE_val={best_rmse:.6f}")

# code for 3-e
    best_logT = results[best["idx"]]["logT"]
    plt.figure(figsize=(6.5,4.2))
    plt.plot(best_logT[:,0], best_logT[:,2], label="RMSE_train (std)")
    plt.plot(best_logT[:,0], best_logT[:,4], label="RMSE_val (std)")
    plt.xlabel("epoch"); plt.ylabel("RMSE")
    plt.title(f"Best config: B={results[best['idx']]['B']}, eta={results[best['idx']]['eta']:.1e}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show(block=False)

# code for 3-h

    plt.figure(figsize=(6.5,4.2))
    plt.plot(best_logT[:,0], best_logT[:,2], label="RMSE_train (std)")
    plt.plot(best_logT[:,0], best_logT[:,4], label="RMSE_val (std)")
    plt.xlabel("epoch"); plt.ylabel("RMSE")
    plt.title(f"Best config: B={best_B}, eta={best_eta:.1e}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()

    
    gW1 = best_logT[:, 5]  
    gW2 = best_logT[:, 6]  
    epochs = best_logT[:, 0]

    
    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(epochs, np.maximum(gW1, 1e-16), label=r"$\|g_{W1}\|_F$")
    plt.axhline(1e-6, ls="--", color='gray', alpha=0.4)
    plt.axhline(1e+2, ls="--", color='gray', alpha=0.4)
    plt.xlabel("epoch"); plt.ylabel("‖gW1‖ Frobenius norm (log)")
    plt.title(f"Gradient norm: W1 (B={best_B}, η={best_eta:.1e})")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout()

    
    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(epochs, np.maximum(gW2, 1e-16), label=r"$\|g_{W2}\|_F$")
    plt.axhline(1e-6, ls="--", color='gray', alpha=0.4)
    plt.axhline(1e+2, ls="--", color='gray', alpha=0.4)
    plt.xlabel("epoch"); plt.ylabel("‖gW2‖ Frobenius norm (log)")
    plt.title(f"Gradient norm: W2 (B={best_B}, η={best_eta:.1e})")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout()

    
    TAU_VANISH  = 2e-7   
    TAU_EXPLODE = 1e+2  

    vanish_epochs  = epochs[(gW1 < TAU_VANISH) | (gW2 < TAU_VANISH)].astype(int)
    explode_epochs = epochs[(gW1 > TAU_EXPLODE) | (gW2 > TAU_EXPLODE)].astype(int)

    print("\n(h) Thresholds:")
    print(f"  Vanishing if ||g||_F < {TAU_VANISH:g}")
    print(f"  Exploding  if ||g||_F > {TAU_EXPLODE:g}")
    print("\n(h) Detected epochs:")
    print("  Vanishing epochs :", vanish_epochs.tolist() if vanish_epochs.size else "None")
    print("  Exploding epochs :", explode_epochs.tolist() if explode_epochs.size else "None")

    plt.show()

if __name__ == "__main__":
    main()
    print("end")

