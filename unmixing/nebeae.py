#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEBEAE - Fast & Stable Vectorized Implementation with Strict Constraints (ANC & ASC)
Based on: Campos-Delgado et al. (2022)
"""
import numpy as np
from scipy.linalg import pinv, svd, sqrtm
from unmixing.endmember_extraction import vca, SVMAX, NFINDR


def initPo(Yo, Ym, initcond, N):
    L = Yo.shape[0]
    Po = np.zeros((L, N))
    eps = 1e-12

    if initcond in [1, 2]:
        if initcond == 1:
            index = 0
            pmax = np.mean(Yo, axis=1)
            Yt = Yo.copy()
            Po[:, index] = pmax
        elif initcond == 2:
            index = 0
            Y1m = np.sum(np.abs(Yo), axis=0)
            Imax = np.argmax(Y1m)
            Imin = np.argmin(Y1m)
            pmax = Yo[:, Imax]
            pmin = Yo[:, Imin]
            indices = np.arange(Yo.shape[1])
            Yt = Yo[:, np.setdiff1d(indices, [Imax, Imin])].copy()
            Po[:, index] = pmax
            index += 1
            Po[:, index] = pmin
        while index < N - 1:
            ymax = np.zeros(index + 1)
            Imax = np.zeros(index + 1, dtype=int)
            for i in range(index + 1):
                den = np.sqrt(np.sum(Yt**2, axis=0)) * np.sqrt(np.sum(Po[:, i]**2))
                e1m = np.sum(Yt * Po[:, i:i+1], axis=0) / np.maximum(den, eps)
                ymax[i], Imax[i] = min((abs(val), idx) for idx, val in enumerate(e1m))
            Immax = np.argmin(ymax)
            IImax = Imax[Immax]
            pmax = Yt[:, IImax]
            index += 1
            Po[:, index] = pmax
            indices = np.arange(Yt.shape[1])
            Yt = Yt[:, np.setdiff1d(indices, [IImax])]
    elif initcond == 3:
        _, _, VV = svd(Ym.T, full_matrices=False)
        W = VV[:N, :].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T
    elif initcond == 4:
        Yom = np.mean(Ym, axis=1, keepdims=True)
        Yon = Ym - Yom
        _, S, VV = svd(Yon.T, full_matrices=False)
        Yo_w = pinv(sqrtm(np.diag(S))) @ VV @ Ym.T
        V, _, _ = svd((Yo_w**2).sum(axis=1, keepdims=True) * Yo_w @ Yo_w.T)
        W = VV.T @ sqrtm(np.diag(S)) @ V[:N, :].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T
    elif initcond == 5:
        Po = NFINDR(Ym, N)
    elif initcond == 6:
        Po = vca(Ym, N)
    elif initcond == 7:
        Po = SVMAX(Ym, N)
    else:
        Po = vca(Ym, N)
    return Po


def probanonlinear_fast(Z, Y, P, A):
    """Cálculo vectorizado del grado de interacción no lineal D_k."""
    E = P @ A  # (L, K)
    T1 = E - Y
    T2 = E - (E * Z)

    num = np.sum(T1 * T2, axis=0)
    den = np.sum(T2 * T2, axis=0)

    den_safe = np.where(np.abs(den) <= 1e-12, 1e-12, den)
    D = np.clip(num / den_safe, 0.0, 1.0)
    return np.nan_to_num(D, nan=0.0)


def abundance_vectorized(Z, Y, P, D, Lambda):
    """
    Estimación de Abundancias vectorizada con cumplimiento estricto de:
    1. No negatividad (A >= 0)
    2. Suma a uno (sum(A, axis=0) == 1)
    """
    L, K = Y.shape
    N = P.shape[1]
    eps = 1e-12

    P_clean = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    Go = P_clean.T @ P_clean
    Go = np.nan_to_num(Go, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals = np.linalg.eigvalsh(Go)
        lmin = np.min(eigvals)
    except np.linalg.LinAlgError:
        lmin = 0.0

    lmin_adj = abs(lmin) if lmin < 0 else 0.0
    G = Go + np.eye(N) * (lmin_adj * Lambda + 1e-8)

    try:
        Gi = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Gi = np.linalg.pinv(G)

    B = P_clean.T @ Y
    A_unconstrained = Gi @ B
    A_unconstrained = np.nan_to_num(A_unconstrained, nan=0.0, posinf=0.0, neginf=0.0)

    ones_N = np.ones((1, N))
    sum_A = ones_N @ A_unconstrained
    A = A_unconstrained / np.maximum(sum_A, eps)

    for _ in range(5):
        A[A < 0] = 0.0
        sums = np.sum(A, axis=0, keepdims=True)
        zero_mask = (sums <= eps)
        sums[zero_mask] = 1.0
        A /= sums
        if np.any(zero_mask):
            A[:, zero_mask.ravel()] = 1.0 / N

    return np.nan_to_num(A, nan=1.0 / N)


def endmember_update_safe(Y_raw, Y_norm, Po_initial, P_current, A, D, rho):
    """
    Actualización de endmembers en la escala normalizada consistente con P y A.
    """
    N, K = A.shape
    L = Y_norm.shape[0]
    eps = 1e-12

    # Residual en espacio normalizado Y_norm
    E = P_current @ A
    Y_hat = (1.0 - D)[None, :] * E + D[None, :] * (E * Y_norm)
    residual = Y_norm - Y_hat

    # Gradiente en la dirección exacta de mínimos cuadrados
    weight = (1.0 - D)[None, :] + D[None, :] * Y_norm
    GradP = (residual * weight) @ A.T / K - rho * (P_current - Po_initial)
    GradP = np.nan_to_num(GradP, nan=0.0, posinf=0.0, neginf=0.0)

    # Paso de gradiente con cota superior estable
    grad_norm = np.linalg.norm(GradP, 'fro')
    alpha = 0.05 / (grad_norm + eps) if grad_norm > eps else 0.05

    P_new = P_current + alpha * GradP
    P_new = np.nan_to_num(P_new, nan=0.0, posinf=0.0, neginf=0.0)
    P_new = np.maximum(P_new, 0.0)

    col_sums = np.sum(P_new, axis=0, keepdims=True)
    invalid_cols = (col_sums <= eps) | (~np.isfinite(col_sums))
    if np.any(invalid_cols):
        P_new[:, invalid_cols.ravel()] = Po_initial[:, invalid_cols.ravel()]
        col_sums = np.sum(P_new, axis=0, keepdims=True)

    col_sums_safe = np.where(col_sums <= eps, 1.0, col_sums)
    P = P_new / col_sums_safe
    return np.nan_to_num(P, nan=1.0 / L)


def nebeae(Yo=[], N=2, parameters=[], Po=[], oae=0):
    initcond = 1
    rho = 0.1
    Lambda = 0
    epsilon = 1e-3
    maxiter = 20
    downsampling = 0.5
    eps = 1e-12

    if parameters is not None and len(parameters) == 8:
        initcond = int(parameters[0])
        rho = parameters[1]
        Lambda = parameters[2]
        epsilon = parameters[3]
        maxiter = parameters[4]
        downsampling = parameters[5]

    if N < 2:
        N = 2
    N = int(N)

    L, K = Yo.shape

    Kdown = int(np.round(K * (1 - downsampling)))
    Is = np.sort(np.random.choice(K, Kdown, replace=False))

    Y = Yo[:, Is]
    mYm = np.sum(Y, axis=0)
    mYmo = np.sum(Yo, axis=0)
    Ym = Y / np.maximum(mYm[np.newaxis, :], eps)
    Ymo = Yo / np.maximum(mYmo[np.newaxis, :], eps)

    if initcond > 0 and (Po is None or len(Po) == 0):
        Po = initPo(Yo, Ym, initcond, N)

    Po = np.nan_to_num(Po, nan=0.0, posinf=0.0, neginf=0.0)
    Po = np.maximum(Po, 0.0)
    mPo = Po.sum(axis=0, keepdims=True)
    mPo = np.where(mPo <= eps, 1.0, mPo)
    Po_normalized = Po / mPo
    P = Po_normalized.copy()

    iter_count = 0
    J = 1e5
    Jp = 1e6
    Ji = []
    Dm = np.zeros(Kdown)

    while ((Jp - J) / Jp) >= epsilon and iter_count < maxiter and oae == 0:
        Am = abundance_vectorized(Y, Ym, P, Dm, Lambda)
        Dm = probanonlinear_fast(Y, Ym, P, Am)

        Pp = P.copy()
        P = endmember_update_safe(Y, Ym, Po_normalized, Pp, Am, Dm, rho)

        Jp = J
        E = P @ Am
        residual = Ym - ((1.0 - Dm)[np.newaxis, :] * E) - (Dm[np.newaxis, :] * (E * Ym))
        J = np.linalg.norm(residual, "fro")
        Ji.append((Jp - J) / Jp)

        if J > Jp:
            P = Pp.copy()
            break
        iter_count += 1

    A = abundance_vectorized(Yo, Ymo, P, probanonlinear_fast(Yo, Ymo, P, np.ones((N, K)) / N), Lambda)
    D = probanonlinear_fast(Yo, Ymo, P, A)

    S = mYmo
    AA = A * mYmo
    Yh = (np.expand_dims(1.0 - D, axis=0) * (P @ AA) + np.expand_dims(D, axis=0) * ((P @ AA) * Yo))

    P = np.nan_to_num(P, nan=1.0 / L, posinf=1.0 / L, neginf=0.0)
    col_norms = np.linalg.norm(P, axis=0, keepdims=True)
    zero_cols = (col_norms <= eps)
    if np.any(zero_cols):
        P[:, zero_cols.ravel()] = Po_normalized[:, zero_cols.ravel()]
        col_norms = np.linalg.norm(P, axis=0, keepdims=True)
    P = P / np.maximum(col_norms, eps)

    A = np.nan_to_num(A, nan=1.0 / N, posinf=1.0 / N, neginf=0.0)

    return P, A, S, Yh, Ji