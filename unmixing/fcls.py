#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCLS - Fully Constrained Least Squares Unmixing
Con Muestreo Estratificado por Clase (ANC & ASC).
"""
import numpy as np
from scipy.optimize import nnls
from unmixing.endmember_extraction import vca


def fcls(Yo=[], N=2, Po=[], percentage=1.0, labels=None):
    """
    Desmezclado FCLS robusto con muestreo estratificado y escala preservada.
    """
    eps = 1e-12
    L, K = Yo.shape

    # 1. MUESTREO ESTRATIFICADO POR CLASE PARA EXTRAER ENDMEMBERS
    if percentage < 1.0 and percentage > 0.0:
        if labels is not None and labels.size == K:
            selected_indices = []
            unique_classes = np.unique(labels)

            for c in unique_classes:
                if c < 0:  # Ignorar fondo o píxeles no etiquetados (-100)
                    continue
                class_indices = np.where(labels == c)[0]
                n_samples = max(1, int(np.round(len(class_indices) * percentage)))
                chosen = np.random.choice(class_indices, n_samples, replace=False)
                selected_indices.extend(chosen)

            indices = np.sort(np.array(selected_indices))
        else:
            K_sub = max(int(np.round(K * percentage)), N)
            indices = np.sort(np.random.choice(K, K_sub, replace=False))

        Y_sub = Yo[:, indices]
    else:
        Y_sub = Yo

    # Preservar magnitud real de la imagen
    S = np.sum(Yo, axis=0)

    # 2. EXTRACCIÓN DE ENDMEMBERS (VCA)
    if Po is None or len(Po) == 0:
        # Normalizar únicamente la muestra para VCA (requerimiento geométrico de VCA)
        mY_sub = np.sum(Y_sub, axis=0)
        Y_sub_vca = Y_sub / np.maximum(mY_sub, eps)
        Po = vca(Y_sub_vca, N)

    # Sanitizar y normalizar Endmembers a suma unitaria
    Po = np.nan_to_num(Po, nan=0.0, posinf=0.0, neginf=0.0)
    Po = np.maximum(Po, 0.0)
    mPo = np.sum(Po, axis=0, keepdims=True)
    mPo_safe = np.where(mPo <= eps, 1.0, mPo)
    P = Po / mPo_safe

    # 3. ESTIMACIÓN DE ABUNDANCIAS FCLS SOBRE FIRMAS REALES (Yo)
    # delta >= 1.0 prioriza la reconstrucción espectral sin romper la ASC
    delta = 10.0
    N_end = P.shape[1]
    
    # Sistema FCLS extendido
    P_fcls = np.vstack([P, delta * np.ones((1, N_end))])

    A = np.zeros((N_end, K), dtype=np.float32)

    for k in range(K):
        y_k = Yo[:, k]
        y_fcls = np.append(y_k, delta)

        # Resolver NNLS (ANC garantizada por definición de scipy.optimize.nnls)
        a_k, _ = nnls(P_fcls, y_fcls)

        # Proyección estricta a la suma unitaria (ASC)
        sum_a = np.sum(a_k)
        if sum_a > eps:
            a_k /= sum_a
        else:
            a_k = np.full(N_end, 1.0 / N_end)

        A[:, k] = a_k

    A = np.nan_to_num(A, nan=1.0 / N_end)

    # Reconstrucción espectral
    AA = A * S
    Yh = P @ AA
    Ji = []

    return P, A, S, Yh, Ji