import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_metrics(
    P,
    P0,
    present_idx=None,
    is_semisupervised=True
):
    """
    Calcula métricas entre los endmembers estimados y el pseudo-ground truth.

    Si `is_semisupervised` es True (ej. EBEAE_S), respeta la alineación biunívoca 1:1
    directa (Endmember k -> GT Class k) y OMITA el Algoritmo Húngaro.

    Si `is_semisupervised` es False (ej. EBEAE ciego, NEBEAE), utiliza el
    Algoritmo Húngaro (linear_sum_assignment) sobre la matriz de SAM.
    """

    eps_val = 1e-12

    # Convertir a numpy
    P = np.asarray(P, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)

    # Verificar dimensiones
    if P.ndim != 2 or P0.ndim != 2:
        raise ValueError(f"P y P0 deben tener 2 dimensiones. Shapes recibidos: P={P.shape}, P0={P0.shape}")

    if P.shape != P0.shape:
        raise ValueError(f"P y P0 deben tener la misma forma. P: {P.shape}, P0: {P0.shape}")

    nBands, N_est = P.shape

    # Verificar present_idx
    if present_idx is None:
        present_idx = np.ones(N_est, dtype=bool)
    else:
        present_idx = np.asarray(present_idx, dtype=bool).reshape(-1)
        if len(present_idx) != N_est:
            raise ValueError(f"present_idx debe tener longitud {N_est}. Longitud: {len(present_idx)}")

    if not np.any(present_idx):
        raise ValueError("No hay clases presentes en present_idx.")

    # Normalizar endmembers estimados
    P_norm = P.copy()
    for n in range(N_est):
        s = np.sum(P_norm[:, n])
        if s > eps_val:
            P_norm[:, n] = P_norm[:, n] / s
        else:
            P_norm[:, n] = 0.0

    # ========================================================
    # ALINEACIÓN DE ENDMEMBERS (1:1 DIRECTA vs HÚNGARO)
    # ========================================================
    if is_semisupervised:
        # En EBEAE_S la correspondencia nace 1:1 biunívoca directa
        row_ind = np.arange(N_est, dtype=int)
        col_ind = np.arange(N_est, dtype=int)
        
        print("\nAsignación de endmembers (EBEAE_S Directa 1:1):")
        for k in range(N_est):
            status = "" if present_idx[k] else " (clase ausente)"
            print(f"  Estimated endmember {k} → GT class {k}{status}")
    else:
        # Algoritmo Húngaro solo para métodos ciegos
        angles = np.zeros((N_est, N_est))
        for i in range(N_est):
            p = P_norm[:, i]
            p_norm = np.linalg.norm(p)
            for j in range(N_est):
                p0 = P0[:, j]
                p0_norm = np.linalg.norm(p0)
                if p_norm > eps_val and p0_norm > eps_val:
                    cosine = np.dot(p, p0) / (p_norm * p0_norm)
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angles[i, j] = np.arccos(cosine)
                else:
                    angles[i, j] = np.pi

        row_ind, col_ind = linear_sum_assignment(angles)

        print("\nAsignación de endmembers (Algoritmo Húngaro):")
        for estimated_idx, gt_idx in zip(row_ind, col_ind):
            status = "" if present_idx[gt_idx] else " (clase ausente)"
            print(f"  Estimated endmember {estimated_idx} → GT class {gt_idx}{status}")

    # Reordenar P según asignación
    P_aligned = np.zeros_like(P_norm)
    P0_aligned = P0.copy()

    for row, col in zip(row_ind, col_ind):
        P_aligned[:, col] = P_norm[:, row]

    # ========================================================
    # CALCULAR MÉTRICAS SOBRE CLASES PRESENTES
    # ========================================================
    sam_vals, euc_vals, cos_vals = [], [], []

    for n in range(N_est):
        if not present_idx[n]:
            continue

        p = P_aligned[:, n]
        p0 = P0_aligned[:, n]

        p_norm = np.linalg.norm(p)
        p0_norm = np.linalg.norm(p0)

        if p_norm > eps_val and p0_norm > eps_val:
            cosine = np.dot(p, p0) / (p_norm * p0_norm)
            cosine = np.clip(cosine, -1.0, 1.0)
            cos_vals.append(cosine)
            sam_vals.append(np.arccos(cosine))
        else:
            cos_vals.append(0.0)
            sam_vals.append(np.pi)

        euc_vals.append(np.linalg.norm(p - p0))

    E_cos = float(np.mean(cos_vals))
    E_euc = float(np.mean(euc_vals))
    E_sam = float(np.mean(sam_vals))

    return E_cos, E_euc, E_sam, row_ind, col_ind