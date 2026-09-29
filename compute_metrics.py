import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_metrics(
    P,
    P0
):
    """
    Calcula métricas entre los endmembers estimados y el
    pseudo-ground truth.

    El matching entre los endmembers estimados y el pseudo-GT
    se realiza mediante Hungarian assignment utilizando SAM.

    Parámetros
    ----------
    P : ndarray
        Endmembers estimados con forma:

            (nBands, N_est)

    P0 : ndarray
        Pseudo-ground truth con forma:

            (nBands, N_est)

    Returns
    -------
    E_cos : float
        Cosine Similarity promedio.

    E_euc : float
        Distancia Euclidiana promedio.

    E_sam : float
        SAM promedio en radianes.
    """

    eps_val = 1e-12

    # ========================================================
    # Convertir a numpy
    # ========================================================

    P = np.asarray(P, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)

    # ========================================================
    # Verificar dimensiones
    # ========================================================

    if P.ndim != 2:
        raise ValueError(
            f"P debe tener 2 dimensiones. "
            f"Shape recibido: {P.shape}"
        )

    if P0.ndim != 2:
        raise ValueError(
            f"P0 debe tener 2 dimensiones. "
            f"Shape recibido: {P0.shape}"
        )

    if P.shape != P0.shape:
        raise ValueError(
            f"P y P0 deben tener la misma forma.\n"
            f"P:  {P.shape}\n"
            f"P0: {P0.shape}"
        )

    nBands, N_est = P.shape

    # ========================================================
    # Normalizar endmembers estimados
    # ========================================================

    P_norm = P.copy()

    for n in range(N_est):

        s = np.sum(
            P_norm[:, n]
        )

        if s > eps_val:

            P_norm[:, n] = (
                P_norm[:, n] / s
            )

        else:

            P_norm[:, n] = 0.0

    # ========================================================
    # Hungarian assignment usando SAM
    # ========================================================

    angles = np.zeros(
        (N_est, N_est)
    )

    for i in range(N_est):

        # Endmember estimado
        p = P_norm[:, i]

        p_norm = np.linalg.norm(p)

        for j in range(N_est):

            # Pseudo-GT
            p0 = P0[:, j]

            p0_norm = np.linalg.norm(p0)

            if (
                p_norm > eps_val
                and p0_norm > eps_val
            ):

                cosine = np.dot(
                    p,
                    p0
                ) / (
                    p_norm * p0_norm
                )

                cosine = np.clip(
                    cosine,
                    -1.0,
                    1.0
                )

                angles[i, j] = np.arccos(
                    cosine
                )

            else:

                angles[i, j] = np.pi

    # ========================================================
    # Hungarian assignment
    # ========================================================

    row_ind, col_ind = (
        linear_sum_assignment(
            angles
        )
    )

    # ========================================================
    # MOSTRAR ASIGNACIÓN DE ENDMEMBERS
    # ========================================================

    print()
    print("Asignación de endmembers:")

    for estimated_idx, gt_idx in zip(
        row_ind,
        col_ind
    ):

        print(
            f"Estimated endmember {estimated_idx} "
            f"→ GT class {gt_idx}"
        )
    # ========================================================
    # Reordenar P según el pseudo-GT
    # ========================================================

    P_aligned = np.zeros_like(
        P_norm
    )

    P0_aligned = P0.copy()

    for row, col in zip(
        row_ind,
        col_ind
    ):

        P_aligned[:, col] = (
            P_norm[:, row]
        )

    # ========================================================
    # Calcular métricas
    # ========================================================

    sam_vals = np.zeros(
        N_est
    )

    euc_vals = np.zeros(
        N_est
    )

    cos_vals = np.zeros(
        N_est
    )

    for n in range(N_est):

        p = P_aligned[:, n]
        p0 = P0_aligned[:, n]

        p_norm = np.linalg.norm(p)
        p0_norm = np.linalg.norm(p0)

        if (
            p_norm > eps_val
            and p0_norm > eps_val
        ):

            # ------------------------------------------------
            # Cosine Similarity
            # ------------------------------------------------

            cosine = np.dot(
                p,
                p0
            ) / (
                p_norm * p0_norm
            )

            cosine = np.clip(
                cosine,
                -1.0,
                1.0
            )

            cos_vals[n] = cosine

            # ------------------------------------------------
            # SAM
            # ------------------------------------------------

            sam_vals[n] = np.arccos(
                cosine
            )

        else:

            cos_vals[n] = 0.0
            sam_vals[n] = np.pi

        # ----------------------------------------------------
        # Euclidean Distance
        # ----------------------------------------------------

        euc_vals[n] = np.linalg.norm(p - p0)

    # ========================================================
    # Promedios
    # ========================================================

    E_cos = np.mean(cos_vals)
    E_euc = np.mean(euc_vals)
    E_sam = np.mean(sam_vals)

    return E_cos, E_euc, E_sam, row_ind, col_ind
