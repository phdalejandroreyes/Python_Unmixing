import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_metrics(
    P,
    P0,
    present_idx=None
):
    """
    Calcula métricas entre los endmembers estimados y el
    pseudo-ground truth.

    El matching entre los endmembers estimados y el pseudo-GT
    se realiza mediante Hungarian assignment utilizando SAM.

    Las clases ausentes del pseudo-ground truth no participan
    en el cálculo final de las métricas.

    Parámetros
    ----------
    P : ndarray
        Endmembers estimados con forma:

            (nBands, N_est)

    P0 : ndarray
        Pseudo-ground truth con forma:

            (nBands, N_est)

    present_idx : ndarray, opcional
        Vector booleano que indica qué clases están presentes
        en el pseudo-ground truth.

        Ejemplo:

            [True, True, True, True, True, True, True, False]

        Si es None, se consideran presentes todas las clases.

    Returns
    -------
    E_cos : float
        Cosine Similarity promedio sobre las clases presentes.

    E_euc : float
        Distancia Euclidiana promedio sobre las clases presentes.

    E_sam : float
        SAM promedio en radianes sobre las clases presentes.

    row_ind : ndarray
        Índices de los endmembers estimados.

    col_ind : ndarray
        Índices de las clases del pseudo-ground truth
        correspondientes al matching.

    P_aligned : ndarray
        Endmembers estimados reordenados de acuerdo con
        el pseudo-ground truth.
    """

    eps_val = 1e-12

    # ========================================================
    # Convertir a numpy
    # ========================================================

    P = np.asarray(
        P,
        dtype=np.float64
    )

    P0 = np.asarray(
        P0,
        dtype=np.float64
    )

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
    # Verificar present_idx
    # ========================================================

    if present_idx is None:

        present_idx = np.ones(
            N_est,
            dtype=bool
        )

    else:

        present_idx = np.asarray(
            present_idx,
            dtype=bool
        ).reshape(-1)

        if len(present_idx) != N_est:
            raise ValueError(
                f"present_idx debe tener longitud {N_est}. "
                f"Longitud recibida: {len(present_idx)}"
            )

    if not np.any(present_idx):
        raise ValueError(
            "No hay clases presentes en present_idx."
        )

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
    #
    # IMPORTANTE:
    # Se mantiene el matching completo N_est x N_est
    # para no romper el alineamiento utilizado posteriormente
    # por train.py y test_transformer.py.
    # ========================================================

    angles = np.zeros(
        (N_est, N_est)
    )

    for i in range(N_est):

        p = P_norm[:, i]

        p_norm = np.linalg.norm(
            p
        )

        for j in range(N_est):

            p0 = P0[:, j]

            p0_norm = np.linalg.norm(
                p0
            )

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

                # Una clase ausente tiene P0 = 0.
                # Se mantiene como una coincidencia
                # de máxima penalización.
                angles[i, j] = np.pi

    # ========================================================
    # Hungarian assignment
    # ========================================================

    row_ind, col_ind = linear_sum_assignment(
        angles
    )

    # ========================================================
    # MOSTRAR ASIGNACIÓN
    # ========================================================

    print()
    print("Asignación de endmembers:")

    for estimated_idx, gt_idx in zip(
        row_ind,
        col_ind
    ):

        if present_idx[gt_idx]:

            print(
                f"Estimated endmember {estimated_idx} "
                f"→ GT class {gt_idx}"
            )

        else:

            print(
                f"Estimated endmember {estimated_idx} "
                f"→ GT class {gt_idx} "
                f"(clase ausente)"
            )

    # ========================================================
    # Reordenar P según el pseudo-GT
    #
    # Se mantiene completo para conservar la compatibilidad
    # con train.py y test_transformer.py.
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
    #
    # SOLAMENTE sobre clases presentes.
    # ========================================================

    sam_vals = []
    euc_vals = []
    cos_vals = []

    for n in range(N_est):

        # ----------------------------------------------------
        # Ignorar clases ausentes
        # ----------------------------------------------------

        if not present_idx[n]:
            continue

        p = P_aligned[:, n]
        p0 = P0_aligned[:, n]

        p_norm = np.linalg.norm(
            p
        )

        p0_norm = np.linalg.norm(
            p0
        )

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

            cos_vals.append(
                cosine
            )

            # ------------------------------------------------
            # SAM
            # ------------------------------------------------

            sam_vals.append(
                np.arccos(cosine)
            )

        else:

            # Si una clase marcada como presente no tiene
            # un endmember válido, se penaliza.
            cos_vals.append(
                0.0
            )

            sam_vals.append(
                np.pi
            )

        # ----------------------------------------------------
        # Euclidean Distance
        # ----------------------------------------------------

        euc_vals.append(
            np.linalg.norm(
                p - p0
            )
        )

    # ========================================================
    # Convertir a numpy
    # ========================================================

    cos_vals = np.asarray(
        cos_vals,
        dtype=np.float64
    )

    sam_vals = np.asarray(
        sam_vals,
        dtype=np.float64
    )

    euc_vals = np.asarray(
        euc_vals,
        dtype=np.float64
    )

    # ========================================================
    # Promedios
    # ========================================================

    E_cos = np.mean(
        cos_vals
    )

    E_euc = np.mean(
        euc_vals
    )

    E_sam = np.mean(
        sam_vals
    )

    return (
        E_cos,
        E_euc,
        E_sam,
        row_ind,
        col_ind
    )