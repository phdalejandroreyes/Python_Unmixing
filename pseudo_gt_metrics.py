import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_pseudo_gt_metrics(
    Y,
    P,
    labels,
    N,
    extra_class=False
):
    """
    Calcula métricas entre los endmembers estimados y un
    pseudo-ground truth construido a partir de las etiquetas.

    Parámetros
    ----------
    Y : ndarray
        Imagen hyperspectral con forma (nBands, numPixels).

    P : ndarray
        Endmembers estimados con forma (nBands, N_est).

    labels : ndarray
        Etiquetas de cada píxel.

        Si extra_class=False:
            clases esperadas = 1 ... N

        Si extra_class=True:
            clases esperadas = 0 ... N
            donde la clase 0 es la clase altamente mezclada.

    N : int
        Número de clases principales.

    extra_class : bool
        False -> estima/evalúa N clases: 1 ... N
        True  -> estima/evalúa N+1 clases: 0 ... N

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

    # ------------------------------------------------------------
    # Dimensiones
    # ------------------------------------------------------------

    nBands, numPixels = Y.shape

    labels = np.asarray(labels).reshape(-1)

    if len(labels) != numPixels:
        raise ValueError(
            f"Y and labels are not aligned: "
            f"{numPixels} pixels vs {len(labels)} labels."
        )

    # ------------------------------------------------------------
    # Número total de endmembers que deben ser evaluados
    # ------------------------------------------------------------

    N_est = N + 1 if extra_class else N

    # ------------------------------------------------------------
    # Verificar que P tenga el número correcto de endmembers
    # ------------------------------------------------------------

    if P.shape[1] != N_est:
        raise ValueError(
            f"P contains {P.shape[1]} endmembers, "
            f"but N_est={N_est} is expected "
            f"(N={N}, extra_class={extra_class})."
        )

    # ------------------------------------------------------------
    # Definir IDs de las clases que serán evaluadas
    #
    # extra_class=False:
    #     [1, 2, ..., N]
    #
    # extra_class=True:
    #     [0, 1, 2, ..., N]
    # ------------------------------------------------------------

    if extra_class:
        class_ids = np.arange(0, N + 1)
    else:
        class_ids = np.arange(1, N + 1)

    # ------------------------------------------------------------
    # Build pseudo Ground Truth
    # ------------------------------------------------------------

    P0 = np.zeros(
        (nBands, N_est)
    )

    present_idx = np.zeros(
        N_est,
        dtype=bool
    )

    for i, class_id in enumerate(class_ids):

        idx = labels == class_id

        if np.any(idx):

            # Promedio espectral de los píxeles
            # pertenecientes a la clase
            spectrum = np.mean(
                Y[:, idx],
                axis=1
            )

            # Normalización espectral
            spectrum = spectrum / (
                np.sum(spectrum) + eps_val
            )

            P0[:, i] = spectrum

            present_idx[i] = True

    # ------------------------------------------------------------
    # Normalizar endmembers estimados
    # ------------------------------------------------------------

    P = P.copy()

    for n in range(N_est):

        s = np.sum(P[:, n])

        if s > eps_val:
            P[:, n] = P[:, n] / s
        else:
            P[:, n] = 0

    # ------------------------------------------------------------
    # Seleccionar solamente clases presentes
    # ------------------------------------------------------------

    present_classes = np.flatnonzero(
        present_idx
    )

    N_present = len(
        present_classes
    )

    if N_present == 0:
        raise ValueError(
            "No ground-truth classes were found."
        )

    # ------------------------------------------------------------
    # Hungarian assignment usando SAM
    # ------------------------------------------------------------

    angles = np.zeros(
        (N_present, N_present)
    )

    for i in range(N_present):

        # Endmember estimado
        p = P[
            :,
            present_classes[i]
        ]

        np_norm = np.linalg.norm(p)

        for j in range(N_present):

            # Pseudo-GT
            p0 = P0[
                :,
                present_classes[j]
            ]

            np0_norm = np.linalg.norm(p0)

            if (
                np_norm > eps_val
                and np0_norm > eps_val
            ):

                val = np.dot(
                    p,
                    p0
                ) / (
                    np_norm * np0_norm
                )

                val = np.clip(
                    val,
                    -1,
                    1
                )

                angles[i, j] = np.arccos(
                    val
                )

            else:

                angles[i, j] = np.pi

    # ------------------------------------------------------------
    # Hungarian assignment
    # ------------------------------------------------------------

    row_ind, col_ind = linear_sum_assignment(
        angles
    )

    assignment = np.empty(
        N_present,
        dtype=int
    )

    assignment[row_ind] = col_ind

    # ------------------------------------------------------------
    # Reordenar endmembers estimados
    # de acuerdo con el pseudo-GT
    # ------------------------------------------------------------

    P_aligned = np.zeros(
        (nBands, N_present)
    )

    P0_aligned = P0[
        :,
        present_classes
    ]

    for i in range(N_present):

        P_aligned[:, i] = P[
            :,
            present_classes[
                assignment[i]
            ]
        ]

    P = P_aligned
    P0 = P0_aligned

    # ------------------------------------------------------------
    # Calcular métricas
    # ------------------------------------------------------------

    sam_vals = np.zeros(
        N_present
    )

    euc_vals = np.zeros(
        N_present
    )

    cos_vals = np.zeros(
        N_present
    )

    for n in range(N_present):

        p = P[:, n]
        p0 = P0[:, n]

        np_norm = np.linalg.norm(p)
        np0_norm = np.linalg.norm(p0)

        if (
            np_norm > eps_val
            and np0_norm > eps_val
        ):

            # ----------------------------------------
            # Cosine Similarity
            # ----------------------------------------

            cosine = np.dot(
                p,
                p0
            ) / (
                np_norm * np0_norm
            )

            cosine = np.clip(
                cosine,
                -1,
                1
            )

            cos_vals[n] = cosine

            # ----------------------------------------
            # SAM
            # ----------------------------------------

            sam_vals[n] = np.arccos(
                cosine
            )

        else:

            cos_vals[n] = 0
            sam_vals[n] = np.pi

        # --------------------------------------------
        # Euclidean Distance
        # --------------------------------------------

        euc_vals[n] = np.linalg.norm(
            p - p0
        )

    # ------------------------------------------------------------
    # Promedios
    # ------------------------------------------------------------

    E_cos = np.mean(
        cos_vals
    )

    E_euc = np.mean(
        euc_vals
    )

    E_sam = np.mean(
        sam_vals
    )

    return E_cos, E_euc, E_sam
