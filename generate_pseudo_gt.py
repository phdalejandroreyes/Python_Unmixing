import numpy as np


def generate_pseudo_gt(
    Y,
    labels,
    N,
    extra_class=False
):
    """
    Construye el pseudo-ground truth espectral a partir
    de las etiquetas de los píxeles.

    Parámetros
    ----------
    Y : ndarray
        Imagen hyperspectral con forma:

            (nBands, numPixels)

    labels : ndarray
        Etiquetas de cada píxel.

        Si extra_class=False:
            clases = 1 ... N

        Si extra_class=True:
            clases = 0 ... N

    N : int
        Número de clases principales.

    extra_class : bool
        False:
            se generan N endmembers correspondientes
            a las clases 1 ... N.

        True:
            se generan N+1 endmembers correspondientes
            a las clases 0 ... N.

    Returns
    -------
    P0 : ndarray
        Pseudo-ground truth con forma:

            (nBands, N_est)

        Cada columna corresponde a la firma espectral
        promedio de una clase.

    present_idx : ndarray
        Vector booleano indicando qué clases están
        presentes en los labels.
    """

    eps_val = 1e-12

    # ========================================================
    # Dimensiones
    # ========================================================

    nBands, numPixels = Y.shape

    labels = np.asarray(
        labels
    ).reshape(-1)

    if len(labels) != numPixels:
        raise ValueError(
            f"Y and labels are not aligned: "
            f"{numPixels} pixels vs {len(labels)} labels."
        )

    # ========================================================
    # Número de endmembers
    # ========================================================

    N_est = (
        N + 1
        if extra_class
        else N
    )

    # ========================================================
    # IDs de las clases
    # ========================================================

    if extra_class:
        class_ids = np.arange(
            0,
            N + 1
        )
    else:
        class_ids = np.arange(
            1,
            N + 1
        )

    # ========================================================
    # Reservar pseudo-GT
    # ========================================================

    P0 = np.zeros(
        (nBands, N_est)
    )

    present_idx = np.zeros(
        N_est,
        dtype=bool
    )

    # ========================================================
    # Calcular firma promedio de cada clase
    # ========================================================

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
                np.sum(spectrum)
                + eps_val
            )

            P0[:, i] = spectrum

            present_idx[i] = True

    # ========================================================
    # Verificar que exista al menos una clase
    # ========================================================

    if not np.any(present_idx):

        raise ValueError(
            "No ground-truth classes were found."
        )

    return P0, present_idx
