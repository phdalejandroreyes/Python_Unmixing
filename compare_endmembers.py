import numpy as np
from scipy.optimize import linear_sum_assignment


def spectral_angle(v1, v2, zero_threshold=1e-8):
    """
    Calcula el ángulo espectral (SAM) entre dos endmembers.

    Menor ángulo = mayor similitud.

    Si ambos vectores son prácticamente cero, se considera
    que tienen SAM = 0 grados.
    """

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Ambos prácticamente nulos
    if norm1 < zero_threshold and norm2 < zero_threshold:
        return 0.0

    # Solo uno es prácticamente nulo
    if norm1 < zero_threshold or norm2 < zero_threshold:
        return np.pi / 2

    cosine = np.dot(v1, v2) / (norm1 * norm2)

    # Evitar errores numéricos
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.arccos(cosine)


def compare_endmembers(P1, P2):
    """
    Compara los endmembers de dos imágenes utilizando SAM
    y obtiene una correspondencia uno-a-uno mediante
    el algoritmo húngaro.

    P1: (bandas, endmembers)
    P2: (bandas, endmembers)

    Retorna:
        sam_matrix : matriz SAM en radianes
        row_ind    : índices de P1
        col_ind    : índices correspondientes de P2
    """

    if P1.shape != P2.shape:
        raise ValueError(
            f"Dimensiones diferentes: P1={P1.shape}, P2={P2.shape}"
        )

    n_endmembers = P1.shape[1]

    # ============================================================
    # LIMPIEZA NUMÉRICA
    # ============================================================

    P1_clean = P1.copy()
    P2_clean = P2.copy()

    # Valores diminutos se consideran cero
    P1_clean[np.abs(P1_clean) < 1e-10] = 0.0
    P2_clean[np.abs(P2_clean) < 1e-10] = 0.0

    # ============================================================
    # NORMAS
    # ============================================================

    norms1 = np.linalg.norm(P1_clean, axis=0)
    norms2 = np.linalg.norm(P2_clean, axis=0)

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE ENDMEMBERS")
    print("=" * 70)

    print(f"P1: {P1.shape}")
    print(f"P2: {P2.shape}")

    print("\nNormas de los endmembers:")
    print("-" * 70)

    for i in range(n_endmembers):
        print(
            f"EM{i}: "
            f"P1 = {norms1[i]:.12e}    "
            f"P2 = {norms2[i]:.12e}"
        )

    # ============================================================
    # DETECTAR ENDMEMBERS CASI NULOS
    # ============================================================

    zero_threshold = 1e-8

    zero_p1 = np.where(norms1 < zero_threshold)[0]
    zero_p2 = np.where(norms2 < zero_threshold)[0]

    print("\nEndmembers prácticamente nulos:")
    print(f"P1: {zero_p1.tolist()}")
    print(f"P2: {zero_p2.tolist()}")

    # ============================================================
    # MATRIZ SAM
    # ============================================================

    sam_matrix = np.zeros(
        (n_endmembers, n_endmembers),
        dtype=float
    )

    for i in range(n_endmembers):
        for j in range(n_endmembers):

            sam_matrix[i, j] = spectral_angle(
                P1_clean[:, i],
                P2_clean[:, j],
                zero_threshold=zero_threshold
            )

    sam_degrees = np.degrees(sam_matrix)

    # ============================================================
    # MOSTRAR MATRIZ SAM
    # ============================================================

    print("\n" + "-" * 70)
    print("MATRIZ SAM (grados)")
    print("-" * 70)

    print(
        np.array2string(
            sam_degrees,
            precision=4,
            suppress_small=True
        )
    )

    # ============================================================
    # HUNGARIAN
    # ============================================================

    row_ind, col_ind = linear_sum_assignment(sam_matrix)

    print("\n" + "-" * 70)
    print("CORRESPONDENCIA ÓPTIMA UNO-A-UNO")
    print("-" * 70)

    for i, j in zip(row_ind, col_ind):

        print(
            f"P1 EM{i} -> P2 EM{j}    "
            f"SAM = {sam_degrees[i, j]:.4f}°"
        )

    # ============================================================
    # COSTO TOTAL Y PROMEDIO
    # ============================================================

    matched_angles = sam_matrix[row_ind, col_ind]
    matched_degrees = sam_degrees[row_ind, col_ind]

    print("\n" + "-" * 70)
    print("CALIDAD DE LA CORRESPONDENCIA")
    print("-" * 70)

    print(
        f"SAM total: "
        f"{np.sum(matched_angles):.6f} rad"
    )

    print(
        f"SAM promedio: "
        f"{np.mean(matched_angles):.6f} rad"
    )

    print(
        f"SAM promedio: "
        f"{np.mean(matched_degrees):.6f}°"
    )

    # Promedio excluyendo endmembers prácticamente nulos
    valid_matches = []

    for i, j in zip(row_ind, col_ind):

        if (
            norms1[i] >= zero_threshold
            and norms2[j] >= zero_threshold
        ):
            valid_matches.append(
                sam_degrees[i, j]
            )

    if len(valid_matches) > 0:

        print(
            f"SAM promedio sin endmembers nulos: "
            f"{np.mean(valid_matches):.6f}°"
        )

    # ============================================================
    # DICCIONARIO
    # ============================================================

    correspondence = {
        int(i): int(j)
        for i, j in zip(row_ind, col_ind)
    }

    print("\nDiccionario de correspondencia:")
    print(correspondence)

    print("=" * 70)

    return sam_matrix, row_ind, col_ind