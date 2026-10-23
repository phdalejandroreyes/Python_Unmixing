
# ============================================================
# test_align.py
#
# Validación del alineamiento espectral y semántico entre
# las imágenes de entrenamiento y prueba.
#
# Alineamiento espectral:
#   1. Similitud coseno: maximizar.
#   2. SAM: minimizar.
#   3. Se conserva COSENO como criterio principal.
#
# Alineamiento semántico:
#   1. Recibe las asignaciones endmember -> clase de train.
#   2. Recibe las asignaciones endmember -> clase de test.
#   3. Reordena el mapping de test con la correspondencia
#      espectral obtenida.
#   4. Compara ambos mappings como diagnóstico.
#
# IMPORTANTE:
#   - El diagnóstico semántico no modifica las abundancias.
#   - El mapping independiente de test no se utiliza para
#     corregir la clasificación.
#   - Si no se proporcionan los mappings, se omite esta
#     validación y se conserva el comportamiento espectral.
#
# Interfaz de retorno compatible:
#   matriz_coseno,
#   correspondencia,
#   A_test_aligned,
#   P_test_aligned,
#   diagnostico
# ============================================================

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def _construir_mapping_semantico(row_ind, col_ind, K, nombre):
    """
    Construye un mapping:
        índice de endmember estimado -> índice de clase.

    Los índices de endmember no asignados permanecen en -1.
    """

    row_ind = np.asarray(row_ind, dtype=int).ravel()
    col_ind = np.asarray(col_ind, dtype=int).ravel()

    if len(row_ind) != len(col_ind):
        raise ValueError(
            f"{nombre}: row_ind y col_ind deben tener "
            "la misma longitud."
        )

    if np.any(row_ind < 0) or np.any(row_ind >= K):
        raise ValueError(
            f"{nombre}: hay índices de endmember fuera de rango."
        )

    if len(np.unique(row_ind)) != len(row_ind):
        raise ValueError(
            f"{nombre}: hay índices de endmember duplicados."
        )

    if np.any(col_ind < 0):
        raise ValueError(
            f"{nombre}: los índices de clase no pueden ser negativos."
        )

    mapping = np.full(K, -1, dtype=int)
    mapping[row_ind] = col_ind

    return mapping


def _validar_alineamiento_semantico(
    correspondencia,
    K_train,
    K_test,
    train_row_ind,
    train_col_ind,
    test_row_ind,
    test_col_ind,
):
    """
    Compara los mappings semánticos de train y test.

    correspondencia[i] indica qué endmember de test corresponde
    al endmember i de train.

    Retorna un diccionario de diagnóstico, sin modificar A_test.
    """

    mapping_train = _construir_mapping_semantico(
        train_row_ind,
        train_col_ind,
        K_train,
        "Mapping de entrenamiento",
    )

    mapping_test = _construir_mapping_semantico(
        test_row_ind,
        test_col_ind,
        K_test,
        "Mapping de prueba",
    )

    # Llevar el mapping independiente de test al orden de train.
    mapping_test_aligned = mapping_test[correspondencia]

    validos = (
        (mapping_train >= 0)
        & (mapping_test_aligned >= 0)
    )

    coincidencias = (
        mapping_train[validos] == mapping_test_aligned[validos]
    )

    num_coincidencias = int(np.sum(coincidencias))
    num_validos = int(np.sum(validos))

    porcentaje_coincidencia = (
        100.0 * num_coincidencias / num_validos
        if num_validos > 0
        else np.nan
    )

    print("\n" + "=" * 84)
    print("VALIDACIÓN DEL ALINEAMIENTO SEMÁNTICO")
    print("=" * 84)

    print(
        f"{'Train':>7} "
        f"{'Test original':>14} "
        f"{'Clase train':>12} "
        f"{'Clase test alineada':>20} "
        f"{'Coinciden':>11}"
    )

    for i in range(K_train):
        j = int(correspondencia[i])
        clase_train = int(mapping_train[i])
        clase_test = int(mapping_test_aligned[i])

        if clase_train < 0 or clase_test < 0:
            coinciden = "N/A"
        else:
            coinciden = str(clase_train == clase_test)

        print(
            f"{i:>7} "
            f"{j:>14} "
            f"{clase_train:>12} "
            f"{clase_test:>20} "
            f"{coinciden:>11}"
        )

    print("-" * 84)
    print(
        f"Coincidencias semánticas: "
        f"{num_coincidencias}/{num_validos}"
    )

    if num_validos > 0:
        print(
            f"Porcentaje de coincidencia: "
            f"{porcentaje_coincidencia:.2f}%"
        )
    else:
        print(
            "No hay suficientes asignaciones válidas para "
            "comparar los mappings."
        )

    print("=" * 84)

    return {
        "mapping_train": mapping_train,
        "mapping_test": mapping_test,
        "mapping_test_aligned": mapping_test_aligned,
        "mascara_mappings_validos": validos,
        "coincidencias_semanticas": num_coincidencias,
        "total_mappings_validos": num_validos,
        "porcentaje_coincidencia_semantica": float(
            porcentaje_coincidencia
        ),
    }


def verificar_correspondencia(
    P_train,
    P_test,
    A_train,
    A_test,
    umbral_coseno=None,
    train_row_ind=None,
    train_col_ind=None,
    test_row_ind=None,
    test_col_ind=None,
):
    """
    Parámetros
    ----------
    P_train : ndarray, (bandas, endmembers)
        Endmembers estimados en entrenamiento.

    P_test : ndarray, (bandas, endmembers)
        Endmembers estimados en prueba.

    A_train : ndarray, (endmembers, pixeles)
        Abundancias de entrenamiento.

    A_test : ndarray, (endmembers, pixeles)
        Abundancias de prueba.

    umbral_coseno : float o None
        Umbral opcional para advertir sobre correspondencias
        espectrales con similitud coseno baja.

    train_row_ind, train_col_ind : arrays o None
        Asignación endmember -> clase obtenida durante train.

    test_row_ind, test_col_ind : arrays o None
        Asignación endmember -> clase obtenida independientemente
        para test. Se utiliza solo como diagnóstico.

    Retorna
    -------
    matriz_coseno : ndarray
        Matriz de similitud coseno (K_train, K_test).

    correspondencia : ndarray
        correspondencia[i] indica el índice del endmember de test
        que corresponde al endmember i de train.

    A_test_aligned : ndarray
        Abundancias de test reordenadas según train.

    P_test_aligned : ndarray
        Endmembers de test reordenados según train.

    diagnostico : dict
        Diagnósticos espectrales y, si se proporcionaron los
        mappings, diagnósticos semánticos.
    """

    # --------------------------------------------------------
    # 1. Conversión y validaciones
    # --------------------------------------------------------
    P_train = np.asarray(P_train, dtype=np.float64)
    P_test = np.asarray(P_test, dtype=np.float64)
    A_train = np.asarray(A_train, dtype=np.float64)
    A_test = np.asarray(A_test, dtype=np.float64)

    if any(
        x.ndim != 2
        for x in (P_train, P_test, A_train, A_test)
    ):
        raise ValueError(
            "Todas las entradas deben ser matrices 2D."
        )

    if not all(
        np.all(np.isfinite(x))
        for x in (P_train, P_test, A_train, A_test)
    ):
        raise ValueError(
            "Se encontraron NaN o valores infinitos."
        )

    bandas_train, K_train = P_train.shape
    bandas_test, K_test = P_test.shape

    if bandas_train != bandas_test:
        raise ValueError(
            "P_train y P_test deben tener las mismas bandas."
        )

    if K_train != K_test:
        raise ValueError(
            "Se requiere el mismo número de endmembers "
            "en train y test."
        )

    if A_train.shape[0] != K_train:
        raise ValueError(
            "Las filas de A_train no coinciden con P_train."
        )

    if A_test.shape[0] != K_test:
        raise ValueError(
            "Las filas de A_test no coinciden con P_test."
        )

    if K_train < 2:
        raise ValueError(
            "Se necesitan al menos dos endmembers."
        )

    if np.any(np.linalg.norm(P_train, axis=0) == 0):
        raise ValueError(
            "P_train contiene endmembers de norma cero."
        )

    if np.any(np.linalg.norm(P_test, axis=0) == 0):
        raise ValueError(
            "P_test contiene endmembers de norma cero."
        )

    # Los mappings semánticos deben recibirse completos por pares.
    argumentos_train = (train_row_ind, train_col_ind)
    argumentos_test = (test_row_ind, test_col_ind)

    if (train_row_ind is None) != (train_col_ind is None):
        raise ValueError(
            "Debes proporcionar train_row_ind y train_col_ind juntos."
        )

    if (test_row_ind is None) != (test_col_ind is None):
        raise ValueError(
            "Debes proporcionar test_row_ind y test_col_ind juntos."
        )

    hay_mapping_train = all(x is not None for x in argumentos_train)
    hay_mapping_test = all(x is not None for x in argumentos_test)

    if hay_mapping_train != hay_mapping_test:
        raise ValueError(
            "Para comparar alineamiento semántico, proporciona "
            "los mappings de train y test. Si no deseas esa "
            "comparación, omite los cuatro argumentos."
        )

    # --------------------------------------------------------
    # 2. Matriz de similitud coseno
    #
    # Filas    = endmembers de train.
    # Columnas = endmembers de test.
    # --------------------------------------------------------
    matriz_coseno = cosine_similarity(
        P_train.T,
        P_test.T,
    )

    matriz_coseno = np.clip(
        matriz_coseno,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # 3. Matriz SAM
    # --------------------------------------------------------
    matriz_sam_rad = np.arccos(matriz_coseno)
    matriz_sam_grados = np.degrees(matriz_sam_rad)

    # --------------------------------------------------------
    # 4. Asignación global óptima mediante Hungarian
    # --------------------------------------------------------
    filas_cos, cols_cos = linear_sum_assignment(
        -matriz_coseno
    )

    correspondencia_coseno = np.empty(
        K_train,
        dtype=int,
    )

    correspondencia_coseno[filas_cos] = cols_cos

    filas_sam, cols_sam = linear_sum_assignment(
        matriz_sam_rad
    )

    correspondencia_sam = np.empty(
        K_train,
        dtype=int,
    )

    correspondencia_sam[filas_sam] = cols_sam

    # Se conserva coseno como criterio principal.
    correspondencia = correspondencia_coseno.copy()

    # --------------------------------------------------------
    # 5. Reordenar abundancias y endmembers de test
    # --------------------------------------------------------
    A_test_aligned = A_test[correspondencia, :].copy()
    P_test_aligned = P_test[:, correspondencia].copy()

    # --------------------------------------------------------
    # 5.1. Validación del alineamiento semántico
    # --------------------------------------------------------
    diagnostico_semantico = None

    if hay_mapping_train and hay_mapping_test:
        diagnostico_semantico = _validar_alineamiento_semantico(
            correspondencia=correspondencia,
            K_train=K_train,
            K_test=K_test,
            train_row_ind=train_row_ind,
            train_col_ind=train_col_ind,
            test_row_ind=test_row_ind,
            test_col_ind=test_col_ind,
        )
    else:
        print(
            "\nValidación semántica omitida: "
            "no se proporcionaron mappings de train y test."
        )

    # --------------------------------------------------------
    # 6. Diagnóstico espectral de los pares seleccionados
    # --------------------------------------------------------
    indices_train = np.arange(K_train)

    coseno_asignado = matriz_coseno[
        indices_train,
        correspondencia_coseno,
    ]

    sam_asignado_rad = matriz_sam_rad[
        indices_train,
        correspondencia_coseno,
    ]

    sam_asignado_grados = np.degrees(
        sam_asignado_rad
    )

    # Margen entre los dos mejores candidatos individuales.
    coseno_ordenado = np.sort(
        matriz_coseno,
        axis=1,
    )

    margen_coseno = (
        coseno_ordenado[:, -1]
        - coseno_ordenado[:, -2]
    )

    sam_ordenado = np.sort(
        matriz_sam_rad,
        axis=1,
    )

    margen_sam_grados = np.degrees(
        sam_ordenado[:, 1]
        - sam_ordenado[:, 0]
    )

    mejor_coseno_individual = np.argmax(
        matriz_coseno,
        axis=1,
    )

    mejor_sam_individual = np.argmin(
        matriz_sam_rad,
        axis=1,
    )

    # --------------------------------------------------------
    # 7. Mostrar matrices completas
    # --------------------------------------------------------
    print("\n" + "=" * 76)
    print(
        "VALIDACIÓN DE ALINEAMIENTO ESPECTRAL: COSENO vs. SAM"
    )
    print("=" * 76)

    print("\nMatriz de similitud coseno:")
    print(np.round(matriz_coseno, 4))

    print("\nMatriz SAM (grados):")
    print(np.round(matriz_sam_grados, 3))

    # --------------------------------------------------------
    # 8. Comparación de asignaciones globales
    # --------------------------------------------------------
    print("\nAsignaciones globales:")
    print(
        f"{'Train':>7} "
        f"{'Test coseno':>12} "
        f"{'Test SAM':>10} "
        f"{'Coinciden':>11} "
        f"{'Coseno elegido':>15} "
        f"{'SAM (°)':>10}"
    )

    for i in range(K_train):
        j_cos = correspondencia_coseno[i]
        j_sam = correspondencia_sam[i]

        coinciden = j_cos == j_sam

        print(
            f"{i:>7} "
            f"{j_cos:>12} "
            f"{j_sam:>10} "
            f"{str(coinciden):>11} "
            f"{matriz_coseno[i, j_cos]:>15.4f} "
            f"{matriz_sam_grados[i, j_cos]:>10.3f}"
        )

    coincidencias_coseno_sam = int(
        np.sum(
            correspondencia_coseno
            == correspondencia_sam
        )
    )

    print(
        f"\nCoincidencias entre asignaciones: "
        f"{coincidencias_coseno_sam}/{K_train}"
    )

    # --------------------------------------------------------
    # 9. Diagnóstico de ambigüedad por endmember
    # --------------------------------------------------------
    print("\nAmbigüedad de los candidatos individuales:")
    print(
        f"{'Train':>7} "
        f"{'Mejor coseno':>13} "
        f"{'Margen coseno':>15} "
        f"{'Mejor SAM':>10} "
        f"{'Margen SAM (°)':>15}"
    )

    for i in range(K_train):
        print(
            f"{i:>7} "
            f"{mejor_coseno_individual[i]:>13} "
            f"{margen_coseno[i]:>15.4f} "
            f"{mejor_sam_individual[i]:>10} "
            f"{margen_sam_grados[i]:>15.4f}"
        )

    print("\nResumen de la asignación basada en coseno:")
    print(
        f"Coseno medio: {np.mean(coseno_asignado):.6f}"
    )
    print(
        f"Coseno mínimo: {np.min(coseno_asignado):.6f}"
    )
    print(
        f"SAM medio: {np.mean(sam_asignado_grados):.4f} grados"
    )
    print(
        f"SAM máximo: {np.max(sam_asignado_grados):.4f} grados"
    )

    if umbral_coseno is not None:
        bajos = np.where(
            coseno_asignado < umbral_coseno
        )[0]

        print(
            f"Correspondencias con coseno < {umbral_coseno}: "
            f"{len(bajos)}"
        )

        if len(bajos):
            print(
                "Índices train:",
                bajos.tolist(),
            )

    print("=" * 76)

    # --------------------------------------------------------
    # 10. Diagnóstico retornado
    # --------------------------------------------------------
    diagnostico = {
        # Matrices espectrales
        "matriz_coseno": matriz_coseno,
        "matriz_sam_rad": matriz_sam_rad,
        "matriz_sam_grados": matriz_sam_grados,

        # Correspondencias
        "correspondencia_coseno": correspondencia_coseno,
        "correspondencia_sam": correspondencia_sam,
        "correspondencia": correspondencia,

        # Ambigüedad
        "mejor_coseno_individual": mejor_coseno_individual,
        "mejor_sam_individual": mejor_sam_individual,
        "margen_coseno": margen_coseno,
        "margen_sam_grados": margen_sam_grados,

        # Métricas espectrales asignadas
        "coseno_asignado": coseno_asignado,
        "sam_asignado_grados": sam_asignado_grados,
        "coincidencias_coseno_sam": coincidencias_coseno_sam,

        # Compatibilidad con test.py
        "similitud_media": float(
            np.mean(coseno_asignado)
        ),
        "similitud_minima": float(
            np.min(coseno_asignado)
        ),
        "similitudes_asignadas": coseno_asignado,
        "num_asignaciones_distintas": int(
            np.sum(
                correspondencia_coseno
                != correspondencia_sam
            )
        ),

        # Diagnóstico semántico
        "diagnostico_semantico": diagnostico_semantico,
    }

    return (
        matriz_coseno,
        correspondencia,
        A_test_aligned,
        P_test_aligned,
        diagnostico,
    )
