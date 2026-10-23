
# ============================================================
# Evaluation/validate_pipeline.py
#
# Validaciones del pipeline de clasificación por abundancias.
# Compatible con las cinco arquitecturas.
# ============================================================

import numpy as np


def _to_numpy(data):
    """Convierte arrays o tensores de PyTorch a NumPy."""
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def validate_abundances(A, expected_pixels, n_endmembers=8,
                        name="abundancias"):
    """Comprueba dimensiones y valores de los mapas de abundancia."""

    A = _to_numpy(A)

    assert A.ndim == 2, (
        f"{name}: se esperaba una matriz 2D; "
        f"forma recibida: {A.shape}"
    )

    assert A.shape == (n_endmembers, expected_pixels), (
        f"{name}: forma incorrecta. "
        f"Esperada: {(n_endmembers, expected_pixels)}, "
        f"recibida: {A.shape}"
    )

    assert np.issubdtype(A.dtype, np.number), (
        f"{name}: los datos deben ser numéricos."
    )

    assert np.isfinite(A).all(), (
        f"{name}: contiene NaN o infinitos."
    )

    print(f"[OK] {name}: {A.shape}, valores finitos.")


def validate_labels(labels, expected_pixels, n_classes=8,
                    name="etiquetas"):
    """Comprueba cantidad de etiquetas y rango de clases."""

    labels = _to_numpy(labels).reshape(-1)

    assert labels.size == expected_pixels, (
        f"{name}: se esperaban {expected_pixels} píxeles, "
        f"pero hay {labels.size} etiquetas."
    )

    assert np.issubdtype(labels.dtype, np.number), (
        f"{name}: las etiquetas deben ser numéricas."
    )

    assert np.isfinite(labels).all(), (
        f"{name}: contiene NaN o infinitos."
    )

    assert np.equal(labels, np.floor(labels)).all(), (
        f"{name}: hay etiquetas que no son enteras."
    )

    assert labels.min() >= 0 and labels.max() < n_classes, (
        f"{name}: las etiquetas deben estar en "
        f"[0, {n_classes - 1}]. "
        f"Rango encontrado: [{labels.min()}, {labels.max()}]."
    )

    print(
        f"[OK] {name}: {labels.size} píxeles, "
        f"clases entre {labels.min()} y {labels.max()}."
    )


def validate_matching(row_ind, col_ind, n_endmembers=8):
    """
    Comprueba que el matching de entrenamiento sea una
    correspondencia uno-a-uno con índices válidos.

    No recalcula el matching ni demuestra por sí solo
    la corrección semántica de las clases.
    """

    row_ind = _to_numpy(row_ind).reshape(-1)
    col_ind = _to_numpy(col_ind).reshape(-1)

    assert row_ind.size == col_ind.size, (
        "El matching tiene diferente número de filas y columnas."
    )

    assert row_ind.size == n_endmembers, (
        f"Se esperaban {n_endmembers} correspondencias, "
        f"pero se encontraron {row_ind.size}."
    )

    assert np.issubdtype(row_ind.dtype, np.integer)
    assert np.issubdtype(col_ind.dtype, np.integer)

    assert np.all((row_ind >= 0) & (row_ind < n_endmembers)), (
        "El matching contiene índices de endmembers inválidos."
    )

    assert np.all((col_ind >= 0) & (col_ind < n_endmembers)), (
        "El matching contiene índices de clases inválidos."
    )

    assert np.unique(row_ind).size == row_ind.size, (
        "Hay endmembers repetidos en el matching."
    )

    assert np.unique(col_ind).size == col_ind.size, (
        "Hay clases repetidas en el matching."
    )

    print(
        f"[OK] Matching de entrenamiento: "
        f"{row_ind.size} correspondencias uno-a-uno."
    )


def validate_full_prediction(prediction, image_shape,
                             n_classes=8):
    """Comprueba que la predicción reconstruida cubra la imagen."""

    prediction = _to_numpy(prediction)

    assert prediction.shape == tuple(image_shape), (
        f"Mapa de predicción incorrecto. "
        f"Esperado: {tuple(image_shape)}, "
        f"recibido: {prediction.shape}."
    )

    assert np.isfinite(prediction).all(), (
        "El mapa de predicción contiene NaN o infinitos."
    )

    assert np.equal(prediction, np.floor(prediction)).all(), (
        "Las predicciones deben ser etiquetas enteras."
    )

    assert np.all(
        (prediction >= 0) & (prediction < n_classes)
    ), (
        f"Las predicciones deben estar entre 0 y "
        f"{n_classes - 1}."
    )

    print(
        f"[OK] Reconstrucción: {prediction.shape}; "
        f"{prediction.size} píxeles clasificados."
    )


def validate_metrics(metrics):
    """Comprueba que las métricas principales sean finitas."""

    required = (
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "iou_macro",
    )

    for key in required:
        assert key in metrics, (
            f"Falta la métrica requerida: {key}."
        )

        value = float(metrics[key])

        assert np.isfinite(value), (
            f"La métrica {key} no es finita: {value}."
        )

        assert 0.0 <= value <= 1.0, (
            f"La métrica {key} está fuera del rango [0, 1]: "
            f"{value}."
        )

    print("[OK] Accuracy, F1 e IoU dentro del rango [0, 1].")