import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
    precision_score,
    recall_score
)


def calculate_classification_metrics(
    prediction,
    labels,
    n_classes
):
    """
    Calcula métricas globales y por clase para clasificación
    de píxeles en imágenes hiperespectrales.

    Parámetros
    ----------
    prediction : array-like
        Mapa de predicciones.

    labels : array-like
        Etiquetas reales.

    n_classes : int
        Número total de clases.

    Retorna
    -------
    metrics : dict
        Diccionario con métricas globales y por clase.
    """

    prediction = np.asarray(prediction)
    labels = np.asarray(labels)

    prediction = prediction.reshape(-1)
    labels = labels.reshape(-1)

    if prediction.shape != labels.shape:
        raise ValueError(
            "prediction y labels deben tener el mismo número de elementos. "
            f"prediction={prediction.shape}, labels={labels.shape}"
        )

    # ========================================================
    # MÉTRICAS GLOBALES
    # ========================================================

    accuracy = accuracy_score(
        labels,
        prediction
    )

    f1_macro = f1_score(
        labels,
        prediction,
        average="macro",
        labels=np.arange(n_classes),
        zero_division=0
    )

    f1_weighted = f1_score(
        labels,
        prediction,
        average="weighted",
        labels=np.arange(n_classes),
        zero_division=0
    )

    iou_macro = jaccard_score(
        labels,
        prediction,
        average="macro",
        labels=np.arange(n_classes),
        zero_division=0
    )

    # ========================================================
    # MÉTRICAS POR CLASE
    # ========================================================

    class_labels = np.arange(n_classes)

    precision_per_class = precision_score(
        labels,
        prediction,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    recall_per_class = recall_score(
        labels,
        prediction,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    f1_per_class = f1_score(
        labels,
        prediction,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    iou_per_class = jaccard_score(
        labels,
        prediction,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    cm = confusion_matrix(
        labels,
        prediction,
        labels=class_labels
    )

    # ========================================================
    # DISTRIBUCIÓN DE CLASES
    # ========================================================

    true_counts = np.bincount(
        labels.astype(int),
        minlength=n_classes
    )

    predicted_counts = np.bincount(
        prediction.astype(int),
        minlength=n_classes
    )

    # ========================================================
    # IMPRESIÓN DEL DIAGNÓSTICO
    # ========================================================

    print()
    print("=" * 70)
    print(" DISTRIBUCIÓN DE CLASES")
    print("=" * 70)

    print()
    print(
        f"{'Clase':<10}"
        f"{'Real':>15}"
        f"{'Predicha':>15}"
    )

    print("-" * 40)

    for c in range(n_classes):

        print(
            f"{c:<10}"
            f"{true_counts[c]:>15}"
            f"{predicted_counts[c]:>15}"
        )

    # ========================================================
    # MÉTRICAS POR CLASE
    # ========================================================

    print()
    print("=" * 70)
    print(" MÉTRICAS POR CLASE")
    print("=" * 70)

    print()
    print(
        f"{'Clase':<10}"
        f"{'Precision':>15}"
        f"{'Recall':>15}"
        f"{'F1':>15}"
        f"{'IoU':>15}"
    )

    print("-" * 70)

    for c in range(n_classes):

        print(
            f"{c:<10}"
            f"{precision_per_class[c]:>15.4f}"
            f"{recall_per_class[c]:>15.4f}"
            f"{f1_per_class[c]:>15.4f}"
            f"{iou_per_class[c]:>15.4f}"
        )

    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    print()
    print("=" * 70)
    print(" MATRIZ DE CONFUSIÓN")
    print("=" * 70)

    print()
    print("Filas = clase real")
    print("Columnas = clase predicha")
    print()

    header = "        " + "".join(
        f"{c:>10}"
        for c in range(n_classes)
    )

    print(header)
    print("-" * len(header))

    for i in range(n_classes):

        row = f"{i:>6} "

        row += "".join(
            f"{cm[i, j]:>10}"
            for j in range(n_classes)
        )

        print(row)

    # ========================================================
    # RESULTADO
    # ========================================================

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "iou_macro": iou_macro,

        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "iou_per_class": iou_per_class,

        "confusion_matrix": cm,

        "true_counts": true_counts,
        "predicted_counts": predicted_counts
    }

    return metrics
