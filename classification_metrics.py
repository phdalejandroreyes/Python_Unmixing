# ============================================================
# classification_metrics.py
#
# Métricas de clasificación del Transformer
#
# Entrada:
#   prediction -> (H, W)
#   labels     -> (H*W,) o (H, W)
#
# Calcula:
#   Accuracy
#   F1-score
#   IoU
#   Matriz de confusión
# ============================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    jaccard_score
)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def calculate_classification_metrics(
    prediction,
    labels,
    n_classes=8
):

    # --------------------------------------------------------
    # Convertir prediction a numpy
    # --------------------------------------------------------

    if hasattr(prediction, "detach"):

        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    prediction = np.asarray(prediction)

    # --------------------------------------------------------
    # Convertir labels a numpy
    # --------------------------------------------------------

    if hasattr(labels, "detach"):

        labels = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

    labels = np.asarray(labels)

    # --------------------------------------------------------
    # Aplanar
    # --------------------------------------------------------

    prediction = prediction.reshape(-1)

    labels = labels.reshape(-1)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        prediction
    )

    # --------------------------------------------------------
    # F1
    #
    # Calculamos:
    #
    #   F1 macro
    #   F1 weighted
    #   F1 por clase
    # --------------------------------------------------------

    f1_macro = f1_score(
        labels,
        prediction,
        labels=np.arange(n_classes),
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        labels,
        prediction,
        labels=np.arange(n_classes),
        average="weighted",
        zero_division=0
    )

    f1_per_class = f1_score(
        labels,
        prediction,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0
    )

    # --------------------------------------------------------
    # IoU
    #
    # Calculamos:
    #
    #   IoU macro
    #   IoU por clase
    # --------------------------------------------------------

    iou_macro = jaccard_score(
        labels,
        prediction,
        labels=np.arange(n_classes),
        average="macro",
        zero_division=0
    )

    iou_per_class = jaccard_score(
        labels,
        prediction,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0
    )

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    confusion = confusion_matrix(
        labels,
        prediction,
        labels=np.arange(n_classes)
    )

    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print()
    print("=" * 60)
    print(" MÉTRICAS DE CLASIFICACIÓN")
    print("=" * 60)

    print()
    print("Accuracy:")
    print(f"  {accuracy:.4f}")

    print()
    print("F1 Macro:")
    print(f"  {f1_macro:.4f}")

    print()
    print("F1 Weighted:")
    print(f"  {f1_weighted:.4f}")

    print()
    print("IoU Macro:")
    print(f"  {iou_macro:.4f}")

    print()
    print("-" * 60)
    print(" MÉTRICAS POR CLASE")
    print("-" * 60)

    for class_id in range(n_classes):

        print(
            f"Clase {class_id}: "
            f"F1 = {f1_per_class[class_id]:.4f} | "
            f"IoU = {iou_per_class[class_id]:.4f}"
        )

    print()
    print("-" * 60)
    print(" MATRIZ DE CONFUSIÓN")
    print("-" * 60)

    print(confusion)

    print()
    print("=" * 60)

    # --------------------------------------------------------
    # Devolver resultados
    # --------------------------------------------------------

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
        "iou_macro": iou_macro,
        "iou_per_class": iou_per_class,
        "confusion_matrix": confusion
    }

    return metrics
