
# ============================================================
# Evaluation/confusion_analysis.py
#
# Matriz de confusion, falsos positivos y falsos negativos.
#
# Filas    = clases reales
# Columnas = clases predichas
# ============================================================

import os
import numpy as np


def analyze_confusion(
    labels,
    prediction_map,
    n_classes=8,
    output_dir="Evaluation/results"
):
    """
    Analiza los errores de clasificación de una arquitectura.

    Parameters
    ----------
    labels : array
        Etiquetas reales de la imagen de prueba.

    prediction_map : array
        Predicciones finales del modelo.

    n_classes : int
        Número total de clases.

    output_dir : str
        Directorio donde se guardarán los resultados.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convertir ambos mapas en vectores de píxeles.
    y_true = np.asarray(labels).reshape(-1).astype(np.int64)
    y_pred = np.asarray(prediction_map).reshape(-1).astype(np.int64)

    if y_true.size != y_pred.size:
        raise ValueError(
            f"Cantidad de píxeles distinta: "
            f"labels={y_true.size}, predictions={y_pred.size}"
        )

    if (
        np.any((y_true < 0) | (y_true >= n_classes))
        or np.any((y_pred < 0) | (y_pred >= n_classes))
    ):
        raise ValueError(
            "Se encontraron etiquetas fuera del rango esperado "
            f"[0, {n_classes - 1}]."
        )

    # Filas = clase real; columnas = clase predicha.
    confusion = np.zeros(
        (n_classes, n_classes),
        dtype=np.int64
    )

    np.add.at(confusion, (y_true, y_pred), 1)

    # Guardar matriz de confusión.
    np.savetxt(
        os.path.join(output_dir, "confusion_matrix.csv"),
        confusion,
        delimiter=",",
        fmt="%d"
    )

    # Conteos por clase.
    tp = np.diag(confusion)
    real_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)

    # FP: predichos como clase c, pero pertenecen a otra clase.
    fp = predicted_count - tp

    # FN: pertenecen realmente a clase c, pero se predijeron
    # como otra clase.
    fn = real_count - tp

    # Métricas por clase.
    precision = np.divide(
        tp,
        predicted_count,
        out=np.full(n_classes, np.nan),
        where=predicted_count != 0
    )

    recall = np.divide(
        tp,
        real_count,
        out=np.full(n_classes, np.nan),
        where=real_count != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full(n_classes, np.nan),
        where=(precision + recall) > 0
    )

    # --------------------------------------------------------
    # 1. Mostrar la matriz de confusión
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSION")
    print("Filas = clase real | Columnas = clase predicha")
    print("=" * 70)
    print("       " + " ".join(f"P{c:>7}" for c in range(n_classes)))

    for c in range(n_classes):
        print(
            f"Real {c:<2} "
            + " ".join(f"{v:8d}" for v in confusion[c])
        )

    # --------------------------------------------------------
    # 2. Falsos positivos y falsos negativos por clase
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("ERRORES Y METRICAS POR CLASE")
    print("=" * 70)

    print(
        f"{'Clase':>5} {'Reales':>9} {'Predichos':>10} "
        f"{'TP':>9} {'FP':>9} {'FN':>9} "
        f"{'Prec.':>9} {'Recall':>9} {'F1':>9}"
    )

    for c in range(n_classes):
        print(
            f"{c:5d} "
            f"{real_count[c]:9d} "
            f"{predicted_count[c]:10d} "
            f"{tp[c]:9d} "
            f"{fp[c]:9d} "
            f"{fn[c]:9d} "
            f"{precision[c]:9.4f} "
            f"{recall[c]:9.4f} "
            f"{f1[c]:9.4f}"
        )

    # --------------------------------------------------------
    # 3. Detalle: ¿a qué clase se van los píxeles incorrectos?
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONFUSIONES ENTRE CLASES")
    print("=" * 70)

    total_errors = 0

    for real_class in range(n_classes):
        for predicted_class in range(n_classes):
            if real_class == predicted_class:
                continue

            count = confusion[real_class, predicted_class]

            if count > 0:
                total_errors += count
                print(
                    f"Clase real {real_class} -> "
                    f"clase predicha {predicted_class}: "
                    f"{count} pixeles"
                )

    print(f"\nTotal de píxeles mal clasificados: {total_errors}")

    # --------------------------------------------------------
    # 4. Desglose de los falsos positivos por clase predicha
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("ORIGEN DE LOS FALSOS POSITIVOS")
    print("=" * 70)

    for predicted_class in range(n_classes):
        if fp[predicted_class] == 0:
            continue

        print(
            f"\nClase predicha {predicted_class}: "
            f"{fp[predicted_class]} falsos positivos"
        )

        for real_class in range(n_classes):
            if real_class == predicted_class:
                continue

            count = confusion[real_class, predicted_class]

            if count > 0:
                print(
                    f"  Provienen de la clase real "
                    f"{real_class}: {count} pixeles"
                )

    # Guardar métricas por clase.
    metrics = np.column_stack([
        np.arange(n_classes),
        real_count,
        predicted_count,
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    ])

    np.savetxt(
        os.path.join(output_dir, "class_errors.csv"),
        metrics,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%d", "%d", "%d",
             "%.6f", "%.6f", "%.6f"],
        header=(
            "class,real_count,predicted_count,TP,FP,FN,"
            "precision,recall,F1"
        ),
        comments=""
    )

    print(f"\nResultados guardados en: {output_dir}")

    return confusion, {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1
    }
