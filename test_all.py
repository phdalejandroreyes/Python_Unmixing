import numpy as np
import tifffile

from unmixing import ebeae
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics
from Evaluation.classification_metrics import calculate_classification_metrics

from inference import test_model


# ============================================================
# CONFIGURACIÓN
# ============================================================

N = 7
extra_class = True
N_est = N + 1 if extra_class else N


# ============================================================
# IMAGEN 1 - ENTRENAMIENTO
# ============================================================

print()
print("=" * 70)
print(" PROCESANDO IMAGEN 1 - ENTRENAMIENTO")
print("=" * 70)

imagen1 = tifffile.imread(
    "Satelital/images/201912_1.tif"
)

labels1 = tifffile.imread(
    "Satelital/labels/201912_1.tif"
)

L, M, B = imagen1.shape

Yo1 = imagen1.reshape(M * L, B).T
labels1 = labels1.reshape(-1)

print("Imagen 1:", imagen1.shape)
print("Yo1:", Yo1.shape)
print("Labels1:", labels1.shape)
print("Endmembers:", N_est)


# ============================================================
# EBEAE - IMAGEN 1
# ============================================================

print()
print("Ejecutando EBEAE - Imagen 1...")

P1, A1, S1, Yh1, Ji1 = ebeae.ebeae(
    Yo1,
    N=N_est
)

print("EBEAE Imagen 1 terminado")
print("P1:", P1.shape)
print("A1:", A1.shape)


# ============================================================
# PSEUDO-GROUND TRUTH Y MATCHING DE ENTRENAMIENTO
# ============================================================

print()
print("Calculando correspondencia de entrenamiento...")

P01, present_idx1 = generate_pseudo_gt(
    Y=Yo1,
    labels=labels1,
    N=N,
    extra_class=extra_class
)

E_cos, E_euc, E_sam, train_row_ind, train_col_ind = compute_metrics(
    P1,
    P01,
    present_idx1
)

print()
print("Correspondencia de entrenamiento:")

for row, col in zip(train_row_ind, train_col_ind):
    print(
        f"Endmember estimado {row} -> clase {col}"
    )


# ============================================================
# IMAGEN 2 - VALIDACIÓN / TEST
# ============================================================

print()
print("=" * 70)
print(" PROCESANDO IMAGEN 2 - VALIDACIÓN")
print("=" * 70)

imagen2 = tifffile.imread(
    "Satelital/images/201912_8.tif"
)

labels2 = tifffile.imread(
    "Satelital/labels/201912_8.tif"
)

L, M, B = imagen2.shape

Yo2 = imagen2.reshape(M * L, B).T
labels2 = labels2.reshape(-1)

print("Imagen 2:", imagen2.shape)
print("Yo2:", Yo2.shape)
print("Labels2:", labels2.shape)
print("Endmembers:", N_est)


# ============================================================
# EBEAE - IMAGEN 2
# ============================================================

print()
print("Ejecutando EBEAE - Imagen 2...")

P2, A2, S2, Yh2, Ji2 = ebeae.ebeae(
    Yo2,
    N=N_est
)

print("EBEAE Imagen 2 terminado")
print("P2:", P2.shape)
print("A2:", A2.shape)


# ============================================================
# TEST DE LOS TRANSFORMERS
# ============================================================

print()
print("=" * 70)
print(" TEST DE LOS TRANSFORMERS")
print("=" * 70)

print()
print("Se utilizará el matching obtenido con la imagen 1.")
print("No se realizará un nuevo matching usando las etiquetas de la imagen 2.")


# ============================================================
# ABUNDANCE TRANSFORMER
# ============================================================

print()
print("Ejecutando Abundance Transformer...")

abundance_prediction_map = test_model(
    A_initial=A2,
    P=P2,
    labels=labels2,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen2.shape[:2],
    architecture="abundance"
)


# ============================================================
# RESIDUAL TRANSFORMER
# ============================================================

print()
print("Ejecutando Residual Transformer...")

residual_prediction_map = test_model(
    A_initial=A2,
    P=P2,
    labels=labels2,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen2.shape[:2],
    architecture="residual"
)


# ============================================================
# CROSS-ATTENTION TRANSFORMER
# ============================================================

print()
print("Ejecutando Cross-Attention Transformer...")

cross_attention_prediction_map = test_model(
    A_initial=A2,
    P=P2,
    labels=labels2,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen2.shape[:2],
    architecture="cross_attention"
)


# ============================================================
# MÉTRICAS - ABUNDANCE TRANSFORMER
# ============================================================

print()
print("=" * 70)
print(" MÉTRICAS - ABUNDANCE TRANSFORMER")
print("=" * 70)

abundance_metrics = calculate_classification_metrics(
    prediction=abundance_prediction_map,
    labels=labels2,
    n_classes=N_est
)

print()
print("Accuracy    =", abundance_metrics["accuracy"])
print("F1 macro    =", abundance_metrics["f1_macro"])
print("F1 weighted =", abundance_metrics["f1_weighted"])
print("IoU macro   =", abundance_metrics["iou_macro"])


# ============================================================
# MÉTRICAS - RESIDUAL TRANSFORMER
# ============================================================

print()
print("=" * 70)
print(" MÉTRICAS - RESIDUAL TRANSFORMER")
print("=" * 70)

residual_metrics = calculate_classification_metrics(
    prediction=residual_prediction_map,
    labels=labels2,
    n_classes=N_est
)

print()
print("Accuracy    =", residual_metrics["accuracy"])
print("F1 macro    =", residual_metrics["f1_macro"])
print("F1 weighted =", residual_metrics["f1_weighted"])
print("IoU macro   =", residual_metrics["iou_macro"])


# ============================================================
# MÉTRICAS - CROSS-ATTENTION TRANSFORMER
# ============================================================

print()
print("=" * 70)
print(" MÉTRICAS - CROSS-ATTENTION TRANSFORMER")
print("=" * 70)

cross_attention_metrics = calculate_classification_metrics(
    prediction=cross_attention_prediction_map,
    labels=labels2,
    n_classes=N_est
)

print()
print("Accuracy    =", cross_attention_metrics["accuracy"])
print("F1 macro    =", cross_attention_metrics["f1_macro"])
print("F1 weighted =", cross_attention_metrics["f1_weighted"])
print("IoU macro   =", cross_attention_metrics["iou_macro"])


# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 70)
print(" RESUMEN FINAL")
print("=" * 70)

print()
print("ABUNDANCE TRANSFORMER")
print("Accuracy    =", abundance_metrics["accuracy"])
print("F1 macro    =", abundance_metrics["f1_macro"])
print("F1 weighted =", abundance_metrics["f1_weighted"])
print("IoU macro   =", abundance_metrics["iou_macro"])

print()
print("RESIDUAL TRANSFORMER")
print("Accuracy    =", residual_metrics["accuracy"])
print("F1 macro    =", residual_metrics["f1_macro"])
print("F1 weighted =", residual_metrics["f1_weighted"])
print("IoU macro   =", residual_metrics["iou_macro"])

print()
print("CROSS-ATTENTION TRANSFORMER")
print("Accuracy    =", cross_attention_metrics["accuracy"])
print("F1 macro    =", cross_attention_metrics["f1_macro"])
print("F1 weighted =", cross_attention_metrics["f1_weighted"])
print("IoU macro   =", cross_attention_metrics["iou_macro"])

print()
print("=" * 70)
print(" PRUEBA FINALIZADA")
print("=" * 70)