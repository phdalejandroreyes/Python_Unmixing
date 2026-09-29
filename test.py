import numpy as np
import tifffile

import ebeae
from generate_pseudo_gt import generate_pseudo_gt
from compute_metrics import compute_metrics
from train import train_model
from test_transformer import test_transformer
from classification_metrics import calculate_classification_metrics

# ============================================================
# IMAGEN 1 - TRAIN
# ============================================================

print()
print("=" * 60)
print(" PROCESANDO IMAGEN 1 - ENTRENAMIENTO")
print("=" * 60)

imagen = tifffile.imread("Satelital/images/201912_1.tif")
labels = tifffile.imread("Satelital/labels/201912_1.tif")

L, M, B = imagen.shape
Yo = imagen.reshape(M * L, B).T
labels = labels.reshape(-1)

print("Imagen:", imagen.shape)
print("Labels:", labels.shape)
print("Yo:", Yo.shape)
print("Labels vector:", labels.shape)

classes, counts = np.unique(labels, return_counts=True)

print()
print("Distribución de clases:")
for c, count in zip(classes, counts):
    print(f"Clase {c}: {count} píxeles")

N = 7
extra_class = True

if extra_class:
    N_est = N + 1
else:
    validos = labels != 0
    Yo = Yo[:, validos]
    labels = labels[validos]
    N_est = N

print(f"Clases principales: {N}")
print(f"Clase extra: {extra_class}")
print(f"Endmembers a estimar: {N_est}")

# ============================================================
# EBEAE - IMAGEN 1
# ============================================================

print()
print("Ejecutando EBEAE...")

P, A, S, Yh, Ji = ebeae.ebeae(Yo, N=N_est)

print("EBEAE terminado")

# ============================================================
# PSEUDO-GROUND TRUTH - IMAGEN 1
# ============================================================

print()
print("Calculando pseudo-ground truth...")

P0, present_idx = generate_pseudo_gt(
    Y=Yo,
    labels=labels,
    N=N,
    extra_class=extra_class
)

# ============================================================
# MÉTRICAS + HUNGARIAN - IMAGEN 1
# ============================================================

E_cos_1, E_euc_1, E_sam_1, train_row_ind, train_col_ind = compute_metrics(
    P,
    P0
)

print()
print("==============================")
print("RESULTADOS IMAGEN 1")
print("==============================")
print(f"Cosine Similarity  : {E_cos_1:.6f}")
print(f"Euclidean Distance : {E_euc_1:.6f}")
print(f"SAM                : {E_sam_1:.6f}")

print()
print("Correspondencia utilizada para entrenar:")

for row, col in zip(train_row_ind, train_col_ind):
    print(f"Estimated endmember {row} → clase {col}")

# ============================================================
# ENTRENAR TRANSFORMER - IMAGEN 1
# ============================================================

A_initial = A

model = train_model(
    A_initial=A_initial,
    labels=labels,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen.shape[:2]
)

# ============================================================
# IMAGEN 2 - TEST
# ============================================================

print()
print("=" * 60)
print(" PROCESANDO IMAGEN 2 - TEST")
print("=" * 60)

imagen = tifffile.imread("Satelital/images/201912_8.tif")
labels = tifffile.imread("Satelital/labels/201912_8.tif")

L, M, B = imagen.shape
Yo = imagen.reshape(M * L, B).T
labels = labels.reshape(-1)

print()
print("Imagen:", imagen.shape)
print("Labels:", labels.shape)
print("Yo:", Yo.shape)

classes, counts = np.unique(labels, return_counts=True)

print()
print("Distribución de clases:")
for c, count in zip(classes, counts):
    print(f"Clase {c}: {count} píxeles")

N = 7
extra_class = True

if extra_class:
    N_est = N + 1
else:
    validos = labels != 0
    Yo = Yo[:, validos]
    labels = labels[validos]
    N_est = N

print(f"Clases principales: {N}")
print(f"Clase extra: {extra_class}")
print(f"Endmembers a estimar: {N_est}")

# ============================================================
# EBEAE - IMAGEN 2
# ============================================================

print()
print("Ejecutando EBEAE...")

P, A, S, Yh, Ji = ebeae.ebeae(Yo, N=N_est)

print("EBEAE terminado")

# ============================================================
# PSEUDO-GROUND TRUTH - IMAGEN 2
# ============================================================

print()
print("Calculando pseudo-ground truth...")

P0, present_idx = generate_pseudo_gt(
    Y=Yo,
    labels=labels,
    N=N,
    extra_class=extra_class
)

# ============================================================
# MÉTRICAS DE ENDMEMBERS - IMAGEN 2
#
# Este Hungarian SOLO sirve para evaluar los endmembers.
# NO se utiliza para alinear las abundancias del Transformer.
# ============================================================

E_cos_2, E_euc_2, E_sam_2, row_ind_2, col_ind_2 = compute_metrics(
    P,
    P0
)

print()
print("==============================")
print("RESULTADOS IMAGEN 2 - ENDMEMBERS")
print("==============================")
print(f"Cosine Similarity  : {E_cos_2:.6f}")
print(f"Euclidean Distance : {E_euc_2:.6f}")
print(f"SAM                : {E_sam_2:.6f}")

# ============================================================
# TEST DEL TRANSFORMER - IMAGEN 2
#
# IMPORTANTE:
# Se utiliza EXCLUSIVAMENTE la correspondencia aprendida
# con la imagen 1.
# ============================================================

A_initial = A

prediction_map = test_transformer(
    A_initial=A_initial,
    labels=labels,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen.shape[:2]
)

# ============================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================

metrics = calculate_classification_metrics(
    prediction=prediction_map,
    labels=labels,
    n_classes=8
)

# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 60)
print(" EXPERIMENTO COMPLETO FINALIZADO")
print("=" * 60)

print()
print("IMAGEN 1 - ENTRENAMIENTO")
print(f"Cosine Similarity  : {E_cos_1:.6f}")
print(f"Euclidean Distance : {E_euc_1:.6f}")
print(f"SAM                : {E_sam_1:.6f}")

print()
print("IMAGEN 2 - ENDMMEMBERS")
print(f"Cosine Similarity  : {E_cos_2:.6f}")
print(f"Euclidean Distance : {E_euc_2:.6f}")
print(f"SAM                : {E_sam_2:.6f}")

print()
print("=" * 60)
