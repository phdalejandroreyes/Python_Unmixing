import numpy as np
import tifffile
import ebeae
from generate_pseudo_gt import generate_pseudo_gt
from compute_metrics import compute_metrics

imagen = tifffile.imread("Satelital/images/201912_8.tif")
labels = tifffile.imread("Satelital/labels/201912_8.tif")

L, M, B = imagen.shape
Yo = imagen.reshape(M * L, B).T
labels = labels.reshape(-1)

N = 7
N_est = 8

print("=" * 60)
print("TEST DE ABUNDANCIAS - IMAGEN 2")
print("=" * 60)
print("Imagen:", imagen.shape)
print("Yo:", Yo.shape)
print("Labels:", labels.shape)

print()
print("Ejecutando EBEAE...")

P, A, S, Yh, Ji = ebeae.ebeae(Yo, N=N_est)

print("EBEAE terminado")
print("P:", np.asarray(P).shape)
print("A:", np.asarray(A).shape)

print()
print("Calculando pseudo-ground truth...")

P0, present_idx = generate_pseudo_gt(
    Y=Yo,
    labels=labels,
    N=N,
    extra_class=True
)

E_cos, E_euc, E_sam, row_ind, col_ind = compute_metrics(P, P0)

print()
print("Correspondencia:")

for row, col in zip(row_ind, col_ind):
    print("Endmember", row, "-> clase", col)

A = np.asarray(A, dtype=np.float64)

A_aligned = np.zeros_like(A)

for row, col in zip(row_ind, col_ind):
    A_aligned[col, :] = A[row, :]

sum_A = np.sum(A_aligned, axis=0, keepdims=True)
A_norm = A_aligned / (sum_A + 1e-12)

GT_abundances = np.zeros((N_est, labels.size), dtype=np.float64)

for class_idx in range(N_est):
    GT_abundances[class_idx, labels == class_idx] = 1.0

mae = np.mean(np.abs(A_norm - GT_abundances))
mse = np.mean((A_norm - GT_abundances) ** 2)
rmse = np.sqrt(mse)

print()
print("=" * 60)
print("COMPARACION DE ABUNDANCIAS")
print("=" * 60)
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)

print()
print("METRICAS POR CLASE")

for class_idx in range(N_est):
    error = np.abs(A_norm[class_idx] - GT_abundances[class_idx])
    class_mae = np.mean(error)
    class_rmse = np.sqrt(np.mean(error ** 2))
    print(
        "Clase",
        class_idx,
        ": MAE =",
        class_mae,
        "| RMSE =",
        class_rmse
    )

prediction = np.argmax(A_norm, axis=0)
accuracy = np.mean(prediction == labels)

print()
print("=" * 60)
print("CLASIFICACION DIRECTA DE EBEAE")
print("=" * 60)
print("Accuracy EBEAE + argmax:", accuracy)

classes_pred, counts_pred = np.unique(
    prediction,
    return_counts=True
)

print()
print("Distribucion de predicciones:")

for c, count in zip(classes_pred, counts_pred):
    print("Clase", c, ":", count, "pixeles")

print()
print("=" * 60)
print("METRICAS DE ENDMEMBERS")
print("=" * 60)
print("Cosine Similarity :", E_cos)
print("Euclidean Distance:", E_euc)
print("SAM               :", E_sam)

print()
print("=" * 60)
print("TEST FINALIZADO")
print("=" * 60)