import numpy as np
import tifffile

import ebeae
from pseudo_gt_metrics import compute_pseudo_gt_metrics

# ============================================================
# Cargar imagen hyperspectral y labels
# ============================================================

imagen = tifffile.imread("Satelital/images/201912_1.tif")
labels = tifffile.imread("Satelital/labels/201912_1.tif")


# ============================================================
# Convertir de:
# (filas, columnas, bandas) -> (bandas, píxeles)
# y ajustando labels al formato vectorial
# ============================================================

L, M, B = imagen.shape
Yo = imagen.reshape( M * L, B).T

labels = labels.reshape(-1)


# ============================================================
# Configuración del experimento
# ============================================================

N = 7                    # número de clases principales
extra_class = False     # incluir clase 0 (altamente mezclada)

# ============================================================
# Preparar clases
# ============================================================

if extra_class:
    # Clase 0 + N clases principales
    N_est = N + 1
else:
    # Eliminar clase 0 (altamente mezclada)
    validos = labels != 0
    Yo = Yo[:, validos]
    labels = labels[validos]

    N_est = N

print(f"Clases principales: {N}")
print(f"Clase extra: {extra_class}")
print(f"Endmembers a estimar: {N_est}")



# ============================================================
# Ejecutar EBEAE
# ============================================================

print("\nEjecutando EBEAE...")

P, A, S, Yh, Ji = ebeae.ebeae(Yo,N=N_est)

print("EBEAE terminado")


# ============================================================
# Calcular métricas
# ============================================================

print("\nCalculando pseudo-ground truth...")

ECos, EEuc, ESAM = compute_pseudo_gt_metrics(    Yo,P,labels,N=N,extra_class=extra_class)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Cosine Similarity  : {ECos:.6f}")
print(f"Euclidean Distance : {EEuc:.6f}")
print(f"SAM                : {ESAM:.6f}")
