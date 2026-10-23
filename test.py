import numpy as np
import tifffile

from unmixing import ebeae, nebeae
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics
from Evaluation.classification_metrics import calculate_classification_metrics
from Evaluation.validate_pipeline import (
    validate_abundances,
    validate_labels,
    validate_matching,
    validate_full_prediction,
    validate_metrics,
)
from train import train_model
from inference import test_model
from test_align import verificar_correspondencia
from Evaluation.confusion_analysis import analyze_confusion

from config import load_config


# ============================================================
# CONFIGURACIÓN
# ============================================================

config = load_config()

ARCHITECTURE = config["classification"]["architecture"]

UNMIXING_METHOD = config["unmixing"]["method"]

N = config["unmixing"]["n_endmembers"]
extra_class = config["unmixing"]["extra_class"]
N_est = N + int(extra_class)

N_CLASSES = config["classification"]["n_classes"]
DEVICE = config["hardware"]["device"]

PATCH_SIZE = config["training"]["patch_size"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]

SPECTRAL_BANDS = config["data"]["spectral_bands"]

print("Configuración cargada correctamente")
print("Arquitectura:", ARCHITECTURE)
print("Método de desmezclado:", UNMIXING_METHOD)
print("Dispositivo:", DEVICE)




# ============================================================
# IMAGEN 1 - ENTRENAMIENTO
# ============================================================
print("\n" + "=" * 60)
print("IMAGEN 1 - ENTRENAMIENTO")
print("=" * 60)

imagen_train = tifffile.imread("Satelital/images/201912_1.tif")
labels_train = tifffile.imread("Satelital/labels/201912_1.tif").reshape(-1)

L_train, M_train, B_train = imagen_train.shape
Yo_train = imagen_train.reshape(L_train * M_train, B_train).T
n_pixels_train = L_train * M_train


if UNMIXING_METHOD == "ebeae":
    P_train, A_train, S_train, Yh_train, Ji_train = ebeae.ebeae(
        Yo_train, N=N_est
    )
elif UNMIXING_METHOD == "nebeae":
    P_train, A_train, S_train, Yh_train, Ji_train = nebeae.nebeae(
        Yo_train, N=N_est
    )
else:
    raise ValueError(
        f"Método de desmezclado no válido: {UNMIXING_METHOD!r}. "
        "Opciones válidas: 'ebeae' o 'nebeae'."
    )


P_train = P_train.copy()
A_train = A_train.copy()

P0_train, present_idx_train = generate_pseudo_gt(
    Y=Yo_train,
    labels=labels_train,
    N=N,
    extra_class=extra_class,
)

E_cos_1, E_euc_1, E_sam_1, train_row_ind, train_col_ind = compute_metrics(
    P_train, P0_train
)

validate_abundances(
    A_train,
    expected_pixels=n_pixels_train,
    n_endmembers=N_est,
    name="Abundancias de entrenamiento",
)
validate_labels(
    labels_train,
    expected_pixels=n_pixels_train,
    n_classes=N_CLASSES,
    name="Etiquetas de entrenamiento",
)
validate_matching(train_row_ind, train_col_ind, n_endmembers=N_est)

nombre_modelo = UNMIXING_METHOD.upper()

print(f"Endmembers: {P_train.shape}")
print(f"Abundancias: {A_train.shape}")
print(
    f"Métricas {nombre_modelo} — coseno: {E_cos_1:.6f}, "
    f"euclidiana: {E_euc_1:.6f}, SAM: {E_sam_1:.6f}"
)
print("Correspondencia semántica de entrenamiento:")
for row, col in zip(train_row_ind, train_col_ind):
    print(f"Endmember estimado {row} -> clase {col}")


# ============================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================
print("\n" + "=" * 60)
print(f"ENTRENANDO {ARCHITECTURE.upper()}")
print("=" * 60)

model = train_model(
    A_initial=A_train,
    P=P_train,
    labels=labels_train,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen_train.shape[:2],
    architecture=ARCHITECTURE,
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
)



# ============================================================
# IMAGEN 2 - TEST
# ============================================================
print("\n" + "=" * 60)
print("IMAGEN 2 - TEST")
print("=" * 60)

imagen_test = tifffile.imread("Satelital/images/201912_8.tif")
labels_test = tifffile.imread("Satelital/labels/201912_8.tif").reshape(-1)

L_test, M_test, B_test = imagen_test.shape
Yo_test = imagen_test.reshape(L_test * M_test, B_test).T
n_pixels_test = L_test * M_test


if UNMIXING_METHOD == "ebeae":
    P_test, A_test, S_test, Yh_test, Ji_test = ebeae.ebeae(
        Yo_test, N=N_est
    )
elif UNMIXING_METHOD == "nebeae":
    P_test, A_test, S_test, Yh_test, Ji_test = nebeae.nebeae(
        Yo_test, N=N_est
    )
else:
    raise ValueError(
        f"Método de desmezclado no válido: {UNMIXING_METHOD!r}. "
        "Opciones válidas: 'ebeae' o 'nebeae'."
    )

P_test = P_test.copy()
A_test = A_test.copy()

validate_abundances(
    A_test,
    expected_pixels=n_pixels_test,
    n_endmembers=N_est,
    name="Abundancias de prueba",
)
validate_labels(
    labels_test,
    expected_pixels=n_pixels_test,
    n_classes=N_CLASSES,
    name="Etiquetas de prueba",
)

print(f"Endmembers: {P_test.shape}")
print(f"Abundancias: {A_test.shape}")


# ============================================================
# PSEUDO-GROUND TRUTH Y MÉTRICAS - IMAGEN 2
# ============================================================
P0_test, present_idx_test = generate_pseudo_gt(
    Y=Yo_test,
    labels=labels_test,
    N=N,
    extra_class=extra_class,
)

E_cos_2, E_euc_2, E_sam_2, test_row_ind, test_col_ind = compute_metrics(
    P_test, P0_test
)

validate_matching(test_row_ind, test_col_ind, n_endmembers=N_est)

print(
    f"Métricas {nombre_modelo} — coseno: {E_cos_2:.6f}, "
    f"euclidiana: {E_euc_2:.6f}, SAM: {E_sam_2:.6f}"
)
print("Correspondencia semántica independiente de la imagen 2:")
for row, col in zip(test_row_ind, test_col_ind):
    print(f"Endmember estimado {row} -> clase {col}")


# ============================================================
# DIAGNÓSTICO DE LOS VALORES DE LOS ENDMEMBERS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO DE P_train Y P_test")
print("=" * 70)

for nombre, P_actual in [
    ("P_train", P_train),
    ("P_test", P_test)
]:
    print(f"\n{nombre}")
    print("Shape:", P_actual.shape)
    print("Mínimo:", np.min(P_actual))
    print("Máximo:", np.max(P_actual))
    print("Media:", np.mean(P_actual))
    print("Desviación estándar:", np.std(P_actual))
    print("Valores negativos:", np.sum(P_actual < 0))
    print("Valores no finitos:", np.sum(~np.isfinite(P_actual)))

    print("\nPor endmember:")

    for j in range(P_actual.shape[1]):
        print(
            f"  Endmember {j}: "
            f"min={P_actual[:, j].min():.6f}, "
            f"max={P_actual[:, j].max():.6f}, "
            f"norma={np.linalg.norm(P_actual[:, j]):.6f}"
        )


# ============================================================
# CORRESPONDENCIA ENTRE IMÁGENES
# ============================================================
print("\n" + "=" * 60)
print("CORRESPONDENCIA TRAIN-TEST")
print("=" * 60)

(
    matriz_similitud,
    correspondencia,
    A_test_aligned,
    P_test_aligned,
    diagnostico,
) = verificar_correspondencia(
    P_train=P_train,
    P_test=P_test,
    A_train=A_train,
    A_test=A_test,
    umbral_coseno=None,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    test_row_ind=test_row_ind,
    test_col_ind=test_col_ind,
)

# ============================================================
# DIAGNÓSTICO DEL ORDEN DE LOS ENDMEMBERS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO: ORDEN DE ENDMEMBERS ANTES DE INFERENCIA")
print("=" * 70)

P_train_norm = P_train / (
    np.linalg.norm(P_train, axis=0, keepdims=True) + 1e-12
)

P_test_norm = P_test_aligned / (
    np.linalg.norm(P_test_aligned, axis=0, keepdims=True) + 1e-12
)

cos_aligned = np.sum(
    P_train_norm * P_test_norm,
    axis=0
)

print("\nSimilitud coseno por canal después del alineamiento:")

print(f"Similitud media: {diagnostico['similitud_media']:.6f}")
print(f"Similitud mínima: {diagnostico['similitud_minima']:.6f}")
print(f"Asignaciones distintas: {diagnostico['num_asignaciones_distintas']}")

if diagnostico.get("diagnostico_semantico") is not None:
    print("\nDiagnóstico de correspondencia semántica:")
    print(diagnostico["diagnostico_semantico"])

validate_abundances(
    A_test_aligned,
    expected_pixels=n_pixels_test,
    n_endmembers=N_est,
    name="Abundancias de prueba alineadas",
)


# ============================================================
# INFERENCIA - IMAGEN 2
# ============================================================
print("\n" + "=" * 60)
print(f"EVALUANDO {ARCHITECTURE.upper()}")
print("=" * 60)

prediction_map = test_model(
    A_initial=A_test_aligned,
    P=P_test_aligned,
    labels=labels_test,
    train_row_ind=train_row_ind,
    train_col_ind=train_col_ind,
    image_shape=imagen_test.shape[:2],
    architecture=ARCHITECTURE,
    model=model,
    device=DEVICE,
)

# Analizar los errores de clasificación.
analyze_confusion(
    labels=labels_test,
    prediction_map=prediction_map,
    n_classes=8,
    output_dir=f"Evaluation/results/{ARCHITECTURE}"
)

validate_full_prediction(
    prediction_map,
    image_shape=imagen_test.shape[:2],
    n_classes=N_CLASSES,
)


# ============================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================
metrics = calculate_classification_metrics(
    prediction=prediction_map,
    labels=labels_test,
    n_classes=N_CLASSES,
)
validate_metrics(metrics)

print("\n" + "=" * 60)
print("RESULTADOS FINALES")
print("=" * 60)
print(f"Método de unmixing: {nombre_modelo}")
print(f"Arquitectura: {ARCHITECTURE}")
print(f"Imagen 1 — coseno: {E_cos_1:.6f}, euclidiana: {E_euc_1:.6f}, SAM: {E_sam_1:.6f}")
print(f"Imagen 2 — coseno: {E_cos_2:.6f}, euclidiana: {E_euc_2:.6f}, SAM: {E_sam_2:.6f}")
print(f"Accuracy    : {metrics['accuracy']:.6f}")
print(f"F1 macro    : {metrics['f1_macro']:.6f}")
print(f"F1 weighted : {metrics['f1_weighted']:.6f}")
print(f"IoU macro   : {metrics['iou_macro']:.6f}")
print("=" * 60)