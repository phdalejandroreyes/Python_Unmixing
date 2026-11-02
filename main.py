from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn


from sklearn.metrics import accuracy_score

from config import load_config
from unmixing import ebeae, nebeae
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics

from training import run_training_pipeline
from dataset import create_dataloader
from Models.cross_attention_transformer import create_model

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ============================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN
# ============================================================
def validate_abundances(A, expected_pixels, n_endmembers, name="A"):
    if A.shape != (n_endmembers, expected_pixels):
        raise ValueError(f"{name}: Forma esperada ({n_endmembers}, {expected_pixels}), obtenida {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name}: Contiene valores NaN o Inf.")

def validate_labels(labels, expected_pixels, n_classes, name="Labels"):
    if labels.size != expected_pixels:
        raise ValueError(f"{name}: Tamaño incorrecto ({labels.size} vs {expected_pixels}).")

def validate_matching(row_ind, col_ind, n_endmembers):
    if len(row_ind) != n_endmembers or len(col_ind) != n_endmembers:
        raise ValueError("El número de emparejamientos no coincide con N_EST.")


# ============================================================
# 1. CARGAR CONFIGURACIÓN Y DEFINIR CONSTANTES
# ============================================================
config = load_config()

ARCHITECTURE = str(config["classification"]["architecture"]).strip().lower()
UNMIXING_METHOD = str(config["unmixing"]["method"]).strip().lower()

N = int(config["unmixing"]["n_endmembers"])
EXTRA_CLASS = bool(config["unmixing"]["extra_class"])
N_EST = N + int(EXTRA_CLASS)

N_CLASSES = int(config["classification"]["n_classes"])
DEVICE = str(config["hardware"]["device"])

PATCH_SIZE = int(config["training"]["patch_size"])
BATCH_SIZE = int(config["training"]["batch_size"])
EPOCHS = int(config["training"]["epochs"])
LEARNING_RATE = float(config["training"]["learning_rate"])

SPECTRAL_BANDS = int(config["data"]["spectral_bands"])

TRAIN_IMAGE_PATH = Path(config["data"]["train_image"])
TRAIN_LABELS_PATH = Path(config["data"]["train_labels"])
TEST_IMAGE_PATH = Path(config["data"]["test_image"])
TEST_LABELS_PATH = Path(config["data"].get("test_labels", "Satelital/labels/201912_9.tif"))


# ============================================================
# 2. VALIDAR CONFIGURACIÓN
# ============================================================
VALID_ARCHITECTURES = {"abundance", "residual", "cross_attention", "mamba", "resunet"}
VALID_UNMIXING_METHODS = {"ebeae", "nebeae"}

if ARCHITECTURE not in VALID_ARCHITECTURES:
    raise ValueError(f"Arquitectura no válida: {ARCHITECTURE!r}. Opciones: {sorted(VALID_ARCHITECTURES)}")

if UNMIXING_METHOD not in VALID_UNMIXING_METHODS:
    raise ValueError(f"Método de desmezclado no válido: {UNMIXING_METHOD!r}. Opciones: {sorted(VALID_UNMIXING_METHODS)}")

if N_EST != N_CLASSES:
    raise ValueError(f"N_EST={N_EST}, pero N_CLASSES={N_CLASSES}. Revisa n_endmembers, extra_class y n_classes.")


# ============================================================
# 3. FUNCIÓN DE DESMEZCLADO
# ============================================================
def run_unmixing(image, image_name):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"{image_name}: se esperaba una imagen 3D; se recibió {image.shape}.")

    if image.shape[2] != SPECTRAL_BANDS:
        raise ValueError(f"{image_name}: se esperaban {SPECTRAL_BANDS} bandas; se recibieron {image.shape[2]}.")

    if not np.isfinite(image).all():
        raise ValueError(f"{image_name}: la imagen contiene NaN o Inf.")

    height, width, bands = image.shape
    Y = image.reshape(height * width, bands).T

    print(f"\nDesmezclado: {image_name}")
    print(f"Imagen: {image.shape}")
    print(f"Y:      {Y.shape}")
    print(f"Método: {UNMIXING_METHOD.upper()}")
    print(f"Arquitectura: {ARCHITECTURE.upper()}")

    if UNMIXING_METHOD == "ebeae":
        P, A, S, Yh, Ji = ebeae.ebeae(Y, N=N_EST)
    elif UNMIXING_METHOD == "nebeae":
        P, A, S, Yh, Ji = nebeae.nebeae(Y, N=N_EST)
    else:
        raise RuntimeError(f"Método no implementado: {UNMIXING_METHOD}")

    P = np.asarray(P).copy()
    A = np.asarray(A).copy()

    validate_abundances(
        A,
        expected_pixels=height * width,
        n_endmembers=N_EST,
        name=f"Abundancias de {image_name}",
    )

    return P, A, Y


# ============================================================
# 4. IMAGEN DE ENTRENAMIENTO (ANCLAJE DE REFERENCIA)
# ============================================================
print("\n" + "=" * 70)
print("IMAGEN DE ENTRENAMIENTO")
print("=" * 70)

image_train = tifffile.imread(TRAIN_IMAGE_PATH)
labels_train = tifffile.imread(TRAIN_LABELS_PATH).reshape(-1).astype(np.int64)

height_train, width_train, _ = image_train.shape
validate_labels(labels_train, expected_pixels=height_train * width_train, n_classes=N_CLASSES)

P_train, A_train, Y_train = run_unmixing(image_train, "imagen de entrenamiento")

P0_train, _ = generate_pseudo_gt(
    Y=Y_train,
    labels=labels_train,
    N=N,
    extra_class=EXTRA_CLASS,
)

(
    cosine_train,
    euclidean_train,
    sam_train,
    train_row_ind,
    train_col_ind,
) = compute_metrics(P_train, P0_train)

train_row_ind = np.asarray(train_row_ind, dtype=np.int64)
train_col_ind = np.asarray(train_col_ind, dtype=np.int64)

validate_matching(train_row_ind, train_col_ind, n_endmembers=N_EST)

# Mapeo espectral exacto P_train -> Pseudo-GT (Clase c = Canal c)
A_train_aligned_raw = np.zeros_like(A_train)
P_train_aligned = np.zeros_like(P_train)

for est_idx, gt_class in zip(train_row_ind, train_col_ind):
    A_train_aligned_raw[gt_class, :] = A_train[est_idx, :]
    P_train_aligned[:, gt_class] = P_train[:, est_idx]

print("\nMatching semántico de entrenamiento (P_train vs P0):")
for row, col in zip(train_row_ind, train_col_ind):
    print(f"Endmember estimado {row} -> clase semántica {col}")

print(f"\nMétricas de Endmembers Entrenamiento vs Pseudo GT:")
print(f"Coseno:     {cosine_train:.6f}")
print(f"Euclidiana: {euclidean_train:.6f}")
print(f"SAM:        {sam_train:.6f} rad ({np.degrees(sam_train):.2f}°)")


# ============================================================
# 5. IMAGEN DE PRUEBA (ALINEACIÓN DIRECTA P_test -> P_train_aligned)
# ============================================================
print("\n" + "=" * 70)
print("IMAGEN DE PRUEBA (TEST)")
print("=" * 70)

image_test = tifffile.imread(TEST_IMAGE_PATH)
labels_test = tifffile.imread(TEST_LABELS_PATH).reshape(-1).astype(np.int64)

height_test, width_test, _ = image_test.shape
validate_labels(labels_test, expected_pixels=height_test * width_test, n_classes=N_CLASSES)

P_test, A_test, Y_test = run_unmixing(image_test, "imagen de prueba")

# Se compara P_test contra P_train_aligned para garantizar la misma firma por canal
cos_dist_test = cdist(P_train_aligned.T, P_test.T, metric='cosine')
test_row_ind, test_col_ind = linear_sum_assignment(cos_dist_test)

validate_matching(test_row_ind, test_col_ind, n_endmembers=N_EST)

A_test_aligned_raw = np.zeros_like(A_test)
P_test_aligned = np.zeros_like(P_test)

for gt_class, est_idx in zip(test_row_ind, test_col_ind):
    A_test_aligned_raw[gt_class, :] = A_test[est_idx, :]
    P_test_aligned[:, gt_class] = P_test[:, est_idx]

# Métricas de congruencia física entre P_test y P_train
cosine_test, euclidean_test, sam_test, _, _ = compute_metrics(P_test_aligned, P_train_aligned)

print("\nMatching físico/semántico de prueba (P_test -> P_train):")
for gt_class, est_idx in zip(test_row_ind, test_col_ind):
    print(f"Endmember Test {est_idx} -> asignado a canal {gt_class} (Clase {gt_class})")

print(f"\nMétricas de Congruencia Espectral (P_test vs P_train):")
print(f"Coseno:     {cosine_test:.6f}")
print(f"Euclidiana: {euclidean_test:.6f}")
print(f"SAM:        {sam_test:.6f} rad ({np.degrees(sam_test):.2f}°)")


# ============================================================
# 5.5. CALIBRACIÓN SOFTMAX SOBRE ABUNDANCIAS ANCLADAS
# ============================================================
def calibrate_abundances_softmax(A_raw, n_classes=8, temperature=0.1):
    A_tensor = torch.tensor(A_raw, dtype=torch.float32)
    return torch.softmax(A_tensor / temperature, dim=0).numpy()

# Calibración del rango dinámico por píxel
A_train_aligned = calibrate_abundances_softmax(A_train_aligned_raw, N_CLASSES)
A_test_aligned  = calibrate_abundances_softmax(A_test_aligned_raw, N_CLASSES)

preds_train_direct = np.argmax(A_train_aligned.reshape(N_CLASSES, height_train, width_train), axis=0).flatten()
preds_test_direct  = np.argmax(A_test_aligned.reshape(N_CLASSES, height_test, width_test), axis=0).flatten()

print("\n" + "=" * 70)
print("EVALUACIÓN DE ABUNDANCIAS ANCLADAS Y CALIBRADAS")
print("=" * 70)
print(f"Accuracy Directo Calibrado en Train: {accuracy_score(labels_train, preds_train_direct) * 100:.2f}%")
print(f"Accuracy Directo Calibrado en Test:  {accuracy_score(labels_test, preds_test_direct) * 100:.2f}%")



# ============================================================
# 6. CREACIÓN DE DATALOADERS Y VERIFICACIÓN DEL TRANSFORMER
# ============================================================
print("\n" + "=" * 70)
print("CONFIGURACIÓN DE MODELO Y DATALOADERS")
print("=" * 70)

DEVICE = "cpu"
device = torch.device(DEVICE)
print(f"Dispositivo de ejecución seleccionado: {device}")

# ------------------------------------------------------------
# A. Crear DataLoaders de Entrenamiento y Prueba
# ------------------------------------------------------------
train_dataset, train_loader = create_dataloader(
    A_initial=A_train_aligned,
    labels=labels_train,
    image_shape=(height_train, width_train),
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True  # <-- Activa la aumentación espacial (rotaciones y flips)
)

test_dataset, test_loader = create_dataloader(
    A_initial=A_test_aligned,
    labels=labels_test,
    image_shape=(height_test, width_test),
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False # <-- Prueba se mantiene intacta para evaluación rigurosa
)



print(f"Train Parches: {len(train_dataset)} | Batches: {len(train_loader)}")
print(f"Test Parches:  {len(test_dataset)} | Batches: {len(test_loader)}")

# ------------------------------------------------------------
# B. Preparar Matriz de Endmembers (8, 32) para Cross-Attention
# ------------------------------------------------------------
endmembers_tensor = torch.from_numpy(P_train_aligned.T).float().to(device)

# ------------------------------------------------------------
# C. Instanciar el Transformer
# ------------------------------------------------------------
model = create_model(
    n_endmembers=N_EST,
    n_classes=N_CLASSES,
    spectral_bands=SPECTRAL_BANDS,
    device=device
)

# ------------------------------------------------------------
# D. Prueba de Integración (Forward Pass con 1 Batch)
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    sample_patches, sample_labels = next(iter(train_loader))
    sample_patches = sample_patches.to(device)
    
    sample_logits = model(sample_patches, endmembers_tensor)

print("\n--- Verificación de Dimensiones ---")
print(f"Entrada Parches (x):    {sample_patches.shape}")
print(f"Entrada Endmembers (P): {endmembers_tensor.shape}")
print(f"Salida Logits:         {sample_logits.shape}")

if sample_logits.shape == (BATCH_SIZE, N_CLASSES, PATCH_SIZE, PATCH_SIZE):
    print("\n[ÉXITO] El Transformer procesó los datos correctamente y las dimensiones son congruentes.")
else:
    raise ValueError("Error en las dimensiones del tensor de salida del Transformer.")
# ============================================================
# 7. CÁLCULO DE PESOS DE CLASE Y FUNCIÓN DE PÉRDIDA PONDERADA
# ============================================================

# 1. Extraer las etiquetas de entrenamiento en formato plano
if isinstance(labels_train, torch.Tensor):
    y_train_flat = labels_train.cpu().numpy().flatten()
else:
    y_train_flat = np.array(labels_train).flatten()

# 2. Contar la cantidad de píxeles por clase (0 a 7)
class_counts = np.bincount(y_train_flat, minlength=N_CLASSES)

# 3. Calcular pesos inversos a la frecuencia: w_c = N_total / (N_clases * N_c)
total_pixels = len(y_train_flat)
class_weights = total_pixels / (N_CLASSES * (class_counts.astype(np.float32) + 1e-6))

# 4. Normalizar pesos para mantener el promedio en 1.0
class_weights = class_weights / class_weights.mean()

# 5. Convertir a Tensor de PyTorch y enviar al dispositivo
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\n" + "=" * 70)
print("PESOS ASIGNADOS A LA FUNCIÓN DE PÉRDIDA (WEIGHTED CROSS-ENTROPY)")
print("=" * 70)
for c in range(N_CLASSES):
    print(f"Clase {c}: {class_counts[c]:8d} píxeles | Peso: {class_weights[c]:.4f}")

# 6. Instanciar el criterio de pérdida ponderado
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ============================================================
# 8. EJECUTAR BUCLE DE ENTRENAMIENTO E INFERENCIA
# ============================================================
oa, f1, kappa, cm = run_training_pipeline(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    endmembers_tensor=endmembers_tensor,
    criterion=criterion,  # <-- Pasar la función de pérdida ponderada
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    device=device
)