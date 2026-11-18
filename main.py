from pathlib import Path
import numpy as np
import torch
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

# Configuración y Unmixing
from config import load_config
from unmixing import ebeae, nebeae, fcls, ebeae_S
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics

# Dataset y Entrenamiento
from dataset import HyperspectralPatchDataset, create_dataloader
from training import run_training_pipeline

# Módulo de funciones de pérdida unificado
from losses import get_loss_function

# Modelo Builder
from Models.model_builder import create_model

# Módulo de firmas espectrales y dispersión
from Evaluation.spectral_signatures import spectral_signatures


# ============================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN Y PROCESAMIENTO
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

def prepare_labels(labels_raw, extra_class):
    if not extra_class:
        labels_remapped = np.full_like(labels_raw, fill_value=-100, dtype=np.int64)
        for old_class in range(1, 8):
            labels_remapped[labels_raw == old_class] = old_class - 1
        return labels_remapped
    return labels_raw

def calibrate_abundances_softmax(A_raw, temperature=1.0):
    A_sum = np.sum(A_raw, axis=0, keepdims=True) + 1e-6
    return A_raw / A_sum


# ============================================================
# 1. CARGAR CONFIGURACIÓN Y DEFINIR CONSTANTES
# ============================================================
config = load_config()

# BANDERA ACTIVADORA / DESACTIVADORA DE ANÁLISIS ESPECTRAL
SHOW_SPECTRAL_ANALYSIS = False

ARCHITECTURE = str(config["classification"]["architecture"]).strip().lower()
UNMIXING_METHOD = str(config["unmixing"]["method"]).strip().lower()

N = int(config["unmixing"]["n_endmembers"])
EXTRA_CLASS = bool(config["unmixing"]["extra_class"])
N_EST = N + int(EXTRA_CLASS)

N_CLASSES = int(config["classification"]["n_classes"])

# Definición del Dispositivo
DEVICE_STR = str(config["hardware"]["device"])
device = torch.device(DEVICE_STR)

PATCH_SIZE = int(config["training"]["patch_size"])
BATCH_SIZE = int(config["training"]["batch_size"])
EPOCHS = int(config["training"]["epochs"])
LEARNING_RATE = float(config["training"]["learning_rate"])
SPECTRAL_BANDS = int(config["data"]["spectral_bands"])

# Slicing de imágenes de entrenamiento según config.yaml
max_imgs = config["data"].get("num_train_images", None)
if max_imgs is None or max_imgs <= 0:
    train_img_list = config["data"]["train_images"]
    train_lbl_list = config["data"]["train_labels"]
else:
    train_img_list = config["data"]["train_images"][:max_imgs]
    train_lbl_list = config["data"]["train_labels"][:max_imgs]

TRAIN_IMAGE_PATHS = [Path(p) for p in train_img_list]
TRAIN_LABELS_PATHS = [Path(p) for p in train_lbl_list]
TEST_IMAGE_PATH = Path(config["data"]["test_image"])
TEST_LABELS_PATH = Path(config["data"].get("test_labels", "Satelital/labels/201912_9.tif"))

assert len(TRAIN_IMAGE_PATHS) == len(TRAIN_LABELS_PATHS), "Número de imágenes y etiquetas no coincide."
print(f"[INFO] Dataset de Entrenamiento configurado con {len(TRAIN_IMAGE_PATHS)} escena(s).")
print(f"[INFO] Método de Desmezclado Activo: {UNMIXING_METHOD.upper()}")


# ============================================================
# 2. FUNCIONES DE DESMEZCLADO CON SOPORTE EBEAE_S
# ============================================================
def run_unmixing(image, labels=None, percentage=0.20, P_ref=None, lambda_ref=0.1):
    image = np.asarray(image)
    height, width, bands = image.shape
    Y = image.reshape(height * width, bands).T

    if UNMIXING_METHOD == "ebeae":
        P, A, S, Yh, Ji = ebeae.ebeae(Y, N=N_EST)
    elif UNMIXING_METHOD == "nebeae":
        P, A, S, Yh, Ji = nebeae.nebeae(Y, N=N_EST)
    elif UNMIXING_METHOD == "fcls":
        P, A, S, Yh, Ji = fcls.fcls(Y, N=N_EST, percentage=percentage, labels=labels)
    elif UNMIXING_METHOD == "ebeae_s":
        P, A, S, Yh, Ji = ebeae_S.ebeae(Yo=Y, N=N_EST, P_ref=P_ref, lambda_ref=lambda_ref)
    else:
        raise RuntimeError(f"Método no implementado: {UNMIXING_METHOD}")

    return np.asarray(P).copy(), np.asarray(A).copy(), Y


def run_unmixing_adaptive(image, labels, extra_class, n_endmembers, description="", P_ref=None):
    height, width, bands = image.shape
    Y = image.reshape(-1, bands).T.astype(np.float32)

    if not extra_class:
        valid_mask = labels >= 0
        Y_valid = Y[:, valid_mask]
        labels_valid = labels[valid_mask]

        if UNMIXING_METHOD == "ebeae":
            P, A_valid, _, _, _ = ebeae.ebeae(Y_valid, N=n_endmembers)
        elif UNMIXING_METHOD == "nebeae":
            P, A_valid, _, _, _ = nebeae.nebeae(Y_valid, N=n_endmembers)
        elif UNMIXING_METHOD == "fcls":
            P, A_valid, _, _, _ = fcls.fcls(Y_valid, N=n_endmembers, percentage=0.20, labels=labels_valid)
        elif UNMIXING_METHOD == "ebeae_s":
            P, A_valid, _, _, _ = ebeae_S.ebeae(Yo=Y_valid, N=n_endmembers, P_ref=P_ref, lambda_ref=0.1)

        A = np.zeros((n_endmembers, height * width), dtype=np.float32)
        A[:, valid_mask] = A_valid
    else:
        P, A, Y = run_unmixing(image, labels=labels, percentage=0.20, P_ref=P_ref, lambda_ref=0.1)

    return P, A, Y


# ============================================================
# 3. PROCESAMIENTO DE LAS N IMÁGENES DE ENTRENAMIENTO
# ============================================================
print("\n" + "=" * 70)
print(f"PROCESANDO {len(TRAIN_IMAGE_PATHS)} ESCENA(S) DE ENTRENAMIENTO")
print("=" * 70)

A_train_aligned_list = []
labels_train_list = []
shapes_train_list = []

P_train_reference = None

for idx, (img_path, lbl_path) in enumerate(zip(TRAIN_IMAGE_PATHS, TRAIN_LABELS_PATHS)):
    print(f"\n---> Cargando escena {idx + 1}/{len(TRAIN_IMAGE_PATHS)}: {img_path.name}")
    img = tifffile.imread(img_path)
    lbls_raw = tifffile.imread(lbl_path).reshape(-1).astype(np.int64)
    lbls = prepare_labels(lbls_raw, extra_class=EXTRA_CLASS)

    h, w, bands = img.shape
    shapes_train_list.append((h, w))
    labels_train_list.append(lbls)
    
    Y_img = img.reshape(h * w, bands).T.astype(np.float32)

    # 1. Generar la referencia de endmembers (Pseudo-GT) desde las etiquetas
    P0_train, _ = generate_pseudo_gt(Y=Y_img, labels=lbls, N=N, extra_class=EXTRA_CLASS)

    # 2. Desmezclado adaptativo (pasando P0_train como P_ref si se usa ebeae_s)
    P_img, A_img, Y_img = run_unmixing_adaptive(
        image=img,
        labels=lbls,
        extra_class=EXTRA_CLASS,
        n_endmembers=N_EST,
        description=f"Entrenamiento Escena {idx+1}",
        P_ref=P0_train if UNMIXING_METHOD == "ebeae_s" else None
    )

    if idx == 0:
        if UNMIXING_METHOD == "ebeae_s":
            P_train_reference = P_img.copy()
        else:
            _, _, _, train_row_ind, train_col_ind = compute_metrics(P_img, P0_train)
            P_train_aligned_ref = np.zeros_like(P_img)
            for est_idx, gt_class in zip(train_row_ind, train_col_ind):
                P_train_aligned_ref[:, gt_class] = P_img[:, est_idx]
            P_train_reference = P_train_aligned_ref

    # 3. Alineación de Abundancias
    if UNMIXING_METHOD == "ebeae_s":
        print("\n[INFO] EBEAE Semisupervisado: Los endmembers nacen alineados con GT (Alineación biunívoca 1:1 directa).")
        A_aligned_raw = A_img.copy()
    else:
        eps = 1e-12
        norm_P = np.linalg.norm(P_img, axis=0, keepdims=True)
        norm_P0 = np.linalg.norm(P0_train, axis=0, keepdims=True)

        norm_P = np.where(norm_P <= eps, 1.0, norm_P)
        norm_P0 = np.where(norm_P0 <= eps, 1.0, norm_P0)

        P_normalized = P_img / norm_P
        P0_normalized = P0_train / norm_P0

        cos_sim = np.dot(P_normalized.T, P0_normalized)
        cos_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)
        cos_dist = np.nan_to_num(cos_dist, nan=1.0, posinf=1.0, neginf=1.0)

        row_ind, col_ind = linear_sum_assignment(cos_dist)

        print("\nAsignación de endmembers (Algoritmo Húngaro):")
        for est_idx, gt_idx in zip(row_ind, col_ind):
            print(f"Estimated endmember {est_idx} → GT class {gt_idx}")

        A_aligned_raw = np.zeros_like(A_img)
        for est_idx, gt_idx in zip(row_ind, col_ind):
            A_aligned_raw[gt_idx, :] = A_img[est_idx, :]

    A_calibrated = calibrate_abundances_softmax(A_aligned_raw)
    A_train_aligned_list.append(A_calibrated)

print(f"\n[ÉXITO] {len(A_train_aligned_list)} escenas de entrenamiento alineadas y calibradas.")

# ============================================================
# 3.1 ANÁLISIS ESPECTRAL Y EVALUACIÓN DE DESMEZCLADO
# ============================================================
if SHOW_SPECTRAL_ANALYSIS:
    print("\n" + "=" * 70)
    print("EJECUTANDO ANÁLISIS DE RESPUESTA ESPECTRAL Y DISPERSIÓN")
    print("=" * 70)
    spectral_signatures(
        Y_img=Y_img,
        labels=lbls,
        P_ref=P_train_reference,
        n_classes=N_CLASSES
    )

print("\n" + "=" * 70)
print(f"EVALUACIÓN DE MÉTRICAS DE DESMEZCLADO ({UNMIXING_METHOD.upper()}) VS PSEUDO-GT")
print("=" * 70)

P0_train, present_idx = generate_pseudo_gt(
    Y=Y_img, 
    labels=lbls, 
    N=N, 
    extra_class=EXTRA_CLASS
)

E_cos, E_euc, E_sam, _, _ = compute_metrics(
    P=P_img,
    P0=P0_train,
    present_idx=present_idx
)

print("\n" + "-" * 50)
print("MÉTRICAS PROMEDIO SOBRE CLASES PRESENTES:")
print(f"  • Cosine Similarity (E_cos): {E_cos:.4f}  (Ideal: 1.0)")
print(f"  • Distancia Euclidiana (E_euc): {E_euc:.4f}  (Ideal: 0.0)")
print(f"  • Ángulo Espectral SAM (E_sam): {E_sam:.4f} rad ({np.degrees(E_sam):.2f}°) (Ideal: 0.0°)")
print("-" * 50)


# ============================================================
# 4. IMAGEN DE PRUEBA (TEST)
# ============================================================
print("\n" + "=" * 70)
print("IMAGEN DE PRUEBA (TEST)")
print("=" * 70)

image_test = tifffile.imread(TEST_IMAGE_PATH)
labels_test_raw = tifffile.imread(TEST_LABELS_PATH).reshape(-1).astype(np.int64)
labels_test = prepare_labels(labels_test_raw, extra_class=EXTRA_CLASS)
height_test, width_test, bands_test = image_test.shape

# Usar P_train_reference como P_ref para proyectar el desmezclado de TEST sobre el mismo espacio de entrenamiento
P_test, A_test, Y_test = run_unmixing_adaptive(
    image=image_test,
    labels=labels_test,
    extra_class=EXTRA_CLASS,
    n_endmembers=N_EST,
    description="imagen de prueba",
    P_ref=P_train_reference if UNMIXING_METHOD == "ebeae_s" else None
)

if UNMIXING_METHOD == "ebeae_s":
    A_test_aligned_raw = A_test.copy()
else:
    cos_dist_test = cdist(P_train_reference.T, P_test.T, metric='cosine')
    test_row_ind, test_col_ind = linear_sum_assignment(cos_dist_test)

    A_test_aligned_raw = np.zeros_like(A_test)
    for gt_class, est_idx in zip(test_row_ind, test_col_ind):
        A_test_aligned_raw[gt_class, :] = A_test[est_idx, :]

A_test_aligned = calibrate_abundances_softmax(A_test_aligned_raw)


# ============================================================
# 5. DATALOADERS Y FÁBRICA DE MODELOS
# ============================================================
print("\n" + "=" * 70)
print("CONFIGURACIÓN DE MODELO Y DATALOADERS MULTI-IMAGEN")
print("=" * 70)

train_dataset, train_loader = create_dataloader(
    A_initial=A_train_aligned_list,
    labels=labels_train_list,
    image_shape=shapes_train_list,
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True
)

test_dataset, test_loader = create_dataloader(
    A_initial=A_test_aligned,
    labels=labels_test,
    image_shape=(height_test, width_test),
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False
)

print(f"Train Total Parches Multi-Imagen: {len(train_dataset)} | Batches: {len(train_loader)}")
print(f"Test Parches:                     {len(test_dataset)} | Batches: {len(test_loader)}")

endmembers_tensor = torch.from_numpy(P_train_reference.T).float().to(device)

# Instanciación del modelo a través del Builder
model = create_model(config, device=device)

# ============================================================
# 6. FUNCIÓN DE PÉRDIDA SOBRE DATASET COMBINADO
# ============================================================
all_train_labels = np.concatenate([lbl.ravel() for lbl in labels_train_list])
valid_train_labels = all_train_labels[all_train_labels >= 0]

class_counts = np.bincount(valid_train_labels, minlength=N_CLASSES)
total_valid_pixels = len(valid_train_labels)

raw_weights = total_valid_pixels / (N_CLASSES * (class_counts.astype(np.float32) + 1e-6))
smoothed_weights = np.sqrt(raw_weights)
smoothed_weights = smoothed_weights / smoothed_weights.mean()
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

print("\n" + "=" * 70)
print("PESOS GLOBALES ATENUADOS (SMOOTHED WEIGHTS)")
print("=" * 70)
for c in range(N_CLASSES):
    print(f"Clase {c}: {class_counts[c]:8d} píxeles | Peso Suavizado: {smoothed_weights[c]:.4f}")

# INSTANCIACIÓN AUTOMÁTICA DE LA FUNCIÓN DE PÉRDIDA DESDE CONFIG.YAML
criterion = get_loss_function(config, class_weights=weights_tensor, device=device)


# ============================================================
# 7. EJECUTAR ENTRENAMIENTO MULTI-IMAGEN
# ============================================================
oa, f1, kappa, cm = run_training_pipeline(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    endmembers_tensor=endmembers_tensor,
    criterion=criterion,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    device=device
)