#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
pilot_pglobal_abundance.py

Descripción:
    Experimento Piloto Fase 1 con AbundanceTransformer (stride=32 intacto):
      1. Construye un diccionario de endmembers global de consenso (P_global)
         acumulando los píxeles etiquetados de las 7 escenas de entrenamiento.
      2. Desmezcla las 7 escenas de Train y las 3 escenas de Test proyectando
         todas sobre el MISMO P_global fijo (invarianza de base del símplex).
      3. Concatena un 9.º canal con la norma radiométrica normalizada por escena
         para recuperar la información de brillo/reflectancia perdida en el símplex.
      4. Entrena AbundanceTransformer (150 épocas) y evalúa en las 3 escenas
         reservadas (201912_8, 201912_9, 201912_10) con y sin Clase 0.
=============================================================================
"""

import os
import csv
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
import tifffile
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config import load_config
from unmixing import ebeae_S
from dataset import create_dataloader
from training import run_training_pipeline
from losses import get_loss_function
from Models.abundance_transformer import AbundanceTransformer


# Definición limpia en Python basada en tu convención de nombres y carpetas
test_scene_ids = ["8", "9", "10"]
TEST_SCENES = [
    (
        f"201912_{sid}",
        Path(f"Satelital/images/201912_{sid}.tif"),
        Path(f"Satelital/labels/201912_{sid}.tif")
    )
    for sid in test_scene_ids
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_labels(labels_raw, extra_class):
    if not extra_class:
        labels_remapped = np.full_like(labels_raw, fill_value=-100, dtype=np.int64)
        for old_class in range(1, 8):
            labels_remapped[labels_raw == old_class] = old_class - 1
        return labels_remapped
    return labels_raw


def build_global_endmembers(image_paths, label_paths, n_est, extra_class):
    """
    Construye P_global (L, n_est) promediando todos los píxeles de cada clase
    a lo largo de las 7 escenas de entrenamiento (centroide espectral global).
    """
    sum_spectra = None
    pixel_counts = np.zeros(n_est, dtype=np.float64)

    for img_p, lbl_p in zip(image_paths, label_paths):
        img = tifffile.imread(img_p).astype(np.float64)
        lbl = prepare_labels(tifffile.imread(lbl_p).reshape(-1).astype(np.int64), extra_class)
        h, w, bands = img.shape
        Y = img.reshape(h * w, bands)  # (N_pix, L)

        if sum_spectra is None:
            sum_spectra = np.zeros((bands, n_est), dtype=np.float64)

        for c in range(n_est):
            mask = lbl == c
            cnt = int(mask.sum())
            if cnt > 0:
                sum_spectra[:, c] += Y[mask].sum(axis=0)
                pixel_counts[c] += cnt

    P_global = np.zeros_like(sum_spectra, dtype=np.float32)
    for c in range(n_est):
        if pixel_counts[c] > 0:
            P_global[:, c] = (sum_spectra[:, c] / pixel_counts[c]).astype(np.float32)
        else:
            raise ValueError(f"La clase {c} no tiene píxeles en ninguna escena de entrenamiento.")

    return P_global, pixel_counts


def extract_9ch_features(image: np.ndarray, P_global: np.ndarray, lambda_ref: float = 0.5) -> np.ndarray:
    """
    Proyecta la imagen sobre P_global fijo mediante ebeae_S y concatena la
    norma radiométrica normalizada por escena como 9.º canal.
    Retorna array de forma (n_est + 1, H * W).
    """
    h, w, bands = image.shape
    Y = image.reshape(h * w, bands).T.astype(np.float32)  # (L, N_pix)
    n_est = P_global.shape[1]

    # Desmezclado proyectado sobre la base global fija P_global
    _, A_raw, _, _, _ = ebeae_S.ebeae(Yo=Y, N=n_est, P_ref=P_global, lambda_ref=lambda_ref)
    A_raw = np.asarray(A_raw, dtype=np.float32)

    # Calibración en el símplex
    A_sum = np.sum(A_raw, axis=0, keepdims=True) + 1e-6
    A_cal = A_raw / A_sum

    # Canal extra: Norma L2 radiométrica estandarizada de forma robusta por escena
    pix_norm = np.linalg.norm(Y, axis=0, keepdims=True)  # (1, N_pix)
    med = np.median(pix_norm)
    q75, q25 = np.percentile(pix_norm, [75, 25])
    iqr = max(float(q75 - q25), 1e-6)
    norm_z = np.clip((pix_norm - med) / iqr, -3.0, 3.0)
    # Reescalar a [0, 1] para mantener la misma escala que las abundancias
    norm_01 = ((norm_z + 3.0) / 6.0).astype(np.float32)

    features_9ch = np.vstack([A_cal, norm_01])  # (9, N_pix)
    return features_9ch


def evaluate_scene_logits(model, loader, endmembers_tensor, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            logits = model(patches)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu().numpy()
            targets_flat = labels.numpy().reshape(-1)
            valid = targets_flat >= 0
            all_logits.append(logits_flat[valid])
            all_targets.append(targets_flat[valid])
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def metrics_summary(logits: np.ndarray, y_true: np.ndarray):
    y_pred_all = np.argmax(logits, axis=1)
    labels_all = np.sort(np.unique(y_true))
    oa_all = accuracy_score(y_true, y_pred_all) * 100.0
    f1_all = f1_score(y_true, y_pred_all, labels=labels_all, average="macro", zero_division=0) * 100.0
    kappa_all = cohen_kappa_score(y_true, y_pred_all, labels=labels_all)

    mask_no_c0 = y_true > 0
    y_true_c17 = y_true[mask_no_c0]
    labels_c17 = np.sort(np.unique(y_true_c17))

    y_pred_c17_dir = y_pred_all[mask_no_c0]
    oa_c17_dir = accuracy_score(y_true_c17, y_pred_c17_dir) * 100.0

    y_pred_c17_mask = np.argmax(logits[mask_no_c0, 1:], axis=1) + 1
    oa_c17_mask = accuracy_score(y_true_c17, y_pred_c17_mask) * 100.0
    f1_c17_mask = f1_score(y_true_c17, y_pred_c17_mask, labels=labels_c17, average="macro", zero_division=0) * 100.0
    kappa_c17_mask = cohen_kappa_score(y_true_c17, y_pred_c17_mask, labels=labels_c17)

    return {
        "oa_all": oa_all,
        "f1_all": f1_all,
        "kappa_all": kappa_all,
        "oa_c17_dir": oa_c17_dir,
        "oa_c17_mask": oa_c17_mask,
        "f1_c17_mask": f1_c17_mask,
        "kappa_c17_mask": kappa_c17_mask,
    }


def main():
    set_seed(42)
    config = load_config()

    n_endmembers = int(config["unmixing"]["n_endmembers"])
    extra_class = bool(config["unmixing"]["extra_class"])
    n_est = n_endmembers + int(extra_class)
    n_classes = int(config["classification"]["n_classes"])

    device = torch.device(str(config["hardware"]["device"]))
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    patch_size = int(config["training"]["patch_size"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])

    max_imgs = config["data"].get("num_train_images", None)
    if max_imgs is None or max_imgs <= 0:
        train_img_paths = [Path(p) for p in config["data"]["train_images"]]
        train_lbl_paths = [Path(p) for p in config["data"]["train_labels"]]
    else:
        train_img_paths = [Path(p) for p in config["data"]["train_images"][:max_imgs]]
        train_lbl_paths = [Path(p) for p in config["data"]["train_labels"][:max_imgs]]

    print("=" * 90)
    print("PILOTO FASE 1: ABUNDANCE TRANSFORMER + P_GLOBAL (7 ESCENAS) + NORMA RADIOMÉTRICA")
    print("=" * 90)

    # 1. Construir P_global desde las 7 escenas de entrenamiento
    t0 = time.time()
    print(f"[1/4] Calculando P_global de consenso sobre {len(train_img_paths)} escenas de entrenamiento...")
    P_global, global_counts = build_global_endmembers(train_img_paths, train_lbl_paths, n_est, extra_class)
    print(f"      P_global forma: {P_global.shape} | Píxeles globales por clase: {global_counts.astype(int).tolist()}")

    # 2. Extraer características de 9 canales (8 abundancias proyectadas en P_global + 1 norma)
    print("\n[2/4] Proyectando escenas de entrenamiento sobre P_global fijo (lambda_ref=0.5)...")
    X_train_list = []
    labels_train_list = []
    shapes_train_list = []

    for idx, (img_p, lbl_p) in enumerate(zip(train_img_paths, train_lbl_paths), start=1):
        print(f"  ---> Train {idx}/{len(train_img_paths)}: {img_p.name}")
        img = tifffile.imread(img_p)
        lbl = prepare_labels(tifffile.imread(lbl_p).reshape(-1).astype(np.int64), extra_class)
        h, w, _ = img.shape

        feats_9ch = extract_9ch_features(img, P_global, lambda_ref=0.5)
        X_train_list.append(feats_9ch)
        labels_train_list.append(lbl)
        shapes_train_list.append((h, w))

    print("\n[3/4] Proyectando las 3 escenas de prueba sobre el MISMO P_global fijo...")
    test_loaders = {}
    for scene_name, img_p, lbl_p in TEST_SCENES:
        print(f"  ---> Test: {scene_name}")
        img_t = tifffile.imread(img_p)
        lbl_t = prepare_labels(tifffile.imread(lbl_p).reshape(-1).astype(np.int64), extra_class)
        ht, wt, _ = img_t.shape

        feats_t = extract_9ch_features(img_t, P_global, lambda_ref=0.5)
        _, loader_t = create_dataloader(
            A_initial=feats_t,
            labels=lbl_t,
            image_shape=(ht, wt),
            patch_size=patch_size,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )
        test_loaders[scene_name] = loader_t

    train_dataset, train_loader = create_dataloader(
        A_initial=X_train_list,
        labels=labels_train_list,
        image_shape=shapes_train_list,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )
    print(f"[OK] Preprocesamiento completo en {time.time() - t0:.1f}s | Parches Train (stride=32): {len(train_dataset)}")

    # 3. Instanciar AbundanceTransformer con in_channels=9 (8 abundancias + 1 norma)
    in_channels = n_est + 1
    model = AbundanceTransformer(
        in_channels=in_channels,
        n_endmembers=n_est,
        n_classes=n_classes,
        patch_size=4,
    ).to(device)

    all_train_labels = np.concatenate([lbl.ravel() for lbl in labels_train_list])
    valid_train_labels = all_train_labels[all_train_labels >= 0]
    class_counts = np.bincount(valid_train_labels, minlength=n_classes)
    raw_weights = len(valid_train_labels) / (n_classes * (class_counts.astype(np.float32) + 1e-6))
    smoothed_weights = np.sqrt(raw_weights)
    smoothed_weights = smoothed_weights / smoothed_weights.mean()
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    criterion = get_loss_function(config, class_weights=weights_tensor, device=device)
    endmembers_tensor = torch.from_numpy(P_global.T).float().to(device)

    print(f"\n[4/4] Entrenando AbundanceTransformer (in_channels={in_channels}) por {epochs} épocas...")
    t_train = time.time()
    run_training_pipeline(
        model=model,
        train_loader=train_loader,
        test_loader=test_loaders["201912_9"],
        endmembers_tensor=endmembers_tensor,
        criterion=criterion,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    elapsed_min = (time.time() - t_train) / 60.0

    # Guardar checkpoint y P_global
    ckpt_dir = Path("checkpoints")
    out_dir = Path("results") / f"pilot_pglobal_{epochs}ep"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / f"pilot_pglobal_abundance_{epochs}ep.pth")
    np.save(out_dir / "P_global.npy", P_global)

    # 4. Evaluación final en las 3 escenas reservadas
    print("\n" + "=" * 90)
    print(f"RESULTADOS PILOTO P_GLOBAL + NORMA — ABUNDANCE TRANSFORMER ({elapsed_min:.2f} min)")
    print("=" * 90)
    print(f"{'Escena':<14} | {'OA (8 cls)':>10} | {'OA (C1-7 Dir)':>13} | {'OA (C1-7 Mask)':>14} | {'F1 (8 cls)':>10} | {'F1 (C1-7 Mask)':>14}")
    print("-" * 90)

    logits_all_scenes = []
    targets_all_scenes = []
    csv_rows = []

    for scene_name, loader in test_loaders.items():
        lg, tg = evaluate_scene_logits(model, loader, endmembers_tensor, device)
        logits_all_scenes.append(lg)
        targets_all_scenes.append(tg)
        m = metrics_summary(lg, tg)
        print(
            f"{scene_name:<14} | {m['oa_all']:9.2f}% | {m['oa_c17_dir']:12.2f}% | "
            f"{m['oa_c17_mask']:13.2f}% | {m['f1_all']:9.2f}% | {m['f1_c17_mask']:13.2f}%"
        )
        csv_rows.append([
            scene_name,
            f"{m['oa_all']:.2f}", f"{m['f1_all']:.2f}", f"{m['kappa_all']:.4f}",
            f"{m['oa_c17_dir']:.2f}", f"{m['oa_c17_mask']:.2f}", f"{m['f1_c17_mask']:.2f}", f"{m['kappa_c17_mask']:.4f}",
        ])

    lg_cat = np.concatenate(logits_all_scenes, axis=0)
    tg_cat = np.concatenate(targets_all_scenes, axis=0)
    mg = metrics_summary(lg_cat, tg_cat)

    print("-" * 90)
    print(
        f"{'GLOBAL (3 ESC)':<14} | {mg['oa_all']:9.2f}% | {mg['oa_c17_dir']:12.2f}% | "
        f"{mg['oa_c17_mask']:13.2f}% | {mg['f1_all']:9.2f}% | {mg['f1_c17_mask']:13.2f}%"
    )
    print("=" * 90)

    csv_rows.append([
        "GLOBAL_3_ESCENAS",
        f"{mg['oa_all']:.2f}", f"{mg['f1_all']:.2f}", f"{mg['kappa_all']:.4f}",
        f"{mg['oa_c17_dir']:.2f}", f"{mg['oa_c17_mask']:.2f}", f"{mg['f1_c17_mask']:.2f}", f"{mg['kappa_c17_mask']:.4f}",
    ])

    csv_path = out_dir / f"resultados_piloto_pglobal_{epochs}ep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "Escena", "OA_8clases_%", "MacroF1_8clases_%", "Kappa_8clases",
            "OA_SinC0_Directo_%", "OA_SinC0_Mask_%", "MacroF1_SinC0_Mask_%", "Kappa_SinC0_Mask",
        ])
        writer.writerows(csv_rows)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as jf:
        json.dump({"train_time_min": round(elapsed_min, 2), "global_3_scenes": mg}, jf, indent=2)

    print(f"[OK] Resultados guardados en: {csv_path}")


if __name__ == "__main__":
    main()