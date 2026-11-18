#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
eval_3_test_scenes.py

Descripción:
    Evalúa los 5 modelos ya entrenados (checkpoints/multi_*_150ep.pth) sobre
    las 3 escenas reservadas de prueba (201912_8, 201912_9, 201912_10),
    reportando métricas por escena y globales (3 escenas combinadas):
      - Con Clase 0 (todas las 8 clases)
      - Sin Clase 0 (Clases 1 a 7, excluyendo cobertura mixta/fondo)

Uso:
    python eval_3_test_scenes.py
=============================================================================
"""

import csv
import copy
import inspect
from pathlib import Path
import numpy as np
import torch
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config import load_config
from unmixing import ebeae, nebeae, fcls, ebeae_S
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics
from dataset import create_dataloader
from Models.model_builder import create_model


ARCHITECTURES = [
    "resunet",
    "abundance",
    "residual",
    "cross_attention",
    "mamba",
]

TEST_SCENES = [
    ("201912_8", Path("Satelital/images/201912_8.tif"), Path("Satelital/labels/201912_8.tif")),
    ("201912_9", Path("Satelital/images/201912_9.tif"), Path("Satelital/labels/201912_9.tif")),
    ("201912_10", Path("Satelital/images/201912_10.tif"), Path("Satelital/labels/201912_10.tif")),
]


def prepare_labels(labels_raw, extra_class):
    if not extra_class:
        labels_remapped = np.full_like(labels_raw, fill_value=-100, dtype=np.int64)
        for old_class in range(1, 8):
            labels_remapped[labels_raw == old_class] = old_class - 1
        return labels_remapped
    return labels_raw


def calibrate_abundances_softmax(A_raw):
    A_sum = np.sum(A_raw, axis=0, keepdims=True) + 1e-6
    return A_raw / A_sum


def run_unmixing_adaptive(image, labels, extra_class, n_endmembers, method, P_ref=None):
    height, width, bands = image.shape
    Y = image.reshape(-1, bands).T.astype(np.float32)

    if not extra_class:
        valid_mask = labels >= 0
        Y_valid = Y[:, valid_mask]
        labels_valid = labels[valid_mask]

        if method == "ebeae":
            P, A_valid, _, _, _ = ebeae.ebeae(Y_valid, N=n_endmembers)
        elif method == "nebeae":
            P, A_valid, _, _, _ = nebeae.nebeae(Y_valid, N=n_endmembers)
        elif method == "fcls":
            P, A_valid, _, _, _ = fcls.fcls(Y_valid, N=n_endmembers, percentage=0.20, labels=labels_valid)
        elif method == "ebeae_s":
            P, A_valid, _, _, _ = ebeae_S.ebeae(Yo=Y_valid, N=n_endmembers, P_ref=P_ref, lambda_ref=0.1)
        else:
            raise RuntimeError(f"Método no implementado: {method}")

        A = np.zeros((n_endmembers, height * width), dtype=np.float32)
        A[:, valid_mask] = A_valid
    else:
        if method == "ebeae":
            P, A, _, _, _ = ebeae.ebeae(Y, N=n_endmembers)
        elif method == "nebeae":
            P, A, _, _, _ = nebeae.nebeae(Y, N=n_endmembers)
        elif method == "fcls":
            P, A, _, _, _ = fcls.fcls(Y, N=n_endmembers, percentage=0.20, labels=labels)
        elif method == "ebeae_s":
            P, A, _, _, _ = ebeae_S.ebeae(Yo=Y, N=n_endmembers, P_ref=P_ref, lambda_ref=0.1)
        else:
            raise RuntimeError(f"Método no implementado: {method}")

    return np.asarray(P).copy(), np.asarray(A).copy()


def _forward_pass(model, patches, endmembers_tensor):
    sig = inspect.signature(model.forward)
    if len(sig.parameters) >= 2:
        return model(patches, endmembers_tensor)
    return model(patches)


def get_logits_and_targets(model, loader, endmembers_tensor, device):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            logits = _forward_pass(model, patches, endmembers_tensor)
            # Reordenar (B, C, H, W) -> (B*H*W, C)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu().numpy()
            targets_flat = labels.numpy().reshape(-1)

            valid = targets_flat >= 0
            all_logits.append(logits_flat[valid])
            all_targets.append(targets_flat[valid])

    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics_with_and_without_c0(logits: np.ndarray, y_true: np.ndarray):
    """Calcula métricas en 3 modos:
       1) 8 clases completas (incluyendo Clase 0)
       2) Sin Clase 0 (píxeles y_true >= 1, predicción directa de 8 clases)
       3) Sin Clase 0 restringiendo argmax a salidas 1..7 (simulando máscara de Clase 0)
    """
    # 1. Con Clase 0 (todas las clases)
    y_pred_all = np.argmax(logits, axis=1)
    labels_all = np.sort(np.unique(y_true))
    oa_all = float(accuracy_score(y_true, y_pred_all))
    f1_all = float(f1_score(y_true, y_pred_all, labels=labels_all, average="macro", zero_division=0))
    kappa_all = float(cohen_kappa_score(y_true, y_pred_all, labels=labels_all))

    # 2 & 3. Sin Clase 0 (solo píxeles pertenecientes a Clases 1..7)
    mask_no_c0 = y_true > 0
    y_true_no_c0 = y_true[mask_no_c0]
    labels_no_c0 = np.sort(np.unique(y_true_no_c0))

    # Directo (puede predecir 0 por error)
    y_pred_no_c0_raw = y_pred_all[mask_no_c0]
    oa_no_c0_raw = float(accuracy_score(y_true_no_c0, y_pred_no_c0_raw))
    f1_no_c0_raw = float(f1_score(y_true_no_c0, y_pred_no_c0_raw, labels=labels_no_c0, average="macro", zero_division=0))
    kappa_no_c0_raw = float(cohen_kappa_score(y_true_no_c0, y_pred_no_c0_raw, labels=labels_no_c0))

    # Enmascarando la salida 0 (argmax sobre canales 1..7)
    y_pred_no_c0_masked = np.argmax(logits[mask_no_c0, 1:], axis=1) + 1
    oa_no_c0_masked = float(accuracy_score(y_true_no_c0, y_pred_no_c0_masked))
    f1_no_c0_masked = float(f1_score(y_true_no_c0, y_pred_no_c0_masked, labels=labels_no_c0, average="macro", zero_division=0))
    kappa_no_c0_masked = float(cohen_kappa_score(y_true_no_c0, y_pred_no_c0_masked, labels=labels_no_c0))

    return {
        "oa_all": oa_all * 100.0,
        "f1_all": f1_all * 100.0,
        "kappa_all": kappa_all,
        "oa_no_c0_raw": oa_no_c0_raw * 100.0,
        "f1_no_c0_raw": f1_no_c0_raw * 100.0,
        "kappa_no_c0_raw": kappa_no_c0_raw,
        "oa_no_c0_masked": oa_no_c0_masked * 100.0,
        "f1_no_c0_masked": f1_no_c0_masked * 100.0,
        "kappa_no_c0_masked": kappa_no_c0_masked,
    }


def main():
    config = load_config()
    unmixing_method = str(config["unmixing"]["method"]).strip().lower()
    n_endmembers = int(config["unmixing"]["n_endmembers"])
    extra_class = bool(config["unmixing"]["extra_class"])
    n_est = n_endmembers + int(extra_class)
    n_classes = int(config["classification"]["n_classes"])
    patch_size = int(config["training"]["patch_size"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])

    device = torch.device(str(config["hardware"]["device"]))
    ckpt_dir = Path("checkpoints")
    out_dir = Path("results") / f"benchmark_{epochs}ep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("EVALUACIÓN MULTI-ESCENA EN TEST (201912_8, 201912_9, 201912_10) — CON Y SIN CLASE 0")
    print("=" * 90)

    # 1. Reconstruir P_train_reference desde la primera imagen de entrenamiento (idéntico al entrenamiento)
    ref_img_path = Path(config["data"]["train_images"][0])
    ref_lbl_path = Path(config["data"]["train_labels"][0])
    print(f"[1/3] Obteniendo espacio de endmembers de referencia desde {ref_img_path.name}...")

    img_ref = tifffile.imread(ref_img_path)
    lbl_ref = prepare_labels(tifffile.imread(ref_lbl_path).reshape(-1).astype(np.int64), extra_class)
    h_r, w_r, b_r = img_ref.shape
    Y_ref = img_ref.reshape(h_r * w_r, b_r).T.astype(np.float32)
    P0_ref, _ = generate_pseudo_gt(Y=Y_ref, labels=lbl_ref, N=n_endmembers, extra_class=extra_class)
    P_img_ref, _ = run_unmixing_adaptive(
        img_ref, lbl_ref, extra_class, n_est, unmixing_method, P_ref=P0_ref if unmixing_method == "ebeae_s" else None
    )

    if unmixing_method == "ebeae_s":
        P_train_reference = P_img_ref.copy()
    else:
        _, _, _, r_ind, c_ind = compute_metrics(P_img_ref, P0_ref)
        P_train_reference = np.zeros_like(P_img_ref)
        for est_i, gt_c in zip(r_ind, c_ind):
            P_train_reference[:, gt_c] = P_img_ref[:, est_i]

    endmembers_tensor = torch.from_numpy(P_train_reference.T).float().to(device)

    # 2. Preparar los DataLoaders de las 3 escenas de prueba
    print("\n[2/3] Desmezclando y analizando distribución de clases en las 3 escenas de prueba...")
    scene_loaders = {}
    for scene_name, img_p, lbl_p in TEST_SCENES:
        if not img_p.exists() or not lbl_p.exists():
            print(f"  [AVISO] No se encontró {img_p} o {lbl_p}, se omite.")
            continue

        img_t = tifffile.imread(img_p)
        lbl_t = prepare_labels(tifffile.imread(lbl_p).reshape(-1).astype(np.int64), extra_class)
        ht, wt, _ = img_t.shape

        counts = np.bincount(lbl_t[lbl_t >= 0], minlength=n_classes)
        pct_c0 = (counts[0] / counts.sum()) * 100.0
        missing = [c for c in range(n_classes) if counts[c] == 0]
        print(f"  • {scene_name}: Píxeles por clase = {counts.tolist()} | Clase 0 = {pct_c0:.1f}% | Ausentes = {missing}")

        P_t, A_t = run_unmixing_adaptive(
            img_t, lbl_t, extra_class, n_est, unmixing_method, P_ref=P_train_reference if unmixing_method == "ebeae_s" else None
        )
        if unmixing_method == "ebeae_s":
            A_t_aligned = calibrate_abundances_softmax(A_t.copy())
        else:
            r_t, c_t = linear_sum_assignment(cdist(P_train_reference.T, P_t.T, metric="cosine"))
            A_raw_t = np.zeros_like(A_t)
            for gt_c, est_i in zip(r_t, c_t):
                A_raw_t[gt_c, :] = A_t[est_i, :]
            A_t_aligned = calibrate_abundances_softmax(A_raw_t)

        _, loader_t = create_dataloader(
            A_initial=A_t_aligned,
            labels=lbl_t,
            image_shape=(ht, wt),
            patch_size=patch_size,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )
        scene_loaders[scene_name] = loader_t

    # 3. Evaluar cada checkpoint en las 3 escenas
    print("\n[3/3] Evaluando checkpoints guardados en las 3 escenas...\n")
    csv_rows = []

    for arch in ARCHITECTURES:
        ckpt_path = ckpt_dir / f"multi_{arch}_{epochs}ep.pth"
        if not ckpt_path.exists():
            print(f"[AVISO] No se encontró checkpoint: {ckpt_path}")
            continue

        cfg_m = copy.deepcopy(config)
        cfg_m["classification"]["architecture"] = arch
        model = create_model(cfg_m, device=device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        print("=" * 90)
        print(f"ARQUITECTURA: {arch.upper()}")
        print("=" * 90)
        print(f"{'Escena':<14} | {'OA (8 cls)':>10} | {'OA (C1-7 Dir)':>13} | {'OA (C1-7 Mask)':>14} | {'F1 (8 cls)':>10} | {'F1 (C1-7 Mask)':>14}")
        print("-" * 90)

        logits_global = []
        targets_global = []

        for scene_name, loader in scene_loaders.items():
            logits_s, targets_s = get_logits_and_targets(model, loader, endmembers_tensor, device)
            logits_global.append(logits_s)
            targets_global.append(targets_s)

            m = compute_metrics_with_and_without_c0(logits_s, targets_s)
            print(
                f"{scene_name:<14} | {m['oa_all']:9.2f}% | {m['oa_no_c0_raw']:12.2f}% | "
                f"{m['oa_no_c0_masked']:13.2f}% | {m['f1_all']:9.2f}% | {m['f1_no_c0_masked']:13.2f}%"
            )
            csv_rows.append([
                arch, scene_name,
                f"{m['oa_all']:.2f}", f"{m['f1_all']:.2f}", f"{m['kappa_all']:.4f}",
                f"{m['oa_no_c0_raw']:.2f}", f"{m['f1_no_c0_raw']:.2f}",
                f"{m['oa_no_c0_masked']:.2f}", f"{m['f1_no_c0_masked']:.2f}", f"{m['kappa_no_c0_masked']:.4f}",
            ])

        # Global acumulado (3 escenas juntas)
        logits_cat = np.concatenate(logits_global, axis=0)
        targets_cat = np.concatenate(targets_global, axis=0)
        mg = compute_metrics_with_and_without_c0(logits_cat, targets_cat)
        print("-" * 90)
        print(
            f"{'GLOBAL (3 ESC)':<14} | {mg['oa_all']:9.2f}% | {mg['oa_no_c0_raw']:12.2f}% | "
            f"{mg['oa_no_c0_masked']:13.2f}% | {mg['f1_all']:9.2f}% | {mg['f1_no_c0_masked']:13.2f}%"
        )
        print("=" * 90 + "\n")

        csv_rows.append([
            arch, "GLOBAL_3_ESCENAS",
            f"{mg['oa_all']:.2f}", f"{mg['f1_all']:.2f}", f"{mg['kappa_all']:.4f}",
            f"{mg['oa_no_c0_raw']:.2f}", f"{mg['f1_no_c0_raw']:.2f}",
            f"{mg['oa_no_c0_masked']:.2f}", f"{mg['f1_no_c0_masked']:.2f}", f"{mg['kappa_no_c0_masked']:.4f}",
        ])

    csv_path = out_dir / f"evaluacion_3_escenas_test_{epochs}ep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "Arquitectura", "Escena",
            "OA_8clases_%", "MacroF1_8clases_%", "Kappa_8clases",
            "OA_SinC0_Directo_%", "MacroF1_SinC0_Directo_%",
            "OA_SinC0_Enmascarado_%", "MacroF1_SinC0_Enmascarado_%", "Kappa_SinC0_Enmascarado",
        ])
        writer.writerows(csv_rows)

    print(f"[OK] Tabla completa guardada en: {csv_path}")


if __name__ == "__main__":
    main()