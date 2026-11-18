#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
main_multi.py

Descripción:
    Benchmark automatizado multi-arquitectura para clasificación hiperespectral
    basada en desmezclado (EBEAE_S).
    - Usa exactamente la misma lógica de preprocesamiento, desmezclado,
      calibración softmax, dataloaders con rotaciones (augment=True),
      pesos suavizados y run_training_pipeline de main.py.
    - Procesa las 7 escenas de entrenamiento y la escena de prueba UNA sola vez.
    - Entrena secuencialmente las 5 arquitecturas configuradas en create_model:
        1. resunet
        2. abundancetransformer
        3. residualtransformer
        4. crossattention
        5. mambatransformer
    - Emite un anuncio claro de finalización con progreso porcentual (20% c/u).
    - Guarda los checkpoints (.pth), métricas (.json) y tabla consolidada (.csv).
=============================================================================
"""

import os
import csv
import copy
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import load_config
from unmixing import ebeae, nebeae, fcls, ebeae_S
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics
from dataset import create_dataloader
from training import run_training_pipeline
from losses import get_loss_function
from Models.model_builder import create_model


ARCHITECTURES = [
    "resunet",
    "abundance",
    "residual",
    "cross_attention",
    "mamba",
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


def calibrate_abundances_softmax(A_raw, temperature=1.0):
    A_sum = np.sum(A_raw, axis=0, keepdims=True) + 1e-6
    return A_raw / A_sum


def run_unmixing(image, method, n_est, labels=None, percentage=0.20, P_ref=None, lambda_ref=0.1):
    image = np.asarray(image)
    height, width, bands = image.shape
    Y = image.reshape(height * width, bands).T

    if method == "ebeae":
        P, A, S, Yh, Ji = ebeae.ebeae(Y, N=n_est)
    elif method == "nebeae":
        P, A, S, Yh, Ji = nebeae.nebeae(Y, N=n_est)
    elif method == "fcls":
        P, A, S, Yh, Ji = fcls.fcls(Y, N=n_est, percentage=percentage, labels=labels)
    elif method == "ebeae_s":
        P, A, S, Yh, Ji = ebeae_S.ebeae(Yo=Y, N=n_est, P_ref=P_ref, lambda_ref=lambda_ref)
    else:
        raise RuntimeError(f"Método no implementado: {method}")

    return np.asarray(P).copy(), np.asarray(A).copy(), Y


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
        P, A, Y = run_unmixing(
            image, method=method, n_est=n_endmembers, labels=labels, percentage=0.20, P_ref=P_ref, lambda_ref=0.1
        )

    return P, A, Y


def announce_progress(arch_name: str, idx: int, total: int, elapsed_min: float, total_elapsed_sec: float):
    """Imprime un banner claro de finalización sin errores y progreso porcentual (20% por arquitectura)."""
    pct = idx * 20
    blocks_done = idx * 4
    blocks_left = (total - idx) * 4
    bar = "█" * blocks_done + "░" * blocks_left

    avg_sec_per_model = total_elapsed_sec / idx
    eta_min = (avg_sec_per_model * (total - idx)) / 60.0

    print("\n" + "*" * 85)
    print(f"[✓] ARQUITECTURA TERMINADA SIN PROBLEMAS: {arch_name.upper()} ({idx}/{total}) — {elapsed_min:.2f} min")
    print(f"    PROGRESO GLOBAL DEL BENCHMARK: [{bar}] {pct}% COMPLETADO")
    if idx < total:
        print(f"    Tiempo acumulado: {total_elapsed_sec / 60.0:.2f} min | Estimado restante: ~{eta_min:.1f} min")
    else:
        print("    ¡TODAS LAS ARQUITECTURAS HAN FINALIZADO AL 100%!")
    print("*" * 85 + "\n", flush=True)


def run_multi_architecture_benchmark():
    set_seed(42)
    config = load_config()

    unmixing_method = str(config["unmixing"]["method"]).strip().lower()
    n_endmembers = int(config["unmixing"]["n_endmembers"])
    extra_class = bool(config["unmixing"]["extra_class"])
    n_est = n_endmembers + int(extra_class)
    n_classes = int(config["classification"]["n_classes"])

    device_str = str(config["hardware"]["device"])
    device = torch.device(device_str)
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    patch_size = int(config["training"]["patch_size"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])

    max_imgs = config["data"].get("num_train_images", None)
    if max_imgs is None or max_imgs <= 0:
        train_img_list = config["data"]["train_images"]
        train_lbl_list = config["data"]["train_labels"]
    else:
        train_img_list = config["data"]["train_images"][:max_imgs]
        train_lbl_list = config["data"]["train_labels"][:max_imgs]

    train_image_paths = [Path(p) for p in train_img_list]
    train_labels_paths = [Path(p) for p in train_lbl_list]
    test_image_path = Path(config["data"]["test_image"])
    test_labels_path = Path(config["data"].get("test_labels", "Satelital/labels/201912_9.tif"))

    assert len(train_image_paths) == len(train_labels_paths), "Número de imágenes y etiquetas no coincide."

    print("\n" + "=" * 85)
    print("BENCHMARK MULTI-ARQUITECTURA (5 MODELOS) — DESMEZCLADO + CLASIFICACIÓN PROFUNDA")
    print("=" * 85)
    print(f"• Dispositivo       : {device}")
    print(f"• Escenas Train     : {len(train_image_paths)} imágenes (augment=True / rotaciones)")
    print(f"• Escena Test       : {test_image_path.name}")
    print(f"• Desmezclado       : {unmixing_method.upper()} (N_EST={n_est})")
    print(f"• Épocas por modelo : {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print("=" * 85)

    # ============================================================
    # 1. DESMEZCLADO ÚNICO DE LAS ESCENAS DE ENTRENAMIENTO Y PRUEBA
    # ============================================================
    t0_unmix = time.time()
    print(f"\n[1/2] Procesando desmezclado de {len(train_image_paths)} escena(s) de entrenamiento...")

    A_train_aligned_list = []
    labels_train_list = []
    shapes_train_list = []
    P_train_reference = None

    for idx, (img_path, lbl_path) in enumerate(zip(train_image_paths, train_labels_paths)):
        print(f"  ---> Escena {idx + 1}/{len(train_image_paths)}: {img_path.name}")
        img = tifffile.imread(img_path)
        lbls_raw = tifffile.imread(lbl_path).reshape(-1).astype(np.int64)
        lbls = prepare_labels(lbls_raw, extra_class=extra_class)

        h, w, bands = img.shape
        shapes_train_list.append((h, w))
        labels_train_list.append(lbls)

        Y_img = img.reshape(h * w, bands).T.astype(np.float32)
        P0_train, _ = generate_pseudo_gt(Y=Y_img, labels=lbls, N=n_endmembers, extra_class=extra_class)

        P_img, A_img, _ = run_unmixing_adaptive(
            image=img,
            labels=lbls,
            extra_class=extra_class,
            n_endmembers=n_est,
            method=unmixing_method,
            P_ref=P0_train if unmixing_method == "ebeae_s" else None,
        )

        if idx == 0:
            if unmixing_method == "ebeae_s":
                P_train_reference = P_img.copy()
            else:
                _, _, _, train_row_ind, train_col_ind = compute_metrics(P_img, P0_train)
                P_train_aligned_ref = np.zeros_like(P_img)
                for est_idx, gt_class in zip(train_row_ind, train_col_ind):
                    P_train_aligned_ref[:, gt_class] = P_img[:, est_idx]
                P_train_reference = P_train_aligned_ref

        if unmixing_method == "ebeae_s":
            A_aligned_raw = A_img.copy()
        else:
            eps = 1e-12
            norm_P = np.where(np.linalg.norm(P_img, axis=0, keepdims=True) <= eps, 1.0, np.linalg.norm(P_img, axis=0, keepdims=True))
            norm_P0 = np.where(np.linalg.norm(P0_train, axis=0, keepdims=True) <= eps, 1.0, np.linalg.norm(P0_train, axis=0, keepdims=True))
            cos_dist = np.nan_to_num(1.0 - np.clip(np.dot((P_img / norm_P).T, P0_train / norm_P0), -1.0, 1.0), nan=1.0)
            row_ind, col_ind = linear_sum_assignment(cos_dist)
            A_aligned_raw = np.zeros_like(A_img)
            for est_idx, gt_idx in zip(row_ind, col_ind):
                A_aligned_raw[gt_idx, :] = A_img[est_idx, :]

        A_train_aligned_list.append(calibrate_abundances_softmax(A_aligned_raw))

    # Imagen de prueba (TEST)
    print(f"  ---> Procesando imagen de prueba: {test_image_path.name}")
    image_test = tifffile.imread(test_image_path)
    labels_test_raw = tifffile.imread(test_labels_path).reshape(-1).astype(np.int64)
    labels_test = prepare_labels(labels_test_raw, extra_class=extra_class)
    height_test, width_test, _ = image_test.shape

    P_test, A_test, _ = run_unmixing_adaptive(
        image=image_test,
        labels=labels_test,
        extra_class=extra_class,
        n_endmembers=n_est,
        method=unmixing_method,
        P_ref=P_train_reference if unmixing_method == "ebeae_s" else None,
    )

    if unmixing_method == "ebeae_s":
        A_test_aligned_raw = A_test.copy()
    else:
        cos_dist_test = cdist(P_train_reference.T, P_test.T, metric="cosine")
        test_row_ind, test_col_ind = linear_sum_assignment(cos_dist_test)
        A_test_aligned_raw = np.zeros_like(A_test)
        for gt_class, est_idx in zip(test_row_ind, test_col_ind):
            A_test_aligned_raw[gt_class, :] = A_test[est_idx, :]

    A_test_aligned = calibrate_abundances_softmax(A_test_aligned_raw)

    # DataLoaders con rotaciones (augment=True)
    train_dataset, train_loader = create_dataloader(
        A_initial=A_train_aligned_list,
        labels=labels_train_list,
        image_shape=shapes_train_list,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )
    test_dataset, test_loader = create_dataloader(
        A_initial=A_test_aligned,
        labels=labels_test,
        image_shape=(height_test, width_test),
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    endmembers_tensor = torch.from_numpy(P_train_reference.T).float().to(device)

    # Pesos globales suavizados
    all_train_labels = np.concatenate([lbl.ravel() for lbl in labels_train_list])
    valid_train_labels = all_train_labels[all_train_labels >= 0]
    class_counts = np.bincount(valid_train_labels, minlength=n_classes)
    total_valid_pixels = len(valid_train_labels)

    raw_weights = total_valid_pixels / (n_classes * (class_counts.astype(np.float32) + 1e-6))
    smoothed_weights = np.sqrt(raw_weights)
    smoothed_weights = smoothed_weights / smoothed_weights.mean()
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    print(
        f"[OK] Desmezclado y DataLoaders listos en {time.time() - t0_unmix:.1f}s | "
        f"Parches Train: {len(train_dataset)} | Parches Test: {len(test_dataset)}"
    )

    # ============================================================
    # 2. ENTRENAMIENTO SECUENCIAL DE LAS 5 ARQUITECTURAS
    # ============================================================
    ckpt_dir = Path("checkpoints")
    out_dir = Path("results") / f"benchmark_{epochs}ep"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total_start = time.time()
    total_models = len(ARCHITECTURES)

    print("\n[2/2] Iniciando entrenamiento secuencial de las 5 arquitecturas...\n")
    for idx, arch_name in enumerate(ARCHITECTURES, start=1):
        print("=" * 85)
        print(f"MODELO [{idx}/{total_models}]: {arch_name.upper()} ({epochs} épocas) — Avance previo: {(idx - 1) * 20}%")
        print("=" * 85)

        t_model_start = time.time()
        try:
            # Fijar semilla por modelo para comparación justa entre arquitecturas
            set_seed(42)

            cfg_model = copy.deepcopy(config)
            cfg_model["classification"]["architecture"] = arch_name

            model = create_model(cfg_model, device=device)
            criterion = get_loss_function(cfg_model, class_weights=weights_tensor, device=device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"• Parámetros entrenables: {n_params:,}")

            oa, f1, kappa, cm = run_training_pipeline(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                endmembers_tensor=endmembers_tensor,
                criterion=criterion,
                epochs=epochs,
                lr=learning_rate,
                device=device,
            )

            elapsed_min = (time.time() - t_model_start) / 60.0

            # Guardar checkpoint del modelo entrenado
            ckpt_path = ckpt_dir / f"multi_{arch_name}_{epochs}ep.pth"
            torch.save(model.state_dict(), ckpt_path)

            cm_arr = np.asarray(cm)
            # Cálculo de F1 por clase desde la matriz de confusión
            per_class_f1 = []
            for c in range(n_classes):
                tp = float(cm_arr[c, c])
                fp = float(cm_arr[:, c].sum() - tp)
                fn = float(cm_arr[c, :].sum() - tp)
                denom = 2 * tp + fp + fn
                per_class_f1.append(round((2 * tp / denom) if denom > 0 else 0.0, 4))

            print(
                f"\n>>> RESULTADOS {arch_name.upper()}: "
                f"OA = {oa * 100:.2f}% | "
                f"Macro F1 = {f1 * 100:.2f}% | "
                f"Kappa = {kappa:.4f}"
            )
            print(f">>> F1 por clase: {per_class_f1}")

            full_report = {
                "architecture": arch_name,
                "epochs": epochs,
                "n_train_scenes": len(train_image_paths),
                "trainable_params": n_params,
                "train_time_minutes": round(elapsed_min, 2),
                "oa": float(oa),
                "macro_f1": float(f1),
                "kappa": float(kappa),
                "per_class_f1": per_class_f1,
                "confusion_matrix": cm_arr.tolist(),
            }
            json_path = out_dir / f"{arch_name}_metrics.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(full_report, jf, indent=2)

            summary_rows.append([
                arch_name,
                n_params,
                f"{elapsed_min:.2f}",
                f"{oa * 100:.2f}",
                f"{f1 * 100:.2f}",
                f"{kappa:.4f}",
                *[f"{v * 100:.2f}" for v in per_class_f1],
            ])

            # Anuncio de finalización sin problemas y avance del 20%
            announce_progress(
                arch_name=arch_name,
                idx=idx,
                total=total_models,
                elapsed_min=elapsed_min,
                total_elapsed_sec=time.time() - t_total_start,
            )

        except Exception as exc:
            elapsed_min = (time.time() - t_model_start) / 60.0
            print(f"\n[ERROR] Falló {arch_name} tras {elapsed_min:.2f} min: {exc}")
            print(f"        Avance alcanzado hasta este punto: {idx * 20}% procesado.\n", flush=True)
            summary_rows.append([arch_name, "ERROR", f"{elapsed_min:.2f}", "N/A", "N/A", "N/A"])

    # ============================================================
    # 3. EXPORTAR TABLA CONSOLIDADA
    # ============================================================
    csv_path = out_dir / f"resumen_5_modelos_{epochs}ep.csv"
    header = [
        "Arquitectura",
        "Parametros",
        "Tiempo_min",
        "OA_%",
        "Macro_F1_%",
        "Kappa",
        *[f"F1_C{c}_%" for c in range(n_classes)],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(header)
        writer.writerows(summary_rows)

    total_hours = (time.time() - t_total_start) / 3600.0
    print("=" * 85)
    print(f"BENCHMARK FINALIZADO EN {total_hours:.2f} HORAS (100% COMPLETADO)")
    print(f"Tabla consolidada guardada en: {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_multi_architecture_benchmark()