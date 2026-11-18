#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Evaluation/spectral_main.py

Descripción:
    Script principal para el Análisis de Respuesta Espectral, Dispersión de
    Endmembers y Evaluación de Métricas de Desmezclado (SAM, Cosine, Euc)
    respecto a las firmas de referencia Ground Truth (Pseudo-GT).

Uso:
    python -m Evaluation.spectral_main
    o desde la carpeta Evaluation:
    python spectral_main.py
=============================================================================
"""

from pathlib import Path
import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

# Configuración y Unmixing
from config import load_config
from unmixing import ebeae, nebeae, fcls, ebeae_S
from unmixing.generate_pseudo_gt import generate_pseudo_gt
from Evaluation.compute_metrics import compute_metrics
from Evaluation.spectral_signatures import spectral_signatures


def prepare_labels(labels_raw, extra_class):
    if not extra_class:
        labels_remapped = np.full_like(labels_raw, fill_value=-100, dtype=np.int64)
        for old_class in range(1, 8):
            labels_remapped[labels_raw == old_class] = old_class - 1
        return labels_remapped
    return labels_raw


def run_spectral_analysis():
    config = load_config()

    UNMIXING_METHOD = str(config["unmixing"]["method"]).strip().lower()
    N = int(config["unmixing"]["n_endmembers"])
    EXTRA_CLASS = bool(config["unmixing"]["extra_class"])
    N_EST = N + int(EXTRA_CLASS)
    N_CLASSES = int(config["classification"]["n_classes"])

    # Rutas de datos
    train_img_path = Path(config["data"]["train_images"][0])
    train_lbl_path = Path(config["data"]["train_labels"][0])

    print("\n" + "=" * 70)
    print(f"ANÁLISIS ESPECTRAL Y MÉTRICAS DE DESMEZCLADO ({UNMIXING_METHOD.upper()})")
    print(f"Escena evaluada: {train_img_path.name}")
    print("=" * 70)

    # Cargar datos espectrales y etiquetas
    img = tifffile.imread(train_img_path)
    lbls_raw = tifffile.imread(train_lbl_path).reshape(-1).astype(np.int64)
    lbls = prepare_labels(lbls_raw, extra_class=EXTRA_CLASS)

    h, w, bands = img.shape
    Y_img = img.reshape(h * w, bands).T.astype(np.float32)

    # 1. Firmas de Referencia Pseudo-GT
    P0_train, present_idx = generate_pseudo_gt(
        Y=Y_img, 
        labels=lbls, 
        N=N, 
        extra_class=EXTRA_CLASS
    )

    # 2. Algoritmo de Desmezclado
    if UNMIXING_METHOD == "ebeae_s":
        P_img, A_img, _, _, _ = ebeae_S.ebeae(Yo=Y_img, N=N_EST, P_ref=P0_train, lambda_ref=0.1)
    elif UNMIXING_METHOD == "ebeae":
        P_img, A_img, _, _, _ = ebeae.ebeae(Y_img, N=N_EST)
    elif UNMIXING_METHOD == "nebeae":
        P_img, A_img, _, _, _ = nebeae.nebeae(Y_img, N=N_EST)
    elif UNMIXING_METHOD == "fcls":
        P_img, A_img, _, _, _ = fcls.fcls(Y_img, N=N_EST, percentage=0.20, labels=lbls)
    else:
        raise RuntimeError(f"Método de desmezclado no soportado: '{UNMIXING_METHOD}'")

    # 3. Visualización de Firmas Espectrales y Dispersión
    print("\n[INFO] Generando curvas espectrales y gráfico de dispersión...")
    spectral_signatures(
        Y_img=Y_img,
        labels=lbls,
        P_ref=P_img,
        n_classes=N_CLASSES
    )

    # 4. Cálculo de Métricas (Manejando alineación directa 1:1 para EBEAE_S)
    print("\n" + "=" * 70)
    print(f"EVALUACIÓN DE MÉTRICAS ESPECTRALES VS PSEUDO-GT ({UNMIXING_METHOD.upper()})")
    print("=" * 70)

    E_cos, E_euc, E_sam, row_ind, col_ind = compute_metrics(
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


if __name__ == "__main__":
    run_spectral_analysis()