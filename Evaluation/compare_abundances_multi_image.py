#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Evaluation/inter_scene_consistency_analysis.py

Descripción:
    Análisis unificado de consistencia inter-escena sobre las imágenes de
    entrenamiento (train_images), manteniendo test_image completamente aislada.
    Evalúa en una sola pasada:
      1. Dominio Espectral (Y_img): Cosine Similarity y SAM (rad/grados).
      2. Dominio de Abundancias (A_img): Cosine Similarity, JSD y TVD.
    Genera tablas en consola, exporta los resultados a CSV y guarda ambas
    figuras superpuestas por clase en Evaluation/results/.

Uso:
    python -m Evaluation.inter_scene_consistency_analysis
=============================================================================
"""

import csv
from pathlib import Path
from itertools import combinations
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from config import load_config
from unmixing import ebeae, nebeae, fcls, ebeae_S
from unmixing.generate_pseudo_gt import generate_pseudo_gt


def prepare_labels(labels_raw, extra_class):
    if not extra_class:
        labels_remapped = np.full_like(labels_raw, fill_value=-100, dtype=np.int64)
        for old_class in range(1, 8):
            labels_remapped[labels_raw == old_class] = old_class - 1
        return labels_remapped
    return labels_raw


def collect_scene_paths(config):
    """Recopila únicamente las escenas de entrenamiento (train_images) para evitar fuga de datos."""
    scenes = []
    train_imgs = config["data"].get("train_images", [])
    train_lbls = config["data"].get("train_labels", [])
    for idx, (img_p, lbl_p) in enumerate(zip(train_imgs, train_lbls), start=1):
        p_img = Path(img_p)
        scenes.append((f"Train {idx} ({p_img.stem})", p_img, Path(lbl_p)))
    return scenes


def extract_abundances(Y_img, lbls, method, N, extra_class):
    """Ejecuta el desmezclado configurado y devuelve matriz A de forma (N_EST, pixeles)."""
    N_EST = N + int(extra_class)
    P0, _ = generate_pseudo_gt(Y=Y_img, labels=lbls, N=N, extra_class=extra_class)

    if method == "ebeae_s":
        _, A_img, _, _, _ = ebeae_S.ebeae(Yo=Y_img, N=N_EST, P_ref=P0, lambda_ref=0.1)
    elif method == "ebeae":
        _, A_img, _, _, _ = ebeae.ebeae(Y_img, N=N_EST)
    elif method == "nebeae":
        _, A_img, _, _, _ = nebeae.nebeae(Y_img, N=N_EST)
    elif method == "fcls":
        _, A_img, _, _, _ = fcls.fcls(Y_img, N=N_EST, percentage=0.20, labels=lbls)
    else:
        raise RuntimeError(f"Método de desmezclado no soportado: '{method}'")

    return A_img


def compute_and_export_tables(spec_results, abund_results, n_classes, out_dir, method):
    """Imprime y exporta a CSV las tablas comparativas de Respuesta Espectral y Abundancias."""
    scene_names = list(spec_results.keys())

    # =========================================================================
    # TABLA 1: CONSISTENCIA DE RESPUESTA ESPECTRAL (Y_img)
    # =========================================================================
    print("\n" + "=" * 82)
    print("1. CONSISTENCIA INTER-ESCENA: RESPUESTA ESPECTRAL CRUDA (Y_img)")
    print("=" * 82)
    print(f"{'Clase':<8} | {'Escenas':<8} | {'Cosine Sim (↑)':<16} | {'SAM rad (↓)':<14} | {'SAM grados (↓)':<14}")
    print("-" * 82)

    spec_rows = []
    all_s_cos, all_s_sam = [], []

    for c in range(n_classes):
        valid_vecs = [spec_results[s][c][0] for s in scene_names if spec_results[s][c] is not None]
        n_sc = len(valid_vecs)
        if n_sc < 2:
            print(f"Clase {c:<2} | {n_sc:<8} | {'N/A (presente en < 2 escenas)':<48}")
            spec_rows.append([f"Clase {c}", n_sc, "N/A", "N/A", "N/A"])
            continue

        cos_l, sam_l = [], []
        for v1, v2 in combinations(valid_vecs, 2):
            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            sam_rad = float(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
            cos_l.append(cos_sim)
            sam_l.append(sam_rad)

        m_cos, m_sam = float(np.mean(cos_l)), float(np.mean(sam_l))
        m_deg = float(np.degrees(m_sam))
        all_s_cos.append(m_cos)
        all_s_sam.append(m_sam)

        print(f"Clase {c:<2} | {n_sc:<8} | {m_cos:<16.4f} | {m_sam:<14.4f} | {m_deg:<13.2f}°")
        spec_rows.append([f"Clase {c}", n_sc, f"{m_cos:.4f}", f"{m_sam:.4f}", f"{m_deg:.2f}"])

    print("-" * 82)
    if all_s_cos:
        g_cos, g_sam = float(np.mean(all_s_cos)), float(np.mean(all_s_sam))
        g_deg = float(np.degrees(g_sam))
        print(f"{'GLOBAL':<8} | {'-':<8} | {g_cos:<16.4f} | {g_sam:<14.4f} | {g_deg:<13.2f}°")
        spec_rows.append(["GLOBAL", "-", f"{g_cos:.4f}", f"{g_sam:.4f}", f"{g_deg:.2f}"])
    print("=" * 82)

    spec_csv = out_dir / "1_spectral_consistency_metrics.csv"
    with open(spec_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clase", "Escenas", "Cosine_Sim", "SAM_rad", "SAM_deg"])
        writer.writerows(spec_rows)

    # =========================================================================
    # TABLA 2: CONSISTENCIA DE PERFILES DE ABUNDANCIA (A_img)
    # =========================================================================
    print("\n" + "=" * 82)
    print(f"2. CONSISTENCIA INTER-ESCENA: PERFILES DE ABUNDANCIA ({method.upper()})")
    print("=" * 82)
    print(f"{'Clase':<8} | {'Escenas':<8} | {'Cosine Sim (↑)':<16} | {'JSD (↓)':<14} | {'TVD / L1 (↓)':<14}")
    print("-" * 82)

    abund_rows = []
    all_a_cos, all_a_jsd, all_a_tvd = [], [], []

    for c in range(n_classes):
        valid_vecs = [abund_results[s][c][0] for s in scene_names if abund_results[s][c] is not None]
        n_sc = len(valid_vecs)
        if n_sc < 2:
            print(f"Clase {c:<2} | {n_sc:<8} | {'N/A (presente en < 2 escenas)':<48}")
            abund_rows.append([f"Clase {c}", n_sc, "N/A", "N/A", "N/A"])
            continue

        cos_l, jsd_l, tvd_l = [], [], []
        for v1, v2 in combinations(valid_vecs, 2):
            p = np.clip(v1, 1e-12, None); p = p / p.sum()
            q = np.clip(v2, 1e-12, None); q = q / q.sum()

            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            jsd_val = float(jensenshannon(p, q, base=2) ** 2)
            tvd_val = float(0.5 * np.sum(np.abs(p - q)))

            cos_l.append(cos_sim)
            jsd_l.append(jsd_val)
            tvd_l.append(tvd_val)

        m_cos, m_jsd, m_tvd = float(np.mean(cos_l)), float(np.mean(jsd_l)), float(np.mean(tvd_l))
        all_a_cos.append(m_cos)
        all_a_jsd.append(m_jsd)
        all_a_tvd.append(m_tvd)

        print(f"Clase {c:<2} | {n_sc:<8} | {m_cos:<16.4f} | {m_jsd:<14.4f} | {m_tvd:<14.4f}")
        abund_rows.append([f"Clase {c}", n_sc, f"{m_cos:.4f}", f"{m_jsd:.4f}", f"{m_tvd:.4f}"])

    print("-" * 82)
    if all_a_cos:
        g_cos, g_jsd, g_tvd = float(np.mean(all_a_cos)), float(np.mean(all_a_jsd)), float(np.mean(all_a_tvd))
        print(f"{'GLOBAL':<8} | {'-':<8} | {g_cos:<16.4f} | {g_jsd:<14.4f} | {g_tvd:<14.4f}")
        abund_rows.append(["GLOBAL", "-", f"{g_cos:.4f}", f"{g_jsd:.4f}", f"{g_tvd:.4f}"])
    print("=" * 82 + "\n")

    abund_csv = out_dir / f"2_abundance_consistency_metrics_{method}.csv"
    with open(abund_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clase", "Escenas", "Cosine_Sim", "JSD", "TVD"])
        writer.writerows(abund_rows)

    print(f"[OK] Tablas CSV guardadas en:\n  • {spec_csv}\n  • {abund_csv}")


def plot_multi_scene_grid(
    results_per_scene,
    n_classes,
    x_values,
    x_label,
    y_label,
    title,
    out_path,
    x_tick_labels=None,
    y_limits=None,
    show_markers=False
):
    """Genera una cuadrícula superpuesta por clase para espectros o abundancias."""
    cols = 4
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.2 * rows), sharey=True)
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for c in range(n_classes):
        ax = axes[c]
        plotted_any = False

        for s_idx, (scene_label, class_stats) in enumerate(results_per_scene.items()):
            stats = class_stats[c]
            if stats is None:
                continue

            mean_vec, std_vec, count = stats
            color = cmap(s_idx % 10)

            plot_kwargs = dict(
                linewidth=1.9,
                color=color,
                label=f"{scene_label} (n={count})"
            )
            if show_markers:
                plot_kwargs["marker"] = "o"
                plot_kwargs["markersize"] = 5

            ax.plot(x_values, mean_vec, **plot_kwargs)

            #lower = mean_vec - std_vec
            #if y_limits is not None:
            #upper = mean_vec + std_vec
            #    lower = np.clip(lower, y_limits[0], y_limits[1])
            #    upper = np.clip(upper, y_limits[0], y_limits[1])

            #ax.fill_between(x_values, lower, upper, color=color, alpha=0.14)
            plotted_any = True

        ax.set_title(f"Clase {c}", fontsize=11, fontweight="bold")
        if x_tick_labels is not None:
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_tick_labels, fontsize=9)
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])

        ax.grid(True, linestyle="--", alpha=0.45)

        if c % cols == 0:
            ax.set_ylabel(y_label, fontsize=10)
        if c >= (rows - 1) * cols:
            ax.set_xlabel(x_label, fontsize=9)

        if plotted_any:
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "Clase ausente en todas las escenas",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")

    for idx in range(n_classes, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Gráfica guardada en: {out_path}")


def run_unified_consistency_analysis():
    config = load_config()

    method = str(config["unmixing"]["method"]).strip().lower()
    N = int(config["unmixing"]["n_endmembers"])
    extra_class = bool(config["unmixing"]["extra_class"])
    n_classes = int(config["classification"]["n_classes"])
    n_est = N + int(extra_class)

    scenes = collect_scene_paths(config)
    if not scenes:
        raise RuntimeError("No se encontraron rutas en data.train_images en config.yaml.")

    print("\n" + "=" * 82)
    print(f"ANÁLISIS INTEGRAL DE CONSISTENCIA INTER-ESCENA (ESPECTROS + ABUNDANCIAS {method.upper()})")
    print(f"Escenas de entrenamiento a procesar: {len(scenes)}")
    print("=" * 82)

    spec_results = {}
    abund_results = {}
    n_bands = None

    for label, img_path, lbl_path in scenes:
        print(f"-> Procesando {label}...")
        img = tifffile.imread(img_path)
        lbls_raw = tifffile.imread(lbl_path).reshape(-1).astype(np.int64)
        lbls = prepare_labels(lbls_raw, extra_class=extra_class)

        h, w, bands = img.shape
        n_bands = bands
        Y_img = img.reshape(h * w, bands).T.astype(np.float32)  # (bands, H*W)
        A_img = extract_abundances(Y_img, lbls, method, N, extra_class)  # (n_est, H*W)

        s_stats = {}
        a_stats = {}
        for c in range(n_classes):
            mask = (lbls == c)
            count = int(np.sum(mask))
            if count > 0:
                Y_c = Y_img[:, mask]
                s_stats[c] = (np.mean(Y_c, axis=1), np.std(Y_c, axis=1), count)

                A_c = A_img[:, mask]
                a_stats[c] = (np.mean(A_c, axis=1), np.std(A_c, axis=1), count)
            else:
                s_stats[c] = None
                a_stats[c] = None

        spec_results[label] = s_stats
        abund_results[label] = a_stats

    out_dir = Path("Evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Imprimir y exportar ambas tablas comparativas
    compute_and_export_tables(spec_results, abund_results, n_classes, out_dir, method)

    # 2. Gráfica 1: Respuesta Espectral Cruda (Y_img)
    plot_multi_scene_grid(
        results_per_scene=spec_results,
        n_classes=n_classes,
        x_values=np.arange(n_bands),
        x_label="Banda Espectral",
        y_label="Reflectancia / Intensidad",
        title="1. Consistencia Inter-Escena de Respuesta Espectral por Clase (Y_img)",
        out_path=out_dir / "1_spectral_consistency_multi_image.png",
        x_tick_labels=None,
        y_limits=None,
        show_markers=False
    )

    # 3. Gráfica 2: Perfiles de Abundancia (A_img)
    endmember_idx = np.arange(n_est)
    plot_multi_scene_grid(
        results_per_scene=abund_results,
        n_classes=n_classes,
        x_values=endmember_idx,
        x_label="Canal de Endmember",
        y_label="Abundancia Promedio",
        title=f"2. Consistencia Inter-Escena de Perfiles de Abundancia ({method.upper()})",
        out_path=out_dir / f"2_abundance_consistency_{method}.png",
        x_tick_labels=[f"E{i}" for i in endmember_idx],
        y_limits=(-0.03, 1.05),
        show_markers=True
    )

    plt.show()


if __name__ == "__main__":
    run_unified_consistency_analysis()