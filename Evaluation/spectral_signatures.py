import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def spectral_signatures(Y_img, labels, P_ref, n_classes):
    """
    Y_img: Matriz espectral (Banderas, Píxeles)
    labels: Array 1D con las etiquetas (Píxeles,)
    P_ref: Matriz de Endmembers (Banderas, n_classes)
    n_classes: Número total de clases/endmembers
    """
    n_bands = Y_img.shape[0]
    bands = np.arange(n_bands)

    # ============================================================
    # 1. IMPRESIÓN DE ESTADÍSTICAS Y GRÁFICA DE 8 SUBPLOTS
    # ============================================================
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS ESPECTRALES POR CLASE")
    print("=" * 60)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.ravel()  # Aplanar cuadrícula a 1D (índices 0 a 7)

    for c in range(n_classes):
        ax = axes[c]
        mask = (labels == c)

        if not np.any(mask):
            ax.set_title(f"Clase {c} (Sin píxeles)")
            continue

        pixels_class = Y_img[:, mask]  # Forma: (Banderas, N_pixeles_clase)

        mean_spec = np.mean(pixels_class, axis=1)
        std_spec = np.std(pixels_class, axis=1)

        print(f"--- Clase {c} ({np.sum(mask)} píxeles) ---")
        print(f"  Intensidad Media Global: {mean_spec.mean():.4f}")
        print(f"  Dispersión Promedio (Std Dev): {std_spec.mean():.4f}")
        print(f"  Rango Promedio [Min, Max]: [{mean_spec.min():.4f}, {mean_spec.max():.4f}]")

        # Endmember puro (Firma central de referencia)
        ax.plot(bands, P_ref[:, c], label='Endmember', color='tab:blue', linewidth=2)

        # Media real de los píxeles
        ax.plot(bands, mean_spec, label='Media real', color='tab:red', linestyle='--')

        # Sombra de dispersión (±1 desviación estándar)
        ax.fill_between(
            bands,
            mean_spec - std_spec,
            mean_spec + std_spec,
            color='tab:red',
            alpha=0.2,
            label='Dispersión (±1 σ)'
        )

        ax.set_title(f"Clase {c} (N={np.sum(mask)})")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)

    fig.text(0.5, 0.02, 'Índice de Banda Espectral', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Reflectancia / Intensidad', va='center', rotation='vertical', fontsize=12)
    plt.suptitle("Respuesta Espectral y Dispersión por Clase", fontsize=16)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.95])
    plt.show()

    # ============================================================
    # 2. ANÁLISIS MATEMÁTICO DE SIMILITUD ENTRE ENDMEMBERS
    # ============================================================
    # Distancia coseno y conversión a Ángulo Espectral (SAM) en grados
    cos_dist = cdist(P_ref.T, P_ref.T, metric='cosine')
    cos_sim = np.clip(1.0 - cos_dist, -1.0, 1.0)
    sam_degrees = np.degrees(np.arccos(cos_sim))

    # Correlación de Pearson entre curvas
    corr_matrix = np.corrcoef(P_ref.T)

    print("\n" + "=" * 60)
    print("ANÁLISIS DE SIMILITUD ESPECTRAL (SAM Y CORRELACIÓN)")
    print("Menor SAM (°) / Mayor Correlación = Mayor Parecido")
    print("=" * 60)

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            print(f"Clase {i} vs Clase {j}: SAM = {sam_degrees[i, j]:6.2f}° | Correlación = {corr_matrix[i, j]:.4f}")

    # Visualización con Mapas de Calor (Heatmaps)
    fig_sim, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap de SAM
    im1 = ax1.imshow(sam_degrees, cmap='viridis_r')
    ax1.set_title("Ángulo Espectral SAM (°)\n(0° = Idénticos)")
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    fig_sim.colorbar(im1, ax=ax1)

    # Heatmap de Correlación
    im2 = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title("Correlación de Pearson\n(1.0 = Misma Forma)")
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    fig_sim.colorbar(im2, ax=ax2)

    # Añadir valores dentro de cada celda
    for i in range(n_classes):
        for j in range(n_classes):
            ax1