# ============================================================
# test_transformer.py
# Evaluación del Abundance Transformer
# ============================================================

import numpy as np
import torch

from dataset import create_dataloader
from abundance_transformer import create_model


def test_transformer(
    A_initial,
    labels,
    train_row_ind,
    train_col_ind,
    image_shape,
    patch_size=32,
    batch_size=4,
    model_path="abundance_transformer.pth",
    device=None
):
    # ========================================================
    # Device
    # ========================================================

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("=" * 60)
    print(" TEST ABUNDANCE TRANSFORMER")
    print("=" * 60)
    print("Device:", device)

    # ========================================================
    # Alinear abundancias usando el matching de TRAIN
    # ========================================================

    print()
    print("Correspondencia de abundancias usada del TRAIN:")

    A_initial = np.asarray(
        A_initial,
        dtype=np.float32
    )

    if A_initial.ndim != 2:
        raise ValueError(
            f"A_initial debe tener 2 dimensiones. Shape: {A_initial.shape}"
        )

    n_endmembers, num_pixels = A_initial.shape

    if n_endmembers != 8:
        raise ValueError(
            f"Se esperaban 8 endmembers. Recibidos: {n_endmembers}"
        )

    if num_pixels != image_shape[0] * image_shape[1]:
        raise ValueError(
            "El número de píxeles de A_initial no coincide con image_shape."
        )

    A_aligned = np.zeros_like(A_initial)

    for row, col in zip(
        train_row_ind,
        train_col_ind
    ):
        A_aligned[col, :] = A_initial[row, :]

        print(
            f"A_aligned[{col}, :] <- "
            f"A_initial[{row}, :]"
        )

    print()
    print("Abundancias alineadas correctamente.")

    # ========================================================
    # Dataset + DataLoader
    # ========================================================

    dataset, test_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=False
    )

    print(
        "Número de parches:",
        len(dataset)
    )

    print(
        "Batch size:",
        batch_size
    )

    # ========================================================
    # Crear modelo
    # ========================================================

    model = create_model(
        n_endmembers=8,
        n_classes=8,
        device=device
    )

    # ========================================================
    # Cargar modelo entrenado
    # ========================================================

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    # ========================================================
    # Modo evaluación
    # ========================================================

    model.eval()

    # ========================================================
    # Mapa de predicciones
    # ========================================================

    prediction_map = torch.zeros(
        image_shape,
        dtype=torch.long
    )

    # ========================================================
    # Inferencia
    # ========================================================

    patch_index = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            logits = model(X)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            for prediction in predictions:
                row, col = dataset.patch_coordinates[
                    patch_index
                ]

                prediction_map[
                    row:row + patch_size,
                    col:col + patch_size
                ] = prediction.cpu()

                patch_index += 1

    # ========================================================
    # Resultados
    # ========================================================

    print()
    print("=" * 60)
    print(" RESULTADO DEL TRANSFORMER")
    print("=" * 60)

    print(
        "Prediction shape:",
        prediction_map.shape
    )

    print(
        "Clases predichas:",
        torch.unique(prediction_map)
    )

    print("=" * 60)
    print(" TEST FINALIZADO")
    print("=" * 60)

    return prediction_map
