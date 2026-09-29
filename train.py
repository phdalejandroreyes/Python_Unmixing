# ============================================================
# train.py
# Entrenamiento del Abundance Transformer
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from dataset import create_dataloader
from abundance_transformer import create_model


def train_model(
    A_initial,
    labels,
    train_row_ind,
    train_col_ind,
    image_shape,
    patch_size=32,
    batch_size=4,
    num_epochs=20,
    learning_rate=1e-4,
    device=None
):
    # ========================================================
    # Device
    # ========================================================

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("=" * 60)
    print(" ENTRENAMIENTO ABUNDANCE TRANSFORMER")
    print("=" * 60)
    print("Device:", device)

    # ========================================================
    # Alinear abundancias usando la correspondencia de TRAIN
    # ========================================================

    print()
    print("Alineando abundancias con correspondencia de TRAIN...")

    A_initial = np.asarray(A_initial, dtype=np.float32)

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

    # ========================================================
    # Crear abundancias alineadas
    #
    # row = endmember estimado por EBEAE
    # col = clase correspondiente
    # ========================================================

    A_aligned = np.zeros_like(A_initial)

    for row, col in zip(train_row_ind, train_col_ind):
        A_aligned[col, :] = A_initial[row, :]

    # ========================================================
    # Mostrar correspondencia
    # ========================================================

    print()
    print("Correspondencia utilizada:")
    for row, col in zip(train_row_ind, train_col_ind):
        print(
            f"A_aligned[{col}, :] <- A_initial[{row}, :]"
        )

    print()
    print("Abundancias alineadas correctamente.")

    # ========================================================
    # Dataset + DataLoader
    # ========================================================

    dataset, train_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=True
    )

    print("Número de parches:", len(dataset))
    print("Batch size:", batch_size)
    print("Número de batches:", len(train_loader))

    # ========================================================
    # Crear modelo
    # ========================================================

    model = create_model(
        n_endmembers=8,
        n_classes=8,
        device=device
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ========================================================
    # Entrenamiento
    # ========================================================

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.6f}"
        )

    # ========================================================
    # Guardar modelo
    # ========================================================

    torch.save(
        model.state_dict(),
        "abundance_transformer.pth"
    )

    print()
    print("=" * 60)
    print(" ENTRENAMIENTO FINALIZADO")
    print("=" * 60)
    print("Modelo guardado en: abundance_transformer.pth")

    return model
