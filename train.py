import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataset import create_dataloader

from Models.abundance_transformer import create_model
from Models.residual_transformer import create_model as create_residual_model
from Models.cross_attention_transformer import create_model as create_cross_attention_model
from Models.mamba_transformer import create_model as create_mamba_model


# ============================================================
# CONFIGURACIÓN ESPECÍFICA DE MAMBA
# ============================================================

MAMBA_BATCH_SIZE = 32
MAMBA_EPOCHS = 20
MAMBA_LEARNING_RATE = 1e-5
MAMBA_WEIGHT_DECAY = 0.0
MAMBA_NUM_SAMPLES = 256
MAMBA_PATCH_SIZE = 32


# ============================================================
# FUNCIONES AUXILIARES DE MAMBA
# ============================================================

def create_mamba_patches(
    A,
    labels,
    image_shape,
    patch_size=32
):
    """
    Extrae parches espaciales de 32x32 para Mamba.

    A:
        (n_channels, n_pixels)

    labels:
        (n_pixels,)

    Retorna:
        X: (n_patches, n_channels, patch_size, patch_size)
        y: (n_patches, patch_size, patch_size)
    """

    n_channels, n_pixels = A.shape
    height, width = image_shape

    if n_pixels != height * width:
        raise ValueError(
            f"A tiene {n_pixels} píxeles, "
            f"pero image_shape requiere {height * width}."
        )

    A_image = A.reshape(n_channels, height, width)
    labels_image = labels.reshape(height, width)

    half_patch = patch_size // 2

    X_patches = []
    y_patches = []

    for row in range(
        half_patch,
        height - half_patch,
        patch_size
    ):
        for col in range(
            half_patch,
            width - half_patch,
            patch_size
        ):

            row_start = row - half_patch
            row_end = row + half_patch

            col_start = col - half_patch
            col_end = col + half_patch

            X_patch = A_image[
                :,
                row_start:row_end,
                col_start:col_end
            ]

            y_patch = labels_image[
                row_start:row_end,
                col_start:col_end
            ]

            if X_patch.shape == (
                n_channels,
                patch_size,
                patch_size
            ):
                X_patches.append(X_patch)
                y_patches.append(y_patch)

    X_patches = np.asarray(X_patches, dtype=np.float32)
    y_patches = np.asarray(y_patches, dtype=np.int64)

    return X_patches, y_patches


def validate_mamba_data(X, y):
    """
    Validaciones básicas de los datos utilizados por Mamba.
    """

    print()
    print("=" * 70)
    print(" VALIDACIÓN DE DATOS MAMBA")
    print("=" * 70)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    print("X min:", X.min())
    print("X max:", X.max())
    print("X mean:", X.mean())
    print("X std:", X.std())

    print("NaN en X:", np.isnan(X).any())
    print("Inf en X:", np.isinf(X).any())

    print("NaN en y:", np.isnan(y).any())
    print("Inf en y:", np.isinf(y).any())

    print("Clases presentes:", np.unique(y))

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contiene NaN o Inf.")

    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contiene NaN o Inf.")

    if X.ndim != 4:
        raise ValueError(
            f"X debe tener 4 dimensiones. Recibido: {X.ndim}"
        )

    if y.ndim != 3:
        raise ValueError(
            f"y debe tener 3 dimensiones. Recibido: {y.ndim}"
        )


def print_mamba_class_distribution(labels, title):
    """
    Imprime distribución de clases.
    """

    print()
    print(title)
    print("-" * 70)

    classes, counts = np.unique(labels, return_counts=True)

    for cls, count in zip(classes, counts):
        print(
            f"Clase {cls}: {count:8d} "
            f"({100.0 * count / len(labels):6.2f}%)"
        )


# ============================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================

def train_model(
    A_initial,
    P,
    labels,
    train_row_ind,
    train_col_ind,
    image_shape,
    architecture="abundance",
    patch_size=32,
    batch_size=4,
    num_epochs=20,
    learning_rate=1e-4,
    device=None
):
    """
    Entrena cualquiera de las arquitecturas disponibles:

        abundance
        residual
        cross_attention
        mamba

    Para Mamba se conserva su pipeline específico de entrenamiento.
    """

    # ========================================================
    # ARQUITECTURAS VÁLIDAS
    # ========================================================

    valid_architectures = {
        "abundance",
        "residual",
        "cross_attention",
        "mamba"
    }

    if architecture not in valid_architectures:
        raise ValueError(
            f"Arquitectura no válida: '{architecture}'. "
            f"Opciones disponibles: "
            f"{sorted(valid_architectures)}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    if device is None:

        if torch.cuda.is_available():
            device = torch.device("cuda")

        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        else:
            device = torch.device("cpu")

    elif isinstance(device, str):
        device = torch.device(device)

    print("Device:", device)

    # ========================================================
    # CONVERSIÓN DE DATOS
    # ========================================================

    A_initial = np.asarray(
        A_initial,
        dtype=np.float32
    )

    P = np.asarray(
        P,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    # ========================================================
    # VALIDACIÓN DE DIMENSIONES
    # ========================================================

    if A_initial.ndim != 2:
        raise ValueError(
            f"A_initial debe ser 2D. "
            f"Recibido: {A_initial.shape}"
        )

    if P.ndim != 2:
        raise ValueError(
            f"P debe ser 2D. "
            f"Recibido: {P.shape}"
        )

    n_endmembers, num_pixels = A_initial.shape

    if n_endmembers != 8:
        raise ValueError(
            f"Se esperaban 8 endmembers. "
            f"Recibido: {n_endmembers}"
        )

    if P.shape != (32, 8):
        raise ValueError(
            f"P debe tener shape (32, 8). "
            f"Recibido: {P.shape}"
        )

    if num_pixels != image_shape[0] * image_shape[1]:
        raise ValueError(
            f"A_initial tiene {num_pixels} píxeles, "
            f"pero image_shape requiere "
            f"{image_shape[0] * image_shape[1]}."
        )

    if labels.size != num_pixels:
        raise ValueError(
            f"labels tiene {labels.size} elementos, "
            f"pero A_initial tiene {num_pixels} píxeles."
        )

    # ========================================================
    # ALINEACIÓN SEGÚN EL MATCHING DE ENTRENAMIENTO
    # ========================================================

    A_aligned = np.zeros_like(
        A_initial,
        dtype=np.float32
    )

    P_aligned = np.zeros_like(
        P,
        dtype=np.float32
    )

    for row, col in zip(
        train_row_ind,
        train_col_ind
    ):
        A_aligned[col, :] = A_initial[row, :]
        P_aligned[:, col] = P[:, row]

    print()
    print("Correspondencia utilizada:")
    
    for row, col in zip(
        train_row_ind,
        train_col_ind
    ):
        print(
            f"Endmember {row} -> clase {col}"
        )

    # ========================================================
    # RUTA MAMBA
    # ========================================================

    if architecture == "mamba":

        print()
        print("=" * 70)
        print(" ENTRENANDO: MAMBA")
        print("=" * 70)

        # ----------------------------------------------------
        # PARÁMETROS ESPECÍFICOS DE MAMBA
        # ----------------------------------------------------

        mamba_batch_size = MAMBA_BATCH_SIZE
        mamba_epochs = MAMBA_EPOCHS
        mamba_learning_rate = MAMBA_LEARNING_RATE
        mamba_weight_decay = MAMBA_WEIGHT_DECAY

        # ----------------------------------------------------
        # CREACIÓN DE PARCHES
        # ----------------------------------------------------

        X_patches, y_patches = create_mamba_patches(
            A=A_aligned,
            labels=labels,
            image_shape=image_shape,
            patch_size=MAMBA_PATCH_SIZE
        )

        print()
        print("Parches generados:")
        print("X:", X_patches.shape)
        print("y:", y_patches.shape)

        # ----------------------------------------------------
        # VALIDACIÓN
        # ----------------------------------------------------

        validate_mamba_data(
            X_patches,
            y_patches
        )

        # ----------------------------------------------------
        # DISTRIBUCIÓN GLOBAL
        # ----------------------------------------------------

        print_mamba_class_distribution(
            labels,
            "Distribución global de clases"
        )

        # ----------------------------------------------------
        # LIMITAR NÚMERO DE MUESTRAS
        # ----------------------------------------------------

        num_samples = min(
            MAMBA_NUM_SAMPLES,
            len(X_patches)
        )

        X_patches = X_patches[:num_samples]
        y_patches = y_patches[:num_samples]

        print()
        print("Número de muestras utilizadas:", num_samples)

        # ----------------------------------------------------
        # DATASET Y DATALOADER
        # ----------------------------------------------------

        X_tensor = torch.from_numpy(
            X_patches
        )

        y_tensor = torch.from_numpy(
            y_patches
        )

        train_dataset = TensorDataset(
            X_tensor,
            y_tensor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=mamba_batch_size,
            shuffle=True
        )

        print(
            "Batch size:",
            mamba_batch_size
        )

        print(
            "Número de batches:",
            len(train_loader)
        )

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        model = create_mamba_model(
            n_endmembers=8,
            n_classes=8,
            device=device
        )

        model = model.to(device)

        # ----------------------------------------------------
        # NÚMERO DE PARÁMETROS
        # ----------------------------------------------------

        num_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print()
        print(
            "Parámetros entrenables:",
            f"{num_parameters:,}"
        )

        # ----------------------------------------------------
        # LOSS Y OPTIMIZADOR
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=mamba_learning_rate,
            weight_decay=mamba_weight_decay
        )

        # ----------------------------------------------------
        # DIAGNÓSTICO DEL PRIMER BATCH
        # ----------------------------------------------------

        first_batch_done = False

        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        for epoch in range(mamba_epochs):

            model.train()

            running_loss = 0.0
            total_grad_norm = 0.0
            num_batches = 0

            for X, y in train_loader:

                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(X)

                if not first_batch_done:

                    print()
                    print("=" * 70)
                    print(" DIAGNÓSTICO PRIMER BATCH")
                    print("=" * 70)

                    print(
                        "Input:",
                        X.shape
                    )

                    print(
                        "Labels:",
                        y.shape
                    )

                    print(
                        "Logits:",
                        logits.shape
                    )

                    print(
                        "Logits min:",
                        logits.min().item()
                    )

                    print(
                        "Logits max:",
                        logits.max().item()
                    )

                    print(
                        "Logits mean:",
                        logits.mean().item()
                    )

                    print(
                        "Logits std:",
                        logits.std().item()
                    )

                    first_batch_done = True

                if (
                    torch.isnan(logits).any()
                    or torch.isinf(logits).any()
                ):
                    raise ValueError(
                        "Logits contienen NaN o Inf."
                    )

                loss = criterion(
                    logits,
                    y
                )

                if (
                    torch.isnan(loss)
                    or torch.isinf(loss)
                ):
                    raise ValueError(
                        "Loss contiene NaN o Inf."
                    )

                loss.backward()

                # ------------------------------------------------
                # NORMA DE GRADIENTES
                # ------------------------------------------------

                grad_norm = 0.0

                for parameter in model.parameters():

                    if parameter.grad is not None:

                        param_grad_norm = (
                            parameter.grad.data.norm(2).item()
                        )

                        grad_norm += (
                            param_grad_norm ** 2
                        )

                grad_norm = grad_norm ** 0.5

                total_grad_norm += grad_norm

                # ------------------------------------------------
                # CLIPPING
                # ------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                running_loss += loss.item()

                num_batches += 1

            average_loss = (
                running_loss / num_batches
            )

            average_grad_norm = (
                total_grad_norm / num_batches
            )

            print(
                f"Epoch [{epoch + 1}/{mamba_epochs}] "
                f"Loss: {average_loss:.6f} "
                f"GradNorm: {average_grad_norm:.6f}"
            )

        print()
        print("=" * 70)
        print(" ENTRENAMIENTO MAMBA TERMINADO")
        print("=" * 70)

        return model

    # ========================================================
    # RUTA TRANSFORMERS
    # ========================================================

    endmembers_tensor = torch.from_numpy(
        P_aligned.T
    ).to(device)

    dataset, train_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=True
    )

    print()
    print("Número de parches:", len(dataset))
    print("Batch size:", batch_size)
    print("Número de batches:", len(train_loader))

    print()
    print("=" * 70)
    print(
        f" ENTRENANDO: {architecture.upper()}"
    )
    print("=" * 70)

    # ========================================================
    # CREACIÓN DEL MODELO
    # ========================================================

    if architecture == "abundance":

        model = create_model(
            n_endmembers=8,
            n_classes=8,
            device=device
        )

        output_filename = (
            "abundance_transformer.pth"
        )

    elif architecture == "residual":

        model = create_residual_model(
            n_endmembers=8,
            n_classes=8,
            device=device
        )

        output_filename = (
            "residual_transformer.pth"
        )

    elif architecture == "cross_attention":

        model = create_cross_attention_model(
            n_endmembers=8,
            n_classes=8,
            spectral_bands=32,
            device=device
        )

        output_filename = (
            "cross_attention_transformer.pth"
        )

    # ========================================================
    # LOSS Y OPTIMIZADOR
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ========================================================
    # ENTRENAMIENTO TRANSFORMER
    # ========================================================

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if architecture == "cross_attention":

                logits = model(
                    X,
                    endmembers_tensor
                )

            else:

                logits = model(X)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss / len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.6f}"
        )

    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    torch.save(
        model.state_dict(),
        output_filename
    )

    print()
    print(
        f"Modelo guardado: {output_filename}"
    )

    return model