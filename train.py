
import numpy as np
import torch
import torch.nn as nn

from dataset import create_dataloader

from Models.abundance_transformer import create_model
from Models.residual_transformer import create_model as create_residual_model
from Models.cross_attention_transformer import create_model as create_cross_attention_model
from Models.mamba_transformer import create_model as create_mamba_model
from Models.resunet import create_model as create_resunet_model


# ============================================================
# CONFIGURACIÓN COMÚN DE ENTRENAMIENTO
# ============================================================

DEFAULT_PATCH_SIZE = 32
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-4
N_ENDMEMBERS = 8
N_CLASSES = 8
SPECTRAL_BANDS = 32


# ============================================================
# VALIDACIÓN DE DATOS
# ============================================================

def validate_training_data(A, P, labels, image_shape):
    """Comprueba las dimensiones y los valores de entrada."""

    if A.ndim != 2:
        raise ValueError(f"A debe ser 2D. Recibido: {A.shape}")

    if P.ndim != 2:
        raise ValueError(f"P debe ser 2D. Recibido: {P.shape}")

    if A.shape[0] != N_ENDMEMBERS:
        raise ValueError(
            f"Se esperaban {N_ENDMEMBERS} mapas de abundancia; "
            f"recibidos: {A.shape[0]}."
        )

    if P.shape != (SPECTRAL_BANDS, N_ENDMEMBERS):
        raise ValueError(
            f"P debe tener shape {(SPECTRAL_BANDS, N_ENDMEMBERS)}; "
            f"recibido: {P.shape}."
        )

    expected_pixels = image_shape[0] * image_shape[1]

    if A.shape[1] != expected_pixels:
        raise ValueError(
            f"A contiene {A.shape[1]} píxeles; "
            f"se esperaban {expected_pixels}."
        )

    if labels.ndim != 1 or labels.size != expected_pixels:
        raise ValueError(
            f"labels debe tener shape ({expected_pixels},); "
            f"recibido: {labels.shape}."
        )

    if not np.isfinite(A).all():
        raise ValueError("A contiene NaN o Inf.")

    if not np.isfinite(P).all():
        raise ValueError("P contiene NaN o Inf.")

    if np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise ValueError(
            f"Las etiquetas deben estar entre 0 y {N_CLASSES - 1}."
        )

    if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2:
        raise ValueError("image_shape debe ser (alto, ancho).")


def validate_matching(row_ind, col_ind, n_endmembers):
    """Verifica que el matching forme una permutación completa."""

    rows = np.asarray(row_ind, dtype=np.int64)
    cols = np.asarray(col_ind, dtype=np.int64)

    if rows.ndim != 1 or cols.ndim != 1 or len(rows) != len(cols):
        raise ValueError("Los índices de correspondencia no son válidos.")

    if len(rows) != n_endmembers:
        raise ValueError(
            f"El matching debe contener {n_endmembers} parejas; "
            f"recibidas: {len(rows)}."
        )

    expected = np.arange(n_endmembers)

    if (
        not np.array_equal(np.sort(rows), expected)
        or not np.array_equal(np.sort(cols), expected)
    ):
        raise ValueError(
            "El matching debe incluir una vez cada índice de "
            "endmember y cada índice de clase."
        )

    return rows, cols


def validate_batch(X, y, logits):
    """Comprueba la compatibilidad de entradas, etiquetas y salidas."""

    if logits.ndim != 4:
        raise ValueError(
            f"Los logits deben tener forma (B,C,H,W); "
            f"recibido: {tuple(logits.shape)}."
        )

    if y.ndim != 3:
        raise ValueError(
            f"Las etiquetas deben tener forma (B,H,W); "
            f"recibido: {tuple(y.shape)}."
        )

    if logits.shape[0] != y.shape[0] or logits.shape[2:] != y.shape[1:]:
        raise ValueError(
            f"Dimensiones incompatibles: logits={tuple(logits.shape)}, "
            f"labels={tuple(y.shape)}."
        )

    if logits.shape[1] != N_CLASSES:
        raise ValueError(
            f"El modelo debe producir {N_CLASSES} clases; "
            f"produjo {logits.shape[1]}."
        )

    if not torch.isfinite(logits).all():
        raise ValueError("Los logits contienen NaN o Inf.")


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
    patch_size=DEFAULT_PATCH_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    num_epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    device=None
):
    """
    Entrena una de las cinco arquitecturas con una configuración común:

        abundance
        residual
        cross_attention
        mamba
        resunet

    Todos los modelos utilizan los mismos mapas de abundancia alineados,
    parches, tamaño de lote, épocas, pérdida y optimizador.

    Cross-Attention recibe adicionalmente los endmembers alineados,
    como requiere su arquitectura.
    """

    valid_architectures = {
        "abundance",
        "residual",
        "cross_attention",
        "mamba",
        "resunet"
    }

    if architecture not in valid_architectures:
        raise ValueError(
            f"Arquitectura no válida: '{architecture}'. "
            f"Opciones: {sorted(valid_architectures)}."
        )

    if patch_size <= 0 or batch_size <= 0 or num_epochs <= 0:
        raise ValueError(
            "patch_size, batch_size y num_epochs deben ser positivos."
        )

    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo.")

    # --------------------------------------------------------
    # DISPOSITIVO
    # --------------------------------------------------------

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print("=" * 70)
    print(f"ENTRENAMIENTO: {architecture.upper()}")
    print("=" * 70)
    print("Dispositivo:", device)

    # --------------------------------------------------------
    # CONVERSIÓN Y VALIDACIÓN
    # --------------------------------------------------------

    A_initial = np.asarray(A_initial, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2:
        raise ValueError("image_shape debe ser (alto, ancho).")

    validate_training_data(A_initial, P, labels, image_shape)

    # --------------------------------------------------------
    # ALINEACIÓN CON EL MATCHING DE ENTRENAMIENTO
    # --------------------------------------------------------

    train_row_ind, train_col_ind = validate_matching(
        train_row_ind,
        train_col_ind,
        N_ENDMEMBERS
    )

    A_aligned = np.zeros_like(A_initial)
    P_aligned = np.zeros_like(P)

    for row, col in zip(train_row_ind, train_col_ind):
        A_aligned[col, :] = A_initial[row, :]
        P_aligned[:, col] = P[:, row]

    #print("\nCorrespondencia utilizada:")

    #for row, col in zip(train_row_ind, train_col_ind):
    #    print(f"Endmember {row} -> clase {col}")

    # --------------------------------------------------------
    # DATASET Y DATALOADER COMPARTIDOS POR LOS CINCO MODELOS
    # --------------------------------------------------------

    dataset, train_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=batch_size,
        shuffle=True
    )

    print("\nConfiguración común:")
    print("Parches:", len(dataset))
    print("Tamaño de parche:", patch_size)
    print("Batch size:", batch_size)
    print("Épocas:", num_epochs)
    print("Learning rate:", learning_rate)
    print("Batches por época:", len(train_loader))

    if len(dataset) == 0 or len(train_loader) == 0:
        raise ValueError("El DataLoader no contiene parches de entrenamiento.")

    # --------------------------------------------------------
    # CREACIÓN DEL MODELO
    # --------------------------------------------------------

    if architecture == "abundance":
        model = create_model(
            n_endmembers=N_ENDMEMBERS,
            n_classes=N_CLASSES,
            device=device
        )
        output_filename = "abundance_transformer.pth"

    elif architecture == "residual":
        model = create_residual_model(
            n_endmembers=N_ENDMEMBERS,
            n_classes=N_CLASSES,
            device=device
        )
        output_filename = "residual_transformer.pth"

    elif architecture == "cross_attention":
        model = create_cross_attention_model(
            n_endmembers=N_ENDMEMBERS,
            n_classes=N_CLASSES,
            spectral_bands=SPECTRAL_BANDS,
            device=device
        )
        output_filename = "cross_attention_transformer.pth"

    elif architecture == "mamba":
        model = create_mamba_model(
            n_endmembers=N_ENDMEMBERS,
            n_classes=N_CLASSES,
            device=device
        )
        output_filename = "mamba_transformer.pth"

    elif architecture == "resunet":
        model = create_resunet_model(
            n_endmembers=N_ENDMEMBERS,
            n_classes=N_CLASSES,
            device=device
        )
        output_filename = "resunet.pth"

    model = model.to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Parámetros entrenables:", f"{num_parameters:,}")

    # --------------------------------------------------------
    # END MEMBERS PARA CROSS-ATTENTION
    # --------------------------------------------------------

    endmembers_tensor = torch.from_numpy(
        P_aligned.T.copy()
    ).to(device)

    # --------------------------------------------------------
    # PÉRDIDA Y OPTIMIZADOR COMUNES
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        labels=labels,
        n_classes=N_CLASSES,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    first_batch_checked = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X, y in train_loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            if architecture == "cross_attention":
                logits = model(X, endmembers_tensor)
            else:
                logits = model(X)

            if not first_batch_checked:
                validate_batch(X, y, logits)

                #print("\nValidación del primer batch:")
                #print("Entrada:", tuple(X.shape))
                #print("Etiquetas:", tuple(y.shape))
                #print("Logits:", tuple(logits.shape))

                first_batch_checked = True
            elif not torch.isfinite(logits).all():
                raise ValueError(
                    f"{architecture}: los logits contienen NaN o Inf."
                )

            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"{architecture}: la pérdida contiene NaN o Inf."
                )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError("No se procesaron batches en esta época.")

        average_loss = running_loss / num_batches

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.6f}"
        )

    # --------------------------------------------------------
    # GUARDAR CHECKPOINT
    # --------------------------------------------------------

    torch.save(model.state_dict(), output_filename)

    print("\n" + "=" * 70)
    print(f"ENTRENAMIENTO {architecture.upper()} TERMINADO")
    print("Modelo guardado:", output_filename)
    print("=" * 70)

    return model


def calculate_class_weights(labels, n_classes, device):
    """
    Calcula pesos inversamente proporcionales a la frecuencia
    de las clases presentes en el conjunto de entrenamiento.

    Las clases ausentes reciben peso cero.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if labels.size == 0:
        raise ValueError("No hay etiquetas de entrenamiento.")

    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError(
            f"Las etiquetas deben estar entre 0 y {n_classes - 1}."
        )

    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0

    if not np.any(present):
        raise ValueError("No hay clases presentes en entrenamiento.")

    weights = np.zeros(n_classes, dtype=np.float64)

    # Calcular pesos solo para clases presentes.
    weights[present] = (
        len(labels) /
        (np.sum(present) * counts[present])
    )

    # Normalizar para que la media de los pesos presentes sea 1.
    weights[present] /= weights[present].mean()

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    print("\nDistribución de clases en entrenamiento:")

    for class_id, count in enumerate(counts):
        if count == 0:
            print(
                f"Clase {class_id}: 0 píxeles; "
                "ausente en entrenamiento, peso = 0"
            )
        else:
            print(
                f"Clase {class_id}: {count} píxeles "
                f"({count / len(labels):.2%}), "
                f"peso = {weights[class_id]:.4f}"
            )

    return class_weights