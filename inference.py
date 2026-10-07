import numpy as np
import torch

from dataset import create_dataloader

from Models.abundance_transformer import create_model
from Models.residual_transformer import create_model as create_residual_model
from Models.cross_attention_transformer import (
    create_model as create_cross_attention_model
)


def test_model(
    A_initial,
    P,
    labels,
    train_row_ind,
    train_col_ind,
    image_shape,
    architecture="abundance",
    model=None,
    patch_size=32,
    batch_size=4,
    abundance_model_path="abundance_transformer.pth",
    residual_model_path="residual_transformer.pth",
    cross_attention_model_path="cross_attention_transformer.pth",
    device=None
):

    # ========================================================
    # VALIDAR ARQUITECTURA
    # ========================================================

    valid_architectures = {
        "abundance",
        "residual",
        "cross_attention",
        "mamba"
    }

    if architecture not in valid_architectures:
        raise ValueError(
            f"Arquitectura no válida: {architecture}. "
            f"Opciones: {sorted(valid_architectures)}"
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

    print()
    print("=" * 60)
    print(" TEST DEL MODELO")
    print("=" * 60)

    print("Arquitectura:", architecture)
    print("Device:", device)

    # ========================================================
    # CONVERTIR A NUMPY
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
    # VALIDACIONES
    # ========================================================

    if A_initial.ndim != 2:
        raise ValueError(
            f"A_initial debe tener 2 dimensiones. "
            f"Shape: {A_initial.shape}"
        )

    if P.ndim != 2:
        raise ValueError(
            f"P debe tener 2 dimensiones. "
            f"Shape: {P.shape}"
        )

    if A_initial.shape[0] != 8:
        raise ValueError(
            f"Se esperaban 8 endmembers en A. "
            f"Shape: {A_initial.shape}"
        )

    if P.shape != (32, 8):
        raise ValueError(
            f"P debe tener shape (32, 8). "
            f"Shape: {P.shape}"
        )

    if A_initial.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError(
            "El número de píxeles de A_initial "
            "no coincide con image_shape."
        )

    if labels.size != image_shape[0] * image_shape[1]:
        raise ValueError(
            "El número de labels no coincide "
            "con image_shape."
        )

    # ========================================================
    # ALINEACIÓN
    #
    # IMPORTANTE:
    #
    # Se utiliza EXCLUSIVAMENTE el matching obtenido
    # durante el entrenamiento con la IMAGEN 1.
    #
    # NO se calcula Hungarian nuevamente para Imagen 2.
    # ========================================================

    print()
    print(
        "Alineando abundancias y endmembers "
        "con el matching de TRAIN..."
    )

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

        print(
            f"Endmember {row} -> clase {col}"
        )

    print()
    print("Alineación completada.")

    # ========================================================
    # ENDMEMBERS PARA CROSS-ATTENTION
    # ========================================================

    endmembers_tensor = torch.from_numpy(
        P_aligned.T
    ).to(device)

    # ========================================================
    # PARÁMETROS ESPECÍFICOS DE MAMBA
    # ========================================================

    if architecture == "mamba":

        mamba_batch_size = 32

    else:

        mamba_batch_size = batch_size

    # ========================================================
    # DATASET / DATALOADER
    # ========================================================

    dataset, test_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=mamba_batch_size,
        shuffle=False
    )

    print()
    print("Número de parches:", len(dataset))
    print("Batch size:", mamba_batch_size)
    print("Número de batches:", len(test_loader))

    # ========================================================
    # CREAR / RECIBIR MODELO
    # ========================================================

    if architecture == "mamba":

        # ----------------------------------------------------
        # MAMBA
        #
        # El modelo ya fue entrenado en train.py y se recibe
        # directamente mediante el argumento model.
        # ----------------------------------------------------

        if model is None:
            raise ValueError(
                "Para la arquitectura 'mamba' se debe "
                "proporcionar el modelo entrenado mediante "
                "el argumento model."
            )

        print()
        print("=" * 60)
        print(" USANDO MAMBA TRANSFORMER")
        print("=" * 60)

        model = model.to(device)
        model.eval()

    else:

        # ----------------------------------------------------
        # ABUNDANCE
        # ----------------------------------------------------

        if architecture == "abundance":

            print()
            print("=" * 60)
            print(" CARGANDO ABUNDANCE TRANSFORMER")
            print("=" * 60)

            model = create_model(
                n_endmembers=8,
                n_classes=8,
                device=device
            )

            model_path = abundance_model_path

        # ----------------------------------------------------
        # RESIDUAL
        # ----------------------------------------------------

        elif architecture == "residual":

            print()
            print("=" * 60)
            print(" CARGANDO RESIDUAL TRANSFORMER")
            print("=" * 60)

            model = create_residual_model(
                n_endmembers=8,
                n_classes=8,
                device=device
            )

            model_path = residual_model_path

        # ----------------------------------------------------
        # CROSS-ATTENTION
        # ----------------------------------------------------

        elif architecture == "cross_attention":

            print()
            print("=" * 60)
            print(" CARGANDO CROSS-ATTENTION TRANSFORMER")
            print("=" * 60)

            model = create_cross_attention_model(
                n_endmembers=8,
                n_classes=8,
                spectral_bands=32,
                device=device
            )

            model_path = cross_attention_model_path

        # ----------------------------------------------------
        # CARGAR PESOS
        # ----------------------------------------------------

        print()
        print("Cargando:", model_path)

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        model = model.to(device)
        model.eval()

        print("Modelo cargado correctamente.")

    # ========================================================
    # MAPA DE PREDICCIONES
    # ========================================================

    prediction_map = torch.zeros(
        image_shape,
        dtype=torch.long
    )

    # ========================================================
    # INFERENCIA
    # ========================================================

    patch_index = 0

    print()
    print("Ejecutando inferencia...")

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(
                device,
                dtype=torch.float32
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            if architecture == "cross_attention":

                logits = model(
                    X,
                    endmembers_tensor
                )

            else:

                logits = model(X)

            # ------------------------------------------------
            # PREDICCIÓN
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )
            # Mantiene el mismo esquema de reconstrucción utilizado
            # en la inferencia de Mamba.

            if architecture == "mamba":

                predictions_numpy = (
                    predictions.cpu().numpy()
                )

                # Guardamos temporalmente las predicciones
                # para reconstruir el mapa después.
                if patch_index == 0:

                    mamba_predictions = []

                mamba_predictions.append(
                    predictions_numpy
                )

                patch_index += X.shape[0]

            # ------------------------------------------------
            # RECONSTRUCCIÓN PARA TRANSFORMERS
            # ------------------------------------------------

            else:

                for i in range(X.shape[0]):

                    row, col = dataset.patch_coordinates[
                        patch_index
                    ]

                    prediction_map[
                        row:row + patch_size,
                        col:col + patch_size
                    ] = predictions[i].cpu()

                    patch_index += 1

    # ========================================================
    # RECONSTRUCCIÓN FINAL DE MAMBA
    # ========================================================

    if architecture == "mamba":

        predictions = np.concatenate(
            mamba_predictions,
            axis=0
        )

        print()
        print(
            "Predicciones por parche:",
            predictions.shape
        )

        H, W = image_shape

        patches_per_row = H // patch_size
        patches_per_col = W // patch_size

        expected_patches = (
            patches_per_row *
            patches_per_col
        )

        if predictions.shape[0] != expected_patches:

            raise ValueError(
                "El número de predicciones no coincide "
                "con el número esperado de parches. "
                f"Predicciones: {predictions.shape[0]}, "
                f"esperadas: {expected_patches}"
            )

        prediction_map = np.zeros(
            (H, W),
            dtype=np.int64
        )

        index = 0

        for i in range(patches_per_row):

            for j in range(patches_per_col):

                row_start = (
                    i * patch_size
                )

                row_end = (
                    row_start + patch_size
                )

                col_start = (
                    j * patch_size
                )

                col_end = (
                    col_start + patch_size
                )

                prediction_map[
                    row_start:row_end,
                    col_start:col_end
                ] = predictions[index]

                index += 1

        print()
        print(
            "Prediction map:",
            prediction_map.shape
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print()
    print("=" * 60)
    print(" RESULTADOS")
    print("=" * 60)

    print()
    print("Arquitectura:", architecture)

    if isinstance(
        prediction_map,
        torch.Tensor
    ):

        unique_predictions = torch.unique(
            prediction_map
        )

    else:

        unique_predictions = np.unique(
            prediction_map
        )

    print(
        "Clases predichas:",
        unique_predictions
    )

    return (
        prediction_map.numpy()
        if isinstance(prediction_map, torch.Tensor)
        else prediction_map
    )

