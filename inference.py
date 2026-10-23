
# ============================================================
# inference.py
#
# Inferencia y validaciones del pipeline de clasificación
# basado en mapas de abundancias.
#
# Arquitecturas:
#   - abundance
#   - residual
#   - cross_attention
#   - mamba
#   - resunet
#
# IMPORTANTE:
# La inferencia utiliza exclusivamente el matching obtenido
# durante el entrenamiento con la imagen 1.
# ============================================================

import numpy as np
import torch

from dataset import create_dataloader

from Models.abundance_transformer import create_model
from Models.residual_transformer import (
    create_model as create_residual_model
)
from Models.cross_attention_transformer import (
    create_model as create_cross_attention_model
)
from Models.resunet import (
    create_model as create_resunet_model
)


# ============================================================
# VALIDACIONES AUXILIARES
# ============================================================

def validate_input_data(
    A_initial,
    P,
    labels,
    image_shape,
    n_endmembers=8,
    n_bands=32,
    n_classes=8
):
    """Valida las dimensiones y los valores de entrada."""

    if len(image_shape) != 2:
        raise ValueError(
            f"image_shape debe ser (H, W). Recibido: {image_shape}"
        )

    H, W = image_shape

    if H <= 0 or W <= 0:
        raise ValueError(
            f"Dimensiones de imagen inválidas: {image_shape}"
        )

    A_initial = np.asarray(A_initial, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)
    labels = np.asarray(labels)

    if A_initial.ndim != 2:
        raise ValueError(
            f"A_initial debe ser 2D. Shape: {A_initial.shape}"
        )

    if A_initial.shape != (n_endmembers, H * W):
        raise ValueError(
            f"A_initial debe tener shape "
            f"{(n_endmembers, H * W)}; recibido: {A_initial.shape}"
        )

    if P.shape != (n_bands, n_endmembers):
        raise ValueError(
            f"P debe tener shape {(n_bands, n_endmembers)}; "
            f"recibido: {P.shape}"
        )

    if not np.isfinite(A_initial).all():
        raise ValueError(
            "A_initial contiene NaN o valores infinitos."
        )

    if not np.isfinite(P).all():
        raise ValueError(
            "P contiene NaN o valores infinitos."
        )

    labels = labels.reshape(-1)

    if labels.size != H * W:
        raise ValueError(
            f"Se esperaban {H * W} etiquetas; "
            f"recibidas: {labels.size}"
        )

    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Las etiquetas deben ser numéricas.")

    if not np.isfinite(labels).all():
        raise ValueError(
            "Las etiquetas contienen NaN o valores infinitos."
        )

    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError(
            "Las etiquetas contienen valores no enteros."
        )

    if labels.size and (
        labels.min() < 0 or labels.max() >= n_classes
    ):
        raise ValueError(
            f"Las etiquetas deben estar entre 0 y "
            f"{n_classes - 1}. "
            f"Rango recibido: [{labels.min()}, {labels.max()}]"
        )

    labels = labels.astype(np.int64, copy=False)

    print(
        f"[OK] Entradas: A={A_initial.shape}, "
        f"P={P.shape}, labels={labels.shape}"
    )
    print(
        f"[OK] Imagen: {H}x{W}; "
        f"{H * W} píxeles; datos finitos."
    )

    return A_initial, P, labels


def validate_matching(
    train_row_ind,
    train_col_ind,
    n_endmembers=8
):
    """Comprueba que el matching sea una permutación válida."""

    row_ind = np.asarray(train_row_ind)
    col_ind = np.asarray(train_col_ind)

    if row_ind.ndim != 1 or col_ind.ndim != 1:
        raise ValueError(
            "Los índices del matching deben ser vectores 1D."
        )

    if row_ind.size != n_endmembers:
        raise ValueError(
            f"Se esperaban {n_endmembers} índices de filas; "
            f"recibidos: {row_ind.size}"
        )

    if col_ind.size != n_endmembers:
        raise ValueError(
            f"Se esperaban {n_endmembers} índices de columnas; "
            f"recibidos: {col_ind.size}"
        )

    if not np.issubdtype(row_ind.dtype, np.integer):
        raise ValueError(
            "train_row_ind debe contener índices enteros."
        )

    if not np.issubdtype(col_ind.dtype, np.integer):
        raise ValueError(
            "train_col_ind debe contener índices enteros."
        )

    expected = np.arange(n_endmembers)

    if not np.array_equal(np.sort(row_ind), expected):
        raise ValueError(
            "train_row_ind no es una permutación válida."
        )

    if not np.array_equal(np.sort(col_ind), expected):
        raise ValueError(
            "train_col_ind no es una permutación válida."
        )

    print(
        f"[OK] Matching de entrenamiento: "
        f"{n_endmembers} correspondencias uno-a-uno."
    )

    return row_ind, col_ind


def validate_logits(logits, batch_size, n_classes=8):
    """Valida la salida del modelo antes de aplicar argmax."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "El modelo debe devolver logits como un tensor de PyTorch."
        )

    if logits.ndim != 4:
        raise ValueError(
            f"Los logits deben tener forma (B,C,H,W); "
            f"recibida: {tuple(logits.shape)}"
        )

    if logits.shape[0] != batch_size:
        raise ValueError(
            f"Batch incorrecto en logits: {logits.shape[0]}; "
            f"esperado: {batch_size}"
        )

    if logits.shape[1] != n_classes:
        raise ValueError(
            f"El modelo debe producir {n_classes} clases; "
            f"produjo {logits.shape[1]}."
        )

    if not torch.isfinite(logits).all().item():
        raise ValueError(
            "Los logits contienen NaN o valores infinitos."
        )


def validate_prediction_map(
    prediction_map,
    image_shape,
    n_classes=8
):
    """Valida el mapa final de clasificación."""

    if isinstance(prediction_map, torch.Tensor):
        prediction_map = prediction_map.detach().cpu().numpy()

    prediction_map = np.asarray(prediction_map)

    if prediction_map.shape != tuple(image_shape):
        raise ValueError(
            f"El mapa final debe tener forma {tuple(image_shape)}; "
            f"recibida: {prediction_map.shape}"
        )

    if not np.issubdtype(prediction_map.dtype, np.integer):
        raise ValueError(
            "El mapa de predicción debe contener etiquetas enteras."
        )

    if not np.isfinite(prediction_map).all():
        raise ValueError(
            "El mapa de predicción contiene valores no finitos."
        )

    if prediction_map.size and (
        prediction_map.min() < 0
        or prediction_map.max() >= n_classes
    ):
        raise ValueError(
            f"Predicciones fuera del rango [0, {n_classes - 1}]."
        )

    print(
        f"[OK] Mapa final: {prediction_map.shape}; "
        f"{prediction_map.size} píxeles clasificados."
    )

    print(
        "[OK] Clases predichas:",
        np.unique(prediction_map)
    )

    return prediction_map


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

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
    resunet_model_path="resunet.pth",
    device=None
):

    # --------------------------------------------------------
    # 1. VALIDAR ARQUITECTURA Y PARÁMETROS
    # --------------------------------------------------------

    valid_architectures = {
        "abundance",
        "residual",
        "cross_attention",
        "mamba",
        "resunet"
    }

    if architecture not in valid_architectures:
        raise ValueError(
            f"Arquitectura no válida: {architecture}. "
            f"Opciones: {sorted(valid_architectures)}"
        )

    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(
            f"patch_size debe ser un entero positivo: {patch_size}"
        )

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"batch_size debe ser un entero positivo: {batch_size}"
        )

    H, W = image_shape

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"La imagen {image_shape} debe ser divisible por "
            f"patch_size={patch_size}. "
            "Esta implementación reconstruye parches completos."
        )

    # --------------------------------------------------------
    # 2. DEVICE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3. VALIDAR ENTRADAS
    # --------------------------------------------------------

    A_initial, P, labels = validate_input_data(
        A_initial=A_initial,
        P=P,
        labels=labels,
        image_shape=image_shape,
        n_endmembers=8,
        n_bands=32,
        n_classes=8
    )

    train_row_ind, train_col_ind = validate_matching(
        train_row_ind,
        train_col_ind,
        n_endmembers=8
    )

    # --------------------------------------------------------
    # 4. ALINEAR ABUNDANCIAS CON EL MATCHING DE TRAIN
    # --------------------------------------------------------

    print()
    print(
        "Alineando abundancias con el matching "
        "obtenido durante el entrenamiento..."
    )

    A_aligned = np.zeros_like(A_initial, dtype=np.float32)
    P_aligned = np.zeros_like(P, dtype=np.float32)

    for row, col in zip(train_row_ind, train_col_ind):
        A_aligned[col, :] = A_initial[row, :]
        P_aligned[:, col] = P[:, row]

        print(f"Endmember estimado {row} -> clase {col}")

    if not np.isfinite(A_aligned).all():
        raise ValueError(
            "Las abundancias alineadas contienen valores no finitos."
        )

    if not np.isfinite(P_aligned).all():
        raise ValueError(
            "Los endmembers alineados contienen valores no finitos."
        )

    # Comprobar que no hayan quedado canales completamente a cero
    # por una correspondencia incompleta.
    if np.any(np.all(A_aligned == 0, axis=1)):
        raise ValueError(
            "Hay canales de abundancia completamente a cero después "
            "de la alineación. Revisa el matching de entrenamiento."
        )

    print("[OK] Alineación de abundancias completada.")

    # --------------------------------------------------------
    # 5. ENDMEMBERS PARA CROSS-ATTENTION
    # --------------------------------------------------------

    endmembers_tensor = torch.from_numpy(
        P_aligned.T.copy()
    ).to(device=device, dtype=torch.float32)

    # --------------------------------------------------------
    # 6. DATASET / DATALOADER
    # --------------------------------------------------------

    inference_batch_size = 32 if architecture == "mamba" else batch_size

    dataset, test_loader = create_dataloader(
        A_initial=A_aligned,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        batch_size=inference_batch_size,
        shuffle=False
    )

    expected_patches = (
        (H // patch_size) * (W // patch_size)
    )

    if len(dataset) != expected_patches:
        raise ValueError(
            f"El dataset contiene {len(dataset)} parches; "
            f"se esperaban {expected_patches}."
        )

    if len(dataset.patch_coordinates) != len(dataset):
        raise ValueError(
            "La cantidad de coordenadas no coincide con "
            "la cantidad de parches."
        )

    print()
    print("Número de parches:", len(dataset))
    print("Batch size:", inference_batch_size)
    print("Número de batches:", len(test_loader))

    # --------------------------------------------------------
    # 7. CREAR O RECIBIR MODELO
    # --------------------------------------------------------

    if architecture == "mamba":

        if model is None:
            raise ValueError(
                "Para architecture='mamba' debes proporcionar "
                "el modelo entrenado mediante model."
            )

        print()
        print("=" * 60)
        print(" USANDO MAMBA TRANSFORMER")
        print("=" * 60)

        model = model.to(device)
        model.eval()

    else:

        if architecture == "abundance":
            print("\nCARGANDO ABUNDANCE TRANSFORMER")

            model = create_model(
                n_endmembers=8,
                n_classes=8,
                device=device
            )

            model_path = abundance_model_path

        elif architecture == "residual":
            print("\nCARGANDO RESIDUAL TRANSFORMER")

            model = create_residual_model(
                n_endmembers=8,
                n_classes=8,
                device=device
            )

            model_path = residual_model_path

        elif architecture == "cross_attention":
            print("\nCARGANDO CROSS-ATTENTION TRANSFORMER")

            model = create_cross_attention_model(
                n_endmembers=8,
                n_classes=8,
                spectral_bands=32,
                device=device
            )

            model_path = cross_attention_model_path

        elif architecture == "resunet":
            print("\nCARGANDO RES-U-NET")

            model = create_resunet_model(
                n_endmembers=8,
                n_classes=8,
                device=device
            )

            model_path = resunet_model_path

        print("Cargando:", model_path)

        # weights_only=True evita deserializar objetos arbitrarios
        # en versiones de PyTorch que soportan esta opción.
        try:
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
        except TypeError:
            # Compatibilidad con versiones antiguas de PyTorch.
            state_dict = torch.load(
                model_path,
                map_location=device
            )

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print("[OK] Modelo cargado correctamente.")

    # --------------------------------------------------------
    # 8. INFERENCIA Y RECONSTRUCCIÓN
    # --------------------------------------------------------

    prediction_map = np.full(
        (H, W),
        fill_value=-1,
        dtype=np.int64
    )

    coverage = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    mamba_predictions = []
    patch_index = 0

    print("\nEjecutando inferencia...")

    with torch.no_grad():

        for X, _ in test_loader:

            X = X.to(
                device=device,
                dtype=torch.float32
            )

            if not torch.isfinite(X).all().item():
                raise ValueError(
                    "El batch de entrada contiene NaN o infinitos."
                )

            # Forward
            if architecture == "cross_attention":
                logits = model(X, endmembers_tensor)
            else:
                logits = model(X)

            # Validar salida del modelo
            validate_logits(
                logits=logits,
                batch_size=X.shape[0],
                n_classes=8
            )

            if logits.shape[2:] != (patch_size, patch_size):
                raise ValueError(
                    f"Resolución espacial de logits incorrecta: "
                    f"{tuple(logits.shape[2:])}; "
                    f"esperada: {(patch_size, patch_size)}."
                )

            predictions = torch.argmax(logits, dim=1)

            if predictions.min().item() < 0 or predictions.max().item() >= 8:
                raise ValueError(
                    "El modelo produjo etiquetas fuera del rango [0, 7]."
                )

            predictions_numpy = predictions.cpu().numpy()

            # -----------------------------------------------
            # MAMBA: acumular parches y reconstruir después
            # -----------------------------------------------

            if architecture == "mamba":
                mamba_predictions.append(predictions_numpy)
                patch_index += X.shape[0]
                continue

            # -----------------------------------------------
            # RESTO DE ARQUITECTURAS: reconstrucción por coordenadas
            # -----------------------------------------------

            for i in range(X.shape[0]):

                if patch_index >= len(dataset.patch_coordinates):
                    raise ValueError(
                        "Se recibieron más predicciones que "
                        "coordenadas de parches."
                    )

                row, col = dataset.patch_coordinates[patch_index]

                row_end = row + patch_size
                col_end = col + patch_size

                if (
                    row < 0 or col < 0
                    or row_end > H or col_end > W
                ):
                    raise ValueError(
                        f"Parche fuera de la imagen: "
                        f"({row}, {col}) - "
                        f"({row_end}, {col_end})."
                    )

                if np.any(
                    coverage[row:row_end, col:col_end] != 0
                ):
                    raise ValueError(
                        f"Solapamiento detectado al reconstruir "
                        f"el parche {patch_index}."
                    )

                prediction_map[
                    row:row_end,
                    col:col_end
                ] = predictions_numpy[i]

                coverage[
                    row:row_end,
                    col:col_end
                ] += 1

                patch_index += 1

    # --------------------------------------------------------
    # 9. RECONSTRUCCIÓN DE MAMBA
    # --------------------------------------------------------

    if architecture == "mamba":

        if not mamba_predictions:
            raise ValueError(
                "Mamba no produjo predicciones."
            )

        predictions = np.concatenate(
            mamba_predictions,
            axis=0
        )

        if predictions.shape[0] != expected_patches:
            raise ValueError(
                "Número de predicciones de Mamba incorrecto. "
                f"Recibidas: {predictions.shape[0]}; "
                f"esperadas: {expected_patches}."
            )

        patch_index = 0

        for i in range(H // patch_size):
            for j in range(W // patch_size):

                row_start = i * patch_size
                row_end = row_start + patch_size
                col_start = j * patch_size
                col_end = col_start + patch_size

                if np.any(
                    coverage[
                        row_start:row_end,
                        col_start:col_end
                    ] != 0
                ):
                    raise ValueError(
                        "Solapamiento detectado durante "
                        "la reconstrucción de Mamba."
                    )

                prediction_map[
                    row_start:row_end,
                    col_start:col_end
                ] = predictions[patch_index]

                coverage[
                    row_start:row_end,
                    col_start:col_end
                ] += 1

                patch_index += 1

    # --------------------------------------------------------
    # 10. COMPROBAR COBERTURA COMPLETA
    # --------------------------------------------------------

    if patch_index != expected_patches:
        raise ValueError(
            f"Se procesaron {patch_index} parches; "
            f"se esperaban {expected_patches}."
        )

    if not np.all(coverage == 1):
        uncovered = int(np.sum(coverage == 0))
        overlapping = int(np.sum(coverage > 1))

        raise ValueError(
            "La reconstrucción no cubre cada píxel exactamente una vez. "
            f"Sin cubrir: {uncovered}; "
            f"con solapamiento: {overlapping}."
        )

    if np.any(prediction_map == -1):
        raise ValueError(
            "Hay píxeles sin clasificar en el mapa final."
        )

    # --------------------------------------------------------
    # 11. VALIDAR RESULTADO FINAL
    # --------------------------------------------------------

    prediction_map = validate_prediction_map(
        prediction_map=prediction_map,
        image_shape=image_shape,
        n_classes=8
    )

    print()
    print("=" * 60)
    print(" RESULTADOS")
    print("=" * 60)
    print("Arquitectura:", architecture)
    print("Clases predichas:", np.unique(prediction_map))
    print("Distribución de predicciones:")

    classes, counts = np.unique(
        prediction_map,
        return_counts=True
    )

    for class_id, count in zip(classes, counts):
        print(
            f"  Clase {class_id}: {count} píxeles "
            f"({100.0 * count / prediction_map.size:.2f} %)"
        )

    print("\n[OK] Inferencia y reconstrucción completadas.")

    return prediction_map