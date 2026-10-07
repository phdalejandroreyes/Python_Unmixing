# ================================================================
# TEST DE INTEGRACIÓN: PIPELINE DE ABUNDANCIAS
# ================================================================
#
# Verifica la integración entre los componentes principales del
# pipeline de clasificación basado en mapas de abundancias:
#
#   A_initial
#       ↓
#   AbundanceDataset
#       ↓
#   DataLoader
#       ↓
#   AbundanceTransformer
#       ↓
#   logits
#       ↓
#   predicción
#       ↓
#   CrossEntropyLoss
#
# Este script no realiza entrenamiento ni inferencia sobre un
# modelo previamente entrenado. Su objetivo es comprobar que las
# interfaces entre Dataset, DataLoader y modelo funcionan
# correctamente y que las dimensiones de entrada y salida son
# compatibles.
#
# Uso:
#   - Pruebas durante el desarrollo.
#   - Detección de errores después de modificar dataset.py o
#     abundance_transformer.py.
#   - Verificación rápida de la integración del pipeline.
#
# No forma parte del flujo principal de entrenamiento/inferencia.
# ================================================================


# ============================================================
# integration_test.py
#
# Prueba de integración:
#
# A_initial
#    ↓
# AbundanceDataset
#    ↓
# DataLoader
#    ↓
# AbundanceTransformer
#    ↓
# Prediction
#    ↓
# CrossEntropyLoss
#
# IMPORTANTE:
# Aquí todavía NO entrenamos el modelo.
# Solo comprobamos que todo está conectado correctamente
# utilizando las abundancias reales generadas por el método
# de unmixing.
#
# Entrada esperada:
#
#   A_initial = (8, 262144)
#   labels    = (262144,)
#
# Salida del Dataset:
#
#   X = (4, 8, 32, 32)
#   y = (4, 32, 32)
#
# Salida del Transformer:
#
#   logits     = (4, 8, 32, 32)
#   prediction = (4, 32, 32)
#
# ============================================================

import torch
import torch.nn as nn

from abundance_transformer import create_model
from dataset import create_dataloader


# ============================================================
# CONFIGURACIÓN
# ============================================================

BATCH_SIZE = 4

N_ENDMEMBERS = 8
N_CLASSES = 8

PATCH_SIZE = 32

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FUNCIÓN PRINCIPAL DE INTEGRACIÓN
# ============================================================

def test_integration(
    A_initial,
    labels,
    image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)
):
    """
    Comprueba la integración completa:

        A_initial
            ↓
        Dataset
            ↓
        DataLoader
            ↓
        Transformer
            ↓
        Loss

    Parámetros
    ----------
    A_initial : numpy.ndarray
        Abundancias iniciales.

        Shape esperado:
            (8, 262144)

    labels : numpy.ndarray
        Labels de cada píxel.

        Shape esperado:
            (262144,)

    image_shape : tuple
        Dimensiones espaciales de la imagen.

        Por defecto:
            (512, 512)
    """

    # ========================================================
    # INFORMACIÓN INICIAL
    # ========================================================

    print()
    print("=" * 60)
    print(" TEST DE INTEGRACIÓN")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # COMPROBAR A_initial
    # ========================================================

    print()
    print("A_initial")
    print("-" * 60)

    print(
        f"Shape: {A_initial.shape}"
    )

    print(
        f"Min:   {A_initial.min()}"
    )

    print(
        f"Max:   {A_initial.max()}"
    )

    # --------------------------------------------------------
    # Comprobación
    # --------------------------------------------------------

    assert A_initial.ndim == 2, (
        f"A_initial debe ser 2D. "
        f"Shape recibido: {A_initial.shape}"
    )

    assert A_initial.shape[0] == N_ENDMEMBERS, (
        f"Se esperaban {N_ENDMEMBERS} "
        f"mapas de abundancia, "
        f"pero se recibieron "
        f"{A_initial.shape[0]}"
    )

    # ========================================================
    # COMPROBAR LABELS
    # ========================================================

    print()
    print("Labels")
    print("-" * 60)

    print(
        f"Shape: {labels.shape}"
    )

    print(
        f"Min:   {labels.min()}"
    )

    print(
        f"Max:   {labels.max()}"
    )

    # ========================================================
    # CREAR DATASET Y DATALOADER
    # ========================================================

    dataset, dataloader = create_dataloader(
        A_initial=A_initial,
        labels=labels,
        image_shape=image_shape,
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print()
    print("Dataset")
    print("-" * 60)

    print(
        f"Número de parches: "
        f"{len(dataset)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    # ========================================================
    # CREAR MODELO
    # ========================================================

    model = create_model(
        n_endmembers=N_ENDMEMBERS,
        n_classes=N_CLASSES,
        device=DEVICE
    )

    model.eval()

    print()
    print("Modelo")
    print("-" * 60)

    print(
        f"Endmembers / canales: "
        f"{N_ENDMEMBERS}"
    )

    print(
        f"Clases: "
        f"{N_CLASSES}"
    )

    # ========================================================
    # OBTENER UN BATCH REAL
    # ========================================================

    X, y = next(
        iter(dataloader)
    )

    print()
    print("Datos reales")
    print("-" * 60)

    print(
        f"X shape: "
        f"{tuple(X.shape)}"
    )

    print(
        f"y shape: "
        f"{tuple(y.shape)}"
    )

    print(
        f"X dtype: "
        f"{X.dtype}"
    )

    print(
        f"y dtype: "
        f"{y.dtype}"
    )

    # ========================================================
    # COMPROBAR BATCH
    # ========================================================

    assert X.shape == (
        BATCH_SIZE,
        N_ENDMEMBERS,
        PATCH_SIZE,
        PATCH_SIZE
    ), (
        f"Shape de X incorrecto: "
        f"{X.shape}"
    )

    assert y.shape == (
        BATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE
    ), (
        f"Shape de y incorrecto: "
        f"{y.shape}"
    )

    # ========================================================
    # MOVER DATOS AL DEVICE
    # ========================================================

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    # ========================================================
    # FORWARD
    # ========================================================

    with torch.no_grad():

        logits = model(X)

    print()
    print("Salida del modelo")
    print("-" * 60)

    print(
        f"logits shape: "
        f"{tuple(logits.shape)}"
    )

    # ========================================================
    # COMPROBAR LOGITS
    # ========================================================

    assert logits.shape == (
        BATCH_SIZE,
        N_CLASSES,
        PATCH_SIZE,
        PATCH_SIZE
    ), (
        f"Shape de logits incorrecto: "
        f"{logits.shape}"
    )

    # ========================================================
    # CALCULAR PREDICCIÓN
    # ========================================================

    prediction = torch.argmax(
        logits,
        dim=1
    )

    print(
        f"prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    # ========================================================
    # COMPROBAR PREDICCIÓN
    # ========================================================

    assert prediction.shape == (
        BATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE
    ), (
        f"Shape de prediction incorrecto: "
        f"{prediction.shape}"
    )

    # ========================================================
    # CALCULAR LOSS
    #
    # CrossEntropyLoss espera:
    #
    # logits:
    #     (batch, clases, alto, ancho)
    #
    # labels:
    #     (batch, alto, ancho)
    #
    # Exactamente lo que tenemos.
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        y
    )

    print()
    print("Loss")
    print("-" * 60)

    print(
        f"Loss: "
        f"{loss.item():.6f}"
    )

    # ========================================================
    # COMPROBAR LABELS
    # ========================================================

    assert y.min() >= 0, (
        f"Se encontraron labels negativos: "
        f"{y.min().item()}"
    )

    assert y.max() < N_CLASSES, (
        f"Se encontró un label "
        f"{y.max().item()}, "
        f"pero el modelo tiene "
        f"{N_CLASSES} clases."
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    print()
    print("=" * 60)
    print("✓ A_initial recibida correctamente")
    print("✓ Dataset creado correctamente")
    print("✓ Parches de abundancia correctos")
    print("✓ DataLoader funcionando correctamente")
    print("✓ Transformer conectado correctamente")
    print("✓ Forward con abundancias reales correcto")
    print("✓ Prediction calculada correctamente")
    print("✓ Loss calculada correctamente")
    print("=" * 60)
    print()

    return model, X, y, logits, prediction, loss
