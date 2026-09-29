# ============================================================
# abundance_transformer.py
#
# Abundance Transformer para clasificación de píxeles
#
# Entrada:
#   (batch, n_endmembers, 32, 32)
#
# En nuestro caso:
#   (batch, 8, 32, 32)
#
# Los canales de entrada representan los mapas de abundancia
# producidos por el método de unmixing.
#
# Salida:
#   (batch, n_classes, 32, 32)
#
# Predicción:
#   (batch, 32, 32)
#
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 1. PATCH EMBEDDING
# ============================================================

class AbundancePatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim=128,
        patch_size=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ----------------------------------------------------
        # Convierte los mapas de abundancia en embeddings.
        #
        # Entrada:
        #   (B, n_endmembers, 32, 32)
        #
        # Salida:
        #   (B, embed_dim, 8, 8)
        #
        # Con:
        #   patch_size = 4
        #
        # 32 / 4 = 8
        # ----------------------------------------------------

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        return self.projection(x)


# ============================================================
# 2. ABUNDANCE TRANSFORMER
# ============================================================

class AbundanceTransformer(nn.Module):

    def __init__(
        self,
        in_channels=8,
        n_endmembers=8,
        n_classes=8,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        patch_size=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ----------------------------------------------------
        # Patch Embedding
        #
        # Entrada:
        #   (B, 8, 32, 32)
        #
        # Salida:
        #   (B, 128, 8, 8)
        # ----------------------------------------------------

        self.patch_embed = AbundancePatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # ----------------------------------------------------
        # Transformer Encoder
        #
        # Posteriormente convertiremos:
        #
        # (B, 128, 8, 8)
        #
        # en:
        #
        # (B, 64, 128)
        #
        # donde:
        #
        # 64  = número de tokens
        # 128 = dimensión del embedding
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Decoder
        #
        # Entrada:
        #   (B, 128, 8, 8)
        #
        # Primera transposed convolution:
        #
        #   8 x 8 → 16 x 16
        #
        # Segunda:
        #
        #   16 x 16 → 32 x 32
        #
        # Salida:
        #
        #   (B, n_classes, 32, 32)
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                embed_dim,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),

            nn.GELU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),

            nn.Conv2d(
                32,
                n_classes,
                kernel_size=1
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Comprobación de entrada
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                f"Se esperaba una entrada 4D "
                f"(batch, canales, alto, ancho), "
                f"pero se recibió {x.shape}"
            )

        # ----------------------------------------------------
        # Comprobar número de canales
        # ----------------------------------------------------

        if x.shape[1] != self.in_channels:

            raise ValueError(
                f"El modelo espera "
                f"{self.in_channels} mapas de abundancia, "
                f"pero recibió {x.shape[1]}"
            )

        # ----------------------------------------------------
        # Patch embedding
        #
        # (B, 8, 32, 32)
        #       ↓
        # (B, 128, 8, 8)
        # ----------------------------------------------------

        x = self.patch_embed(x)

        # ----------------------------------------------------
        # Guardar dimensiones
        # ----------------------------------------------------

        B, C, H, W = x.shape

        # ----------------------------------------------------
        # Convertir mapa espacial a secuencia de tokens
        #
        # (B, 128, 8, 8)
        #
        #       ↓ flatten
        #
        # (B, 128, 64)
        #
        #       ↓ transpose
        #
        # (B, 64, 128)
        # ----------------------------------------------------

        x = x.flatten(2).transpose(1, 2)

        # ----------------------------------------------------
        # Transformer
        #
        # (B, 64, 128)
        #       ↓
        # (B, 64, 128)
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # Volver a representación espacial
        #
        # (B, 64, 128)
        #
        #       ↓ transpose
        #
        # (B, 128, 64)
        #
        #       ↓ reshape
        #
        # (B, 128, 8, 8)
        # ----------------------------------------------------

        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        # ----------------------------------------------------
        # Decoder
        #
        # (B, 128, 8, 8)
        #       ↓
        # (B, 64, 16, 16)
        #       ↓
        # (B, 32, 32, 32)
        #       ↓
        # (B, n_classes, 32, 32)
        # ----------------------------------------------------

        x = self.decoder(x)

        return x


# ============================================================
# 3. CREAR MODELO
# ============================================================

def create_model(
    n_endmembers=8,
    n_classes=8,
    device="cpu"
):
    """
    Crea el AbundanceTransformer.

    Parámetros
    ----------
    n_endmembers : int
        Número de mapas de abundancia utilizados como
        características de entrada.

    n_classes : int
        Número total de clases que debe predecir el modelo.

    device : str
        'cpu' o 'cuda'.

    Retorna
    -------
    model : AbundanceTransformer
    """

    # --------------------------------------------------------
    # En nuestro caso:
    #
    # n_endmembers = 8
    #
    # Por tanto:
    #
    # in_channels = 8
    # --------------------------------------------------------

    model = AbundanceTransformer(
        in_channels=n_endmembers,
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        patch_size=4
    )

    model = model.to(device)

    return model


# ============================================================
# 4. TEST DEL MODELO
# ============================================================

def test_model(
    n_endmembers=8,
    n_classes=8,
    batch_size=4,
    device="cpu"
):
    """
    Realiza una prueba rápida del modelo.

    Crea una entrada artificial:

        (batch, n_endmembers, 32, 32)

    En nuestro caso:

        (4, 8, 32, 32)

    y verifica que la salida tenga:

        (batch, n_classes, 32, 32)
    """

    print()
    print("=" * 60)
    print(" TEST DEL ABUNDANCE TRANSFORMER")
    print("=" * 60)

    # --------------------------------------------------------
    # Crear modelo
    # --------------------------------------------------------

    model = create_model(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        device=device
    )

    # --------------------------------------------------------
    # Entrada de prueba
    #
    # batch = 4
    # canales = número de mapas de abundancia
    # alto = 32
    # ancho = 32
    # --------------------------------------------------------

    x = torch.randn(
        batch_size,
        n_endmembers,
        32,
        32,
        device=device
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(x)

    # --------------------------------------------------------
    # Predicción
    #
    # Selecciona la clase con mayor logit
    # para cada píxel.
    # --------------------------------------------------------

    prediction = torch.argmax(
        logits,
        dim=1
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print(
        f"Device:       {device}"
    )

    print(
        f"Endmembers:   {n_endmembers}"
    )

    print(
        f"Classes:      {n_classes}"
    )

    print(
        f"Input:        {tuple(x.shape)}"
    )

    print(
        f"Logits:       {tuple(logits.shape)}"
    )

    print(
        f"Prediction:   {tuple(prediction.shape)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Comprobaciones
    # --------------------------------------------------------

    assert x.shape == (
        batch_size,
        n_endmembers,
        32,
        32
    ), (
        f"Input incorrecto: {x.shape}"
    )

    assert logits.shape == (
        batch_size,
        n_classes,
        32,
        32
    ), (
        f"Logits incorrectos: {logits.shape}"
    )

    assert prediction.shape == (
        batch_size,
        32,
        32
    ), (
        f"Prediction incorrecta: {prediction.shape}"
    )

    print()
    print("✓ Input correcto")
    print("✓ Forward correcto")
    print("✓ Dimensiones de salida correctas")
    print("✓ Test completado correctamente")
    print()

    return model, x, logits, prediction


# ============================================================
# 5. PRUEBA DIRECTA
#
# Este bloque solamente se ejecuta si ejecutamos:
#
#     python3 abundance_transformer.py
#
# No se ejecuta cuando lo importamos desde otro archivo.
# ============================================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    test_model(
        n_endmembers=8,
        n_classes=8,
        batch_size=4,
        device=device
    )
