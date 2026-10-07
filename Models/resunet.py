import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BLOQUE RESIDUAL
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.activation = nn.GELU()

        # Si cambian los canales, adaptamos
        # la conexión residual.
        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual

        out = self.activation(out)

        return out


# ============================================================
# BLOQUE DE ENCODER
# ============================================================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.residual = ResidualBlock(
            in_channels,
            out_channels
        )

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.residual(x)

        down = self.downsample(
            features
        )

        return features, down


# ============================================================
# BLOQUE DE DECODER
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.residual = ResidualBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        # Seguridad ante pequeñas diferencias
        # de tamaño espacial.
        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        x = self.residual(x)

        return x


# ============================================================
# RES-U-NET
# ============================================================

class ResUNet(nn.Module):

    def __init__(
        self,
        in_channels=8,
        n_classes=8,
        base_channels=32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder1 = EncoderBlock(
            in_channels,
            base_channels
        )

        self.encoder2 = EncoderBlock(
            base_channels,
            base_channels * 2
        )

        self.encoder3 = EncoderBlock(
            base_channels * 2,
            base_channels * 4
        )

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = ResidualBlock(
            base_channels * 4,
            base_channels * 8
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder3 = DecoderBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4
        )

        self.decoder2 = DecoderBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2
        )

        self.decoder1 = DecoderBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels
        )

        # ----------------------------------------------------
        # CLASIFICACIÓN
        # ----------------------------------------------------

        self.classifier = nn.Conv2d(
            base_channels,
            n_classes,
            kernel_size=1
        )

        self._initialize_weights()

    # ========================================================
    # INICIALIZACIÓN
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(
                module,
                nn.ConvTranspose2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        if x.ndim != 4:

            raise ValueError(
                f"Se esperaba una entrada 4D "
                f"(batch, canales, alto, ancho), "
                f"pero se recibió {x.shape}"
            )

        if x.shape[1] != self.in_channels:

            raise ValueError(
                f"El modelo espera "
                f"{self.in_channels} canales, "
                f"pero recibió {x.shape[1]}"
            )

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        skip1, x = self.encoder1(x)

        skip2, x = self.encoder2(x)

        skip3, x = self.encoder3(x)

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        x = self.bottleneck(x)

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        x = self.decoder3(
            x,
            skip3
        )

        x = self.decoder2(
            x,
            skip2
        )

        x = self.decoder1(
            x,
            skip1
        )

        # ----------------------------------------------------
        # LOGITS
        # ----------------------------------------------------

        logits = self.classifier(x)

        return logits

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    def predict(
        self,
        x
    ):

        logits = self.forward(x)

        prediction = torch.argmax(
            logits,
            dim=1
        )

        return prediction


# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

def create_model(
    n_endmembers=8,
    n_classes=8,
    device="cpu"
):

    model = ResUNet(
        in_channels=n_endmembers,
        n_classes=n_classes,
        base_channels=32
    )

    model = model.to(device)

    return model


# ============================================================
# PRUEBA DEL MODELO
# ============================================================

def test_model(
    n_endmembers=8,
    n_classes=8,
    batch_size=4,
    device="cpu"
):

    print()
    print("=" * 60)
    print("TEST DEL RES-U-NET")
    print("=" * 60)
    print()

    model = create_model(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        device=device
    )

    x = torch.rand(
        batch_size,
        n_endmembers,
        32,
        32,
        device=device
    )

    model.eval()

    with torch.no_grad():

        logits = model(x)

        prediction = model.predict(x)

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

    print()

    assert x.shape == (
        batch_size,
        n_endmembers,
        32,
        32
    )

    assert logits.shape == (
        batch_size,
        n_classes,
        32,
        32
    )

    assert prediction.shape == (
        batch_size,
        32,
        32
    )

    print("✓ Entrada correcta")
    print("✓ Encoder correcto")
    print("✓ Bottleneck correcto")
    print("✓ Decoder correcto")
    print("✓ Skip connections correctas")
    print("✓ Logits correctos")
    print("✓ Prediction correcta")
    print()
    print("TEST COMPLETADO")
    print()

    return (
        model,
        x,
        logits,
        prediction
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    if torch.backends.mps.is_available():

        device = "mps"

    elif torch.cuda.is_available():

        device = "cuda"

    else:

        device = "cpu"

    test_model(
        n_endmembers=8,
        n_classes=8,
        batch_size=4,
        device=device
    )