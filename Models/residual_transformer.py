import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFeatureExtractor(nn.Module):

    def __init__(
        self,
        in_channels,
        feature_dim=64
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feature_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        return x


class ResidualPatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim=128,
        patch_size=2
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        return self.projection(x)


class ResidualTransformer(nn.Module):

    def __init__(
        self,
        in_channels=8,
        n_endmembers=8,
        n_classes=8,
        feature_dim=64,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        patch_size=2,
        image_size=32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        if image_size % patch_size != 0:

            raise ValueError(
                "image_size debe ser divisible por patch_size"
            )

        self.patch_grid_size = image_size // patch_size
        self.num_tokens = self.patch_grid_size ** 2

        self.local_features = ResidualFeatureExtractor(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        self.patch_embed = ResidualPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.transformer_projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                feature_dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                feature_dim * 2,
                feature_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        self.residual_head = nn.Conv2d(
            feature_dim,
            n_classes,
            kernel_size=1
        )

        self.residual_scale = nn.Parameter(
            torch.tensor(0.01)
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.residual_head.weight,
            mean=0.0,
            std=1e-3
        )

        if self.residual_head.bias is not None:

            nn.init.zeros_(
                self.residual_head.bias
            )

    def _prepare_base_abundances(self, x):

        if torch.isnan(x).any():

            raise ValueError(
                "La entrada contiene NaN"
            )

        if torch.isinf(x).any():

            raise ValueError(
                "La entrada contiene infinitos"
            )

        x = torch.clamp(
            x,
            min=0.0
        )

        abundance_sum = torch.sum(
            x,
            dim=1,
            keepdim=True
        )

        abundance_sum = torch.clamp(
            abundance_sum,
            min=1e-8
        )

        base_abundances = (
            x / abundance_sum
        )

        return base_abundances

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

        if x.shape[2] != self.image_size:

            raise ValueError(
                f"El modelo espera una altura de "
                f"{self.image_size}, "
                f"pero recibió {x.shape[2]}"
            )

        if x.shape[3] != self.image_size:

            raise ValueError(
                f"El modelo espera un ancho de "
                f"{self.image_size}, "
                f"pero recibió {x.shape[3]}"
            )

        base_abundances = self._prepare_base_abundances(x)

        local_features = self.local_features(
            x
        )

        tokens = self.patch_embed(x)

        B, C, H, W = tokens.shape

        tokens = tokens.flatten(
            2
        ).transpose(
            1,
            2
        )

        tokens = (
            tokens
            + self.positional_embedding
        )

        tokens = self.transformer(
            tokens
        )

        tokens = tokens.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        transformer_features = self.transformer_projection(
            tokens
        )

        transformer_features = F.interpolate(
            transformer_features,
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        )

        fused_features = torch.cat(
            [
                local_features,
                transformer_features
            ],
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )

        delta_logits = self.residual_head(
            fused_features
        )

        delta_logits = (
            self.residual_scale
            * delta_logits
        )

        base_logits = torch.log(
            base_abundances
            + 1e-8
        )

        logits = (
            base_logits
            + delta_logits
        )

        return logits

    def predict_abundances(self, x):

        logits = self.forward(x)

        abundances = torch.softmax(
            logits,
            dim=1
        )

        return abundances

    def predict(self, x):

        logits = self.forward(x)

        prediction = torch.argmax(
            logits,
            dim=1
        )

        return prediction


def create_model(
    n_endmembers=8,
    n_classes=8,
    device="cpu"
):

    model = ResidualTransformer(
        in_channels=n_endmembers,
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        feature_dim=64,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        patch_size=2,
        image_size=32
    )

    model = model.to(device)

    return model


def test_model(
    n_endmembers=8,
    n_classes=8,
    batch_size=4,
    device="cpu"
):

    print()
    print("TEST DEL RESIDUAL TRANSFORMER")
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

        abundances = model.predict_abundances(
            x
        )

        prediction = model.predict(
            x
        )

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
        f"Abundances:   {tuple(abundances.shape)}"
    )

    print(
        f"Prediction:   {tuple(prediction.shape)}"
    )

    print(
        f"Tokens:       {model.num_tokens}"
    )

    print(
        f"Residual scale: "
        f"{model.residual_scale.item():.6f}"
    )

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

    assert abundances.shape == (
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

    abundance_sum = abundances.sum(
        dim=1
    )

    assert torch.allclose(
        abundance_sum,
        torch.ones_like(abundance_sum),
        atol=1e-5
    )

    print()
    print("Input correcto")
    print("Forward correcto")
    print("Abundancias correctas")
    print("Sum-to-one correcto")
    print("Dimensiones correctas")
    print("Test completado")
    print()

    return (
        model,
        x,
        logits,
        abundances,
        prediction
    )


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
