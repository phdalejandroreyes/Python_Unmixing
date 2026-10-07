import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EXTRACTOR DE CARACTERÍSTICAS DE ABUNDANCIAS
# ============================================================

class AbundanceFeatureExtractor(nn.Module):

    def __init__(
        self,
        in_channels,
        feature_dim=64
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.features(x)


# ============================================================
# EMBEDDING DE TOKENS ESPACIALES
# ============================================================

class SpatialTokenEmbedding(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim=128,
        patch_size=2
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.projection(x)

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        return tokens, H, W


# ============================================================
# EMBEDDING DE ENDMEMBERS
# ============================================================

class EndmemberEmbedding(nn.Module):

    def __init__(
        self,
        spectral_bands=32,
        embed_dim=128
    ):
        super().__init__()

        self.embedding = nn.Sequential(

            nn.Linear(
                spectral_bands,
                embed_dim
            ),

            nn.LayerNorm(embed_dim),

            nn.GELU(),

            nn.Linear(
                embed_dim,
                embed_dim
            ),

            nn.LayerNorm(embed_dim)
        )

    def forward(self, endmembers):

        return self.embedding(endmembers)


# ============================================================
# CROSS ATTENTION
# ============================================================

class EndmemberCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                embed_dim * 2,
                embed_dim
            )
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries,
        keys,
        values
    ):

        attention_output, attention_weights = self.attention(
            query=queries,
            key=keys,
            value=values
        )

        x = self.norm1(
            queries + attention_output
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x, attention_weights


# ============================================================
# CROSS-ATTENTION TRANSFORMER
# ============================================================

class CrossAttentionTransformer(nn.Module):

    def __init__(
        self,
        n_endmembers=8,
        n_classes=8,
        spectral_bands=32,
        feature_dim=64,
        embed_dim=128,
        num_heads=4,
        num_transformer_layers=2,
        patch_size=2,
        dropout=0.1
    ):
        super().__init__()

        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.spectral_bands = spectral_bands
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # CARACTERÍSTICAS LOCALES DE ABUNDANCIAS
        # ----------------------------------------------------

        self.abundance_feature_extractor = (
            AbundanceFeatureExtractor(
                in_channels=n_endmembers,
                feature_dim=feature_dim
            )
        )

        # ----------------------------------------------------
        # TOKENS ESPACIALES
        # ----------------------------------------------------

        self.spatial_embedding = SpatialTokenEmbedding(
            in_channels=feature_dim,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # ----------------------------------------------------
        # POSICIÓN ESPACIAL
        # ----------------------------------------------------

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                256,
                embed_dim
            )
        )

        nn.init.trunc_normal_(
            self.positional_embedding,
            std=0.02
        )

        # ----------------------------------------------------
        # TRANSFORMER ESPACIAL
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # ----------------------------------------------------
        # EMBEDDING DE ENDMEMBERS
        # ----------------------------------------------------

        self.endmember_embedding = EndmemberEmbedding(
            spectral_bands=spectral_bands,
            embed_dim=embed_dim
        )

        # ----------------------------------------------------
        # CROSS ATTENTION
        #
        # Query  = endmembers
        # Key    = spatial tokens
        # Value  = spatial tokens
        # ----------------------------------------------------

        self.cross_attention = EndmemberCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ----------------------------------------------------
        # PROYECCIÓN DE ENDMEMBERS
        # ----------------------------------------------------

        self.endmember_projection = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim,
                feature_dim
            )
        )

        # ----------------------------------------------------
        # PROYECCIÓN DE CARACTERÍSTICAS ESPACIALES
        # ----------------------------------------------------

        self.spatial_projection = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                feature_dim,
                kernel_size=1
            ),

            nn.BatchNorm2d(feature_dim),

            nn.GELU()
        )

        # ----------------------------------------------------
        # FUSIÓN
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(
                feature_dim * 2,
                feature_dim,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(feature_dim),

            nn.GELU(),

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(feature_dim),

            nn.GELU()
        )

        # ----------------------------------------------------
        # CABEZA DE CLASIFICACIÓN
        # ----------------------------------------------------

        self.abundance_head = nn.Conv2d(
            feature_dim,
            n_classes,
            kernel_size=1
        )

        # ----------------------------------------------------
        # ESCALA DE LA CORRECCIÓN RESIDUAL
        # ----------------------------------------------------

        self.residual_scale = nn.Parameter(
            torch.tensor(0.01)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        endmembers
    ):

        # ----------------------------------------------------
        # x:
        # (B, 8, 32, 32)
        #
        # endmembers:
        # (8, 32)
        # ----------------------------------------------------

        B, _, H, W = x.shape

        # ----------------------------------------------------
        # CARACTERÍSTICAS LOCALES
        # ----------------------------------------------------

        local_features = (
            self.abundance_feature_extractor(x)
        )

        # Shape:
        # (B, 64, 32, 32)

        # ----------------------------------------------------
        # TOKENS ESPACIALES
        # ----------------------------------------------------

        spatial_tokens, token_H, token_W = (
            self.spatial_embedding(
                local_features
            )
        )

        # Shape:
        # (B, 256, 128)

        # ----------------------------------------------------
        # POSICIÓN
        # ----------------------------------------------------

        spatial_tokens = (
            spatial_tokens
            +
            self.positional_embedding[
                :, :spatial_tokens.shape[1], :
            ]
        )

        # ----------------------------------------------------
        # TRANSFORMER ESPACIAL
        # ----------------------------------------------------

        spatial_tokens = self.spatial_transformer(
            spatial_tokens
        )

        # ----------------------------------------------------
        # RESTAURAR MAPA ESPACIAL
        # ----------------------------------------------------

        spatial_features = (
            spatial_tokens
            .transpose(1, 2)
            .reshape(
                B,
                self.embed_dim,
                token_H,
                token_W
            )
        )

        # ----------------------------------------------------
        # PROYECCIÓN ESPACIAL
        # ----------------------------------------------------

        spatial_features = self.spatial_projection(
            spatial_features
        )

        # Shape:
        # (B, 64, 16, 16)

        # ----------------------------------------------------
        # INTERPOLAR A 32x32
        # ----------------------------------------------------

        spatial_features = F.interpolate(
            spatial_features,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # Shape:
        # (B, 64, 32, 32)

        # ----------------------------------------------------
        # ENDMEMBERS
        # ----------------------------------------------------

        if endmembers.dim() == 2:

            endmembers = endmembers.unsqueeze(0)

        # Shape:
        # (1, 8, 32)

        if endmembers.shape[0] == 1:

            endmembers = endmembers.expand(
                B,
                -1,
                -1
            )

        # ----------------------------------------------------
        # EMBEDDING DE ENDMEMBERS
        # ----------------------------------------------------

        endmember_features = (
            self.endmember_embedding(
                endmembers
            )
        )

        # Shape:
        # (B, 8, 128)

        # ----------------------------------------------------
        # CROSS ATTENTION
        # ----------------------------------------------------
        #
        # Query:
        #   endmembers
        #
        # Key / Value:
        #   spatial tokens
        #
        # ----------------------------------------------------

        endmember_features, attention_weights = (
            self.cross_attention(
                queries=endmember_features,
                keys=spatial_tokens,
                values=spatial_tokens
            )
        )

        # Shape:
        # (B, 8, 128)

        # ----------------------------------------------------
        # PROYECCIÓN DE ENDMEMBERS
        # ----------------------------------------------------

        endmember_features = (
            self.endmember_projection(
                endmember_features
            )
        )

        # Shape:
        # (B, 8, 64)

        # ====================================================
        # AGREGACIÓN ORIGINAL
        # ====================================================
        #
        # Se conserva la arquitectura original:
        #
        # (B, 8, 64)
        #       ↓ mean(dim=1)
        # (B, 64)
        #
        # Esto genera un contexto global de endmembers.
        # ====================================================

        endmember_context = (
            endmember_features.mean(
                dim=1
            )
        )

        # Shape:
        # (B, 64)

        # ----------------------------------------------------
        # EXPANDIR CONTEXTO A MAPA ESPACIAL
        # ----------------------------------------------------

        endmember_context = (
            endmember_context
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                H,
                W
            )
        )

        # Shape:
        # (B, 64, 32, 32)

        # ----------------------------------------------------
        # FUSIÓN
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                spatial_features,
                endmember_context
            ],
            dim=1
        )

        # Shape:
        # (B, 128, 32, 32)

        fused_features = self.fusion(
            fused_features
        )

        # Shape:
        # (B, 64, 32, 32)

        # ----------------------------------------------------
        # DELTA LOGITS
        # ----------------------------------------------------

        delta_logits = self.abundance_head(
            fused_features
        )

        # Shape:
        # (B, 8, 32, 32)

        # ----------------------------------------------------
        # BASE LOGITS
        # ----------------------------------------------------
        #
        # Las abundancias se convierten en logits.
        #
        # Se utiliza log para mantener la información
        # relativa de las abundancias.
        # ----------------------------------------------------

        eps = 1e-6

        base_logits = torch.log(
            x + eps
        )

        # ----------------------------------------------------
        # CORRECCIÓN RESIDUAL
        # ----------------------------------------------------

        logits = (
            base_logits
            +
            self.residual_scale * delta_logits
        )

        return logits

    # ========================================================
    # PREDICCIÓN DE ABUNDANCIAS
    # ========================================================

    def predict_abundances(
        self,
        x,
        endmembers
    ):

        logits = self.forward(
            x,
            endmembers
        )

        return torch.softmax(
            logits,
            dim=1
        )


# ============================================================
# CREAR MODELO
# ============================================================

def create_model(
    n_endmembers=8,
    n_classes=8,
    spectral_bands=32,
    device=None
):

    model = CrossAttentionTransformer(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        spectral_bands=spectral_bands
    )

    if device is not None:

        model = model.to(device)

    return model


# ============================================================
# TEST DEL MODELO
# ============================================================

def test_model():

    print("=" * 60)
    print("TEST CROSS-ATTENTION TRANSFORMER")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Datos ficticios
    # --------------------------------------------------------

    batch_size = 2
    n_endmembers = 8
    n_classes = 8
    spectral_bands = 32

    x = torch.rand(
        batch_size,
        n_endmembers,
        32,
        32,
        device=device
    )

    endmembers = torch.rand(
        n_endmembers,
        spectral_bands,
        device=device
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = create_model(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        spectral_bands=spectral_bands,
        device=device
    )

    model.eval()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            x,
            endmembers
        )

    print()
    print("Input:", x.shape)
    print("Endmembers:", endmembers.shape)
    print("Output:", logits.shape)

    # --------------------------------------------------------
    # Verificación
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        n_classes,
        32,
        32
    )

    assert logits.shape == expected_shape

    print()
    print("Modelo funcionando correctamente.")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_model()