# ============================================================
# mamba_transformer.py
#
# SELECTIVE SSM / MAMBA PARA ABUNDANCE MAPPING
#
# Versión:
#   PyTorch puro
#
# Diseñada para:
#   - CPU
#   - Apple Silicon / MPS
#   - NVIDIA / CUDA
#
# NO requiere:
#   - mamba-ssm
#   - CUDA
#   - nvcc
#   - causal-conv1d
#
# IMPORTANTE:
#
# Esta implementación reproduce el PRINCIPIO de un
# Selective State Space Model utilizado por Mamba,
# pero no es la implementación CUDA optimizada oficial
# de mamba-ssm.
#
# ============================================================


import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 2. PATCH EMBEDDING
# ============================================================

class AbundancePatchEmbedding(nn.Module):
    """
    Convierte mapas de abundancia en tokens espaciales.

    Entrada:
        (B, N, 32, 32)

    Salida:
        (B, D, 8, 8)

    Con patch_size = 4:

        32 / 4 = 8

    Por tanto:

        8 x 8 = 64 tokens
    """

    def __init__(
        self,
        n_endmembers=8,
        embed_dim=128,
        patch_size=4
    ):

        super().__init__()

        self.n_endmembers = n_endmembers
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            n_endmembers,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.BatchNorm2d(
            embed_dim
        )

    def forward(self, x):

        x = self.proj(x)

        x = self.norm(x)

        x = F.gelu(x)

        return x


# ============================================================
# 3. POSITIONAL EMBEDDING 2D
# ============================================================

class SpatialPositionalEmbedding(nn.Module):
    """
    Positional embedding aprendido.

    Para 32x32 con patch_size=4:

        grid = 8x8
        tokens = 64
    """

    def __init__(
        self,
        grid_size=8,
        embed_dim=128
    ):

        super().__init__()

        self.grid_size = grid_size

        self.num_tokens = (
            grid_size * grid_size
        )

        self.embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                embed_dim
            )
        )

        nn.init.normal_(
            self.embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, x):

        return x + self.embedding


# ============================================================
# 4. SELECTIVE SSM
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Implementación PyTorch pura de un Selective State Space Model.

    Entrada:

        (B, L, D)

    Salida:

        (B, L, D)

    El estado se actualiza secuencialmente:

        h_t =
            A_t * h_{t-1}
            +
            B_t * x_t

        y_t =
            C_t * h_t
            +
            D * x_t

    La selección depende de la entrada:

        dt = f(x)

        B  = f(x)

        C  = f(x)

    Esto es lo que permite que el estado sea
    dependiente de la entrada.
    """

    def __init__(
        self,
        dim=128,
        state_dim=16,
        dt_rank=8,
        dropout=0.0
    ):

        super().__init__()

        self.dim = dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        # ----------------------------------------------------
        # Proyección para dt
        # ----------------------------------------------------

        self.dt_proj = nn.Sequential(

            nn.Linear(
                dim,
                dt_rank
            ),

            nn.GELU(),

            nn.Linear(
                dt_rank,
                dim
            )
        )

        # ----------------------------------------------------
        # B(x)
        # ----------------------------------------------------

        self.B_proj = nn.Linear(
            dim,
            state_dim
        )

        # ----------------------------------------------------
        # C(x)
        # ----------------------------------------------------

        self.C_proj = nn.Linear(
            dim,
            state_dim
        )

        # ----------------------------------------------------
        # A
        #
        # Queremos A < 0 para estabilidad.
        #
        # A = -exp(log_A)
        # ----------------------------------------------------

        self.log_A = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    state_dim + 1,
                    dtype=torch.float32
                )
                .unsqueeze(0)
                .repeat(
                    dim,
                    1
                )
            )
        )

        # ----------------------------------------------------
        # D
        #
        # Conexión directa.
        # ----------------------------------------------------

        self.D = nn.Parameter(
            torch.ones(dim)
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            dim,
            dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(
            self.dt_proj[0].weight,
            gain=0.5
        )

        nn.init.zeros_(
            self.dt_proj[0].bias
        )

        nn.init.xavier_uniform_(
            self.dt_proj[2].weight,
            gain=0.5
        )

        nn.init.zeros_(
            self.dt_proj[2].bias
        )

        nn.init.xavier_uniform_(
            self.B_proj.weight,
            gain=0.5
        )

        nn.init.zeros_(
            self.B_proj.bias
        )

        nn.init.xavier_uniform_(
            self.C_proj.weight,
            gain=0.5
        )

        nn.init.zeros_(
            self.C_proj.bias
        )

        nn.init.xavier_uniform_(
            self.out_proj.weight,
            gain=0.5
        )

        nn.init.zeros_(
            self.out_proj.bias
        )

    def forward(self, x):

        B, L, D = x.shape

        # ----------------------------------------------------
        # dt
        # ----------------------------------------------------

        dt = self.dt_proj(x)

        dt = F.softplus(
            dt
        )

        # Limitar dt para evitar inestabilidad
        dt = torch.clamp(
            dt,
            min=1e-4,
            max=1.0
        )

        # ----------------------------------------------------
        # B(x)
        # ----------------------------------------------------

        B_t = torch.tanh(
            self.B_proj(x)
        )

        # ----------------------------------------------------
        # C(x)
        # ----------------------------------------------------

        C_t = torch.tanh(
            self.C_proj(x)
        )

        # ----------------------------------------------------
        # A
        #
        # A siempre negativo.
        # ----------------------------------------------------

        A = -torch.exp(
            self.log_A
        )

        # ----------------------------------------------------
        # Estado inicial
        #
        # (B, D, state_dim)
        # ----------------------------------------------------

        state = torch.zeros(
            B,
            D,
            self.state_dim,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        # ----------------------------------------------------
        # Recurrencia
        #
        # Se mantiene explícitamente en Python para:
        #
        # - compatibilidad MPS
        # - evitar CUDA
        # - facilitar diagnóstico
        #
        # No es tan rápida como un kernel Mamba optimizado.
        # ----------------------------------------------------

        for t in range(L):

            x_t = x[:, t, :]

            dt_t = dt[:, t, :]

            B_current = B_t[:, t, :]

            C_current = C_t[:, t, :]

            # ------------------------------------------------
            # discretización simple:
            #
            # A_bar = exp(dt * A)
            # ------------------------------------------------

            A_bar = torch.exp(
                dt_t.unsqueeze(-1)
                * A.unsqueeze(0)
            )

            # ------------------------------------------------
            # input contribution
            # ------------------------------------------------

            input_term = (
                dt_t.unsqueeze(-1)
                * B_current.unsqueeze(1)
                * x_t.unsqueeze(-1)
            )

            # ------------------------------------------------
            # estado
            # ------------------------------------------------

            state = (
                A_bar * state
                +
                input_term
            )

            # ------------------------------------------------
            # salida
            # ------------------------------------------------

            y_t = (
                torch.sum(
                    C_current.unsqueeze(1)
                    * state,
                    dim=-1
                )
                +
                self.D.unsqueeze(0)
                * x_t
            )

            outputs.append(
                y_t.unsqueeze(1)
            )

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        y = torch.cat(
            outputs,
            dim=1
        )

        y = self.out_proj(
            y
        )

        y = self.dropout(
            y
        )

        return y


# ============================================================
# 5. BIDIRECTIONAL SELECTIVE SSM
# ============================================================

class BidirectionalSelectiveSSM(nn.Module):
    """
    Procesa los tokens:

        izquierda -> derecha

    y también:

        derecha -> izquierda

    Esto es especialmente útil para imágenes porque la
    secuencia rasterizada introduce una dirección artificial.
    """

    def __init__(
        self,
        dim=128,
        state_dim=16,
        dt_rank=8,
        dropout=0.1
    ):

        super().__init__()

        self.forward_ssm = SelectiveSSM(
            dim=dim,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        self.backward_ssm = SelectiveSSM(
            dim=dim,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        self.output_projection = nn.Linear(
            dim * 2,
            dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        y_forward = self.forward_ssm(
            x
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        x_reverse = torch.flip(
            x,
            dims=[1]
        )

        y_backward = self.backward_ssm(
            x_reverse
        )

        y_backward = torch.flip(
            y_backward,
            dims=[1]
        )

        # ----------------------------------------------------
        # Combinar
        # ----------------------------------------------------

        y = torch.cat(
            [
                y_forward,
                y_backward
            ],
            dim=-1
        )

        y = self.output_projection(
            y
        )

        return y


# ============================================================
# 6. MAMBA BLOCK
# ============================================================

class MambaBlock(nn.Module):

    def __init__(
        self,
        dim=128,
        state_dim=16,
        dt_rank=8,
        dropout=0.1,
        mlp_ratio=2
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.ssm = BidirectionalSelectiveSSM(
            dim=dim,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                dim
            ),

            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # SSM residual
        # ----------------------------------------------------

        residual = x

        x = self.norm1(
            x
        )

        x = self.ssm(
            x
        )

        x = residual + x

        # ----------------------------------------------------
        # MLP residual
        # ----------------------------------------------------

        residual = x

        x = self.norm2(
            x
        )

        x = self.mlp(
            x
        )

        x = residual + x

        return x


# ============================================================
# 7. VISION MAMBA
# ============================================================

class VisionMamba(nn.Module):

    def __init__(
        self,
        dim=128,
        depth=3,
        state_dim=16,
        dt_rank=8,
        dropout=0.1
    ):

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    dim=dim,
                    state_dim=state_dim,
                    dt_rank=dt_rank,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            dim
        )

    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        x = self.norm(
            x
        )

        return x


# ============================================================
# 8. DECODER
# ============================================================

class MambaDecoder(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        n_classes=8
    ):

        super().__init__()

        self.decoder = nn.Sequential(

            # 8 -> 16

            nn.ConvTranspose2d(
                embed_dim,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.GELU(),

            # 16 -> 32

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),

            # clasificación

            nn.Conv2d(
                32,
                n_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        return self.decoder(
            x
        )


# ============================================================
# 9. MAMBA TRANSFORMER
# ============================================================

class MambaTransformer(nn.Module):

    def __init__(
        self,
        n_endmembers=8,
        n_classes=8,
        image_size=32,
        embed_dim=128,
        patch_size=4,
        depth=3,
        state_dim=16,
        dt_rank=8,
        dropout=0.1,
        use_abundance_prior=True
    ):

        super().__init__()

        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_abundance_prior = (
            use_abundance_prior
        )

        # ----------------------------------------------------
        # Validaciones
        # ----------------------------------------------------

        if image_size % patch_size != 0:

            raise ValueError(
                "image_size debe ser divisible "
                "por patch_size"
            )

        if use_abundance_prior:

            if n_endmembers != n_classes:

                raise ValueError(
                    "El abundance prior requiere "
                    "n_endmembers == n_classes"
                )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        self.grid_size = (
            image_size // patch_size
        )

        self.num_tokens = (
            self.grid_size
            * self.grid_size
        )

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = (
            AbundancePatchEmbedding(
                n_endmembers=n_endmembers,
                embed_dim=embed_dim,
                patch_size=patch_size
            )
        )

        # ----------------------------------------------------
        # Posición
        # ----------------------------------------------------

        self.positional_embedding = (
            SpatialPositionalEmbedding(
                grid_size=self.grid_size,
                embed_dim=embed_dim
            )
        )

        # ----------------------------------------------------
        # Mamba
        # ----------------------------------------------------

        self.mamba = VisionMamba(
            dim=embed_dim,
            depth=depth,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = MambaDecoder(
            embed_dim=embed_dim,
            n_classes=n_classes
        )

        # ----------------------------------------------------
        # Residual scale
        #
        # NO lo hacemos extremadamente pequeño.
        #
        # 0.1 permite que el prior sea importante al inicio
        # pero que Mamba pueda aprender una corrección.
        # ----------------------------------------------------

        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self._initialize_weights()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear
            ):

                if module.weight is not None:

                    nn.init.xavier_uniform_(
                        module.weight,
                        gain=0.5
                    )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # PREPARE ABUNDANCES
    # ========================================================

    def _prepare_base_abundances(
        self,
        x
    ):
        """
        Convierte los canales de entrada en una distribución
        composicional por píxel.

        x >= 0

        base_abundances =
            x / sum(x)
        """

        if torch.isnan(x).any():

            raise ValueError(
                "La entrada contiene NaN"
            )

        if torch.isinf(x).any():

            raise ValueError(
                "La entrada contiene Inf"
            )

        # ----------------------------------------------------
        # Abundancias no negativas
        # ----------------------------------------------------

        x = torch.clamp(
            x,
            min=0.0
        )

        # ----------------------------------------------------
        # Suma por píxel
        # ----------------------------------------------------

        abundance_sum = torch.sum(
            x,
            dim=1,
            keepdim=True
        )

        abundance_sum = torch.clamp(
            abundance_sum,
            min=1e-8
        )

        # ----------------------------------------------------
        # Normalización composicional
        # ----------------------------------------------------

        base_abundances = (
            x / abundance_sum
        )

        return base_abundances

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                f"Se esperaba "
                f"(B,C,H,W), recibido {x.shape}"
            )

        if x.shape[1] != self.n_endmembers:

            raise ValueError(
                f"Se esperaban "
                f"{self.n_endmembers} canales, "
                f"recibidos {x.shape[1]}"
            )

        if x.shape[2] != self.image_size:

            raise ValueError(
                f"Altura incorrecta: "
                f"{x.shape[2]}"
            )

        if x.shape[3] != self.image_size:

            raise ValueError(
                f"Anchura incorrecta: "
                f"{x.shape[3]}"
            )

        # ====================================================
        # ABUNDANCE PRIOR
        # ====================================================

        if self.use_abundance_prior:

            base_abundances = (
                self._prepare_base_abundances(
                    x
                )
            )

        else:

            base_abundances = None

        # ====================================================
        # PATCH EMBEDDING
        # ====================================================

        x = self.patch_embed(
            x
        )

        # ====================================================
        # FEATURE MAP
        # ====================================================

        B, C, H, W = x.shape

        # ====================================================
        # FEATURE MAP -> TOKENS
        # ====================================================

        x = x.flatten(
            2
        )

        x = x.transpose(
            1,
            2
        )

        # Ahora:

        # (B, 64, 128)

        # ====================================================
        # POSITIONAL EMBEDDING
        # ====================================================

        x = self.positional_embedding(
            x
        )

        # ====================================================
        # MAMBA
        # ====================================================

        x = self.mamba(
            x
        )

        # ====================================================
        # TOKENS -> FEATURE MAP
        # ====================================================

        x = x.transpose(
            1,
            2
        )

        x = x.reshape(
            B,
            C,
            H,
            W
        )

        # ====================================================
        # DECODER
        # ====================================================

        delta_logits = self.decoder(
            x
        )

        # ====================================================
        # RESIDUAL SCALE
        # ====================================================

        delta_logits = (
            self.residual_scale
            * delta_logits
        )

        # ====================================================
        # ABUNDANCE PRIOR
        # ====================================================

        if self.use_abundance_prior:

            base_logits = torch.log(
                base_abundances
                + 1e-8
            )

            logits = (
                base_logits
                + delta_logits
            )

        else:

            logits = delta_logits

        # ====================================================
        # SANITY CHECK
        # ====================================================

        if torch.isnan(logits).any():

            raise RuntimeError(
                "MambaTransformer produjo NaN"
            )

        if torch.isinf(logits).any():

            raise RuntimeError(
                "MambaTransformer produjo Inf"
            )

        return logits

    # ========================================================
    # PREDICT ABUNDANCES
    # ========================================================

    def predict_abundances(
        self,
        x
    ):

        logits = self.forward(
            x
        )

        abundances = torch.softmax(
            logits,
            dim=1
        )

        return abundances

    # ========================================================
    # PREDICT CLASS
    # ========================================================

    def predict(
        self,
        x
    ):

        logits = self.forward(
            x
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )

        return prediction


# ============================================================
# 10. MODEL FACTORY
# ============================================================

def create_model(
    n_endmembers=8,
    n_classes=8,
    device=None
):

    if device is None:

        device = get_device()

    print()
    print("=" * 70)
    print("MAMBA TRANSFORMER")
    print("=" * 70)

    print(
        "Implementation: PyTorch Selective SSM"
    )

    print(
        "Official mamba-ssm: NO"
    )

    print(
        "CUDA dependency: NO"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = MambaTransformer(

        n_endmembers=n_endmembers,

        n_classes=n_classes,

        image_size=32,

        embed_dim=128,

        patch_size=4,

        depth=3,

        state_dim=16,

        dt_rank=8,

        dropout=0.1,

        use_abundance_prior=True
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{n_params:,}"
    )

    print(
        "Tokens:",
        model.num_tokens
    )

    print(
        "Patch size:",
        model.patch_size
    )

    print(
        "Embedding dimension:",
        model.embed_dim
    )

    print(
        "Mamba depth:",
        3
    )

    print(
        "SSM state dimension:",
        16
    )

    print(
        "Abundance prior:",
        model.use_abundance_prior
    )

    print("=" * 70)
    print()

    return model


# ============================================================
# 11. TEST DEL MODELO
# ============================================================

def test_model(
    n_endmembers=8,
    n_classes=8,
    batch_size=2,
    device=None
):

    if device is None:

        device = get_device()

    print()
    print("=" * 70)
    print("TEST MAMBA TRANSFORMER")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Crear modelo
    # --------------------------------------------------------

    model = create_model(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        device=device
    )

    # --------------------------------------------------------
    # Entrada
    #
    # Usamos rand y no randn porque las abundancias deben
    # ser no negativas.
    # --------------------------------------------------------

    x = torch.rand(
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

        logits = model(
            x
        )

        abundances = (
            model.predict_abundances(
                x
            )
        )

        prediction = model.predict(
            x
        )

    # ========================================================
    # SHAPES
    # ========================================================

    print()
    print("Input:")
    print(
        "   ",
        tuple(x.shape)
    )

    print()
    print("Logits:")
    print(
        "   ",
        tuple(logits.shape)
    )

    print()
    print("Abundances:")
    print(
        "   ",
        tuple(abundances.shape)
    )

    print()
    print("Prediction:")
    print(
        "   ",
        tuple(prediction.shape)
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print()
    print("Logits statistics:")

    print(
        "   min :",
        logits.min().item()
    )

    print(
        "   max :",
        logits.max().item()
    )

    print(
        "   mean:",
        logits.mean().item()
    )

    print(
        "   std :",
        logits.std().item()
    )

    # ========================================================
    # ABUNDANCE SUM
    # ========================================================

    abundance_sum = (
        abundances.sum(
            dim=1
        )
    )

    print()
    print(
        "Abundance sum statistics:"
    )

    print(
        "   min:",
        abundance_sum.min().item()
    )

    print(
        "   max:",
        abundance_sum.max().item()
    )

    print(
        "   mean:",
        abundance_sum.mean().item()
    )

    # ========================================================
    # PREDICTED CLASSES
    # ========================================================

    unique_classes = torch.unique(
        prediction
    )

    print()
    print(
        "Classes predicted:"
    )

    print(
        "   ",
        unique_classes.detach().cpu().tolist()
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

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

    # --------------------------------------------------------
    # Sum-to-one
    # --------------------------------------------------------

    assert torch.allclose(
        abundance_sum,
        torch.ones_like(
            abundance_sum
        ),
        atol=1e-5
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    assert not torch.isnan(
        logits
    ).any()

    assert not torch.isnan(
        abundances
    ).any()

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    assert not torch.isinf(
        logits
    ).any()

    assert not torch.isinf(
        abundances
    ).any()

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("✓ Input correcto")
    print("✓ Patch embedding correcto")
    print("✓ Selective SSM correcto")
    print("✓ Mamba blocks correctos")
    print("✓ Decoder correcto")
    print("✓ Logits correctos")
    print("✓ Softmax correcto")
    print("✓ Sum-to-one correcto")
    print("✓ No NaN")
    print("✓ No Inf")
    print("✓ Dimensiones correctas")
    print()
    print("TEST COMPLETADO")
    print("=" * 70)
    print()

    return (
        model,
        x,
        logits,
        abundances,
        prediction
    )


# ============================================================
# 12. TEST DE PROPORCIONES
# ============================================================

def test_abundance_prior(
    device=None
):

    if device is None:

        device = get_device()

    print()
    print("=" * 70)
    print("TEST DEL ABUNDANCE PRIOR")
    print("=" * 70)

    model = create_model(
        n_endmembers=8,
        n_classes=8,
        device=device
    )

    model.eval()

    # --------------------------------------------------------
    # Crear abundancias controladas
    # --------------------------------------------------------

    x = torch.zeros(
        1,
        8,
        32,
        32,
        device=device
    )

    # Clase/endmember 0 dominante

    x[:, 0] = 0.80

    # Endmember 1 moderado

    x[:, 1] = 0.10

    # Endmember 2 pequeño

    x[:, 2] = 0.05

    # Endmember 3 pequeño

    x[:, 3] = 0.03

    # Endmember 4

    x[:, 4] = 0.01

    # Endmember 5

    x[:, 5] = 0.005

    # Endmember 6

    x[:, 6] = 0.003

    # Endmember 7

    x[:, 7] = 0.002

    with torch.no_grad():

        base_abundances = (
            model._prepare_base_abundances(
                x
            )
        )

        predicted = (
            model.predict_abundances(
                x
            )
        )

    print()
    print(
        "Abundancias base:"
    )

    print(
        base_abundances[
            0,
            :,
            16,
            16
        ]
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(
        "Abundancias predichas:"
    )

    print(
        predicted[
            0,
            :,
            16,
            16
        ]
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(
        "✓ Test de proporciones completado"
    )

    print("=" * 70)
    print()

    return (
        base_abundances,
        predicted
    )


# ============================================================
# 13. TEST DE BACKWARD
# ============================================================

def test_backward(
    device=None
):

    if device is None:

        device = get_device()

    print()
    print("=" * 70)
    print("TEST BACKWARD")
    print("=" * 70)

    model = create_model(
        n_endmembers=8,
        n_classes=8,
        device=device
    )

    model.train()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    x = torch.rand(
        2,
        8,
        32,
        32,
        device=device
    )

    # --------------------------------------------------------
    # Target
    #
    # Clasificación por píxel.
    # --------------------------------------------------------

    target = torch.randint(
        0,
        8,
        (
            2,
            32,
            32
        ),
        device=device
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits = model(
        x
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        target
    )

    print()
    print(
        "Loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Gradient check
    # --------------------------------------------------------

    gradients_found = 0

    gradients_nan = 0

    gradients_inf = 0

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        gradients_found += 1

        if torch.isnan(
            parameter.grad
        ).any():

            gradients_nan += 1

        if torch.isinf(
            parameter.grad
        ).any():

            gradients_inf += 1

    print(
        "Parámetros con gradiente:",
        gradients_found
    )

    print(
        "Gradientes NaN:",
        gradients_nan
    )

    print(
        "Gradientes Inf:",
        gradients_inf
    )

    assert gradients_found > 0

    assert gradients_nan == 0

    assert gradients_inf == 0

    print()
    print(
        "✓ Forward correcto"
    )

    print(
        "✓ Loss correcta"
    )

    print(
        "✓ Backward correcto"
    )

    print(
        "✓ Gradientes válidos"
    )

    print("=" * 70)
    print()

    return model, loss


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("#" * 70)
    print("# MAMBA TRANSFORMER")
    print("# SELECTIVE SSM - PYTORCH PURO")
    print("#")

    print(
        "# CUDA:",
        torch.cuda.is_available()
    )

    print(
        "# MPS:",
        torch.backends.mps.is_available()
    )

    print("#" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print(
        ">>> Device seleccionado:",
        device
    )

    # --------------------------------------------------------
    # Test principal
    # --------------------------------------------------------

    test_model(
        n_endmembers=8,
        n_classes=8,
        batch_size=2,
        device=device
    )

    # --------------------------------------------------------
    # Test abundance prior
    # --------------------------------------------------------

    test_abundance_prior(
        device=device
    )

    # --------------------------------------------------------
    # Test backward
    # --------------------------------------------------------

    test_backward(
        device=device
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print("# TODOS LOS TESTS TERMINADOS")
    print("#" * 70)
    print()
