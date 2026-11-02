
# ============================================================
# mamba_transformer.py
#
# SELECTIVE SSM / MAMBA PARA ABUNDANCE MAPPING
#
# Implementacion PyTorch puro.
#
# Las abundancias son obligatorias como conexion residual:
#
# logits = log(abundances_normalizadas + eps) + delta_logits
#
# No requiere mamba-ssm, CUDA ni causal-conv1d.
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
    Entrada: (B, N, 32, 32)
    Salida:  (B, D, 8, 8), con patch_size=4.
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

        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return F.gelu(x)


# ============================================================
# 3. POSITIONAL EMBEDDING 2D
# ============================================================

class SpatialPositionalEmbedding(nn.Module):
    """Positional embedding espacial aprendido."""

    def __init__(
        self,
        grid_size=8,
        embed_dim=128
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        self.embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )

        nn.init.normal_(
            self.embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, x):
        if x.shape[1] != self.num_tokens:
            raise ValueError(
                f"Se esperaban {self.num_tokens} tokens; "
                f"se recibieron {x.shape[1]}."
            )

        return x + self.embedding


# ============================================================
# 4. SELECTIVE SSM
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Implementacion recurrente de un Selective State Space Model.

    Entrada y salida: (B, L, D).

    Esta version es PyTorch puro y no es el kernel optimizado
    oficial de Mamba.
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

        self.dt_proj = nn.Sequential(
            nn.Linear(dim, dt_rank),
            nn.GELU(),
            nn.Linear(dt_rank, dim)
        )

        self.B_proj = nn.Linear(dim, state_dim)
        self.C_proj = nn.Linear(dim, state_dim)

        self.log_A = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    state_dim + 1,
                    dtype=torch.float32
                ).unsqueeze(0).repeat(dim, 1)
            )
        )

        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in (
            self.dt_proj[0],
            self.dt_proj[2],
            self.B_proj,
            self.C_proj,
            self.out_proj
        ):
            nn.init.xavier_uniform_(
                module.weight,
                gain=0.5
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        if dim != self.dim:
            raise ValueError(
                f"Dimension esperada {self.dim}; recibida {dim}."
            )

        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=1e-4, max=1.0)

        B_t = torch.tanh(self.B_proj(x))
        C_t = torch.tanh(self.C_proj(x))

        A = -torch.exp(self.log_A)

        state = torch.zeros(
            batch_size,
            dim,
            self.state_dim,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_current = B_t[:, t, :]
            C_current = C_t[:, t, :]

            A_bar = torch.exp(
                dt_t.unsqueeze(-1) * A.unsqueeze(0)
            )

            input_term = (
                dt_t.unsqueeze(-1)
                * B_current.unsqueeze(1)
                * x_t.unsqueeze(-1)
            )

            state = A_bar * state + input_term

            y_t = (
                torch.sum(
                    C_current.unsqueeze(1) * state,
                    dim=-1
                )
                + self.D.unsqueeze(0) * x_t
            )

            outputs.append(y_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.out_proj(y)
        return self.dropout(y)


# ============================================================
# 5. BIDIRECTIONAL SELECTIVE SSM
# ============================================================

class BidirectionalSelectiveSSM(nn.Module):
    """Procesa la secuencia en ambos sentidos."""

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

        self.output_projection = nn.Linear(dim * 2, dim)

    def forward(self, x):
        y_forward = self.forward_ssm(x)

        x_reverse = torch.flip(x, dims=[1])
        y_backward = self.backward_ssm(x_reverse)
        y_backward = torch.flip(y_backward, dims=[1])

        y = torch.cat(
            [y_forward, y_backward],
            dim=-1
        )

        return self.output_projection(y)


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

        self.norm1 = nn.LayerNorm(dim)

        self.ssm = BidirectionalSelectiveSSM(
            dim=dim,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
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

        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                state_dim=state_dim,
                dt_rank=dt_rank,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


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

    def forward(self, x):
        return self.decoder(x)


# ============================================================
# 9. MAMBA TRANSFORMER
# ============================================================

class MambaTransformer(nn.Module):
    """
    Clasificador de mapas de abundancias.

    La conexión residual de abundancias es obligatoria.

    logits = log(base_abundances + eps) + delta_logits

    Requiere n_endmembers == n_classes porque cada canal
    de abundancia debe corresponder al canal de clase
    alineado utilizado por la clasificación.
    """

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
        dropout=0.1
    ):
        super().__init__()

        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ----------------------------------------------------
        # Validaciones
        # ----------------------------------------------------

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size debe ser divisible por patch_size."
            )

        if n_endmembers != n_classes:
            raise ValueError(
                "La conexion residual de abundancias requiere "
                "n_endmembers == n_classes."
            )

        # ----------------------------------------------------
        # Grid y tokens
        # ----------------------------------------------------

        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size ** 2

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = AbundancePatchEmbedding(
            n_endmembers=n_endmembers,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # ----------------------------------------------------
        # Posicion
        # ----------------------------------------------------

        self.positional_embedding = SpatialPositionalEmbedding(
            grid_size=self.grid_size,
            embed_dim=embed_dim
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

        # Escala aprendible de la correccion residual.
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(
                        module.weight,
                        gain=0.5
                    )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ========================================================
    # NORMALIZAR ABUNDANCIAS
    # ========================================================

    def _prepare_base_abundances(self, x):
        """
        Normaliza los canales de abundancia por pixel.

        Entrada: (B, N, H, W)
        Salida:  (B, N, H, W)

        Los valores negativos se recortan a cero.
        Un pixel con suma cero queda como vector de ceros.
        """

        if not torch.isfinite(x).all():
            raise ValueError(
                "La entrada contiene NaN o Inf."
            )

        nonnegative = torch.clamp(x, min=0.0)

        abundance_sum = nonnegative.sum(
            dim=1,
            keepdim=True
        )

        base_abundances = (
            nonnegative
            / abundance_sum.clamp_min(1e-8)
        )

        return base_abundances

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):
        # ----------------------------------------------------
        # Validacion de entrada
        # ----------------------------------------------------

        if x.ndim != 4:
            raise ValueError(
                f"Se esperaba (B,C,H,W); recibido {tuple(x.shape)}."
            )

        if x.shape[1] != self.n_endmembers:
            raise ValueError(
                f"Se esperaban {self.n_endmembers} canales; "
                f"se recibieron {x.shape[1]}."
            )

        if x.shape[2] != self.image_size:
            raise ValueError(
                f"Altura incorrecta: {x.shape[2]}."
            )

        if x.shape[3] != self.image_size:
            raise ValueError(
                f"Anchura incorrecta: {x.shape[3]}."
            )

        # ====================================================
        # ABUNDANCIAS BASE OBLIGATORIAS
        # ====================================================

        base_abundances = self._prepare_base_abundances(x)

        # ====================================================
        # PATCH EMBEDDING
        # ====================================================

        features = self.patch_embed(x)

        batch_size, channels, height, width = features.shape

        # ====================================================
        # FEATURE MAP -> TOKENS
        # ====================================================

        tokens = features.flatten(2).transpose(1, 2)

        # ====================================================
        # POSITIONAL EMBEDDING
        # ====================================================

        tokens = self.positional_embedding(tokens)

        # ====================================================
        # MAMBA
        # ====================================================

        tokens = self.mamba(tokens)

        # ====================================================
        # TOKENS -> FEATURE MAP
        # ====================================================

        features = tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width
        )

        # ====================================================
        # CORRECCION APRENDIDA
        # ====================================================

        delta_logits = self.decoder(features)
        delta_logits = self.residual_scale * delta_logits

        # ====================================================
        # CONEXION RESIDUAL OBLIGATORIA
        # ====================================================

        base_logits = torch.log(
            base_abundances + 1e-8
        )

        logits = base_logits + delta_logits

        # ====================================================
        # VALIDACION DE SALIDA
        # ====================================================

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "MambaTransformer produjo NaN o Inf."
            )

        return logits

    # ========================================================
    # PREDICCION DE ABUNDANCIAS
    # ========================================================

    def predict_abundances(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    # ========================================================
    # PREDICCION DE CLASE
    # ========================================================

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


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
    print("Implementation: PyTorch Selective SSM")
    print("Official mamba-ssm: NO")
    print("CUDA dependency: NO")
    print("Device:", device)
    print("Mandatory abundance residual: YES")

    model = MambaTransformer(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        image_size=32,
        embed_dim=128,
        patch_size=4,
        depth=3,
        state_dim=16,
        dt_rank=8,
        dropout=0.1
    ).to(device)

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Trainable parameters:", f"{n_params:,}")
    print("Tokens:", model.num_tokens)
    print("Patch size:", model.patch_size)
    print("Embedding dimension:", model.embed_dim)
    print("Mamba depth:", len(model.mamba.blocks))
    print("SSM state dimension:", 16)
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
    print("Device:", device)

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
        abundances = model.predict_abundances(x)
        prediction = model.predict(x)
        base_abundances = model._prepare_base_abundances(x)

    print("Input:", tuple(x.shape))
    print("Logits:", tuple(logits.shape))
    print("Abundances:", tuple(abundances.shape))
    print("Prediction:", tuple(prediction.shape))

    print()
    print("Logits:")
    print("  min:", logits.min().item())
    print("  max:", logits.max().item())
    print("  mean:", logits.mean().item())
    print("  std:", logits.std().item())

    abundance_sum = abundances.sum(dim=1)

    print()
    print("Predicted abundance sum:")
    print("  min:", abundance_sum.min().item())
    print("  max:", abundance_sum.max().item())
    print("  mean:", abundance_sum.mean().item())

    print("Classes predicted:", torch.unique(prediction).cpu().tolist())

    assert logits.shape == (
        batch_size, n_classes, 32, 32
    )

    assert abundances.shape == (
        batch_size, n_classes, 32, 32
    )

    assert prediction.shape == (
        batch_size, 32, 32
    )

    assert torch.allclose(
        abundance_sum,
        torch.ones_like(abundance_sum),
        atol=1e-5
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(abundances).all()

    base_sum = base_abundances.sum(dim=1)
    nonzero_pixels = base_sum > 0

    if nonzero_pixels.any():
        assert torch.allclose(
            base_sum[nonzero_pixels],
            torch.ones_like(base_sum[nonzero_pixels]),
            atol=1e-5
        )

    print()
    print("✓ Dimensiones correctas")
    print("✓ Conexion residual de abundancias activa")
    print("✓ Softmax correcto")
    print("✓ Suma de probabilidades igual a uno")
    print("✓ No hay NaN ni Inf")
    print("TEST COMPLETADO")
    print("=" * 70)
    print()

    return model, x, logits, abundances, prediction


# ============================================================
# 12. TEST DE CONEXION RESIDUAL DE ABUNDANCIAS
# ============================================================

def test_abundance_prior(device=None):
    """
    Verifica que las abundancias se normalizan correctamente
    y que el modelo produce logits finitos con la conexion
    residual obligatoria.

    Este test NO exige que la prediccion coincida exactamente
    con las abundancias de entrada: el decoder puede corregirlas.
    """

    if device is None:
        device = get_device()

    print()
    print("=" * 70)
    print("TEST DE CONEXION RESIDUAL OBLIGATORIA")
    print("=" * 70)

    model = create_model(
        n_endmembers=8,
        n_classes=8,
        device=device
    )

    model.eval()

    x = torch.zeros(
        1, 8, 32, 32,
        device=device
    )

    proportions = torch.tensor(
        [0.80, 0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002],
        device=device
    ).view(1, 8, 1, 1)

    x[:] = proportions

    with torch.no_grad():
        base_abundances = model._prepare_base_abundances(x)
        logits = model(x)
        predicted = torch.softmax(logits, dim=1)

    print("Base abundances:")
    print(
        base_abundances[0, :, 16, 16].cpu().numpy()
    )

    print("Predicted probabilities:")
    print(
        predicted[0, :, 16, 16].cpu().numpy()
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted).all()

    assert torch.allclose(
        base_abundances.sum(dim=1),
        torch.ones_like(base_abundances[:, 0]),
        atol=1e-5
    )

    print()
    print("✓ Abundancias base normalizadas")
    print("✓ Conexión residual obligatoria ejecutada")
    print("✓ Logits y probabilidades finitos")
    print("=" * 70)
    print()

    return base_abundances, predicted


# ============================================================
# 13. TEST DE BACKWARD
# ============================================================

def test_backward(device=None):
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

    x = torch.rand(2, 8, 32, 32, device=device)

    target = torch.randint(
        0,
        8,
        (2, 32, 32),
        device=device
    )

    logits = model(x)
    loss = F.cross_entropy(logits, target)

    print("Loss:", loss.item())

    loss.backward()

    gradients_found = 0
    gradients_nan = 0
    gradients_inf = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradients_found += 1

        if torch.isnan(parameter.grad).any():
            gradients_nan += 1

        if torch.isinf(parameter.grad).any():
            gradients_inf += 1

    print("Parameters with gradients:", gradients_found)
    print("Gradients NaN:", gradients_nan)
    print("Gradients Inf:", gradients_inf)

    assert gradients_found > 0
    assert gradients_nan == 0
    assert gradients_inf == 0

    assert model.residual_scale.grad is not None
    assert torch.isfinite(model.residual_scale.grad).all()

    print()
    print("✓ Forward correcto")
    print("✓ Loss calculada")
    print("✓ Backward correcto")
    print("✓ Gradientes válidos")
    print("✓ residual_scale recibe gradiente")
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
    print("# CUDA:", torch.cuda.is_available())
    print("# MPS:", torch.backends.mps.is_available())
    print("#" * 70)

    device = get_device()
    print("Device seleccionado:", device)

    test_model(
        n_endmembers=8,
        n_classes=8,
        batch_size=2,
        device=device
    )

    test_abundance_prior(device=device)
    test_backward(device=device)

    print()
    print("#" * 70)
    print("# TODOS LOS TESTS TERMINADOS")
    print("#" * 70)
