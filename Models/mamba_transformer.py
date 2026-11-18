# ============================================================
# mamba_transformer.py
# SELECTIVE SSM / MAMBA - VERSIÓN ORIGINAL COMPATIBLE CON YAML
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DISPOSITIVO
# ============================================================

def get_device(device_setting=None):
    if device_setting == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 2. PATCH EMBEDDING 2D
# ============================================================

class AbundancePatchEmbedding(nn.Module):
    """Proyecta parches de abundancia a dimensión de embedding."""

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

    def __init__(self, grid_size=8, embed_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        self.embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x):
        if x.shape[1] != self.num_tokens:
            raise ValueError(
                f"Se esperaban {self.num_tokens} tokens; se recibieron {x.shape[1]}."
            )
        return x + self.embedding


# ============================================================
# 4. SELECTIVE SSM (Formulación Matemática Corregida)
# ============================================================

class SelectiveSSM(nn.Module):
    def __init__(
        self,
        dim=128,
        state_dim=16,
        dt_rank=8,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        self.dt_proj = nn.Sequential(
            nn.Linear(dim, dt_rank, bias=False),
            nn.GELU(),
            nn.Linear(dt_rank, dim, bias=True)
        )

        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(dim, state_dim, bias=False)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=1e-4, max=1.0)

        B_t = self.B_proj(x)
        C_t = self.C_proj(x)

        A = -torch.exp(self.A_log)

        state = torch.zeros(
            batch_size, dim, self.state_dim, device=x.device, dtype=x.dtype
        )
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_curr = B_t[:, t, :]
            C_curr = C_t[:, t, :]

            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            input_term = dt_t.unsqueeze(-1) * x_t.unsqueeze(-1) * B_curr.unsqueeze(1)

            state = A_bar * state + input_term

            y_t = torch.sum(C_curr.unsqueeze(1) * state, dim=-1) + self.D * x_t
            outputs.append(y_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.out_proj(y)
        return self.dropout(y)


# ============================================================
# 5. BIDIRECTIONAL SELECTIVE SSM
# ============================================================

class BidirectionalSelectiveSSM(nn.Module):
    def __init__(self, dim=128, state_dim=16, dt_rank=8, dropout=0.1):
        super().__init__()
        self.forward_ssm = SelectiveSSM(dim, state_dim, dt_rank, dropout)
        self.backward_ssm = SelectiveSSM(dim, state_dim, dt_rank, dropout)
        self.output_projection = nn.Linear(dim * 2, dim)

    def forward(self, x):
        y_forward = self.forward_ssm(x)
        x_reverse = torch.flip(x, dims=[1])
        y_backward = self.backward_ssm(x_reverse)
        y_backward = torch.flip(y_backward, dims=[1])

        y = torch.cat([y_forward, y_backward], dim=-1)
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
        self.ssm = BidirectionalSelectiveSSM(dim, state_dim, dt_rank, dropout)
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
            MambaBlock(dim=dim, state_dim=state_dim, dt_rank=dt_rank, dropout=dropout)
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
    def __init__(self, embed_dim=128, n_classes=8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)


# ============================================================
# 9. MAMBA TRANSFORMER (Conexión Residual Original Libre)
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
        dropout=0.1
    ):
        super().__init__()

        if n_endmembers != n_classes:
            raise ValueError(
                "La conexión residual de abundancias requiere n_endmembers == n_classes."
            )

        self.n_endmembers = n_endmembers
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size ** 2

        self.patch_embed = AbundancePatchEmbedding(
            n_endmembers=n_endmembers,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.positional_embedding = SpatialPositionalEmbedding(
            grid_size=self.grid_size,
            embed_dim=embed_dim
        )

        self.mamba = VisionMamba(
            dim=embed_dim,
            depth=depth,
            state_dim=state_dim,
            dt_rank=dt_rank,
            dropout=dropout
        )

        self.decoder = MambaDecoder(
            embed_dim=embed_dim,
            n_classes=n_classes
        )

        # Escala aprendible libre (Restaurada como en tu versión original)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_base_abundances(self, x):
        if not torch.isfinite(x).all():
            raise ValueError("La entrada contiene NaN o Inf.")

        nonnegative = torch.clamp(x, min=0.0)
        abundance_sum = nonnegative.sum(dim=1, keepdim=True)
        base_abundances = nonnegative / abundance_sum.clamp_min(1e-8)
        return base_abundances

    def forward(self, x):
        base_abundances = self._prepare_base_abundances(x)

        features = self.patch_embed(x)
        batch_size, channels, height, width = features.shape

        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.positional_embedding(tokens)
        tokens = self.mamba(tokens)

        features = tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )

        delta_logits = self.decoder(features)

        # Multiplicación directa por residual_scale (Sin clamp)
        delta_logits = self.residual_scale * delta_logits

        base_logits = torch.log(base_abundances + 1e-8)
        logits = base_logits + delta_logits

        if not torch.isfinite(logits).all():
            raise RuntimeError("MambaTransformer produjo NaN o Inf.")

        return logits

    def predict_abundances(self, x):
        return torch.softmax(self.forward(x), dim=1)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


# ============================================================
# 10. CONSTRUCTOR ADAPTADO A TU YAML
# ============================================================

def create_model_from_config(config):
    """
    Lee tu config.yaml e instancia MambaTransformer de forma dinámica.
    """
    n_endmembers = config["unmixing"]["n_endmembers"]
    if config["unmixing"].get("extra_class", False):
        n_endmembers += 1

    n_classes = config["classification"]["n_classes"]
    patch_size_training = config["training"]["patch_size"]
    device_setting = config["hardware"]["device"]

    device = get_device(device_setting)

    model = MambaTransformer(
        n_endmembers=n_endmembers,
        n_classes=n_classes,
        image_size=patch_size_training,
        embed_dim=128,
        patch_size=4,
        depth=3,
        state_dim=16,
        dt_rank=8,
        dropout=0.1
    ).to(device)

    return model, device