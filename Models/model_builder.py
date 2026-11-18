# ============================================================
# Models/model_builder.py
#
# Fábrica centralizada para instanciar dinámicamente las 5
# arquitecturas del experimento multi-escena.
# ============================================================

import torch
import torch.nn as nn

def create_model(config, device=None):
    """
    Instancia el modelo según la clave config['classification']['architecture'].
    Arquitecturas soportadas:
      1. 'resunet'
      2. 'mamba'
      3. 'abundance'
      4. 'residual'
      5. 'cross_attention'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = str(config["classification"]["architecture"]).strip().lower()
    
    n_endmembers = int(config["unmixing"]["n_endmembers"])
    extra_class = bool(config["unmixing"]["extra_class"])
    N_EST = n_endmembers + int(extra_class)
    
    n_classes = int(config["classification"]["n_classes"])
    patch_size = int(config["training"]["patch_size"])
    spectral_bands = int(config["data"]["spectral_bands"])

    print(f"\n[INFO] Instanciando arquitectura: '{arch.upper()}' (Patch Size: {patch_size}x{patch_size})")

    # ------------------------------------------------------------
    # 1. RESUNET
    # ------------------------------------------------------------
    if arch == "resunet":
        try:
            from Models.resunet import ResUNet
        except ModuleNotFoundError:
            from resunet import ResUNet

        model = ResUNet(
            in_channels=N_EST,
            n_classes=n_classes
        )

    # ------------------------------------------------------------
    # 2. MAMBA TRANSFORMER (Selective SSM)
    # ------------------------------------------------------------
    elif arch == "mamba":
        try:
            from Models.mamba_transformer import MambaTransformer
        except ModuleNotFoundError:
            from mamba_transformer import MambaTransformer

        model = MambaTransformer(
            n_endmembers=N_EST,
            n_classes=n_classes,
            image_size=patch_size,
            embed_dim=128,
            patch_size=4,
            depth=3,
            state_dim=16,
            dt_rank=8,
            dropout=0.1
        )

    # ------------------------------------------------------------
    # 3. ABUNDANCE TRANSFORMER (Baseline A-Space)
    # ------------------------------------------------------------
    elif arch == "abundance":
        try:
            from Models.abundance_transformer import AbundanceTransformer
        except ModuleNotFoundError:
            from abundance_transformer import AbundanceTransformer

        model = AbundanceTransformer(
            in_channels=N_EST,
            n_endmembers=N_EST,
            n_classes=n_classes,
            patch_size=4  # FIJO EN 4: Proyecta (32x32) -> (8x8) -> Decoder (32x32)
        )
    # ------------------------------------------------------------
    # 4. RESIDUAL TRANSFORMER
    # ------------------------------------------------------------
    elif arch == "residual":
        try:
            from Models.residual_transformer import ResidualTransformer
        except ModuleNotFoundError:
            from residual_transformer import ResidualTransformer

        model = ResidualTransformer(
            n_endmembers=N_EST,
            n_classes=n_classes,
            image_size=patch_size
        )

    # ------------------------------------------------------------
    # 5. CROSS-ATTENTION TRANSFORMER (Hyperspectral + Endmembers)
    # ------------------------------------------------------------
    elif arch == "cross_attention":
        try:
            from Models.cross_attention import CrossAttentionTransformer
        except ModuleNotFoundError:
            try:
                from Models.cross_attention_transformer import CrossAttentionTransformer
            except ModuleNotFoundError:
                from cross_attention import CrossAttentionTransformer

        model = CrossAttentionTransformer(
            n_endmembers=N_EST,
            n_classes=n_classes,
            spectral_bands=config['data']['spectral_bands'],
            patch_size=2  # Patch size interno por defecto de esta red
        )
    else:
        valid_options = ["resunet", "mamba", "abundance", "residual", "cross_attention"]
        raise ValueError(f"Arquitectura no válida: {arch!r}. Opciones soportadas: {valid_options}")

    return model.to(device)
