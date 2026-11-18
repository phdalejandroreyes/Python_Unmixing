# ============================================================
# losses.py
#
# Módulo unificado de Funciones de Pérdida para Clasificación de HSI
# Soporta: Cross Entropy (CE), Weighted CE (WCE), Focal Loss,
#          Tversky Loss y Focal Tversky Loss.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        # 1. CE sin reducción para mantener por-píxel
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # 2. Obtener p_t (estabilidad numérica)
        pt = torch.exp(-ce_loss)
        
        # 3. Factor modulador focal
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        # 4. Ponderación por clase (Alpha / Pesos Atenuados)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            valid_targets = torch.clamp(targets, min=0)
            alpha_t = self.alpha[valid_targets]
            focal_loss = alpha_t * focal_loss

        # 5. Filtrar píxeles ignorados (-100)
        valid_mask = (targets != self.ignore_index)
        focal_loss = focal_loss[valid_mask]

        # 6. Reducción
        if self.reduction == 'mean':
            return focal_loss.mean() if focal_loss.numel() > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# 2. TVERSKY LOSS & FOCAL TVERSKY LOSS
# ============================================================
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma_tversky=1.0, weights=None, eps=1e-6, ignore_index=-100):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_tversky = gamma_tversky
        self.eps = eps
        self.ignore_index = ignore_index
        self.weights = weights

    def forward(self, inputs, targets):
        # Proba softmax por canal
        probs = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        
        # Máscara para ignore_index
        valid_mask = (targets != self.ignore_index).unsqueeze(1) # (B, 1, H, W)
        targets_clean = torch.clamp(targets, min=0)
        
        # Ground Truth One-Hot
        targets_one_hot = F.one_hot(targets_clean, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)

        # Aplicar validez de píxeles
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Métricas por clase: TP, FP, FN
        tp = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        fp = torch.sum(probs * (1.0 - targets_one_hot), dim=(0, 2, 3))
        fn = torch.sum((1.0 - probs) * targets_one_hot, dim=(0, 2, 3))

        # Índice de Tversky
        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        
        # Pérdida Tversky por clase
        loss_per_class = 1.0 - tversky_index
        
        # Exponente Focal opcional (Focal Tversky)
        if self.gamma_tversky != 1.0:
            loss_per_class = loss_per_class ** (1.0 / self.gamma_tversky)

        # Si se pasan pesos atenuados por clase
        if self.weights is not None:
            if self.weights.device != inputs.device:
                self.weights = self.weights.to(inputs.device)
            loss_per_class = loss_per_class * self.weights
            return loss_per_class.sum() / self.weights.sum()

        return loss_per_class.mean()


# ============================================================
# 3. FÁBRICA DE FUNCIONES DE PÉRDIDA (FACTORY)
# ============================================================
def get_loss_function(config, class_weights=None, device="cpu"):
    """
    Construye e instancia la función de pérdida seleccionada desde config.yaml.
    """
    loss_cfg = config.get('loss', {})
    loss_type = str(loss_cfg.get('type', 'ce')).lower()
    
    # Manejar flags de pesos (use_weights o use_alpha en config)
    use_weights = loss_cfg.get('use_weights', False) or loss_cfg.get('use_alpha', False)
    ignore_index = loss_cfg.get('ignore_index', -100)

    # Determinar si aplicamos pesos atenuados de frecuencia
    weights = class_weights.to(device) if (use_weights and class_weights is not None) else None

    # --- 1. Cross Entropy Estándar (Sin Pesos) ---
    if loss_type in ['ce', 'cross_entropy']:
        print("[INFO] Pérdida activada: CrossEntropyLoss (Sin pesos por clase)")
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    # --- 2. Weighted Cross Entropy (Con Pesos Atenuados) ---
    elif loss_type in ['wce', 'weighted_ce', 'weighted_cross_entropy']:
        print("[INFO] Pérdida activada: Weighted CrossEntropyLoss (Con pesos atenuados)")
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    # --- 3. Focal Loss ---
    elif loss_type in ['focal', 'focal_loss']:
        gamma = loss_cfg.get('gamma', 2.0)
        print(f"[INFO] Pérdida activada: FocalLoss (gamma={gamma}, use_weights={weights is not None})")
        return FocalLoss(alpha=weights, gamma=gamma, ignore_index=ignore_index)

    # --- 4. Tversky Loss ---
    elif loss_type in ['tversky', 'tversky_loss']:
        alpha = loss_cfg.get('alpha', 0.3)
        beta = loss_cfg.get('beta', 0.7)
        print(f"[INFO] Pérdida activada: TverskyLoss (alpha={alpha}, beta={beta}, use_weights={weights is not None})")
        return TverskyLoss(alpha=alpha, beta=beta, gamma_tversky=1.0, weights=weights, ignore_index=ignore_index)

    # --- 5. Focal Tversky Loss ---
    elif loss_type in ['focal_tversky', 'focal_tversky_loss']:
        alpha = loss_cfg.get('alpha', 0.3)
        beta = loss_cfg.get('beta', 0.7)
        gamma_tversky = loss_cfg.get('gamma_tversky', 1.33)
        print(f"[INFO] Pérdida activada: FocalTverskyLoss (alpha={alpha}, beta={beta}, gamma_tversky={gamma_tversky}, use_weights={weights is not None})")
        return TverskyLoss(alpha=alpha, beta=beta, gamma_tversky=gamma_tversky, weights=weights, ignore_index=ignore_index)

    else:
        raise ValueError(
            f"Tipo de función de pérdida desconocido: '{loss_type}'. "
            f"Opciones válidas en config.yaml: 'ce', 'wce', 'focal_loss', 'tversky', 'focal_tversky'"
        )