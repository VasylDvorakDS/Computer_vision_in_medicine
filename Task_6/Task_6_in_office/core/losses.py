# loss_funcs.py
# ─────────────────────────────────────────────────────────────────────────────
# Простейшие loss‑классы без «магии» из torch.*.Loss:
#   • BCELoss            (с / без pixel‑weights)
#   • FocalLoss          (γ=2, с / без pixel‑weights)
#   • IoULoss            (с / без class‑weights)
#   • ComboLoss          = ½·(IoU + Focal)  (весовые варианты поддерживаются)
# Каждый класс вызываетcя как функция:  loss = Loss(...)(logits, target)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F


# helper: ручная BCE‑формула с лог‑стабилизацией
def _bce_logits(logits, target, eps=1e-7):
    p = torch.sigmoid(logits)
    return -(target * torch.log(p + eps) + (1 - target) * torch.log(1 - p + eps))


# ─────────────────────────────────────────────────────────────────────────────
class BCELoss:
    """
    pixel_weights — Tensor([w_pos, w_neg])  или None
    """
    def __init__(self, pixel_weights=None):
        self.pixel_weights = pixel_weights

    def __call__(self, logits, target):
        loss = _bce_logits(logits, target)          # (N,1,H,W)
        if self.pixel_weights is not None:
            w_pos, w_neg = self.pixel_weights
            weight = torch.where(target > 0.5, w_pos, w_neg)
            loss = loss * weight
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss:
    """
    gamma = 2  (фиксировано)
    pixel_weights — Tensor([w_pos, w_neg])  или None
    """
    def __init__(self, gamma=2.0, pixel_weights=None):
        self.gamma = gamma
        self.pixel_weights = pixel_weights

    def __call__(self, logits, target):
        ce = _bce_logits(logits, target)
        p  = torch.sigmoid(logits)
        focal = (1 - p) ** self.gamma * ce
        if self.pixel_weights is not None:
            w_pos, w_neg = self.pixel_weights
            weight = torch.where(target > 0.5, w_pos, w_neg)
            focal = focal * weight
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
class IoULoss:
    """
    class_weight — скаляр‑тензор или None
    """
    def __init__(self, class_weight=None, smooth=1e-6):
        self.class_weight = class_weight
        self.smooth = smooth

    def __call__(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(2, 3))
        union = (p + target - p * target).sum(dim=(2, 3))
        iou = (inter + self.smooth) / (union + self.smooth)   # (N,1)
        loss = 1 - iou                                         # (N,1)
        if self.class_weight is not None:
            loss = loss * self.class_weight
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
class ComboLoss:
    """
    0.5·(IoU + Focal).  Поддерживает те же веса, что и входящие лоссы.
    """
    def __init__(self,
                 pixel_weights=None,
                 class_weight=None,
                 gamma=2.0):
        self.focal = FocalLoss(gamma=gamma, pixel_weights=pixel_weights)
        self.iou   = IoULoss(class_weight=class_weight)

    def __call__(self, logits, target):
        return 0.5 * (self.focal(logits, target) +
                      self.iou  (logits, target))
