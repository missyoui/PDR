
import torch
import torch.nn as nn

# ==================== Scale-based Dynamic Loss (SDLoss) ====================
class SDLoss(nn.Module):
    """
    Scale-based Dynamic Loss adapted for VI-ReID.
    Reference: AAAI 2025 Paper
    """

    def __init__(self, delta=0.5, max_area=81, use_mask=False):
        super(SDLoss, self).__init__()
        self.delta = delta
        self.max_area = max_area
        self.use_mask = use_mask

    def forward(self, pred, target):
        if self.use_mask:
            return self._mask_loss(pred, target)
        else:
            return self._bbox_loss(pred, target)

    def _bbox_loss(self, pred, target):
        eps = 1e-6
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + eps
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]) + eps

        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        union_area = pred_area + target_area - inter_area + eps
        iou = inter_area / union_area

        center_pred = (pred[:, :2] + pred[:, 2:]) / 2
        center_target = (target[:, :2] + target[:, 2:]) / 2
        rho2 = torch.sum((center_pred - center_target) ** 2, dim=1)

        c_x1 = torch.min(pred[:, 0], target[:, 0])
        c_y1 = torch.min(pred[:, 1], target[:, 1])
        c_x2 = torch.max(pred[:, 2], target[:, 2])
        c_y2 = torch.max(pred[:, 3], target[:, 3])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

        v = (4 / (torch.pi ** 2)) * (
            torch.atan((pred[:, 2] - pred[:, 0]) / (pred[:, 3] - pred[:, 1] + eps)) -
            torch.atan((target[:, 2] - target[:, 0]) / (target[:, 3] - target[:, 1] + eps))
        ) ** 2
        alpha = v / (v - iou + 1 + eps)

        scale_factor = torch.clamp(target_area / self.max_area * self.delta, max=self.delta)
        beta_iou = self.delta - scale_factor
        beta_ciou = 1 - self.delta + scale_factor

        sdloss = beta_iou * (1 - iou + alpha * v) + beta_ciou * (rho2 / c2)
        return sdloss.mean()

    def _mask_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        eps = 1e-6

        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou = intersection / (union + eps)
        iou_loss = 1 - iou

        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        dp = pred_sum / (target_sum + eps)
        dgt = target_sum / (pred_sum + eps)
        loc_loss = 1 - torch.min(dp, dgt) / (torch.max(dp, dgt) + eps)

        mask_area = target_sum
        scale_factor = torch.clamp(mask_area / self.max_area * self.delta, max=self.delta)

        beta_scale = 1 + scale_factor
        beta_loc = 1 - scale_factor
        loss = beta_scale * iou_loss + beta_loc * loc_loss
        return loss.mean()
