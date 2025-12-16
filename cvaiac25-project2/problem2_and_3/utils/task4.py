import torch
import torch.nn as nn

class RegressionLossPre(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

        self.positive_reg_lb = self.config.get('positive_reg_lb', 0.55)

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive_mask = iou >= self.positive_reg_lb


        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=iou.device)

        
        pred_pos = pred[positive_mask]
        target_pos = target[positive_mask]
        #print("reg inputs:", pred_pos.shape, target_pos.shape)
        return self.loss(pred_pos, target_pos)

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.positive_reg_lb = float(config.get("positive_reg_lb", 0.55))
        self.positive_reg_ub = float(config.get("positive_reg_ub", 0.75))
        self.reg_w_min = float(config.get("reg_w_min", 0.2))
        self.w_yaw = float(config.get("reg_w_yaw", 2.0))

        self.base = nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def _wrap_pi(angle_diff: torch.Tensor) -> torch.Tensor:
        # Differentiable, no modulo, safe
        return torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        iou = iou.view(-1).float()

        pos_mask = iou >= self.positive_reg_lb
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_pos = pred[pos_mask]
        tgt_pos  = target[pos_mask]
        iou_pos  = iou[pos_mask].clamp(0.0, 1.0)

        # Split yaw from other components (NO in-place ops)
        diff = pred_pos - tgt_pos

        diff_xyzwhl = diff[:, :6]                 # (N,6)
        diff_yaw    = self._wrap_pi(diff[:, 6])   # (N,)

        # SmoothL1 per component
        loss_xyzwhl = self.base(diff_xyzwhl, torch.zeros_like(diff_xyzwhl)).mean(dim=1)
        loss_yaw    = self.base(diff_yaw, torch.zeros_like(diff_yaw)) * self.w_yaw

        per_sample_loss = loss_xyzwhl + loss_yaw  # (N,)

        # IoU-aware weight ramp
        ramp = (iou_pos - self.positive_reg_lb) / (
            self.positive_reg_ub - self.positive_reg_lb + 1e-9
        )
        ramp = ramp.clamp(0.0, 1.0)
        w = self.reg_w_min + (1.0 - self.reg_w_min) * ramp

        return (w * per_sample_loss).sum() / (w.sum() + 1e-6)
    

    
class ClassificationLossPrev(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()
        self.positive_cls_lb = self.config.get('positive_cls_lb', 0.6)
        self.negative_cls_ub = self.config.get('negative_cls_ub', 0.45)

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        positive_mask = iou >= self.positive_cls_lb
        negative_mask = iou <= self.negative_cls_ub

        train_mask = positive_mask | negative_mask
        if train_mask.sum() == 0:
            return torch.tensor(0.0, device=iou.device)

        target_full = torch.zeros_like(pred, device=iou.device)
        target_full[positive_mask] = 1.0

        pred_train = pred[train_mask]
        target_train = target_full[train_mask]

        loss_val = self.loss(pred_train, target_train)
        #print("num pos for regression:", (iou >= 0.55).sum().item())
        #print("num cls train:", ((iou >= 0.6) | (iou <= 0.45)).sum().item())
        #print("any NaNs in pred before loss:", torch.isnan(pred).any())

        return loss_val
    


class ClassificationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # pick reasonable defaults if not in config
        self.t_low  = float(cfg.get("cls_t_low", 0.45))
        self.t_high = float(cfg.get("cls_t_high", 0.60))
        self.w_pos  = float(cfg.get("cls_w_pos", 1.0))   # optional
        self.w_neg  = float(cfg.get("cls_w_neg", 1.0))   # optional
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred_prob, iou):
        """
        pred_prob: (B,1) or (B,) already sigmoid-ed
        iou:       (B,) or (B,1) IoU of proposal/ROI with assigned GT
        """
        pred_prob = pred_prob.view(-1).float().clamp(1e-6, 1 - 1e-6)
        iou = iou.view(-1).float()   # <-- THIS IS THE FIX

        y = (iou - self.t_low) / (self.t_high - self.t_low + 1e-9)
        y = y.clamp(0.0, 1.0)

        loss = self.bce(pred_prob, y)

        # weights should match dtype/device too
        w_pos = torch.tensor(self.w_pos, device=loss.device, dtype=loss.dtype)
        w_neg = torch.tensor(self.w_neg, device=loss.device, dtype=loss.dtype)
        w = torch.where(y > 0.5, w_pos, w_neg)

        return (loss * w).mean()
