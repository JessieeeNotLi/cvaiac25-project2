import torch
import torch.nn as nn

class RegressionLoss(nn.Module):
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

class ClassificationLoss(nn.Module):
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