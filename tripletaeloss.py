import torch.nn as nn
import torch
import torchvision

# 构建Triplet AE 损失
class TripletAELoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletAELoss, self).__init__()
        self.margin = margin

    def cal_eucliden(self, x1, x2):
        return (x1-x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative, anchor_img, pred_img, gamma=0.9):
        criterion = nn.MSELoss()
        distance_positive = self.cal_eucliden(anchor, positive)
        distance_negative = self.cal_eucliden(anchor, negative)
        contruct_loss = criterion(anchor_img, pred_img)
        losses = (1-gamma) * torch.relu(distance_positive - distance_negative + self.margin) + gamma * contruct_loss
        return losses.mean()
