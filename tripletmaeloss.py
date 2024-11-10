import torch
import torch.nn as nn
import torchvision


# 构建Triplet AE 损失
class TripletMAELoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletMAELoss, self).__init__()
        self.margin = margin

    def cal_eucliden(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self,
                anchor,
                positive,
                negative,
                predictions,
                targets,
                gamma=1):
        distance_positive = self.cal_eucliden(anchor, positive)
        distance_negative = self.cal_eucliden(anchor, negative)

        construct_criterion = nn.MSELoss()
        contruct_loss = construct_criterion(predictions, targets)

        losses = torch.relu(distance_positive - distance_negative +
                            self.margin) + gamma * contruct_loss.detach()
        return losses.mean()
