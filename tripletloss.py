import torch.nn as nn
import torch
import torchvision

# 定义Triplet损失

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def cal_eucliden(self, x1, x2):
        return (x1-x2).pow(2).sum(1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.cal_eucliden(anchor, positive) # 计算anchor与正样本损失
        distance_negative = self.cal_eucliden(anchor, negative) # 计算anchor与负样本损失
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() #计算anchor与正样本，及负样本损失平均值
