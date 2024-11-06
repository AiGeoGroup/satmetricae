# 设置三元组深度神经网络
import torch.nn as nn

class TripletNet(nn.Module):
    def __init__(self, embedding_net): # embedding_net是通过AutoEncoder编码器获得遥感图像表征网络
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, x1, x2, x3): # x1是Anchor图像, x2是正样本图像，x3是负样本图像
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3 # 对应的表征
    
    def get_embedding(self, x):
        return self.embedding_net(x)