import torch
import torch.nn as nn
import torchvision


class Block(torch.nn.Module):

    def __init__(self, dim_in, dim_out, is_encoder=True):
        super().__init__()

        cnn_type = torch.nn.Conv2d
        if not is_encoder:
            cnn_type = torch.nn.ConvTranspose2d

        def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
            return (
                cnn_type(dim_in,
                         dim_out,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding),
                torch.nn.BatchNorm2d(dim_out),
                torch.nn.LeakyReLU(),
            )

        self.s = torch.nn.Sequential(
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
        )

        self.res = cnn_type(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=2,
                            padding=0)

    def forward(self, x):
        return self.s(x) + self.res(x)

# Block(3, 5)(torch.randn(2, 3, 64, 64)).shape


def get_encoder():
    encoder = torch.nn.Sequential(
        Block(3, 32, True),
        Block(32, 64, True),
        Block(64, 128, True),
        Block(128, 256, True),
        torch.nn.Flatten(),
        torch.nn.Linear(2304, 128),
    )
    return encoder

# encoder = get_encoder()
# encoder(torch.randn(2, 3, 64, 64)).shape


def get_decoder():
    decoder = torch.nn.Sequential(
        torch.nn.Linear(128, 256 * 4 * 4),
        torch.nn.InstanceNorm1d(256 * 4 * 4),
        torch.nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
        Block(256, 128, False),
        Block(128, 64, False),
        Block(64, 32, False),
        Block(32, 3, False),
        torch.nn.UpsamplingNearest2d(size=64),
        torch.nn.Conv2d(in_channels=3,
                        out_channels=3,
                        kernel_size=1,
                        stride=1,
                        padding=0),
        torch.nn.Tanh(),
    )
    return decoder

# decoder = get_decoder()
# decoder(torch.randn(2, 128)).shape

class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()

    def forward(self, data):
        hidden = self.encoder(data)
        output = self.decoder(hidden)
        return output # 重建图像，生成图像