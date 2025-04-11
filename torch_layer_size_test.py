import torch.nn as nn
import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

frame_stack_amount = 4
conv_net = nn.Sequential(
    layer_init(nn.Conv2d(frame_stack_amount, 64, 8, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(64, 128, 8, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(128, 256, 4, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(256, 512, 2, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(512, 1024, 2, stride=1)),
    # nn.Flatten(),
)




height = 72
width = 128

z = conv_net(torch.zeros((1, frame_stack_amount, height, width)))
print(z.shape)
print(z.flatten().shape)
# print(conv_net(torch.zeros((1, channels, height, width))).shape)

