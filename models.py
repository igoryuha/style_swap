import torch
import torch.nn as nn
from collections import namedtuple


normalised_vgg_relu5_1 = nn.Sequential(
    nn.Conv2d(3, 3, 1),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.AvgPool2d(2, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.AvgPool2d(2, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.AvgPool2d(2, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.AvgPool2d(2, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU()
)


class NormalisedVGG(nn.Module):

    def __init__(self, pretrained_path='./encoder/vgg_normalised_conv5_1.pth'):
        super(NormalisedVGG, self).__init__()
        self.net = normalised_vgg_relu5_1
        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, target='relu5_1'):
        relu1_1 = self.net[:4](x)
        if target == 'relu1_1':
            return relu1_1
        relu2_1 = self.net[4:11](relu1_1)
        if target == 'relu2_1':
            return relu2_1
        relu3_1 = self.net[11:18](relu2_1)
        if target == 'relu3_1':
            return relu3_1
        relu4_1 = self.net[18:31](relu3_1)
        if target == 'relu4_1':
            return relu4_1
        relu5_1 = self.net[31:](relu4_1)
        out = namedtuple('out', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1','relu5_1'])
        return out(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
