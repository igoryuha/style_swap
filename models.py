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


class Conv_InstanceNorm_ReLU(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_InstanceNorm_ReLU, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_instance_norm_relu3_1 = Conv_InstanceNorm_ReLU(256, 128)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv_instance_norm_relu2_2 = Conv_InstanceNorm_ReLU(128, 128)
        self.conv_instance_norm_relu2_1 = Conv_InstanceNorm_ReLU(128, 64)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv_instance_norm_relu1_2 = Conv_InstanceNorm_ReLU(64, 64)
        self.conv1_1 = Conv(64, 3)

    def forward(self, x):
        x = self.conv_instance_norm_relu3_1(x)
        x = self.upsample3(x)
        x = self.conv_instance_norm_relu2_2(x)
        x = self.conv_instance_norm_relu2_1(x)
        x = self.upsample2(x)
        x = self.conv_instance_norm_relu1_2(x)
        x = self.conv1_1(x)
        return x
