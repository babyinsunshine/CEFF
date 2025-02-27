
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels),3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class dispHead(nn.Module):
    def __init__(self):
        super(dispHead, self).__init__()
        outG = 3

        self.covd1 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,
                                                         padding=0, bias=True),
                                         torch.nn.LeakyReLU(inplace=True))
        self.covd2 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                                         padding=0, bias=True),
                                         torch.nn.LeakyReLU(inplace=True))
        self.covd3 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=128, out_channels=outG, kernel_size=3, stride=1,
                                                         padding=0, bias=True))

    def forward(self, x):
        return self.covd3(self.covd2(self.covd1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        # inD = 1

        self.convc1 = ConvBlock(256, 256+64)
        self.convc2 = ConvBlock(256+64, 256)

        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))

        self.conv = ConvBlock(256 + 64, 512 - 3)

    def forward(self, depth, corr):
        cor = self.convc1(corr)
        cor = self.convc2(cor)

        dep = self.convf1(depth)
        dep = self.convf2(dep)

        cor_depth = torch.cat([cor, dep], dim=1)
        out = self.conv(cor_depth)
        return torch.cat([out, depth], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()

        self.flow_head = dispHead()
        self.mask = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 64 * 9 * 3, 1, padding=0))

    def forward(self, net, corr, depth):
        net = self.encoder(depth, corr)
        delta_depth = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, mask, delta_depth

