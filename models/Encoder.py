import torch.nn.functional as F
import torch
import torch.nn as nn
import functools
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, feature_dim=1024, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            feature_dim (int)   -- the dimension of the output feature vector
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add downsampling layers
            model += [nn.Conv2d(ngf * mult, ngf * mult,
                                         kernel_size=3, stride=2,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_downsampling):  # add downsampling layers
            mult = mult * (2 ** i)
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                                         kernel_size=3, stride=2,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        model += [
            nn.AdaptiveAvgPool2d((2, 2))
        ]
        # Fully connected layer to flatten the feature map into a feature vector
        self.conv = nn.Sequential(*model)
        self.fc = nn.Linear((ngf * mult * 2) * 2 * 2, feature_dim)

    def forward(self, input):
        # Forward pass through each layer in `self.conv` sequentially
        # and save the feature map after the last `ResnetBlock`
        x = input
        last_resblock_output = None

        for layer in self.conv:
            x = layer(x)
            # If the layer is a ResnetBlock, update last_resblock_output
            if isinstance(layer, ResnetBlock):
                last_resblock_output = x

        # After all layers, x is the result of pooling
        # Proceed with flattening and fully connected layer
        x = x.view(x.size(0), -1)
        feature = self.fc(x)

        # Return the final feature and the output of the last ResnetBlock
        return feature, last_resblock_output

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A Resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Two independent encoders
        self.encoder1 = ResnetEncoder(feature_dim=1024, use_dropout=False)
        self.encoder2 = ResnetEncoder(feature_dim=1024, use_dropout=False)
        # Add MLP layers for classification tasks (optional, not implemented here)

    def forward(self, x1, x2):
        # Input two sets of data
        z1, _ = self.encoder1(x1)  # Output feature from the first encoder
        z2, _ = self.encoder2(x2)  # Output feature from the second encoder
        return z1, z2

def init_Encoder(net=Encoder(), init_type='normal', init_gain=0.02, gpu_ids=[0]):
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

class InfoNCE_Loss(nn.Module):
    """
    InfoNCE Loss class implementation.
    """

    def __init__(self, temperature=0.07):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, query, key):
        """
        Compute InfoNCE Loss.

        Parameters:
            query (torch.Tensor): Query vectors, shape (batch_size, feature_dim).
            key (torch.Tensor): Key vectors, shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Computed InfoNCE Loss.
        """
        # L2 normalize query and key vectors
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        # Compute similarity between query and key vectors
        logits = torch.matmul(query, key.T) / self.temperature  # (batch_size, batch_size)

        # Define target labels; true labels are on the diagonal (query and key are paired)
        labels = torch.arange(query.size(0), device=query.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss