import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = ['OctResNet', 'oct_resnet18', 'oct_resnet34', 'oct_resnet50', 
           'oct_resnet101', 'oct_resnet152']

class TensorTuple():
    '''Stores high and low frequency tensors.'''    
    def __init__(self, hf, lf):
        self.hf, self.lf = hf, lf
        self.shape = (self.hf.shape, self.lf.shape)
        
    def __repr__(self):
        return 'tensor_tuple: '+self.hf.__repr__() + self.lf.__repr__()
        
    def __add__(self, other):
        return TensorTuple(self.hf+other.hf, self.lf+other.lf)
    
def tupify(layer):
    
    '''Turns a layer (say relu of maxpool) into one that operates on TensorTuple'''
    
    if hasattr(layer, 'tupified'):
        if layer.tupified: return layer
    
    class Tuplefied(nn.Module):
        tupified=True
        def __init__(self, *args, **kwargs):        
            super().__init__()
            self.inst1, self.inst2 = layer(*args, **kwargs), layer(*args, **kwargs)
        def forward(self, x):
            return TensorTuple(self.inst1(x.hf), self.inst2(x.lf))        
    return Tuplefied

def tupify_bn(layer, alpha):
    
    '''Turns a batch norm layer into one that operates on Tensortuple'''
    
    if hasattr(layer, 'tupified'):
        if layer.tupified: return layer
    
    class Tuplefied(nn.Module):
        tupified=True
        def __init__(self, num_features, *args, **kwargs):        
            super().__init__()
            num_lf = int(alpha*num_features)
            num_hf = num_features-num_lf
            self.inst1, self.inst2 = layer(num_hf, *args, **kwargs), layer(num_lf, *args, **kwargs)
        def forward(self, x):
            return TensorTuple(self.inst1(x.hf), self.inst2(x.lf))        
    return Tuplefied

class OctConv(nn.Module):
    
    '''Main layer replacing nn.Conv2d'''
    
    def __init__(self, in_channels, out_channels, alpha, **kwargs):
        
        super(OctConv, self).__init__()        
        if not isinstance(alpha, (tuple,list)):
            alpha = (alpha, alpha)
        assert alpha[0] >= 0 and alpha[0] < 1 and alpha[1] >= 0 and alpha[1] < 1
        
        self.low_in, self.low_out = int(alpha[0] * in_channels), int(alpha[1] * out_channels)
        self.high_in, self.high_out = in_channels-self.low_in, out_channels-self.low_out
        
        if self.low_in != 0:
            self.L2H = nn.Conv2d(self.low_in, self.high_out, **kwargs)
            nn.init.kaiming_normal_(self.L2H.weight, mode='fan_out', nonlinearity='relu')
            
            if self.low_out != 0:
                self.L2L = nn.Conv2d(self.low_in, self.low_out, **kwargs) 
                nn.init.kaiming_normal_(self.L2L.weight, mode='fan_out', nonlinearity='relu')
            
        if self.low_out != 0:
            self.H2L = nn.Conv2d(self.high_in, self.low_out, **kwargs)
            nn.init.kaiming_normal_(self.H2L.weight, mode='fan_out', nonlinearity='relu')
            
        self.H2H = nn.Conv2d(self.high_in, self.high_out, **kwargs)
        nn.init.kaiming_normal_(self.H2H.weight, mode='fan_out', nonlinearity='relu')
        
        self.upsample = partial(F.interpolate, mode='nearest')
        self.avg_pool = partial(F.avg_pool2d, kernel_size=2)

    def forward(self, x):
        
        if self.low_in == 0 and self.low_out == 0:
            return self.H2H(x)  # Regular convolution
        elif self.low_in == 0 and self.low_out > 0:
            return TensorTuple(self.H2H(x), self.H2L(self.avg_pool(x)))
        elif self.low_in > 0 and self.low_out == 0:
            hf, lf = x.hf, x.lf
            hh = self.H2H(hf)
            return hh + self.upsample(self.L2H(lf), size=hh.shape[2:])
        else:
            hf, lf = x.hf, x.lf
            hh = self.H2H(hf); ll = self.L2L(lf)
            if hf.shape[2]%2 ==0 and hf.shape[3]%2==0:
                l_out = ll + self.H2L(self.avg_pool(hf))
            else:
                l_out = ll + self.H2L(F.adaptive_avg_pool2d(hf, ll.shape[2:]))
            return TensorTuple(hh + self.upsample(self.L2H(lf), size=hh.shape[2:]), l_out)    
        
def conv3x3(in_planes, out_planes, alpha, stride=1, groups=1, oct_layer=True):
    """3x3 convolution with padding"""
    if oct_layer:
        return OctConv(in_planes, out_planes, alpha, kernel_size=3, stride=stride,
                       padding=1, groups=groups, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, alpha, stride=1, oct_layer=True):
    """1x1 convolution"""
    if oct_layer:
        return OctConv(in_planes, out_planes, alpha, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class OctBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, alpha, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, last_block=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            
        relu_layer = nn.ReLU if last_block else tupify(nn.ReLU)
        norm_layer = norm_layer if last_block else tupify_bn(norm_layer, alpha)
        alpha = (alpha, 0) if last_block else alpha
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, alpha, stride)
        self.bn1 = norm_layer(planes)
        self.relu = relu_layer(inplace=True)
        self.conv2 = conv3x3(planes, planes, alpha, oct_layer=not last_block)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        if isinstance(out, TensorTuple):    
            out += identity
        else:
            hf, lf = identity.hf, identity.lf
            lf = F.interpolate(lf, size=hf.shape[2:], mode='nearest')
            out += torch.cat([hf, lf], dim=1)     
        
        out = self.relu(out)

        return out

class OctBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, last_block=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        relu_layer = nn.ReLU if last_block else tupify(nn.ReLU)
        norm_layer = norm_layer if last_block else tupify_bn(norm_layer, alpha)
        alpha = (alpha, 0) if last_block else alpha
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, alpha)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, alpha, stride, groups, oct_layer=not last_block)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, alpha, oct_layer=not last_block)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = relu_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if isinstance(out, TensorTuple):    
            out += identity
        else:
            hf, lf = identity.hf, identity.lf
            lf = F.interpolate(lf, size=hf.shape[2:], mode='nearest')
            out += torch.cat([hf, lf], dim=1)
            
            
        out = self.relu(out)

        return out


class OctResNet(nn.Module):

    def __init__(self, block, layers, alpha, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.alpha = alpha
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = OctConv(3, self.inplanes, alpha = (0, alpha), kernel_size=7, stride=2, padding=3,
                               bias=False)        
        self.bn1 = tupify_bn(norm_layer, alpha)(self.inplanes)
        self.relu = tupify(nn.ReLU)(inplace=True)
        self.maxpool = tupify(nn.MaxPool2d)(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,last_block=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, last_block=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.alpha, stride),
                tupify_bn(norm_layer, self.alpha)(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.alpha, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for k in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.alpha, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                last_block=last_block and k == (blocks-1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def oct_resnet18(alpha, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(OctBasicBlock, [2, 2, 2, 2], alpha, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def oct_resnet34(alpha, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(OctBasicBlock, [3, 4, 6, 3], alpha,  **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def oct_resnet50(alpha, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(OctBottleneck, [3, 4, 6, 3], alpha, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def oct_resnet101(alpha, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(OctBottleneck, [3, 4, 23, 3], alpha, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def oct_resnet152(alpha, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(OctBottleneck, [3, 8, 36, 3], alpha, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model



