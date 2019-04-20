import torch.nn as nn
import torch.nn.functional as F
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
from octconv import OctConv, tuplify, tuplify_bn, TensorTuple

__all__ = ['OctXResNet', 'octxresnet18', 'octxresnet34', 'octxresnet50', 'octxresnet101', 'octxresnet152']

class Flatten(nn.Module):
    def forward(self, x): 
        if isinstance(x, TensorTuple): x = torch.cat([x.hf, x.lf], dim=1)
        return x.view(x.size(0), -1)

class AdaptivePool(nn.Module):
    def forward(self, x, shape):
        if not isinstance(x, TensorTuple): return F.adaptive_avg_pool2d(x, shape[2:])
        else: 
            return TensorTuple(F.adaptive_avg_pool2d(x.hf, shape[0][2:]), 
                               F.adaptive_avg_pool2d(x.lf, shape[1][2:]))
    
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def init_bn(bn, zero_bn):
    if hasattr(bn, 'tupified') and bn.tupified:
        nn.init.constant_(bn.inst1.weight, 0. if zero_bn else 1.)
        nn.init.constant_(bn.inst2.weight, 0. if zero_bn else 1.)
    else:
        nn.init.constant_(bn.weight, 0. if zero_bn else 1.)      

def noop(x, **kwargs): return x    

def conv(ni, nf, alpha, ks=3, stride=1, bias=False):
    if alpha==(0, 0):
        return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
    else:
        return OctConv(ni, nf, alpha, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def conv_layer(ni, nf, alpha, ks=3, stride=1, zero_bn=False, act=True):
    bn = (nn.BatchNorm2d if alpha==(0, 0) else tuplify_bn(nn.BatchNorm2d, alpha[1]))(nf)
    init_bn(bn, zero_bn)
    layers = [conv(ni, nf, alpha, ks, stride=stride), bn]
    if act: layers.append((nn.ReLU if alpha==(0, 0) else tuplify(nn.ReLU))(inplace=True))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, a=0, stride=1):
        super().__init__()
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, (a, a), 1)]
        layers += [
            conv_layer(nh, nf, (a, a), 3, stride=stride, zero_bn=True, act=False)
        ] if expansion==1 else [
            conv_layer(nh, nh, (a, a), 3, stride=stride),
            conv_layer(nh, nf, (a, a), 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, (a, a), 1, act=False)
        self.pool = noop if stride==1 else AdaptivePool()
        self.act_fn = (nn.ReLU if a==0 else tuplify(nn.ReLU))(inplace=True)
    def forward(self, x): 
        conv_out = self.convs(x)
        return self.act_fn(conv_out + self.idconv(self.pool(x, shape=conv_out.shape)))

def filt_sz(recep): return min(64, 2**math.floor(math.log2(recep*0.75)))


class OctXResNet(nn.Sequential):
    def __init__(self, a, expansion, layers, c_in=3, c_out=1000):
        stem = []
        sizes = [c_in,32,32,64]
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i+1], (0, a) if i==0 else (a, a), stride=2 if i==0 else 1))
            #nf = filt_sz(c_in*9)
            #stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
            #c_in = nf

        block_szs = [64//expansion,64,128,256,512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], a, l, 1 if i==0 else 2)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            (nn.MaxPool2d if a==0 else tuplify(nn.MaxPool2d))(kernel_size=3, stride=2, padding=1),
            *blocks,
            (nn.AdaptiveAvgPool2d if a==0 else tuplify(nn.AdaptiveAvgPool2d))(1), Flatten(),
            nn.Linear(block_szs[-1]*expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, a, blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, a, stride if i==0 else 1)
              for i in range(blocks)])

def octxresnet(a, expansion, n_layers, name, pretrained=False, **kwargs):
    model = OctXResNet(a, expansion, n_layers, **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls[name]))
    return model

me = sys.modules[__name__]
for n,e,l in [
    [ 18 , 1, [2,2,2 ,2] ],
    [ 34 , 1, [3,4,6 ,3] ],
    [ 50 , 4, [3,4,6 ,3] ],
    [ 101, 4, [3,4,23,3] ],
    [ 152, 4, [3,8,36,3] ],
]:
    name = f'octxresnet{n}'
    setattr(me, name, partial(octxresnet, expansion=e, n_layers=l, name=name))