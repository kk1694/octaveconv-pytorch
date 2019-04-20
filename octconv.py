import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class TensorTuple():
    '''Stores high and low frequency tensors.'''    
    def __init__(self, hf, lf):
        self.hf, self.lf = hf, lf
        self.shape = (self.hf.shape, self.lf.shape)
        
    def __repr__(self):
        return 'tensor_tuple: '+self.hf.__repr__() + self.lf.__repr__()
        
    def __add__(self, other):
        return TensorTuple(self.hf+other.hf, self.lf+other.lf)
    
def tuplify(layer):
    
    '''Turns a layer (say relu of maxpool) into one that operates on TensorTuple'''
    
    if hasattr(layer, 'tuplified') and layer.tuplified: return layer
    
    class Tuplefied(nn.Module):
        tuplified=True
        def __init__(self, *args, **kwargs):        
            super().__init__()
            self.inst1, self.inst2 = layer(*args, **kwargs), layer(*args, **kwargs)
        def forward(self, x, *args, **kwargs):
            return TensorTuple(self.inst1(x.hf, *args, **kwargs), self.inst2(x.lf, *args, **kwargs))        
    return Tuplefied

def tuplify_bn(layer, alpha):
    
    '''Turns a batch norm layer into one that operates on Tensortuple'''
    
    if hasattr(layer, 'tuplified') and layer.tuplified: return layer
    
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
    
    def __init__(self, in_channels, out_channels, alpha, kernel_size, **kwargs):
        
        super().__init__()        
        if not isinstance(alpha, (tuple,list)):
            alpha = (alpha, alpha)
        assert alpha[0] >= 0 and alpha[0] < 1 and alpha[1] >= 0 and alpha[1] < 1
        
        self.low_in, self.low_out = int(alpha[0] * in_channels), int(alpha[1] * out_channels)
        self.high_in, self.high_out = in_channels-self.low_in, out_channels-self.low_out
        
        if self.low_in != 0:
            self.L2H = nn.Conv2d(self.low_in, self.high_out, kernel_size, **kwargs)
            
            if self.low_out != 0:
                self.L2L = nn.Conv2d(self.low_in, self.low_out, kernel_size, **kwargs) 
            
        if self.low_out != 0:
            self.H2L = nn.Conv2d(self.high_in, self.low_out, kernel_size, **kwargs)
            
        self.H2H = nn.Conv2d(self.high_in, self.high_out, kernel_size, **kwargs)
        
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
                l_out = ll + self.H2L(F.adaptive_avg_pool2d(hf, [ll.shape[2]*self.H2L.stride[0],
                                                                 ll.shape[3]*self.H2L.stride[1]]))
            return TensorTuple(hh + self.upsample(self.L2H(lf), size=hh.shape[2:]), l_out)    
  