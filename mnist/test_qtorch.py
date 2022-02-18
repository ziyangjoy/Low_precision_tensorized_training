# import numpy as np
# import tensorly as tl
# import scipy as sp
# import cupy as cp
# import time
# from cupyx.time import repeat

import torch
import qtorch
import torch.nn as nn
import torch.nn.functional as F
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

# quant_half = lambda x : float_quantize(x, exp=5, man=2, rounding="stochastic")
# quant_half = lambda x : float_quantize(x, exp=5, man=2, rounding="nearest")

def quant_half(tensor):
    # return float_quantize(tensor, exp=6, man=9, rounding="stochastic")
    # return float_quantize(tensor, exp=4, man=3, rounding="stochastic")
    return fixed_point_quantize(tensor,wl = 8, fl = 4, rounding='nearest')

device = 'cuda:0'
# # device = 'cpu'
x = torch.randn(10,10,10,device=device,dtype=torch.float)
scale = torch.tensor(2,device=device)
y = torch.tensor(-4.1,device=device)
print(quant_half(y))
