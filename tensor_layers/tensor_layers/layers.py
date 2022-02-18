import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from . import low_rank_tensors
from .emb_utils import get_cum_prod,tensorized_lookup

from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


class TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]),self.bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(2/(np.prod(self.shape[0])+np.prod(self.shape[1])))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            tensor_shape = self.shape[0]+self.shape[1]

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        self.parameters = self.tensor.parameters

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

        self.cum_prod = get_cum_prod(shape)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


    def forward(self, x,rank_update=True):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        #x_ind = self.ind2sub(x)

#        full = self.tensor.get_full()
#        full = torch.reshape(full,[self.voc_quant,self.emb_quant])
#        rows = full[x]
        if hasattr(self.tensor,"masks"):
            rows = tensorized_lookup(x,self.tensor.get_masked_factors(),self.cum_prod,self.shape,self.tensor_type)
        else:
            rows = tensorized_lookup(x,self.tensor.factors,self.cum_prod,self.shape,self.tensor_type)
#        rows = gather_rows(self.tensor, x_ind)

        rows = rows.view(x.shape[0], -1)

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        

        return rows.to(x.device)

#---------------------Define Scale Layer-------------------------------------------------------
class scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale, quant, min_q, max_q):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, scale)
        ctx.quant = quant
        ctx.input_div_scale = input/scale
        ctx.q_input = quant(ctx.input_div_scale)
        ctx.min_q = torch.tensor(min_q)
        ctx.max_q = torch.tensor(max_q)
        return scale * ctx.q_input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, scale= ctx.saved_tensors
        grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
        grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))

        grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q, max = ctx.max_q)

        return grad_input, grad_scale, None, None, None

class ScaleLayer(nn.Module):

   def __init__(self, scale=2**(-5), bit = 8):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([scale]))

       max_q = 2.0**(bit-1)-1.0
       min_q = -2.0**(bit-1)
       quant = lambda x : fixed_point_quantize(x, wl=8, fl=0, rounding="nearest")

       self.quant = quant
       self.min_q = min_q
       self.max_q = max_q

   def forward(self, input):
       return scale.apply(input,self.scale,self.quant,self.min_q,self.max_q)


class Quantized_Linear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                eta = None,
                device=None,
                dtype=None,
                bit = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5)
    ):

        super(Quantized_Linear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features

        self.max_q = 2.0**(bit-1)-1.0
        self.min_q = -2.0**(bit-1)
        self.quant = lambda x : fixed_point_quantize(x, wl=8, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

    def forward(self, input):
        self.weight = scale.apply(self.weight,self.scale_w,self.quant,self.min_q,self.max_q)
        self.bias = scale.apply(self.bias,self.scale_b,self.quant,self.min_q,self.max_q)
        
        return F.linear(input,self.weight,self.bias)



class Q_TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                bit = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5)
    ):

        super(Q_TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        self.max_q = 2.0**(bit-1)-1.0
        self.min_q = -2.0**(bit-1)
        self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))


    def forward(self, input, rank_update=True):
        

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.quant,self.min_q,self.max_q))
        Q_bias = nn.Parameter(scale.apply(self.bias,self.scale_b,self.quant,self.min_q,self.max_q))
        
        return F.linear(input,self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]),Q_bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class Quantized_conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 bit = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Quantized_conv2d,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype)

        self.stride = stride
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups

        self.max_q = 2.0**(bit-1)-1.0
        self.min_q = -2.0**(bit-1)
        self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

    def forward(self, input):
        self.weight = scale.apply(self.weight,self.scale_w,self.quant,self.min_q,self.max_q)
        self.bias = scale.apply(self.bias,self.scale_b,self.quant,self.min_q,self.max_q)
        
        return F.conv2d(input,self.weight,self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)