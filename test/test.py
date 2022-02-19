# from pyrsistent import T
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

# from tensor_layers.layers import ScaleLayer, Q_TensorizedLinear


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
    
f = scale.apply
bit = 8
q_max = 2.0**(bit-1)-1.0
q_min = -2.0**(bit-1)
quant = lambda x : fixed_point_quantize(x, wl=8, fl=0, rounding="nearest")

scale = torch.tensor(.3, requires_grad=True)
input = torch.zeros(3,3,3)
input[0,0,0] = 00
input[0,0,1] = 300

input.requires_grad = True

input_q = f(input,scale,quant,q_min,q_max)

print(input_q)

loss = torch.sum(input_q**2)
loss.backward()

print(scale.grad)
print(input.grad)

input = torch.randn(4,3)
weight = torch.randn(1,3)
output = F.linear(input,weight)
print(output.shape)