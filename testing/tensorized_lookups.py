#%%

import tensorly as tl
from torch_bayesian_tensor_layers.layers import TensorizedEmbedding
import torch
import random
import numpy as np

from emb_utils import get_cum_prod, tensorized_lookup

import sys
tensor_type = 'TensorTrainMatrix'

shape = [[100, 10, 13], [4, 2, 8]]

n = np.prod(shape[0])
r = np.prod(shape[1])

layer = TensorizedEmbedding(tensor_type=tensor_type, shape=shape)
factors = layer.tensor.factors
batch_size = 10
x_list = [random.randint(0, n - 1) for _ in range(batch_size)]
x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [
    n,
]
x = x.view(-1)
idx = x
full = layer.tensor.get_full()
rows = layer.forward(idx)

dims = layer.shape[0]

out = []
rem = idx

cum_prod = get_cum_prod(shape)

gathered_rows = tensorized_lookup(idx, factors, cum_prod, shape, tensor_type)

rel_err = torch.norm(rows - gathered_rows) / torch.norm(rows)

print(rel_err.item())
assert (rel_err < 1e-6)
"""

tmp_factors[0]
core.shape

gathered_rows = torch.einsum('ij,jkl->ikl',tmp_factors[0],core)

gathered_rows = torch.einsum('ij,ijk->ijk',tmp_factors[1],gathered_rows)

gathered_rows = gathered_rows.sum(axis=1)

gathered_rows = gathered_rows@full_factors[-1].T

print(torch.norm(rows-gathered_rows)/torch.norm(rows))


tmp_factors[0].shape
core.shape

gathered_rows = torch.einsum('ij,jkl->ikl',tmp_factors[0],core)

gathered_rows = torch.einsum('ij,ijk->ijk',tmp_factors[1],gathered_rows)

gathered_rows = gathered_rows.sum(axis=1)

gathered_rows = gathered_rows@full_factors[-1].T

print(torch.norm(rows-gathered_rows)/torch.norm(rows))



def reduce_fun(x,y):
    return torch.mult(x,y)

from functools import reduce

reduced = reduce(lambda x,y:x*y,tmp_factors)
print(reduced.shape)

tmp_factors = [reduced]

for factor in full_factors[-len(shape[1]):]:
    tmp_factors.append(factor)

gathered_rows = tl.kruskal_to_tensor((None,tmp_factors)).view(-1,np.prod(shape[1]))


batch_tensor = tt_gather_rows(cores,tensorized_indices,layer.shape)

#batch_tensor = tl.tt_to_tensor(tmp_factors).view(-1,np.prod(shape[1]))

gathered_rows = batch_tensor

torch.norm(rows-gathered_rows)/torch.norm(rows)
"""
