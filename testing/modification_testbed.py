#%%
import numpy as np

import matplotlib.pyplot as plt

n = 10000
r = 1
m = 1000

alpha = 0.1
beta = 0.1

X = 2 * (-0.5 + np.random.beta(alpha, beta, size=[n, r]))

for i in range(1, 10):
    plt.figure()
    plt.hist(np.reshape(X, -1), bins=100)
    X *= i * 2 * (-0.5 + np.random.beta(alpha, beta, size=[n, r]))

#%%

X = np.random.uniform(0, 1, size=[n, r])
Y = np.random.uniform(0, 1, size=[n, r])

alpha = 0.4
beta = alpha

X = np.random.beta(alpha, beta, size=[n, r])
Y = np.random.beta(alpha, beta, size=[n, r])

plt.hist(np.reshape(X + Y, -1), bins=100)
#%%
from scipy.stats import loguniform
X = loguniform.rvs(np.exp(-1 / 4), np.exp(1 / 4), size=[n, r])
Y = loguniform.rvs(np.exp(-1 / 4), np.exp(1 / 4), size=[r, m])
plt.hist(np.reshape(np.log(X) @ np.log(Y), -1), bins=100)

# %%

import t3nsor as t3
import torch
import random
import numpy as np

batch_size = 100

n = 10000
r = 64

x_list = [random.randint(0, n) for _ in range(batch_size)]
x = torch.tensor(x_list).view(batch_size, )

layer = t3.TTEmbedding(voc_size=n, emb_size=r, auto_shapes=True)

for core in layer.tt_matrix.tt_cores:
    print(core.shape)

cum_prod = torch.tensor(
    list(reversed([1] + list(np.cumprod(layer.voc_quant[::-1])[:-1]))))
idx = x

for i, y in enumerate(cum_prod):
    print(i, y)


def ind2sub(idx):

    out = []
    rem = idx

    for i, y in enumerate(cum_prod):
        val = torch.floor(rem.float() / y).long()
        rem = torch.fmod(rem, y)
        #        print(idx)

        #        val,rem = divmod(rem,y)
        out.append(torch.tensor(val))

    out = torch.stack(out, dim=1).view(x.shape[0], -1)
    return out


#%%


def a():
    full = layer.tt_matrix.full()
    x_list = [random.randint(0, n - 1) for _ in range(batch_size)]
    x = torch.tensor(x_list).view(batch_size, )
    return full[x]


x_list = [random.randint(0, n - 1) for _ in range(batch_size)]


def b():
    #    x_list = [random.randint(0,n-1) for _ in range(batch_size)]
    x = torch.tensor(x_list).view(batch_size, )
    rows = layer(x)
    """     
    x_ind = ind2sub(x)

    rows = t3.gather_rows(layer.tt_matrix, x_ind)
    rows = rows.view(x.shape[0], -1)
    """
    return rows


full = layer.tt_matrix.full()


def c():
    #    x_list = [random.randint(0,n-1) for _ in range(batch_size)]
    x = torch.tensor(x_list).view(batch_size, )
    return full[x]


torch.norm(b() - c()) / torch.norm(b())
#%%

import matplotlib.pyplot as plt
import seaborn as sns
old = [91e-3, 448e-3, 891e-3, 1.65, 3.1]
mine = [712e-6, 732e-6, 723e-6, 708e-6, 718e-6]
std_lookup = [217e-6, 229e-6, 235e-6, 238e-6, 237e-6]
x = [1000000, 5000000, 10000000, 20000000, 40000000]

plt.loglog(x, old, label='tensorized-naive')
plt.loglog(x, mine, label='tensorized-proposed')
plt.loglog(x, std_lookup, label='not tensorized')
plt.xlabel('Embedding Size')
plt.ylabel('Batch Inference Time')
sns.set_context('poster')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#%%

x_list = [random.randint(0, n - 1) for _ in range(batch_size)]

x = torch.tensor(x_list).view(batch_size, )

b, c = layer(x)

torch.norm(b - c) / torch.norm(b)

#%%
#x_ind = t3.ind2sub(layer.voc_quant, x)
#x_ind = out # torch.tensor([[0,0,1],[0,2,2]]).view([2,3])#t3.ind2sub(layer.voc_quant, x)

torch.norm(rows - std_out) / torch.norm(std_out)

#%%
layer.voc_quant

cum_prod = list(reversed([1] + list(np.cumprod(layer.voc_quant[1:]))))

x

out = []
rem = x.numpy()

for i, y in enumerate(cum_prod):
    val, rem = divmod(rem, y)
    out.append(torch.tensor(val))

out = torch.stack(out, dim=1).view(x.shape[0], -1)

#%%
xshape = list(x.shape)
xshape_new = xshape + [
    n,
]
x = x.view(-1)

tt_matrix = layer.tt_matrix

raw_shapes = tt_matrix.raw_shape

n_bases = [1]
m_bases = [1]

for k in range(1, len(raw_shapes[0])):
    n_bases.append(raw_shapes[0][-k] * n_bases[-1])
    m_bases.append(raw_shapes[1][-k] * m_bases[-1])

n_bases.reverse()
m_bases.reverse()
n_bases = torch.tensor(n_bases)
m_bases = torch.tensor(m_bases)

#%%


def convert_to_tt(idx, dims):
    out = []
    rem = idx

    for x in dims:
        val, rem = divmod(rem, int(x))
        out.append(val)
    return out


idx = 10101

tt_idx = convert_to_tt(idx, n_bases)

#%%

d = 3

full_out = tt_matrix.full()[idx]

my_out = t3.ops.gather_rows(tt_matrix,
                            torch.reshape(torch.tensor(tt_idx), shape=(1, d)))

x_ind = t3.ind2sub(n_bases, idx)
rows = t3.gather_rows(self.tt_matrix, x_ind)
rows = rows.view(x.shape[0], -1)
#core_slices =

#%%


def a():
    x_list = [random.randint(0, n) for _ in range(batch_size)]
    full = t3.naive_full(tt_matrix)
    rows = full[x]


def b():
    x_list = [random.randint(0, n) for _ in range(batch_size)]
    full = tt_matrix.full()
    rows = full[x]


#%%

import t3nsor as t3
import torch
import random
import numpy as np


def convert_to_tt(idx, dims):
    out = []
    rem = idx

    for x in dims:
        val, rem = divmod(rem, int(x))
        out.append(val)
    return out


#%%
n = 1000
r = 2 * 2 * 4

layer = t3.TTEmbedding(voc_size=n, emb_size=r, auto_shapes=True)

for x in layer.tt_matrix.tt_cores:
    print(x.shape)

batch_size = 100

x_list = [random.randint(0, n) for _ in range(batch_size)]

x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [
    n,
]
x = x.view(-1)

tt_matrix = layer.tt_matrix

raw_shapes = tt_matrix.raw_shape

full = layer.tt_matrix.full()

rows = full[x]

idx = x
"""
x = 11
y = 5
print(divmod(x,y))
a = torch.fmod(torch.tensor(x),torch.tensor(y))
b = torch.tensor(x)//torch.tensor(y)
print((b,a))
"""


def torch_divmod(x, y):
    return torch.tensor(x) // torch.tensor(y), torch.fmod(
        torch.tensor(x), torch.tensor(y))


#%%
print(idx)
dims = layer.shape[0]
out = []
rem = idx

cum_prod = torch.tensor([100, 10])

for x in cum_prod:
    val, rem = torch_divmod(rem, x)
    out.append(val)

out.append(rem)
out = torch.stack(out).T

print(out)

tensorized_indices = torch.tensor(out).view(batch_size, -1)

torch.norm(full[idx] - t3.gather_rows(layer.tt_matrix, tensorized_indices).
           view(batch_size, -1)) / torch.norm(full[idx])

#%%
tensorized_indices = t3.ind2sub(layer.voc_quant, idx)

new_rows = t3.gather_rows(layer.tt_matrix, tensorized_indices)
new_rows = new_rows.view(idx.shape[0], -1)

torch.norm(rows - new_rows) / torch.norm(rows)
# %%
