import numpy as np

dims = [[125, 220, 250], [2, 2, 4]]

shape0 = [[200, 220, 250], [125, 130, 136], [200, 200, 209], [166, 175, 188],
          [200, 200, 200]]
shape1 = [4, 4, 8]

ii = 4

dims = [shape0[ii], shape1]

ttm_rank = 16
cp_rank = 335
tt_rank = 25
tucker_rank = 22

order = len(dims[0])
ranks = [[1, ttm_rank]] + [[ttm_rank, ttm_rank]
                           for _ in range(order - 2)] + [[ttm_rank, 1]]
dim_pairs = [[x, y] for x, y in zip(dims[0], dims[1])]
ttm_params = sum([np.prod(x + y) for x, y in zip(dim_pairs, ranks)])
print(dim_pairs, ranks)
print("TTM_params", ttm_params)

tt_params = 0
while tt_params < ttm_params:
    tt_dims = dims[0] + [np.prod(dims[1])]
    order = len(tt_dims)
    ranks = [[1, tt_rank]] + [[tt_rank, tt_rank]
                              for _ in range(order - 2)] + [[tt_rank, 1]]
    tt_params = sum([np.prod([x] + y) for x, y in zip(tt_dims, ranks)])
    print("TT_params ", tt_params, " tt_rank ", tt_rank)
    tt_rank += 1

cp_params = 0.0
while cp_params < ttm_params:
    cp_dims = dims[0] + [np.prod(dims[1])]
    cp_params = cp_rank * sum(cp_dims)
    print("CP_params ", cp_params, " cp rank ", cp_rank)
    cp_rank += 1

tucker_params = 0.0
while tucker_params < ttm_params:
    tucker_dims = dims[0] + [np.prod(dims[1])]
    tucker_params = (tucker_rank**
                     len(tucker_dims)) + tucker_rank * sum(tucker_dims)
    print("Tucker params ", tucker_params, " tucker rank ", tucker_rank)
    tucker_rank += 1

print("CP ", cp_rank, "\nTTM ", ttm_rank, "\nTT ", tt_rank, "\nTucker ",
      tucker_rank)
