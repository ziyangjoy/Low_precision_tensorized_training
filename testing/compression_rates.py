#%%
import numpy as np

log_uniform=False

def get_tucker_params(tucker_dims, tucker_rank):

    if type(tucker_rank) != list:
        tucker_rank = len(tucker_dims) * [tucker_rank]
    
    tucker_params = np.prod(tucker_rank) + sum(
        [x * y for x, y in zip(tucker_rank, tucker_dims)])

    return tucker_params


def get_cp_params(cp_dims, cp_rank):

    cp_params = cp_rank * sum(cp_dims)
    return cp_params


def get_ttm_params(ttm_dims, ttm_rank):

    if type(ttm_rank) != list:
        ttm_rank = [1] + (len(ttm_dims[0]) - 1) * [ttm_rank] + [1]
    order = len(ttm_dims[0])
    ranks = [[ttm_rank[i], ttm_rank[i + 1]] for i in range(order)]
    ttm_params = sum(
        [np.prod(x) * np.prod(y) for x, y in zip(zip(*ttm_dims), ranks)])
    return ttm_params


def get_tt_params(tt_dims, tt_rank):

    if type(tt_rank) != list:
        tt_rank = [1] + (len(tt_dims) - 1) * [tt_rank] + [1]

    order = len(tt_dims)
    ranks = [[tt_rank[i], tt_rank[i + 1]] for i in range(order)]
    tt_params = sum([np.prod([x] + y) for x, y in zip(tt_dims, ranks)])
    return tt_params


def get_params_wrapper(tensor_type, dims, rank):
    if tensor_type == 'Tucker':
        return get_tucker_params(dims, rank)
    elif tensor_type == 'CP':
        return get_cp_params(dims, rank)
    elif tensor_type == 'TensorTrain':
        return get_tt_params(dims, rank)
    elif tensor_type == 'TensorTrainMatrix':
        return get_ttm_params(dims, rank)


shape0 = [[200, 220, 250], [125, 130, 136], [200, 200, 209], [166, 175, 188],
          [200, 200, 200]]
shape1 = [4, 4, 8]

baseline_params = [10131227, 2202608, 8351593, 5461306, 7046547]
baseline_params = [128 * x for x in baseline_params]
print(baseline_params)



max_ranks = {
    'CP': [350, 306, 333, 326, 335],
    'TensorTrainMatrix': [16, 16, 16, 16, 16],
    'TensorTrain': [24, 24, 24, 24, 24],
    'Tucker': [22, 20, 22, 21, 22]
}
if log_uniform:

    true_ranks = {
        'CP': [163, 171, 161, 175, 153],
        'TensorTrainMatrix': [[1,16,4,1], [1,16,1,1], [1,16,2,1], [1,16,2,1],[1,16,1,1]],
        'TensorTrain': [[1,23,5,7,1], [1,20,3,5,1], [1,15,6,4,1], [1,22,6,5,1], [1,23,6,7,1]],
    'Tucker': [[18,17,20,21], [19,19,19,19], [13,19,19,18], [17,18,18,21],[19,20,21,18]]
    }

else:
    true_ranks = {
        'CP': [166, 181, 154, 164, 169],
        'TensorTrainMatrix': [[1,16,2,1], [1,16,2,1], [1,16,1,1], [1,16,1,1],[1,16,2,1]],
        'TensorTrain': [[1,21,5,2,1], [1,22,4,4,1], [1,23,6,5,1], [1,23,7,9,1], [1,22,4,7,1]],
        'Tucker': max_ranks['Tucker']
    }

tensorized_params = {}
ard_params = {}
tensor_types = ['CP', 'TensorTrainMatrix', 'TensorTrain', 'Tucker']

for key in tensor_types:
    tensorized_params[key] = []
    ard_params[key] = []

ii = 0

dims = [shape0[ii], shape1]

ttm_dims = dims
ttm_rank = max_ranks['TensorTrainMatrix'][ii]
print("TTM params", get_params_wrapper('TensorTrainMatrix',ttm_dims, ttm_rank))

tt_dims = dims[0] + [np.prod(dims[1])]
tt_rank = max_ranks['TensorTrain'][ii]
print("TT params", get_params_wrapper('TensorTrain',tt_dims, tt_rank))

cp_dims = tt_dims
cp_rank = max_ranks['CP'][ii]
cp_params = get_cp_params(cp_dims, cp_rank)
print("CP_params ", cp_params)
print("CP params", get_params_wrapper('CP',cp_dims, cp_rank))

print("baseline parameters", sum(baseline_params))
tucker_dims = tt_dims
tucker_rank = max_ranks['Tucker'][ii]
print("Tucker_params ", get_params_wrapper('Tucker', tucker_dims, tucker_rank))


for ii in range(len(shape0)):
    for tensor_type in tensor_types:
        
        dims = [shape0[ii],shape1]
        
        if tensor_type!='TensorTrainMatrix':
            dims = dims[0]+[np.prod(dims[1])]

        max_rank = max_ranks[tensor_type][ii]
        true_rank = true_ranks[tensor_type][ii]

        max_rank_param_count = get_params_wrapper(tensor_type,dims,max_rank) 
        tensorized_param_count = get_params_wrapper(tensor_type,dims,true_rank)

        tensorized_params[tensor_type].append(max_rank_param_count)        
        ard_params[tensor_type].append(tensorized_param_count)        

#%%

total_baseline_params = sum(baseline_params)

for tensor_type in tensor_types:
    total_ard_params = sum(ard_params[tensor_type]) 
    total_tensorized_params = sum(tensorized_params[tensor_type]) 

    
    print(tensor_type)
    ard_string = ["ARD Params" ,total_ard_params,"Compression Ratio",int(total_baseline_params/total_ard_params)]
    tensorized_string = ["Tensorized Params" ,total_tensorized_params,"Compression Ratio",int(total_baseline_params/total_tensorized_params)]

    print(''.join([str(x).ljust(20) for x in ard_string]))
    print(''.join([str(x).ljust(20) for x in tensorized_string]))
    print('\n')


#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



label = 'CP'

ind = np.arange(5)  # the x locations for the groups
num_bars = 3

width = 0.8/num_bars

correction = (width/2)*((num_bars+1)%2)

fig, ax = plt.subplots()

rects = []

labels = tensor_types

inner_labels = ['baseline','FR','ARD-LU']


for ii,means in enumerate([baseline_params,tensorized_params[label],ard_params[label]]):
    print(means)

    rects.append(ax.bar(ind + (ii-num_bars//2)*width+correction, means, width,
                label=inner_labels[ii]))

ax.set_yscale('log')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Parameter Count')
ax.set_xticks(ind)
ax.set_xlabel('DLRM Embedding')
ax.set_xticklabels(('1', '2', '3', '4', '5'))
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

#    for rect in rects:
#        autolabel(rect, "center")

fig.tight_layout()
ax.set_title(label)
plt.show()
plt.figure()
print(label)
