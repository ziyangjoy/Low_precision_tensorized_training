#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

PATH = 'saved_models/'
TRAINING = False
MODEL_NAME = 'CP'

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%

if TRAINING:

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH + MODEL_NAME)
else:
    net.load_state_dict(torch.load(PATH + MODEL_NAME), strict=False)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

#%%

#%%

model = TheModelClass()
model.load_state_dict(torch.load(PATH + model_name))
model.eval()

#%%

import os
import torch
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
import torch.distributions as td
Parameter = torch.nn.Parameter
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torch_bayesian_tensor_layers.low_rank_tensors import TensorTrainMatrix, Tucker, TensorTrain, CP

max_rank = 5
true_rank = 2
EM_STEPSIZE = 1.0

dims = [10, 10, 10]

tensor = Tucker(dims=dims,
                max_rank=max_rank,
                prior_type='log_uniform',
                em_stepsize=EM_STEPSIZE,
                learned_scale=False)

true_tensor = Tucker(dims=dims,
                     max_rank=true_rank,
                     prior_type='log_uniform',
                     em_stepsize=EM_STEPSIZE)

#%%
full = true_tensor.get_full().clone().detach()
#full = tl.tucker_to_tensor(tl.random.random_tucker(shape=dims,rank=true_rank))

import torch.distributions as td
log_likelihood_dist = td.Normal(0.0, 0.001)

tensor.sample_full = tensor.get_full


def log_likelihood():
    return torch.mean(
        torch.stack([
            -torch.mean(
                log_likelihood_dist.log_prob(full - tensor.sample_full()))
            for _ in range(5)
        ]))


def mse():
    return torch.norm(full - tensor.get_full()) / torch.norm(full)


def kl_loss():
    return log_likelihood() + tensor.get_kl_divergence_to_prior()


loss = kl_loss

#loss = log_likelihood
#%%

from allennlp.training.optimizers import DenseSparseAdam

tensor.trainable_variables[0].name

tmp_params = [(None, x) for x in tensor.parameters()]

optimizer = DenseSparseAdam(tmp_params, lr=1e-3)

#optimizer = torch.optim.Adam(tensor.trainable_variables,lr=1e-2)

#%%

for i in range(10000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

    tensor.update_rank_parameters()

    if i % 1000 == 0:
        print('Loss ', loss())
        print('RMSE ', mse())
        print('Rank ', tensor.estimate_rank())
        print(tensor.rank_parameters)

optimizer = DenseSparseAdam(tmp_params, lr=1e-4)

for i in range(10000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

    tensor.update_rank_parameters()

    if i % 1000 == 0:
        print('Loss ', loss())
        print('RMSE ', mse())
        print('Rank ', tensor.estimate_rank())
        print(tensor.rank_parameters)

#%%

i = 0
print(tensor.rank_parameters[i])
print(tensor.factor_distributions[1].stddev)
print(tensor)
#%%

print(tensor.factor_prior_distributions[-1].stddev[:, 0, 0])
print(tensor.rank_parameters[1])

#%%
tensor.update_rank_parameters()

#%%
import torch
import torch.distributions as td

mean = 0.0
std = torch.nn.Parameter(torch.tensor(1.0))

dist = td.Normal(mean, 1.0)

dist.rsample()


def loss():
    return std * torch.norm(torch.relu(std) * dist.sample([100]))


# %%
optimizer = torch.optim.Adam([std], 1e-2)

#%%

for _ in range(100):

    optimizer.zero_grad()
    loss_value = loss()
    loss_value.backward()
    optimizer.step()

    print('std', std.data.numpy())
    print(loss())

#%%
loss()

# %%

i = 0
j = 2
print(tensor.rank_parameters[i])
print(tensor.factor_prior_distributions[j].stddev[:, 8, 0])

#%%

print(tensor.factor_distributions[j].stddev[:, 0, 3, 0])
