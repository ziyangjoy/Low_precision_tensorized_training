from __future__ import print_function
from tensor_layers.layers import ScaleLayer,  Q_TensorizedLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_layers import TensorizedLinear
import qtorch
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize




forward_num = FixedPoint(wl=8, fl=6)
# forward_num = FloatingPoint(exp=8, man=23)

backward_num = FloatingPoint(exp=8, man=23)
Q = Quantizer(forward_number=forward_num, backward_number=backward_num,
              forward_rounding="nearest", backward_rounding="stochastic")



def get_kl_loss(model, args, epoch):

    kl_loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "tensor"):
            
            up = torch.tensor(100.0)

            kl_loss += torch.maximum(torch.minimum(layer.tensor.get_kl_divergence_to_prior(),up),up)
    kl_mult = args.kl_multiplier * torch.clamp(
                            torch.tensor((
                                (epoch - args.no_kl_epochs) / args.warmup_epochs)), 0.0, 1.0)
    """
    print("KL loss ",kl_loss.item())
    print("KL Mult ",kl_mult.item())
    """
    return kl_loss*kl_mult.to(kl_loss.device)



def get_net(args):
    if args.model_type in ['CP','TensorTrain','TensorTrainMatrix','Tucker']:
        if args.scale:
            if args.lp == True:
                return get_TensorizedNet_LP_scale(args)
            else:
                return get_TensorizedNet(args)
        else:
            if args.lp == True:
                return get_TensorizedNet_LP(args)
            else:
                return get_TensorizedNet(args)
    else:
        return Net()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        if args.rank_loss:
            ard_loss = get_kl_loss(model,args,epoch)
            loss += ard_loss
        


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_TensorizedNet(args):

    if args.model_type=='full':
        fc1 = nn.Linear(784, 512)
        fc2 = nn.Linear(512, 10)
     
    else:
        if args.model_type=='TensorTrainMatrix':
            shape1 = [[4,7,4,7], [4,4,8,4]]    
            # shape1 = [[28,28],[16,32]]
            shape2 = [[16,32], [2,5]]

        else:
            shape1 = [28, 28, 16, 32]
            shape2 = [32, 16, 10]

    
        fc1 = TensorizedLinear(784, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc2 = TensorizedLinear(512, 10, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
    
    return TensorizedNet(fc1,fc2)

def get_TensorizedNet_LP(args):

    if args.model_type=='full':
        fc1 = nn.Linear(784, 512)
        fc2 = nn.Linear(512, 10)
     
    else:
        if args.model_type=='TensorTrainMatrix':
            shape1 = [[4,7,4,7], [4,4,8,4]]    
            # shape1 = [[28,28],[16,32]]
            shape2 = [[16,32], [2,5]]

        else:
            shape1 = [28, 28, 16, 32]
            shape2 = [32, 16, 10]

    
        fc1 = TensorizedLinear(784, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc2 = TensorizedLinear(512, 10, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
    
    return TensorizedNet_LP(fc1,fc2)

def get_TensorizedNet_LP_scale(args):

    if args.model_type=='full':
        fc1 = nn.Linear(784, 512)
        fc2 = nn.Linear(512, 10)
     
    else:
        if args.model_type=='TensorTrainMatrix':
            shape1 = [[4,7,4,7], [4,4,8,4]]    
            # shape1 = [[28,28],[16,32]]
            shape2 = [[16,32], [2,5]]

        else:
            shape1 = [28, 28, 16, 32]
            shape2 = [32, 16, 10]

    
        # fc1 = TensorizedLinear(784, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        # fc2 = TensorizedLinear(512, 10, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)

        bit = 8

        fc1 = Q_TensorizedLinear(784, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize, bit = bit, scale_w = 2**(-3), scale_b = 2**(-3))

        fc2 = Q_TensorizedLinear(512, 10, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize, bit = bit, scale_w = 2**(-3), scale_b = 2**(-3))





    
    return TensorizedNet_LP_scale(fc1,fc2,bit)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TensorizedNet(nn.Module):
    def __init__(self,fc1,fc2):
        super(TensorizedNet, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.add_module('fc1',fc1)
        self.bn1 = nn.BatchNorm1d(512)
        self.add_module('fc2',fc2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TensorizedNet_LP(nn.Module):
    def __init__(self,fc1,fc2):
        super(TensorizedNet_LP, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.add_module('fc1',fc1)
        self.bn1 = nn.BatchNorm1d(512)
        self.add_module('fc2',fc2)
        self.relu = nn.ReLU()

    def forward(self, x):
        scale = 0.1

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)

        x = x*scale

        x = Q(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class TensorizedNet_LP_scale(nn.Module):
    def __init__(self,fc1,fc2,bit):
        super(TensorizedNet_LP_scale, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.add_module('fc1',fc1)
        self.bn1 = nn.BatchNorm1d(512)
        self.add_module('fc2',fc2)
        self.relu = nn.ReLU()

        sc1 = ScaleLayer(bit = 8, scale = 2**-3)
        sc2 = ScaleLayer(bit = 8, scale = 2**-3)

        self.add_module('sc1',sc1)
        self.add_module('sc2',sc2)


    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
