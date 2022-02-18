"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
from tensor_layers.layers import ScaleLayer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensor_layers import TensorizedLinear
import qtorch
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


__all__ = ["VGG16LP", "VGG16BNLP", "VGG19LP", "VGG19BNLP"]


forward_num = FixedPoint(wl=8, fl=5)
# forward_num = FixedPoint(wl=4, fl=1)

# forward_num = FloatingPoint(exp=8, man=23)

backward_num = FloatingPoint(exp=8, man=23)
Q = lambda: Quantizer(forward_number=forward_num, backward_number=backward_num,
              forward_rounding="nearest", backward_rounding="stochastic")


def get_kl_loss(model, args, epoch):

    kl_loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "tensor"):
            
            up = torch.tensor(10.0)

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
        if args.lp:
            # return VGG_LP(args,quant = Q)
            return VGG_tensor_LP(args,quant = Q)
        else:
            # return VGG_LP(args, quant = Q)
            return VGG_tensor(args,quant = None)
    else:
        return VGG_LP(quant = Q)

def make_layers(cfg, quant, batch_norm=False, quantized = True):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != "N"
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if use_quant and quantized:
                layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


cfg = {
    16: [
        "64",
        "64",
        "M",
        "128",
        "128",
        "M",
        "256",
        "256",
        "256",
        "M",
        "512",
        "512",
        "512",
        "M",
        "512",
        "512",
        "512",
        "M",
    ],
    19: [
        '64',
        '64',
        "M",
        '128',
        '128',
        "M",
        '256',
        '256',
        '256',
        '256',
        "M",
        '512',
        '512',
        '512',
        '512',
        "M",
        '512',
        '512',
        '512',
        '512',
        "M",
    ],
}

class VGG(nn.Module):
    def __init__(self, quant=None, num_classes=10, depth=16, batch_norm=False):

        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm, quantized=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x



class VGG_LP(nn.Module):
    def __init__(self, args,quant=None, num_classes=10, depth=16, batch_norm=False):

        super(VGG_LP, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm, quantized=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class VGG_tensor_LP(nn.Module):
    def __init__(self, args, quant=Q, num_classes=10, depth=16, batch_norm=True):

        super(VGG_tensor_LP, self).__init__()
        depth = 19
        self.features = make_layers(cfg[depth], quant, batch_norm, quantized=args.lp)
        shape1 = [[4,4,8,4], [4,4,8,4]]
        shape2 = [[4,4,8,4], [4,4,8,4]]
        shape3 = [[16,32], [2,5]]
        fc1 = TensorizedLinear(512, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc2 = TensorizedLinear(512, 512, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc3 = TensorizedLinear(512, 10, shape=shape3, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)

        self.add_module('fc1',fc1)
        self.add_module('fc2',fc2)
        self.add_module('fc3',fc3)

        sc1 = ScaleLayer(init_value=0.1)
        self.add_module('sc1',sc1)
        sc2 = ScaleLayer(init_value=0.1)
        self.add_module('sc2',sc2)


        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.quant = quant()


        self.classifier = nn.Sequential(
            nn.Dropout(),
            fc1,
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            fc2,
            nn.ReLU(True),
            quant(),
            fc3,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # rewrite self.classifier to add scaling part
        scale = 0.1

        x = self.dropout(x)
        x = self.fc1(x)
        x = scale * x
        # x = self.sc1(x)
        x = self.relu(x)
        x = self.quant(x)

        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.sc2(x)
        x = scale * x
        x = self.relu(x)
        x = self.quant(x)

        x = self.fc3(x)


        # x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class VGG_tensor(nn.Module):
    def __init__(self, args, quant=Q, num_classes=10, depth=16, batch_norm=True):

        super(VGG_tensor, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm, quantized=args.lp)
        shape1 = [[4,4,8,4], [4,4,8,4]]
        shape2 = [[4,4,8,4], [4,4,8,4]]
        shape3 = [[16,32], [2,5]]
        fc1 = TensorizedLinear(512, 512, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc2 = TensorizedLinear(512, 512, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc3 = TensorizedLinear(512, 10, shape=shape3, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)

        self.add_module('fc1',fc1)
        self.add_module('fc2',fc2)
        self.add_module('fc3',fc3)


        self.classifier = nn.Sequential(
            nn.Dropout(),
            fc1,
            nn.ReLU(True),
            nn.Dropout(),
            fc2,
            nn.ReLU(True),
            fc3,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


class VGG16LP(Base):
    pass


class VGG16BNLP(Base):
    kwargs = {"batch_norm": True}


class VGG19LP(Base):
    kwargs = {"depth": 19}


class VGG19BNLP(Base):
    kwargs = {"depth": 19, "batch_norm": True}



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.cross_entropy(output,target)
        loss = F.nll_loss(output, target)
        
        if args.rank_loss:
            ard_loss = get_kl_loss(model,args,epoch)
            loss += ard_loss
        


        loss.backward()
        grad_up = 10
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_up)
        optimizer.step()

        #quantize weights
        weight_quant = lambda x : fixed_point_quantize(x, wl=8, fl=5, rounding="stochastic")
        # weight_quant = lambda x : fixed_point_quantize(x, wl=8, fl=5, rounding="nearest")

        if args.lp:
            saved_first = []
            for p in model.features[0].parameters():
                saved_first.append(p.data)

            saved_last = []
            for p in model.fc3.parameters():
                saved_last.append(p.data)
        
            # for group in optimizer.param_groups:
            #     for p in group["params"]:
            #         p.data = weight_quant(p.data).data
            
            for layer in model.modules():
                if hasattr(layer, "tensor"):
                    for p in layer.tensor.factors:
                        p.data = weight_quant(p.data).data
                elif not hasattr(layer, "scale"):
                    for p in layer.parameters():
                        if p.requires_grad:
                            p.data = weight_quant(p.data).data

            for p,q in zip(model.features[0].parameters(), saved_first):
                p.data = q
            for p,q in zip(model.fc3.parameters(), saved_last):
                p.data = q
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
    return correct