from __future__ import print_function
import argparse
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import train,test,get_net
import time
from time import gmtime, strftime


from qtorch.optim import OptimLP
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument(
        '--model-type',
        default='full',
        choices=['CP', 'TensorTrain', 'TensorTrainMatrix','Tucker','full'],
    type=str)
    parser.add_argument('--lp', type=bool, default=True)
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--bit', type=int, default=4)

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--rank-loss', type=bool, default=False)
    parser.add_argument('--kl-multiplier', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
    parser.add_argument('--em-stepsize', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
    parser.add_argument('--no-kl-epochs', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--prior-type', type=str, default='log_uniform')
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--tensorized', type=bool, default=False,
                        help='Run the tensorized model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}


    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform_train)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = get_net(args).to(device)

    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lp:
        weight_quant = lambda x : float_quantize(x, exp=8, man=23, rounding="stochastic")
        # weight_quant = lambda x : fixed_point_quantize(x, wl=8, fl=6, rounding="stochastic")
        # weight_quant = lambda x : fixed_point_quantize(x, wl=8, fl=6, rounding="nearest")
        

        gradient_quant = lambda x : float_quantize(x, exp=8, man=23, rounding="nearest")
        momentum_quant = lambda x : float_quantize(x, exp=8, man=23, rounding="nearest")

        
        # optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.005)
        # optimizer = OptimLP(optimizer, 
        #                 weight_quant=None)

    current_time = strftime("%Y_%m_%d_%H_%M", gmtime())

    path = './saved_models/cifar10_vgg16_tensor'
    if args.lp:
        path = path + '_LP'
        # path = './saved_models/' + 'cifar10_vgg19_tensor_LP_' + 'rank' +str(args.rank) + '_' + current_time + '.pt'
    if args.scale:
        path = path + '_scale_int' + str(args.bit)
    path = path + '_rank' +str(args.rank) + '_' + current_time + '.pt'


    max_acc = -1
    args.save_model = False
    for epoch in range(1, args.epochs + 1):
        

        t = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print("Epoch train time {:.2f}".format(time.time()-t))

        

        acc = test(model, device, test_loader)
        if acc>max_acc and args.save_model:
            torch.save(model.state_dict(), path)
            print('current model saved')
            max_acc = acc

        # print(model.fc1.tensor.factors[0])
        print(model.sc1.scale)
        # print(model.sc2.scale)
        print(model.fc1.scale_w)
        print(model.fc2.scale_w)

        # print(model.fc1.Q_factors[0])

        # print(model.fc1.tensor.factor_distributions[0].mean)
        # print(model.features[0])
        # print(model.fc1.tensor.factors[0])
        # kkk = 0
        # for p in model.fc1.tensor.trainable_variables:
        #     # print(p.data.shape)
        #     if kkk == 0 or kkk == 4:
        #         print(p.data.shape)
        #         print(p.data)
        #     kkk += 1




        # kkk = 0
        # for p in model.fc2.tensor.factors:
        #     kkk += 1
        #     if p.requires_grad and kkk>-1:
        #         print(p.name, p.data.shape)
        #         # print(p.data)
        #         kkk += 1


        if args.model_type == 'full':
            pass
        else:
            print("******Tensor Ranks*******")
            print(model.fc1.tensor.estimate_rank())
            print(model.fc2.tensor.estimate_rank())
            # print("******Param Savings*******")
            # param_savings_1 = model.fc1.tensor.get_parameter_savings(threshold=1e-4)
            # param_savings_2 = model.fc2.tensor.get_parameter_savings(threshold=1e-4)
            # full_params = 784*512+512+512*10+10
            # print(param_savings_1,param_savings_2)
            # total_savings = sum(param_savings_1)+sum(param_savings_2)
            # print("Savings {} ratio {}".format(total_savings,full_params/(full_params-total_savings)))

            print("******End epoch stats*******")


if __name__ == '__main__':
    main()
