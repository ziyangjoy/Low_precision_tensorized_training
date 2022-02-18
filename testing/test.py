from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch_bayesian_tensor_layers
from torch_bayesian_tensor_layers.layers import TensorizedLinear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

        self.tensor_fc1 = TensorizedLinear(shape=[[7,4,7,4],[4,4,8,4]],max_rank=20)
        self.tensor_fc2 = TensorizedLinear(shape=[[16,32],[2,5]],max_rank=20)
    
    def forward(self, x):
        x = torch.flatten(x, 1)

        #x = self.fc1(x)
        x = self.tensor_fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        self.fc2(x)
        output = F.softmax(x, dim=1)
        return output



def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fun(output, target) 

        kl_loss = model.tensor_fc1.tensor.get_kl_divergence_to_prior()#+model.tensor_fc2.tensor.get_kl_divergence_to_prior()

        loss += kl_loss*torch.clamp(torch.tensor(epoch/args.warmup_epochs),0.0,1.0)*(1.0/60000)

        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    loss_fun = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fun(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("Tensor ranks layer 1")
    print(model.tensor_fc1.tensor.get_rank(threshold=1e-3))
    print([torch.square(x) for x in model.tensor_fc1.tensor.rank_parameters])
    """
    print("Tensor ranks layer 2")
    print(model.tensor_fc2.tensor.get_rank(threshold=1e-4))
    print([torch.square(x) for x in model.tensor_fc2.tensor.rank_parameters])
    """
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-kl-epochs', type=int, default=0, metavar='N',
                        help='')
    parser.add_argument('--warmup-epochs', type=int, default=50, metavar='N',
                        help='')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num-samples', type=int, default=10, metavar='N',
                        help='')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()