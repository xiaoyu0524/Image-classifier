
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(data_set, batch_size=64)

class CIFAR10_test(nn.Module):
    def __init__(self):
        super(CIFAR10_test, self).__init__()
        self.conv1 = nn.Conv2d(3,32,5, padding=2)   #3 channel 5*5 kernel*32
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)   #32 channel 5*5 kernel*32
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2) #32 channel 5*5 kernel*64
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input= self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)
        input = self.conv3(input)
        input = self.maxpool3(input)
        input = self.flatten(input)
        input = self.linear1(input)
        input = self.linear2(input)
        output = self.softmax(input)
        return output


CIFAR10_nn = CIFAR10_test()
writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    imgs, target = data
    output = CIFAR10_nn(imgs)
    print(target)

    step += 1
    if step == 5:
        break

writer.close()


