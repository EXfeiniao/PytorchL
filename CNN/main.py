import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    image_size = 28
    num_classes = 10
    num_epochs = 20
    batch_size = 64

    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    # 加载测试数据集
    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # 训练数据集的加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    indices = range(len(test_dataset))
    indices_val = indices[:5000]
    indices_test = indices[5000:]

    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=sampler_val)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=sampler_test)
    '''
    显示图像：
    idx = 100

    muteimg = train_dataset[idx][0].numpy()

    plt.imshow(muteimg[0, ...])
    print(' | label: ', train_dataset[idx][1])
    '''

    depth = [4, 8]


    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
            self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)

            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)

            x = self.pool(x)

            x = self.conv2(x)
            x = F.relu(x)

            self.pool(x)

            x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])

            x = F.relu(self.fc1(x))

            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            return x

        def retrieve_features(self, x):
            feature_map1 = F.relu(self.conv1(x))
            x = self.pool(feature_map1)

            feature_map2 = F.relu(self.conv2(x))
            return feature_map1, feature_map2


    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    







