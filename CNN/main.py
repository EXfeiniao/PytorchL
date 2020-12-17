import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    image_size = 28     # 图像的总尺寸为28*28
    num_classes = 10    # 标签的种类数
    num_epochs = 10     # 训练的总循环周期
    batch_size = 64     # 一个批次的大小，64张图片

    # 加载MNIST 数据，如果没有就会下载，存在当前路径的/data 子目录下
    # MNIST 数据属于torchvision 包自带的的数据，可以直接调用
    # 调用自己的图像数据时，可以用torchvision.datasets.ImageFolder
    # 或torch.utils.data.TensorDataset 来加载
    train_dataset = dataset.MNIST(root='./data',  # 文件存放路径
                                  train=True,  # 提取数据集
                                  # 将图像转化为Tensor，在加载数据时，就可以对图像进行预处理
                                  transform=transforms.ToTensor(),
                                  download=True)  # 当找不到文件时，自动下载

    # 加载测试数据集
    test_dataset = dataset.MNIST(root='./data',
                                 train=False,
                                 transform=transforms.ToTensor())

    # 训练数据集的加载器，自动将数据切分成批，顺序随机打乱
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)    # 随机排列

    '''
    将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据
    校验数据用于检测模型是否过拟合并调整参数，测试数据检验整个模型的工作
    '''

    # 定义下标数组indices，它是对所有test_dataset 中数据的编码
    # 定义下标indices_val 表示校验数据集的下标，indices_test 表示测试数据的下标
    indices = range(len(test_dataset))  # indices，index的复数：索引
    indices_val = indices[:5000]    # 前5000个data 作为校验数据集
    indices_test = indices[5000:]   # 5000之后的data 作为测试数据

    # 根据下标构造两个数据集的SubsetRandomSampler 采样器，它会对下标进行采样
    '''
    会根据后面给的列表从数据集中按照下标取元素
    torch.utils.data.SubsetRandomSampler(indices)：无放回地按照给定的索引列表采样样本元素。
    '''
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    # 根据两个采样器定义加载器
    # 将sample_val和sample_test 分别赋值给了validation_loader，test_loader
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=sampler_val)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=sampler_test)
    '''
    # 随便从数据集挑选一张图片并绘制出来
    idx = 100
    
    # dataset支持下标索引，其中提取出来的元素为features, target 格式，即属性和标签。[0]表示索引features
    muteimg = train_dataset[idx][0].numpy()
    # 一般的图像包含RGB这三个通道，而MNIST数据集的图像都是灰度的，只有一个通道
    # 因此，可以忽略通道，把图像看作一个灰度矩阵
    # 用imshow 绘图，会将灰度图像自动展现为彩色，不同灰度对应不同的颜色：从黄到紫

    plt.imshow(muteimg[0, ...])
    print(' | label: ', train_dataset[idx][1])
    '''

    # 定义卷积神经网络：4和8是人为指定的两个卷积层的厚度
    depth = [4, 8]


    class ConvNet(nn.Module):
        def __init__(self):
            # 该函数在创建一个ConvNet对象的即调用语句net = ConvNet()时就会被调用
            # 首先调用父类相应的构造函数
            super(ConvNet, self).__init__()

            # 其次构造ConvNet需要用倒的各个神经模块
            # 定义组件并不是搭建组件，只是构造好组件
            # 定义一个卷积层，输入通道为1，输出通道为4， 窗口大小为5，padding为2
            self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            # 第二层卷积，输入通道为depth[0]，输出通道为depth[1]，窗口大小为5，padding为2
            self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
            # 一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层512个节点
            self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)

            self.fc2 = nn.Linear(512, num_classes)  # 最后一层线性分类单元，输入为512，输出为要做分类的分类数

        def forward(self, x):   # 该函数完成神经网络真正的前向运行，在这里把各个组件进行实际的拼装

            # x的尺寸：(batch_size, image_channels, image_width, image_height)
            x = self.conv1(x)   # 第一层卷积
            x = F.relu(x)       # 激活函数使用ReLu，防止过拟合
            # x的尺寸：(batch_size, num_filters, image_width, image_height)

            x = self.pool(x)    # 第二层池化，将图片变小
            # x的尺寸：(batch_size, depth[0], image_width/2, image_height/2)

            x = self.conv2(x)   # 第三层又是卷积，窗口为5，输入输出通道分别为depth[0]=4, depth[1]=8
            x = F.relu(x)   # 非线性函数
            # x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)

            x = self.pool(x)    # 第四层池化，将图片缩小到原来的1/4
            # x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)

            # 将特征图tensor压成一个一维的向量
            # view函数可以将一个tensor按指定的方式重新排布
            # 下面的这个命令就是让x按照batch_size * (image_size//4)^2*depth[1]的方式来排布向量
            x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
            # x的尺寸：(batch_size, depth[1]*image_width/4*image_height/4)

            x = F.relu(self.fc1(x))
            # x的尺寸：(batch_size, 512)

            # 以默认的0.5的概率对这一层进行dropout操作，防止过拟合
            '''
            drop操作：可以关闭一部分神经元，避免过拟合，增强模型的泛化能力
            '''
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)     # 全连接
            # x的尺寸：(batch_size, num_classes)

            # 输出层为log_softmax，即概率对数值log(p(x))。采用log_softmax可以使后面的交叉熵计算更快
            x = F.log_softmax(x, dim=1)
            return x

        def retrieve_features(self, x):
            # 该函数用于提取卷积神经网络的特征图，返回feature_map1, feature_map2为全两层卷积层的特征图
            feature_map1 = F.relu(self.conv1(x))    # 完成第一层卷积
            x = self.pool(feature_map1)     # 完成第一层池化
            # 第二层卷积，两层特则图都存储到了feature_map1，feature_map2中
            feature_map2 = F.relu(self.conv2(x))
            return feature_map1, feature_map2


    def rightness(predictions, labels):
        # 计算预测错误率的函数，其中predictions是模型给出的一组预测结果，
        # batch_size行num_classes列的矩阵，labels是数据之中的正确答案
        pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
        rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


    # 运行模型
    net = ConvNet()     # 新建一个卷积神经网络的实例，此时ConvNet的__init__()函数会被自动调用

    criterion = nn.CrossEntropyLoss()   # loss函数的定义，交叉熵 criterion：标准，原则
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
    # 定义优化器，普通的随机梯度下降算法 optimizer：优化器
    '''
    原本使用的是SGD
    SGD：
        基本策略可以理解为随机梯度下降像是一个盲人下山，不用每走一步计算一次梯度，但是他总能下到山底，
    只不过过程会显得扭扭曲曲。
        优点：
        虽然SGD 需要走很多步的样子，但是对梯度的要求很低（计算梯度快）。而对于引入噪声，
    大量的理论和实践工作证明，只要噪声不是特别大，SGD 都能很好地收敛。
        应用大型数据集时，训练速度很快。比如每次从百万数据样本中，取几百个数据点，
    算一个SGD 梯度，更新一下模型参数。相比于标准梯度下降法的遍历全部样本，每输入一个样本更新一次参数，要快得多。
        缺点：
        SGD 在随机选择梯度的同时会引入噪声，使得权值更新的方向不一定正确。
        此外，SGD 也没能单独克服局部最优解的问题。
    我使用了别的算法也试了下：Adagrad 在这个模型中表现得不行，速度慢而且效果不好
                         RMSprop 在这个模型中速度快效果好,
                         原本epochs 是20，用了这个后就换成了10
    RMSProp：
        借鉴了Adagrad 的思想，于取了个加权平均，避免了学习率越来越低的的问题，而且能自适应
    地调节学习率。
    '''

    record = []     # 记录准确率的数值的容器
    weights = []    # 每若干布就记录一次卷积核

    # 开始训练循环
    for epoch in range(num_epochs):

        train_rights = []   # 记录训练数据集准确率的容器

        '''
        下面的enumerate 起到构造一个枚举器的作用。在对train_loader做循环迭代时，enumerate 
        会自动输出一个数字指示循环了几次，并记录在batch_idx中，它就等于0, 1, 2, ...
        train_loader 每迭代一次，就会输出一对数据data和target，分别对应一个批中的手写数字图
        及其对应的标签
        '''
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将Tensor 转化为Valuable, data 为一批图像，target 为一批标签
            data, target = Variable(data), Variable(target)
            # 给网络模型做标记，标志着模型在训练集上上训练
            # 这种区分主要是为了打开关闭net 的training 标志，从而决定是否运行dropout
            net.train()

            output = net(data)  # 神经网络完成一次前馈的计算的过程，得到预测输出output
            loss = criterion(output, target)    # 将output 和target 进行比较，计算误差
            optimizer.zero_grad()   # 清空梯度
            loss.backward()     # 反向传播
            optimizer.step()    # 异步随机梯度下降算法
            right = rightness(output, target)   # 计算准确率所需数值，返回数值为（正确样例数，总样例数）
            train_rights.append(right)  # 将计算结果装到列表容器train_rights 中

            if batch_idx % 100 == 0:    # 每隔100个batch 执行一次打印

                net.eval()  # 给网络模型做标记，标志着模型在训练集上训练
                val_rights = []     # 记录校验数据集准确率的容器

                # 开始在校验集上做循环，计算校验集上的准确率
                for (data, target) in validation_loader:
                    data, target = Variable(data), Variable(target)
                    # 完成一次前馈计算过程，得到目前训练的模型net在校验集上的表现
                    output = net(data)
                    # 计算准确率所需数值，返回正确的数值为（正确样例数，总样例数）
                    right = rightness(output, target)
                    val_rights.append(right)

                # 分别计算目前已经计算过的测试集以及全部校验集上的模型的表现：分类准确率
                # train_r 为一个二元组，分别记录经历过的所有训练集中分类正确的数量和该集合中总的样本数
                # train_r[0]/train_r[1]是训练集的分类精确度，val_r[0]/val_r[1]是校验集的分类精确度
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                # val_r 为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                # 打印准确率等数值，其中正确率为本训练周期epoch开始后到目前批的正确率的平均值
                print('| epoch：{}[{}/{} ({:.0f}%)] | loss: {:.6f} | train acc: {:.2f}% | test acc: {:.2f}%'.format(
                           epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data,
                           100. * train_r[0] / train_r[1],
                           100. * val_r[0] / val_r[1]
                      ))

                # 将准确率和权重等数值加载到容器中，方便后续处理
                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

                # weights 记录了训练周期中所有卷积核的演化过程，net.conv1.weight提取出第一层卷积核的权重
                # clone 是将weight.data 中的数据做一个备份放到列表中
                # 否则当weight.data 变化时，列表中的每一项数值都会联动
                # 这里使用clone 这个函数
                '''
                clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
                '''
                weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                                net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])
