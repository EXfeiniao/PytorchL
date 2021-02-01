import numpy as np
import pandas as pd     # 读取csv文件的库
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


# 从硬盘中读取要导入的数据
data_path = 'hour.csv'  # 读取数据到内存，rides为一个dataframe 对象
rides = pd.read_csv(data_path)
# rides.head()    # 输出部分数据
counts = rides['cnt'][:50]  # 截取数据，前面50个数据
x = np.arange(len(counts))  # 获取变量x
y = np.array(counts)        # 单车数量为y


# 输入变量，1，2，3，...这样的一维数组
x = torch.tensor(np.arange(len(counts), dtype=float) / len(counts), requires_grad=True)
# 输出变量，它是从数据counts中读取的每一时刻的单车数，共50个数据点的一维数组，作为标准答案
y = torch.tensor(np.array(counts, dtype=float), requires_grad=True)

sz = 10     # 设置第一层hidden layers神经元数量
# 初始化所有神经网络的权重（weights）和阈值（biases）
weights = torch.randn((1, sz), dtype=torch.double, requires_grad=True)
# 1*10的输入到隐含层的权重矩阵
biases = torch.randn(sz, dtype=torch.double, requires_grad=True)
# 尺度为10的隐含层节点偏置向量
weights2 = torch.randn((sz, 1), dtype=torch.double, requires_grad=True)
# 10*1的隐含到输出

lr = 0.001  # 设置学习率
losses = []     # 该数组记录每一次迭代的损失函数函数值，方便以后绘图

# 将 x 转换为(50,1)的维度，以便与维度为(1,10)的weights矩阵相乘
x = x.view(50, -1)
# 将 y 转换为(50,1)的维度
y = y.view(50, -1)

for i in range(500000):
    # 从输入层到隐含层的计算
    hidden = x * weights + biases
    # 将sigmoid函数作用在hidden layers的每个神经元上
    hidden = torch.sigmoid(hidden)
    # print(hidden.size())
    # 隐含层输出到输出层，计算得到最终预测
    predictions = hidden.mm(weights2)
    # print(predictions.size())
    # 通过与标签数据y比较，计算均方误差
    loss = torch.mean((predictions - y) ** 2)
    # print(loss.size())
    losses.append(loss.data.numpy())

    if i % 10000 == 0:  # 每隔100000个周期打印一下损失函数数值
        print(' | epoch:{:d} | loss:{:.4f}'.format(int(i/10000), loss.data.numpy()))

    # 接下来开始梯度下降算法，将误差反向传播
    loss.backward()     # 对损失函数进行梯度反转

    # 利用上一步的计算得到的weights，biases等梯度信息更新weights，biases
    weights.data.add_(- lr * weights.grad.data)
    biases.data.add_(- lr * biases.grad.data)
    weights2.data.add_(- lr * weights2.grad.data)

    # 清空梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()

# 将预测和实际的点绘制出来
x_data = x.data.numpy()
plt.figure(figsize=(10, 7))
xplot, = plt.plot(x_data, y.data.numpy(), 'o')  # 绘制原始数据
yplot, = plt.plot(x_data, predictions.data.numpy())   # 绘制预测数据
plt.xlabel('X')
plt.ylabel('Y')
plt.legend([xplot, yplot], ['data', 'predictdions under 1000000 epochs'])
plt.show()
