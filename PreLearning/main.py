import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
# 导入pytorch变量中自动微分的包


x = Variable(torch.linspace(0, 100).type(torch.FloatTensor))
# 使用linspace构造0-100之间的均匀数字作为时间变量
rand = Variable(torch.randn(100)) * 10  # 噪声
y = x + rand

# 数据集
x_train = x[: -10]
x_test = x[-10:]
y_train = y[: -10]
y_test = y[-10:]

'''
# 训练数据集的可视化
plt.figure(figsize=(10, 8))     # 设定绘制窗口大小为10*8 inch
# 绘制数据， 由于x和y都是Valuable，需要用data获取他们的包裹的Tensor，并转成Numpy
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''

a = Variable(torch.rand(1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)
learning_rate = 0.0001

for i in range(1000):
    # 计算在当前a、b条件下的模型预测数值
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    loss = torch.mean((predictions - y_train)**2)   # 通过与标签数据y相比较，计算误差
    print('loss:', loss)
    loss.backward()     # 对损失函数进行梯度反转
    # 利用上一步计算中得到的a的梯度信息更新a中的data数值
    a.data.add_(- learning_rate * a.grad.data)
    # 利用上一步计算中得到的b的梯度信息更新b中的data数值
    b.data.add_(- learning_rate * b.grad.data)
    a.grad.data.zero_()     # 清空a的梯度信息
    b.grad.data.zero_()     # 清空a的梯度信息


# 绘制拟合的曲线
x_data = x_train.data.numpy()
plt.figure(figsize=(10, 7))
xplot, = plt.plot(x_data, y_train.data.numpy(), 'o')
# 绘制x, y散点图
yplot, = plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy())
# 绘制拟合曲线图
plt.xlabel('X')     # 给横坐标轴加标注
plt.ylabel('Y')     # 给纵坐标轴加标注
str1 = str(a.data.numpy()[0]) + 'x + ' + str(b.data.numpy()[0])
# 将拟合直线的参数a，b显示出来
plt.legend([xplot, yplot], ['Data', str1])  # 绘制图例
plt.show()

# 预测
predictions = a.expand_as(x_test) * x_test + b.expand_as(x_test)
# 计算模型的预测结果
print(predictions)

# 看看预测准不准
x_data = x_train.data.numpy()   # 获得x包裹的数据
x_pred = x_test.data.numpy()    # 获得包裹的测试数据的自变量
plt.figure(figsize=(10, 7))
plt.plot(x_data, y_train.data.numpy(), 'o')
# 绘制训练数据
plt.plot(x_pred, y_test.data.numpy(), 's')
# 绘制测试数据
x_data = np.r_[x_data, x_test.data.numpy()]
plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy())  # 绘制拟合数据
plt.plot(x_pred, a.data.numpy() * x_pred + b.data.numpy())  # 绘制预测数据
plt.xlabel('X')     # 给横坐标轴加标注
plt.ylabel('Y')     # 给纵坐标轴加标注
str1 = str(a.data.numpy()[0]) + 'x + ' + str(b.data.numpy()[0])
# 将拟合直线的参数a，b显示出来
plt.legend([xplot, yplot], ['Data', str1])  # 绘制图例
plt.show()







