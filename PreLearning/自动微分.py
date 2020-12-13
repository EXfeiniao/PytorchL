import torch
from torch.autograd import Variable
# 导入pytorch变量中自动微分的包


x = Variable(torch.ones(2, 2), requires_grad=True)
# 自动微分变量，requires_grad=True是为了保证它可以在反向传播算法中获得梯度信息
y = x + 2
print(y)
z = y * y
print(z)
t = torch.mean(z)
print(t)
