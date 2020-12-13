import torch
from torch.autograd import Variable
# 导入pytorch变量中自动微分的包

s = Variable(torch.FloatTensor([[0.01, 0.02]]), requires_grad=True)
x = Variable(torch.ones(2, 2), requires_grad=True)
for i in range(10):
    s = s.mm(x)
z = torch.mean(s)
print(z)
z.backward()
print(x.grad)
print(s.grad)   # 会有警告，只有叶节点才有梯度信息
