import torch
import numpy as np


if __name__ == '__main__':
    x_tensor = torch.randn(2, 3)
    y_numpy = np.random.randn(2, 3)
    x_numpy = x_tensor.numpy()  # 将张量转化为numpy
    y_tensor = torch.from_numpy(y_numpy)    # 将numpy转化为张量
    print(torch.cuda.is_available())
    x = x_tensor.cuda()
    y = y_tensor.cuda()
    print(x + y)
    x = x.cpu()
