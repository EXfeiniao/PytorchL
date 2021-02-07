import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

data_dir = './data/DogsVSCats'
'''
data中的文件被分成了两个：train和valid
train中dog和cat各有10000张
valid中dog和cat各有2500张
'''

# 对数据进行的处理
# Compose: 将多个transform组合起来使用
data_transform = {x: transforms.Compose([transforms.Resize([64, 64]),
                                         transforms.ToTensor()])
                  for x in ["train", "valid"]}
# 数据载入
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "valid"]}
# 数据装载
dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=16,
                                             shuffle=True)  # 随机排列
              for x in ["train", "valid"]}

# 获取一个批次
X_example, y_example = next(iter(dataloader["train"]))
# print(u'X_example个数{}'.format(len(X_example)))
# print(u'y_example个数{}'.format(len(y_example)))
# print(X_example.shape)
# print(y_example.shape)

# 验证独热编码的对应关系
index_classes = image_datasets["train"].class_to_idx
print(index_classes)

example_classes = image_datasets["train"].classes
# print(example_classes)

img = torchvision.utils.make_grid(X_example)
img = img.numpy().transpose([1, 2, 0])

'''
# 图片预览
for i in range(len(y_example)):
    index = y_example[i]
    print(example_classes[index], end='    ')
    if (i+1) % 8 == 0:
        print()
plt.imshow(img)
plt.show()
'''


class Models(torch.nn.Module):

    def __init__(self):
        super(Models, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 参数：channels, output, kernel_size,
            # stride:步长，控制cross-correlation的步长,
            # padding(补0)：控制zero-padding的数目
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 2)
        )

    def forward(self, input):
        # 卷积层
        x = self.Conv(input)
        # -1 表示不确定要几行， 列数为4*4*512
        x = x.view(-1, 4 * 4 * 512)
        # NN分类
        x = self.Classes(x)
        return x


model = Models()
# 查看能不能用GPU
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

epoch_n = 10

for epoch in range(epoch_n):
    print("\nEpoch {}/{}".format(epoch + 1, epoch_n))
    print("-" * 10)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            # 设置为True，会进行Dropout并使用batch mean和batch var
            model.train(True)
        else:
            print("Validing...")
            # 设置为False，不会进行Dropout并使用running mean和running var
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        # enuerate(),返回的是索引和元素值，数字1表明设置start=1，即索引值从1开始
        for batch, data in enumerate(dataloader[phase], 1):
            # X: 图片，16*3*64*64; y: 标签，16
            X, y = data

            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            # y_pred: 预测概率矩阵，16*2
            y_pred = model(X)

            # pred，概率较大值对应的索引值，可看做预测结果
            _, pred = torch.max(y_pred.data, 1)

            # 梯度归零
            optimizer.zero_grad()

            # 计算损失
            loss = loss_f(y_pred, y)

            # 若是在进行模型训练，则需要进行后向传播及梯度更新
            if phase == "train":
                loss.backward()
                optimizer.step()

            # 计算损失和
            running_loss += float(loss)

            # 统计预测正确的图片数
            running_corrects += torch.sum(pred == y.data)

            # 共20000张测试图片，1250个batch，在使用500个及1000个batch对模型进行训练之后，输出训练结果
            if batch % 500 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4F}%".format(batch, running_loss / batch,
                                                                              100 * running_corrects / (16 * batch)))

        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])

        # 输出最终的结果
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))
