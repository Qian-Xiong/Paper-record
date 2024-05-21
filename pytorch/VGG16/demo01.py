##在MNISET数据集上训练VGG16
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torch.utils.data as Data
import torch.optim as optim
import time
from torchvision import datasets, transforms

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.226, 0.224, 0.225))])
##读取数据
train_data = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

##定义数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
)

# 准备测试数据集
test_data = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=False
)


# # 搭建VGG16模型

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(16, 64, 3, 2, 1),  ##反卷积防止卷到最后没有像素
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


##训练
epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VGG16().to(device)
##损失函数
criterion = nn.CrossEntropyLoss()
##优化器
optimizer = optim.Adam(model.parameters(), 0.001)

# optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.8,weight_decay=0.001)
# schedule=optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)


loss_list = []

print("start train ........")
for epoch in range(epochs):
    start_time = time.time()
    running_loss = 0.0
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_list.append(loss.item())
    print(f"epoch {epoch + 1} loss:{running_loss / len(train_loader)}")
    end_time = time.time()
    print(f"time:{end_time - start_time}")

model.eval()
correct = 0.0
total = 0
with torch.no_grad():
    print("*" * 30)
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pre = outputs.argmax(dim=1)
        total += inputs.size(0)
        correct += torch.eq(pre, labels).sum().item()
print(f"acc correct{(correct / total) * 100}%")
