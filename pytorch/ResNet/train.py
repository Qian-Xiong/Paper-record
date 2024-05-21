import torch

from model import ResNet18
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import time

'''
使用cifer-10数据集
'''

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomGrayscale(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = datasets.CIFAR10(
    root='../data/cifer-10',
    download=False,
    train=True,
    transform=transform
)
trainloader = data.DataLoader(
    trainset,
    batch_size=100,
    shuffle=True
)

testset = datasets.CIFAR10(
    root='../data/cifer-10',
    download=False,
    train=False,
    transform=transform
)
testloader = data.DataLoader(
    testset,
    batch_size=5,
    shuffle=True
)

##训练

epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
##损失函数
loss_function = nn.CrossEntropyLoss()
##优化器
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

print(f"train start ......")
for epoch in range(epochs):
    print(f"epoch {epoch} :")
    start_time = time.time()
    batch_loss = 0.0
    for batch, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels).to(device)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    end_time = time.time()
    print(f"time: {end_time - start_time} ")
    # print(f"当前学习率:{scheduler.get_last_lr()}")
    # scheduler.step()
    print(f"loss: {batch_loss / (len(trainloader) / 100):.2f} ")

print(f"test start ......")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred = outputs.argmax(dim=-1)
        correct += torch.eq(pred, labels).sum().item()
        total += inputs.size(0)
    print(f"test acc:{correct / total:.2f}")
