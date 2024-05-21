import torch.nn as nn
import torch.nn.functional as f


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        nn.ReLU(inplace=True)
        return out


class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=int(outchannel / 4), kernel_size=1, stride=stride, padding=0,
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(outchannel / 4), out_channels=int(outchannel / 4), kernel_size=3, stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(outchannel / 4), out_channels=outchannel, stride=1, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        nn.ReLU(inplace=True)
        return out


class ResNet_18(nn.Module):
    def __init__(self, block, num_class=10, ):
        super().__init__()
        self.inchannel = 64
        ###输入的图片通道数是3，卷积成63维数据
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        ###为什么第一个步长为1
        self.layer1 = self.make_layer(block, 64, 2, 1)
        self.layer2 = self.make_layer(block, 128, 2, 2)
        self.layer3 = self.make_layer(block, 256, 2, 2)
        self.layer4 = self.make_layer(block, 512, 2, 2)
        self.fc = nn.Linear(512, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        ###第一层的stride步长为2，作用？
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ResNet_34(nn.Module):
    def __init__(self, block, num_class=10):
        super().__init__()
        self.inchannel = 64
        ###输入的图片通道数是3，卷积成63维数据
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        ###为什么第一个步长为1
        self.layer1 = self.make_layer(block, 64, 3, 1)
        self.layer2 = self.make_layer(block, 128, 4, 2)
        self.layer3 = self.make_layer(block, 256, 6, 2)
        self.layer4 = self.make_layer(block, 512, 3, 2)
        self.fc = nn.Linear(512, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        ###第一层的stride步长为2，作用？
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ResNet_50(nn.Module):
    def __init__(self, block, num_class=10):
        super().__init__()
        self.inchannel = 64
        ###输入的图片通道数是3，卷积成63维数据
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        ###为什么第一个步长为1
        self.layer1 = self.make_layer(block, 64, 3, 1)
        self.layer2 = self.make_layer(block, 256, 4, 2)
        self.layer3 = self.make_layer(block, 1024, 6, 2)
        self.layer4 = self.make_layer(block, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        ###第一层的stride步长为2，作用？
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ResNet_101(nn.Module):
    def __init__(self, block, num_class=10):
        super().__init__()
        self.inchannel = 64
        ###输入的图片通道数是3，卷积成64维数据
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        ###为什么第一个步长为1，第一个不需要下采样做残差连接
        self.layer1 = self.make_layer(block, 64, 3, 1)
        self.layer2 = self.make_layer(block, 256, 4, 2)
        self.layer3 = self.make_layer(block, 1024, 23, 2)
        self.layer4 = self.make_layer(block, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        ###第一层的stride步长为2，作用:下采样,通道数翻倍
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ResNet_152(nn.Module):
    def __init__(self, block, num_class=10):
        super().__init__()
        self.inchannel = 64
        ###输入的图片通道数是3，卷积成63维数据
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block, 64, 3, 1)
        self.layer2 = self.make_layer(block, 256, 8, 2)
        self.layer3 = self.make_layer(block, 1024, 36, 2)
        self.layer4 = self.make_layer(block, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def ResNet18():
    return ResNet_18(BasicBlock)


def ResNet34():
    return ResNet_34(BasicBlock)


def ResNet50():
    return ResNet_50(Bottleneck)


def ResNet101():
    return ResNet_101(Bottleneck)


def ResNet152():
    return ResNet_152(Bottleneck)
