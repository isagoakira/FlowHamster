"""
LeNet-5 - 手写数字识别网络
Reference: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
"""
import torch
import torch.nn as nn


class LeNet(nn.Module):
    """
    LeNet-5 架构

    结构：
        Conv1 (6 filters, 5x5) -> ReLU -> MaxPool
        Conv2 (16 filters, 5x5) -> ReLU -> MaxPool
        FC (120) -> ReLU
        FC (84) -> ReLU
        FC (10)
    """

    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 自适应池化 - 处理任意尺寸输入
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 卷积块 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 卷积块 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 自适应池化
        x = self.adaptive_pool(x)

        # 展平
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        return x


def lenet5(num_classes=10, input_channels=1):
    """LeNet-5 工厂函数"""
    return LeNet(num_classes=num_classes, input_channels=input_channels)
