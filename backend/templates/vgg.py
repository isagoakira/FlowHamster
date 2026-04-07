"""
VGG - 视觉几何群网络
Reference: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
"""
import torch
import torch.nn as nn


# ============================================================================
# VGG 网络主体
# ============================================================================
class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=True, num_classes=1000):
        super().__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3

        for v in cfg:
            if v != 'M':
                # 卷积层
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)

                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))

                layers.append(nn.ReLU(inplace=True))
                in_channels = v
            else:
                # 池化层
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# VGG 配置
# ============================================================================
# 'M' 表示 MaxPool2d
VGG11_CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG13_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG19_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


# ============================================================================
# 预设模型
# ============================================================================
def vgg11(num_classes=1000):
    return VGG(VGG11_CFG, batch_norm=False, num_classes=num_classes)


def vgg11_bn(num_classes=1000):
    return VGG(VGG11_CFG, batch_norm=True, num_classes=num_classes)


def vgg13(num_classes=1000):
    return VGG(VGG13_CFG, batch_norm=False, num_classes=num_classes)


def vgg13_bn(num_classes=1000):
    return VGG(VGG13_CFG, batch_norm=True, num_classes=num_classes)


def vgg16(num_classes=1000):
    return VGG(VGG16_CFG, batch_norm=False, num_classes=num_classes)


def vgg16_bn(num_classes=1000):
    return VGG(VGG16_CFG, batch_norm=True, num_classes=num_classes)


def vgg19(num_classes=1000):
    return VGG(VGG19_CFG, batch_norm=False, num_classes=num_classes)


def vgg19_bn(num_classes=1000):
    return VGG(VGG19_CFG, batch_norm=True, num_classes=num_classes)
