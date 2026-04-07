"""
MobileNetV2 - 轻量级移动网络
Reference: "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018)
"""
import torch
import torch.nn as nn


# ============================================================================
# 基础卷积块 (Conv + BN + ReLU6)
# ============================================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes,
                kernel_size=kernel_size, stride=stride, padding=padding,
                groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


# ============================================================================
# 倒残差块 (Inverted Residual Block)
# ============================================================================
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and inp == oup)

        # hidden dimension
        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # 逐点卷积 - 升维
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        # 深度可分离卷积
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))

        # 投影回低维 (linear bottleneck, 无激活函数)
        layers.append(
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ============================================================================
# MobileNetV2 主网络
# ============================================================================
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()

        # 初始配置
        input_channel = 32
        last_channel = 1280

        # 反向残差块配置: [expand_ratio, 输出通道, 重复次数, 步长]
        inverted_residual_setting = [
            [1, 16, 1, 1],   # stage 1
            [6, 24, 2, 2],   # stage 2
            [6, 32, 3, 2],   # stage 3
            [6, 64, 4, 2],   # stage 4
            [6, 96, 3, 1],   # stage 5
            [6, 160, 3, 2],  # stage 6
            [6, 320, 1, 1],  # stage 7
        ]

        # 初始卷积层
        input_channel = self._round_channels(input_channel, width_mult)
        self.last_channel = self._round_channels(last_channel, width_mult)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        # 构建倒残差块
        for expand_ratio, out_channels, num_blocks, stride in inverted_residual_setting:
            output_channel = self._round_channels(out_channels, width_mult)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, s, expand_ratio
                    )
                )
                input_channel = output_channel

        # 最终卷积层
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def _round_channels(self, channels, width_mult):
        return int(round(channels * width_mult))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(num_classes=1000):
    return MobileNetV2(num_classes=num_classes)
