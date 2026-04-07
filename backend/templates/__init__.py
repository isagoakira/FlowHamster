"""
FlowHamster Module Templates

This package contains pretrained module templates that can be used
in FlowHamster workflows.

Available templates:
- resnet: ResNet (18/34/50/101)
- vgg: VGG (11/13/16/19 with batch norm variants)
- mobilenet: MobileNetV2
- lenet: LeNet-5 for MNIST
- vit: Vision Transformer (Tiny/Small/Base/Large)
"""

from .resnet import resnet18, resnet34, resnet50, resnet101
from .vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from .mobilenet import mobilenet_v2
from .lenet import lenet5
from .vit import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224

__all__ = [
    # ResNet
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    # VGG
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    # MobileNet
    'mobilenet_v2',
    # LeNet
    'lenet5',
    # ViT
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224',
]
