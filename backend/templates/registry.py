"""
FlowHamster 模板注册表

从 backend/templates/ 目录加载所有模板定义
"""

import os
import importlib
import inspect
from typing import Dict, List, Any, Optional
from pathlib import Path

# 模板目录
TEMPLATES_DIR = Path(__file__).parent

# 模板缓存
_template_cache: Dict[str, Dict[str, Any]] = {}

# 模板定义装饰器
def template(name: str, description: str = "", category: str = "generic", emoji: str = "📦"):
    """模板定义装饰器"""
    def decorator(cls):
        cls._template_name = name
        cls._template_description = description
        cls._template_category = category
        cls._template_emoji = emoji
        return cls
    return decorator


def get_all_template_ids() -> List[str]:
    """获取所有模板 ID"""
    templates = []
    for f in TEMPLATES_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        templates.append(f.stem)
    return templates


def load_template(template_id: str) -> Optional[Dict[str, Any]]:
    """加载指定模板"""
    if template_id in _template_cache:
        return _template_cache[template_id]

    template_file = TEMPLATES_DIR / f"{template_id}.py"
    if not template_file.exists():
        return None

    try:
        # 动态导入模块
        spec = importlib.util.spec_from_file_location(template_id, template_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 提取模板元数据
            template_info = {
                "id": template_id,
                "name": getattr(module, "_template_name", template_id),
                "description": getattr(module, "_template_description", ""),
                "category": getattr(module, "_template_category", "generic"),
                "emoji": getattr(module, "_template_emoji", "📦"),
                "version": getattr(module, "_template_version", "1.0.0"),
                "graph": getattr(module, "_template_graph", {"nodes": [], "edges": []}),
                "metadata": {
                    "author": getattr(module, "_template_author", "FlowHamster"),
                    "created_at": getattr(module, "_template_created", ""),
                    "updated_at": getattr(module, "_template_updated", ""),
                }
            }

            _template_cache[template_id] = template_info
            return template_info

    except Exception as e:
        print(f"Error loading template {template_id}: {e}")

    return None


def load_all_templates() -> List[Dict[str, Any]]:
    """加载所有模板"""
    templates = []
    for template_id in get_all_template_ids():
        template = load_template(template_id)
        if template:
            templates.append(template)
    return templates


def get_template_code(template_id: str) -> Optional[str]:
    """获取模板的 Python 代码"""
    template_file = TEMPLATES_DIR / f"{template_id}.py"
    if not template_file.exists():
        return None

    with open(template_file, "r", encoding="utf-8") as f:
        return f.read()


# 预设模板元数据（与前端共享）
PRESET_TEMPLATES = {
    "resnet": {
        "id": "resnet",
        "name": "ResNet",
        "description": "18层残差网络，含4个残差阶段，适合图像分类",
        "category": "cv",
        "emoji": "🔬",
        "version": "1.0.0",
    },
    "vgg16": {
        "id": "vgg16",
        "name": "VGG-16",
        "description": "VGG-16 网络，经典图像分类模型",
        "category": "cv",
        "emoji": "🔶",
        "version": "1.0.0",
    },
    "lenet": {
        "id": "lenet",
        "name": "LeNet-5",
        "description": "LeNet-5 手写数字识别网络",
        "category": "cv",
        "emoji": "🔢",
        "version": "1.0.0",
    },
    "mobilenetv2": {
        "id": "mobilenetv2",
        "name": "MobileNetV2",
        "description": "倒残差结构+深度可分离卷积，适合移动端部署",
        "category": "cv",
        "emoji": "📱",
        "version": "1.0.0",
    },
    "vit": {
        "id": "vit",
        "name": "Vision Transformer",
        "description": "视觉Transformer，Patch Embed + Transformer Encoder",
        "category": "cv",
        "emoji": "🔮",
        "version": "1.0.0",
    },
}
