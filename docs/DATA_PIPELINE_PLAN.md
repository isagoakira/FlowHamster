# Data Pipeline Node System - Project Plan

## Overview
创建独立于模型图的数据加载节点图系统，支持从文件读取、数据整理、预处理到输出给网络的完整流程。

---

## ✅ Phase 1: 扩展现有节点类型 (已完成)

### 1.1 数据源节点 (Data Sources) ✅
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `folder_source` | 从文件夹读取 | `path`, `pattern` | ✅ |
| `csv_source` | 读取 CSV 文件 | `path`, `delimiter` | ✅ |
| `jsonl_source` | 读取 JSONL 文件 | `path` | ✅ |
| `parquet_source` | 读取 Parquet 文件 | `path` | ✅ 已添加 |
| `huggingface_source` | 从 HuggingFace 数据集 | `dataset_name`, `split`, `config` | ✅ 已添加 |

### 1.2 读取器节点 (Readers) ✅
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `read_image` | 解码图像为张量 | `mode` (RGB/L/RGBA) | ✅ |
| `read_text` | 读取文本文件 | `encoding` | ✅ 已添加 |
| `unpack` | 解包嵌套字段 | `key` | ✅ |

---

## ✅ Phase 2: 预处理变换节点 (已完成)

### 2.1 基础变换 ✅
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `resize` | 调整图像大小 | `size` (W,H) | ✅ 已添加 |
| `crop` | 裁剪图像 | `x`, `y`, `width`, `height`, `type` | ✅ 已添加 |
| `flip` | 翻转图像 | `horizontal`, `vertical` | ✅ 已添加 |

### 2.2 数值变换 ✅
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `normalize` | 归一化张量 | `mean`, `std` | ✅ 已添加 |
| `to_tensor` | 转换 PIL/numpy 为张量 | - | ✅ 已添加 |
| `scale` | 缩放值 | `min`, `max` | ✅ 已添加 |

---

## ✅ Phase 3: 数据增强节点 (已完成)

### 3.1 图像增强 ✅
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `random_horizontal_flip` | 随机水平翻转 | `p` | ✅ 已添加 |
| `random_vertical_flip` | 随机垂直翻转 | `p` | ✅ 已添加 |
| `random_crop` | 随机裁剪 | `size`, `padding` | ✅ 已添加 |
| `random_rotation` | 随机旋转 | `degrees` | ✅ 已添加 |
| `color_jitter` | 颜色抖动 | `brightness`, `contrast`, `saturation`, `hue` | ✅ 已添加 |
| `grayscale` | 转灰度图 | - | ✅ 已添加 |

---

## ✅ Phase 4: 数据组织节点 (已完成)

| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `train_val_split` | 训练/验证划分 | `train_ratio`, `val_ratio`, `seed` | ✅ |
| `shuffle` | 打乱数据 | `enabled`, `seed` | ✅ |
| `repeat` | 重复数据集 | `times` | ✅ 已添加 |
| `concat` | 拼接多个数据源 | - | ✅ 已添加 |

---

## ✅ Phase 5: 批处理节点 (已完成)

| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `batch` | 批处理 | `batch_size`, `drop_last` | ✅ |
| `collate` | 自定义批处理函数 | `strategy` | ✅ |
| `dataloader` | PyTorch DataLoader | `batch_size`, `shuffle`, `num_workers`, `pin_memory` | ✅ 已添加 |

---

## ✅ Phase 6: NLP 专用节点 (部分完成)

### 已完成
| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `tokenizer` | 文本分词 | `tokenizer_type`, `max_length`, `padding` | ✅ 已添加 |
| `truncate` | 截断序列 | `max_length` | ✅ 已添加 |

### 待完成
- `add_special_tokens` - 添加特殊token
- `random_mask` - 随机掩码 (MLM)

---

## ✅ Phase 7: 输出节点 (已完成)

| 节点类型 | 说明 | 参数 | 状态 |
|----------|------|------|------|
| `dataset_output` | 声明输出字段用于绑定 | `fields` | ✅ |
| `cache` | 缓存处理后的数据 | `cache_dir` | ✅ 已添加 |

---

## ✅ Phase 8: 后端数据处理服务 (已完成)

### 已完成
- `backend/modules/datasets/data_pipeline.py` - 数据集类和数据处理核心
  - `ImageFolderDataset`
  - `CSVDataset`
  - `JSONLDataset`
  - `ParquetDataset`
  - `ComposeTransforms`
  - `RandomHorizontalFlip`, `RandomVerticalFlip`
  - `Resize`, `Normalize`, `ToTensor`
  - `create_dataloader`

---

## ✅ Phase 9: 前端 UI/UX (已完成)

- 增强 `DataPipelineNode.tsx` 支持分类颜色
- `DataSidebar.tsx` 更新使用新的节点注册表
- 所有节点使用统一的通用组件 `DataPipelineNode`

---

## Phase 10: 代码生成集成 (进行中)

- ✅ 扩展 `dataWorkflowCompiler.ts` 支持所有新节点类型
- ✅ 更新参数校验器
- ✅ 支持 HuggingFace 数据源

---

## 待完成 (Low Priority)

### Phase 6 补充
- `add_special_tokens` - 添加特殊token
- `random_mask` - 随机掩码 (MLM)

### Phase 3 补充
- `mixup` - MixUp 混合
- `cutmix` - CutMix 混合
- `random_erasing` - 随机擦除
- `gaussian_blur` - 高斯模糊

### Phase 4 补充
- `filter` - 按条件过滤样本
- `sampler` - 自定义采样器
- `weighted_sampler` - 加权采样

### 高级功能
- 视频/音频读取器 (`read_video`, `read_audio`)
- GPU 加速的变换
- 数据缓存优化
- 分布式训练支持

---

## 已创建/修改的文件

### 后端
- `backend/modules/__init__.py` - 模块初始化
- `backend/modules/mamba.py` - Mamba-2 实现
- `backend/modules/datasets/__init__.py` - 数据集模块
- `backend/modules/datasets/data_pipeline.py` - 数据处理核心

### 前端
- `src/types/dataGraph.ts` - 扩展节点类型定义
- `src/utils/dataNodeRegistry.ts` - 完整的节点注册表
- `src/utils/dataWorkflowCompiler.ts` - 代码生成器
- `src/components/data/DataSidebar.tsx` - 侧边栏更新
- `src/components/data/DataPipelineNode.tsx` - 增强的节点组件

### 文档
- `docs/DATA_PIPELINE_PLAN.md` - 项目计划

---

## 下一步行动

1. **立即可用**: 所有 P0/P1 功能已完成
2. **可选增强**: 添加 mixup/cutmix 等高级增强
3. **长期**: GPU 加速、分布式训练支持
