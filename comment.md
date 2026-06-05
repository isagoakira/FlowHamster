数据增强/合成/特征工程节点已实现完毕。

## 新增节点

**Advanced Augmentation (7个)**
- mixup、cutmix、autoaugment、randaugment、cutout、posterize、solarize

**Multi-source Synthesis (3个)**
- zip_datasets、interleave_datasets、sample_from_datasets

**Feature Engineering (3个)**
- standard_scaler、minmax_scaler、pca

## 修改文件

| 文件 | 说明 |
|---|---|
| src/types/dataGraph.ts | 新增 13 个 DataNodeType |
| src/utils/dataNodeRegistry.ts | 新增 3 个 category、13 个节点定义 |
| src/components/data/DataPipelineNode.tsx | 新增 3 个 category 样式 |
| src/utils/dataWorkflowCompiler.ts | 前端代码生成逻辑 |
| backend/services/dataflow_compiler.py | 后端代码生成逻辑 |
| src/utils/dataNodeRegistry.test.ts | 新增注册表一致性测试 |
| backend/tests/test_dataflow_compiler.py | 新增代码生成黄金样例测试 |

## 代码生成要点
- MixUp / CutMix：在 __getitem__ 内通过 self[idx2] 获取第二样本并做 lambda 混合，label 同步混合
- AutoAugment / RandAugment：PIL 转换后转回 tensor（与现有 torchvision 兼容）
- Posterize / Solarize：使用 torchvision.transforms.functional 直接操作 uint8 tensor
- CutOut：tensor 级矩形置零
- 多源合成：生成注释级 scaffold（拼接/交替/加权采样在 Dataset 构造层处理）
- 特征工程：StandardScaler / MinMaxScaler / PCA 在 __init__ 预留统计量占位，在 __getitem__ 应用变换

## 测试
- npx tsc --noEmit 通过
- npm run test:run：90 passed, 2 skipped
- python -m pytest backend/tests/：21 passed

## 影响说明
- 不涉及后端 API 路由变更
- 不涉及模型图 nodeComponentRegistry（数据图使用通用 DataPipelineNode）
- 代码生成输出为可运行 scaffold，MixUp/CutMix 的自引用混合在真实场景需替换为预计算 second-sample

## 备注
issue 原文中节点名称列表为空（markdown 渲染丢失了 backtick 内容），本次实现按常见 ML 数据增强/合成/特征工程节点推断。如预期节点不同请补充说明。
