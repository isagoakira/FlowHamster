重构增强已完成，已根据测试反馈补全所有缺口。

## 本次修复内容

### 1. 后端代码生成补全（8个已有增强节点）
`backend/services/dataflow_compiler.py` 现已支持全部前端已有的 augmentation 节点：
- `random_horizontal_flip`、`random_vertical_flip`
- `random_crop`、`random_rotation`
- `color_jitter`、`random_erasing`
- `gaussian_blur`、`grayscale`

同时更新了 `import torchvision` 的触发条件，确保任意图像增强节点存在时都会导入 torchvision。

### 2. 新增 Feature Engineering 节点（3个）
补全了 tester 反馈中提到的缺失节点：
- `normalize_features` — 对 tabular features 做 mean/std 归一化
- `fill_missing_values` — 支持 mean/median/constant 策略填充 NaN/Inf
- `one_hot_encode` — one-hot 编码 scaffold

### 3. 前端代码生成补全
`src/utils/dataWorkflowCompiler.ts` 补充了此前缺失的：
- `random_erasing` 的 `__getitem__` 代码生成
- `gaussian_blur` 的 `__getitem__` 代码生成
- 新增 3 个 feature engineering 节点的代码生成

### 4. 测试覆盖
新增/更新了以下测试：

**前端 vitest**
- `src/utils/dataNodeRegistry.test.ts`：验证全部 augmentation 节点（8+7 个）、多源合成（3 个）、特征工程（6 个）均存在于 registry，且全局类型唯一

**后端 pytest**
- `test_existing_augmentation_nodes_generate_code`：遍历全部 8 个已有增强节点，断言 scaffold 中无 `Unknown node type`
- `test_new_feature_engineering_nodes_generate_code`：验证 normalize_features / fill_missing_values / one_hot_encode 生成正确
- `test_mixup_cutmix_label_blending`：专项验证 label 混合逻辑存在
- 保留原有 advanced_augmentation / multi_source / feature_engineering 测试

## 测试结果
- `npx tsc --noEmit`：通过
- `npm run test:run`：91 passed, 2 skipped
- `python -m pytest backend/tests/test_dataflow_compiler.py`：8 passed
- `python -m pytest backend/tests/test_acceptance_data_augmentation.py`：5 passed
- `python -m pytest backend/tests/`：31 passed（6 个失败为 ISA-207/208 前置 P0 bug，与本次改动无关）

## 修改文件汇总
- `src/types/dataGraph.ts`
- `src/utils/dataNodeRegistry.ts`
- `src/components/data/DataPipelineNode.tsx`
- `src/utils/dataWorkflowCompiler.ts`
- `backend/services/dataflow_compiler.py`
- `src/utils/dataNodeRegistry.test.ts`
- `backend/tests/test_dataflow_compiler.py`
