# 节点图到训练全流程排除计划

本文档用于持续验证 FlowHamster 的“节点图 -> 代码生成 -> DataLoader -> 训练执行”链路。目标不是只证明 UI 图看起来正确，而是逐段排除映射、连接、运行时和鲁棒性风险。

## 当前已补齐并覆盖的环节

- DataLoader 节点现在会参与前端与后端的数据流编译，生成 `DATALOADER_CONFIG`、`FlowHamsterDataset`、`build_flowhamster_dataset()`、`build_flowhamster_dataloader()` 与 `build_demo_batch(device)`。
- 数据图字段会通过 `inputBindings` 映射到模型输入；缺失字段会回退到可运行的 fallback tensor，数值标量也会转换为 device 上的 tensor。
- 训练代码生成已避免重复计算 `loss = loss_fn(...)`，并保留 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()` 的闭环。
- 后端 `/api/execute/execute` 路由已修正为当前 FastAPI include 前缀下的真实路径，并使用当前解释器执行临时代码。
- 已新增端到端倾向的单元测试，覆盖前端数据图编译、前端训练代码生成、后端数据流编译与后端执行路由。

## 逐项排除计划

### 1. 模型节点图到 Python 模块映射

目标：排除“React Flow 图正确，但生成的 `__init__` / `forward` 语义错误”。

测试场景：
- 单输入线性链：`Input -> Conv/Linear -> Activation -> Output`。
- 多分支合流：并行层输出接 `Concat/Add`，检查 forward 顺序和变量名。
- 参数边界：缺省参数、0 值参数、布尔参数、字符串枚举参数。
- 命名冲突：重复节点类型、重复自定义名称、包内外同名节点。

排除步骤：
- 若生成代码缺层，先查 node registry 是否暴露该类型，再查 AST builder 是否解析该节点。
- 若生成顺序错，先查边拓扑排序和输入依赖收集。
- 若参数错，先查节点 data schema，再查代码 emitter 的参数序列化。
- 若运行时报未定义变量，先查 forward 变量命名和输出端口映射。

### 2. 封装节点与复合模块映射

目标：排除“封装显示正常，但复用、嵌套或解包后代码不稳定”。

测试场景：
- 基础封装：选中连续链路后打包，生成一个复合模块类。
- 嵌套封装：包内再次封装，检查内部 `internalStructure` 与 class id。
- 复用封装：同一个自定义复合模块实例化多次，检查类定义只生成一次、实例名唯一。
- 封包解包：package -> expand -> collapse -> unpackage 后，原节点、边和端口恢复一致。
- 包边界连接：外部输入接包输入、包输出接外部节点、多输入多输出端口。

排除步骤：
- 若类定义重复，查 custom composite registry 的 class id 去重。
- 若复用实例串线，查 instance id 与 custom class id 是否混用。
- 若解包丢边，查边界 edge 还原逻辑和端口映射表。
- 若嵌套包展开异常，先固定最小二层结构，再逐层检查 `internalStructure`。

### 3. DataLoader 数据图连接流程

目标：排除“数据图节点可画，但无法编译成可训练 batch”。

测试场景：
- `csv_source -> dataloader -> dataset_output` 基础链路。
- `folder_source -> transforms -> dataloader -> dataset_output` 图像链路。
- 多字段输出：`image`、`target`、`metadata` 同时输出。
- 空数据/不可访问路径：检查是否降级到 fallback batch，并给出 warning。
- DataLoader 参数边界：`batch_size=1`、`num_workers=0`、`shuffle=false`、`drop_last=true`、`pin_memory=true`。

排除步骤：
- 若 DataLoader 配置未出现，查 data node compiler 是否识别 `dataloader` 节点。
- 若 Python 布尔值错误，查 TS/Python 参数序列化。
- 若字段未绑定，查 dataset output 的 field metadata 与 `inputBindings`。
- 若真实数据源失败，先确认 source 节点路径，再确认 `build_demo_batch` 是否回退。

### 4. 数据图字段到模型输入绑定

目标：排除“DataLoader 能产出 batch，但模型输入拿不到正确字段”。

测试场景：
- 单输入：`model.input` 绑定到 `batch["image"]`。
- 多输入：模型 forward 多参数分别绑定到 `image`、`condition`、`lengths`。
- 目标字段：loss 使用 `target` 字段。
- 缺失绑定：应生成 fallback 输入，而不是未定义变量。
- 标量绑定：数值 label 或权重应被转换为 tensor。

排除步骤：
- 若 forward 参数数量错误，查 code generator 的 input binding 收集。
- 若字段名错误，查 data graph output fields 到 binding key 的映射。
- 若缺失字段崩溃，查 `_coerce_bound_value` 与 fallback batch。
- 若 device 不一致，查 tensor 创建和 `.to(device)` 位置。

### 5. 训练闭环生成

目标：排除“生成代码可导入，但不构成有效训练 step”。

测试场景：
- 标准训练 step：`zero_grad -> forward -> loss -> backward -> optimizer.step`。
- 显式 loss 节点：已有 loss 计算时不得重复追加默认 loss。
- 默认 loss 回退：没有显式 loss 时可生成默认 loss 函数和 target。
- Optimizer 节点参数：学习率、weight decay、momentum。
- Scheduler 节点：在 optimizer step 后更新，并避免未定义 optimizer。

排除步骤：
- 若 loss 重复，查 training config generator 是否检测已有 loss。
- 若 backward 前 loss 未定义，查 loss 节点和默认 loss 回退。
- 若 optimizer 未定义，查训练节点与模型参数绑定。
- 若 target 未定义，查 DataLoader binding 和 fallback target。

### 6. 前后端代码生成一致性

目标：排除“前端预览代码与后端 API 生成代码行为不一致”。

测试场景：
- 同一 model graph + data graph 输入前端 `generateLocalCode` 与后端 `/api/generate`。
- 比较关键结构：dataset class、DataLoader builder、binding resolver、train step。
- 对空数据图、仅模型图、仅数据图分别生成。

排除步骤：
- 若关键结构前后端不一致，先比较 dataflow compiler 输出契约。
- 若 API 成功但前端失败，查 TS compiler 与 Python compiler 的默认值差异。
- 若前端成功但 API 失败，查 Pydantic schema 与后端 service 入参。

### 7. 后端执行与运行时隔离

目标：排除“生成代码看起来正确，但后端执行入口不可用或环境不一致”。

测试场景：
- `/api/execute/execute` 执行最小 Python 代码。
- 执行包含 torch import、tensor 创建、简单 forward/loss 的代码。
- 执行错误代码时返回结构化 stderr，而不是吞错。
- conda 环境中执行，确认使用当前解释器而不是系统 Python。

排除步骤：
- 若 404，查 FastAPI router prefix 与 main include prefix。
- 若找不到依赖，查 subprocess 使用的解释器路径。
- 若超时，查 execute timeout 与训练循环是否无限。
- 若错误不可读，查 stderr/stdout 捕获与返回 schema。

### 8. 极限与鲁棒性场景

目标：排除“普通图通过，但复杂/异常图破坏生成器稳定性”。

测试场景：
- 空图、孤立节点、断开的必需输入。
- 图中存在环或非法边。
- 大图：100+ 节点、多个数据源、多个复合模块。
- 端口类型不匹配：image 接 scalar、target 接 feature。
- 节点参数非法：负 batch size、未知 optimizer、空字段名。

排除步骤：
- 若生成器崩溃，先将图最小化到单个非法节点或非法边。
- 若错误信息含糊，补 schema validation 和明确诊断。
- 若大图性能差，分离拓扑排序、代码发射和格式化耗时。
- 若非法图仍生成危险代码，补前置校验阻断。

## 建议执行顺序

1. 先固定最小训练闭环：模型单链 + CSV DataLoader + target loss。
2. 再扩展多输入、多输出、多字段 batch。
3. 接着验证封装节点：基础封装、嵌套、复用、解包。
4. 最后做异常和极限场景：非法边、缺失字段、大图、运行时错误。

每新增一类节点或训练能力，都应至少补一条“图结构 -> 生成代码关键片段 -> 可执行 smoke”的回归测试。
