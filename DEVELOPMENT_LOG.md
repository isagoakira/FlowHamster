# FlowHamster 开发日志

## 2026-04-27 更新 - 自定义类/实例复用与代码生成可读性重构

### 本轮目标

这一轮的核心目标是把 FlowHamster 的 package 功能从“把几个节点压成一个黑盒”推进到更接近 Python 类/实例模型的形态：

- package 后生成的是可复用的自定义类。
- 画布上的节点是该类的实例，实例名和类名需要分离。
- 同一个自定义类可以从节点列表中多次拖拽实例化。
- 后续允许从同源类派生多个不同版本，并让这些派生类同时出现在 Nodes 中。
- 生成代码要从晦涩的线性拼接，走向可读的类定义、实例构建和嵌套调用。

### 已完成

#### 1. 交互与选择行为修复

- 修复节点详情栏关闭后不能再次点击打开的问题。
- 区分点击和拖动，拖动节点不会误打开详情栏。
- Ctrl / Cmd / Shift 多选时不再弹出详情栏。
- package 节点右键菜单补充“重命名实例 / 重命名类 / 解包”。
- 节点详情栏对自定义 package 增加 `Instance Name` 和 `Class Name` 两个独立编辑入口。

#### 2. 代码生成可读性重构

- 新增 `pythonNodeRegistry.ts`，集中维护 Python 节点的 `__init__` 和 `forward` 生成逻辑。
- ViT / Transformer 相关生成逻辑改为可复用类结构，减少不可读的线性代码堆叠。
- 自定义 package 类使用最新 registry 结构实时生成代码，不再依赖陈旧的 `codeTemplate`。
- `parameter` / `constant` / `input` 等节点参与根节点保留，避免 ViT 参数节点被剪枝。
- 添加 ViT 生成代码回归测试，覆盖类定义、参数保留和可读结构。

#### 3. Package / 解包 / 嵌套复用重构

- package 时完整保存节点 `data`、位置、内部边 id、handle、boundary edge 映射。
- 展开 / 收起 / 解包改为按“当前实例真实连接”恢复边，而不是盲目使用历史边 id。
- 修复复制出来的 package 实例解包时接回旧节点的问题。
- 修复多个同类实例存在时，解包当前实例会被错误拦截或破坏其他实例的问题。
- 修复嵌套自定义 package 时 `customClassId` 丢失的问题。
- module 自动命名改为基于当前图状态计算，不再受旧 localStorage 或全局计数器污染。

#### 4. 自定义类 registry 与节点列表

- 自定义类 registry 保存稳定 `id`、`name`、`originClassId`、内部结构、端口、边界映射。
- 新创建的自定义类会自动进入左侧 `Custom` 节点列表。
- 从 `Custom` 列表拖拽会创建新的类实例，允许同一个类多次调用。
- 实例使用独立 `label`；类身份使用 `customClassRegistryId` 和 `customClassId`。
- 类重命名会同步更新同类实例、嵌套快照和 registry；实例自定义标签会被保留。
- 遇到未使用的 stale registry 类名时允许让位，避免类重命名被旧缓存卡住。

#### 5. 测试与验证

- `npm run test:run` 通过：5 个测试文件，53 个测试。
- `npm run build` 通过；仅剩 Vite chunk size warning。
- `git diff --check` 通过。
- 本地开发页 `http://127.0.0.1:5173/` 返回 200。

新增/强化的覆盖包括：

- 自定义类进入节点列表并携带 class id。
- 从 registry 多次实例化同一个自定义类。
- stale registry 被当前 package 覆盖。
- 当前实例解包不影响其他同类实例。
- 复制实例解包时生成唯一节点/边 id。
- 类重命名递归更新嵌套 package 快照。
- boundary edge 按当前实例连接恢复。

### 当前架构判断

FlowHamster 现在已经具备了“类定义 / 实例节点 / 嵌套调用”的基本骨架，但还没有进入完整的“类编辑器”阶段。当前 package 是创建类的入口，`Custom` 节点列表是复用入口，下一步应该把“修改类”和“派生类”做成明确的产品工作流。

推荐继续采用以下边界：

- `CustomCompositeClass` 是类定义。
- `CustomCompositeNodeData` 是画布实例。
- `customClassRegistryId` 是稳定类身份。
- `customClassId` 是 Python 类名，可重命名。
- `originClassId` 用于记录派生链，后续支持同源多版本。
- 实例 `label` 只作为实例名，不参与类查找。

### 下一步计划

#### P0 - 手动回归与小修

- 在浏览器内完整走一遍：普通节点打包 → 自定义列表出现类 → 拖出两个实例 → 分别改实例名 → 改类名 → 生成代码 → 解包其中一个实例。
- 检查 package 节点在详情栏和右键菜单两条路径的命名行为是否一致。
- 检查自定义列表刷新时机：新建类、重命名类、删除/覆盖 stale 类后都应即时更新。
- 对当前未覆盖的浏览器交互补最小 Playwright 测试。

#### P1 - 类编辑与派生工作流

- 增加“编辑类定义”入口：从自定义节点列表或 package 实例进入类编辑视图。
- 增加“保存为派生类”能力：
  - 新类获得新的 registry id。
  - `originClassId` 指向源类或源链根。
  - 默认命名可用 `ClassName_Variant_1`，但允许用户改名。
  - 派生类和源类同时显示在 `Custom` 节点列表。
- 明确“修改当前类”和“另存为派生类”的差异，避免用户误改所有实例。

#### P1 - 代码生成正确性验证

- 为自定义类嵌套生成代码增加更多 snapshot 测试。
- 增加 Python 语法检查或轻量执行测试，至少验证生成代码可 import / instantiate。
- 对 ViT、嵌套 package、自定义类复用三个模板做端到端代码生成回归。
- 统一 frontend codeGenerator 与 backend AST generator 的职责，避免双实现继续分叉。

#### P2 - Registry 与 Workflow 持久化

- 工作流保存时显式记录所依赖的 custom class registry 版本。
- 加载工作流时校验缺失类、重名类、派生类冲突，并给出修复提示。
- 设计自定义类导入/导出格式，方便分享 package 类库。

#### P2 - 大文件拆分与维护性

- 拆分 `useGraphStore.ts`：selection、clipboard、package、history、workflow 分离。
- 拆分 `subgraphPackager.ts`：boundary inference、snapshot restore、id mapping、package commands 分离。
- 继续缩小 `codeGenerator.ts`，让节点级 Python emit 与图级调度彻底分层。

#### P3 - 产品体验

- 自定义节点列表支持分类、搜索、高亮来源类/派生类。
- package 节点视觉上显示实例名和类名，例如 `encoder_0 : EncoderBlock`。
- 类编辑器显示“影响 N 个实例”，降低误操作风险。
- 支持 duplicate class / fork class / replace instances with derived class 等高级操作。

### 推荐立即执行顺序

1. 先做 P0 浏览器回归，把当前类/实例命名链路确认稳定。
2. 再做 P1 “保存为派生类”，这是后续同源多类并存的关键产品语义。
3. 同步补代码生成端到端测试，确保类嵌套和复用不是只在 UI 上成立。
4. 最后再拆大文件，避免在行为仍变化时提前重构目录结构。

---

## 2026-04-01 更新 (第三版) - InternalStructure 重构

### Phase 1: SubModuleType 规范化 ✅

移除了不可展开的类型（multiheadattention黑盒），新增完整的可展开类型：

```typescript
// 新增的 SubModuleType
type SubModuleType =
  // 基础层
  | 'conv1d' | 'conv2d' | 'conv3d' | 'linear' | 'embedding'
  // 归一化
  | 'layernorm' | 'batchnorm2d' | 'groupnorm' | 'instancenorm'
  // 激活
  | 'relu' | 'gelu' | 'silu' | 'sigmoid' | 'tanh' | 'leakyrelu'
  // Dropout
  | 'dropout' | 'droppath'
  // 池化
  | 'maxpool2d' | 'avgpool2d' | 'adaptiveavgpool2d' | 'globalavgpool'
  // 张量操作
  | 'softmax' | 'flatten' | 'reshape' | 'view'
  | 'transpose' | 'permute'
  | 'cat' | 'stack' | 'add' | 'mul' | 'matmul' | 'div' | 'sqrt'
  // Attention 专用
  | 'q_proj' | 'k_proj' | 'v_proj' | 'out_proj'
  | 'attn_score'       // QK^T / sqrt(d_k)
  | 'attn_weight'       // softmax(scores, dim)
  | 'attn_apply'       // attn_weights @ V
  // 残差
  | 'residual_add'     // x + sublayer
  // 多头分割/合并
  | 'split_heads' | 'merge_heads'
  // FFN
  | 'ffn_linear1' | 'ffn_linear2' | 'ffn_gelu' | 'ffn_drop'
  // 特殊
  | 'constant' | 'parameter' | 'scaledotattn'
```

### Phase 2: 复合节点 InternalStructure 重写 ✅

#### SelfAttention (完全展开)
```python
# 内部结构: QKV_proj → split_heads → attn_score → attn_weight → attn_apply → merge_heads → out_proj → drop
```
- 20个subModules
- 正确的 QKV 数据流

#### CrossAttention (完全展开)
```python
# 内部结构: Q_proj(query), K,V_proj(kv) → split_heads → attn_score → attn_weight → attn_apply → merge_heads → out_proj → drop
```
- 支持 query 和 kv 分开的输入

#### TransformerEncoder (完全展开)
```python
# Pre-LN 架构:
# x → SA_QKV → SA_attn → SA_drop → res_add1(x, SA_out) → norm1
# → FFN1 → GELU → FFN_drop → FFN2 → res_add2(norm1, FFN_out) → norm2 → output
```
- 22个subModules
- 包含完整的残差连接

#### TransformerDecoder (完全展开)
```python
# Pre-LN 架构:
# x → SelfAttn → res_add1 → norm1
# norm1 → CrossAttn(memory) → res_add2 → norm2
# norm2 → FFN → res_add3 → norm3 → output
```
- 29个subModules

### Phase 3: 代码生成器重构 ✅

代码生成器已重构，支持展开 `internalStructure`。实现内容:

1. **SubModule → Python 代码映射** ✅
   - `linear` → `nn.Linear(...)`
   - `attn_score` → `torch.matmul(q, k.transpose(-2, -1)) * self.scale`
   - `attn_weight` → `F.softmax(scores, dim=-1)`
   - `attn_apply` → `torch.matmul(attn_weights, v)`
   - `residual_add` → `x + sublayer_output`
   - `transpose` → `.transpose(dim0, dim1)`

2. **InternalEdges → 数据流追踪** ✅
   - `topologicalSortSubModules()` - 拓扑排序subModules
   - 变量名追踪 (`varMap`)
   - 多端口连接处理 (q/k/v, x/sublayer)
   - **`sourceInput` 属性**: 支持 SubModule 接收复合节点的外部输入

3. **生成完整可执行代码** ✅
   - `genCompositeInit()` - 生成复合节点的 `__init__`
   - `genCompositeForward()` - 生成展开的 forward 逻辑
   - 支持 SelfAttention, CrossAttention, TransformerEncoder, TransformerDecoder

### 已知问题/待修复

- [x] `residual_add` 的 x 输入需要正确传递 ✅
  - 修复：检查 xEdge 是否存在，不存在时使用 forward 函数的输入 `x`
- [x] CrossAttention 的 query/kv 分离输入 ✅
  - 修复：添加 `sourceInput` 属性到 SubModule 接口
  - SelfAttention: QKV projections 的 `sourceInput: 'x'`
  - CrossAttention: Q_proj 的 `sourceInput: 'query'`, K/V_proj 的 `sourceInput: 'kv'`
- [x] TransformerEncoder QKV 输入 ✅
  - `sa_q_proj`, `sa_k_proj`, `sa_v_proj` 添加 `sourceInput: 'x'`
- [x] TransformerDecoder QKV 输入 ✅
  - Self-Attention: QKV projections 添加 `sourceInput: 'tgt'`
  - Cross-Attention: K/V projections 添加 `sourceInput: 'memory'`
- [ ] TransformerEncoder/Decoder 多层堆叠未实现
- [ ] 需实际测试生成的代码是否运行正确

---

## 2026-04-01 更新 (第二版)

### 已完成
1. **节点内部结构可视化完善**
   - SelfAttention、CrossAttention、MultiheadAttention 现在是复合节点
   - TransformerEncoder/Decoder 内部结构重新设计
   - 所有复杂节点组件显示参数和内部结构指示器
   - 新增 SubModuleType: q_proj, k_proj, v_proj, out_proj, attn_score, attn_weight, attn_apply, residual_add, layer_norm, ffn_linear1/2, add, matmul

2. **Workflow 设置栏**
   - 新增底部 WorkflowSettingsBar 组件
   - 移除了遮挡弹窗的浮动 Features 面板
   - Multi-Output 开关移到设置栏
   - 统一的训练配置入口：Loss、Optimizer、LR、Scheduler、Epochs、Batch、Device、AMP

### 架构决策: 代码生成 vs 可视化

**当前实现状态 (2026-04-01 已变更):**

1. **代码生成器 (`codeGenerator.ts`)** 现在展开 `internalStructure`：
   - `selfattention` → 展开为 QKV_proj + attn_score + attn_weight + attn_apply 等 20 个 subModules
   - `transformerencoder` → 展开为完整的 Pre-LN 架构 22 个 subModules
   - `ffn` → 展开为 ffn_linear1 + ffn_gelu + ffn_drop + ffn_linear2
   - 图结构与生成的代码完全对应

2. **`internalStructure`** 同时用于：
   - UI 可视化（教育目的）
   - **代码生成（通过展开模式）**
   - 显示概念性的 QKV 流程、残差连接和 LayerNorm 的位置

3. **PyTorch nn.MultiheadAttention 实际内部结构**:
   ```python
   # PyTorch 内部实现（黑盒）：
   forward(query, key, value):
       q_proj = Linear(embed_dim, embed_dim)  # Q 投影
       k_proj = Linear(embed_dim, embed_dim)  # K 投影 (可能带 bias)
       v_proj = Linear(embed_dim, embed_dim)  # V 投影
       # 内部使用 F.scaled_dot_product_attention 或等效实现
       # 返回: (output, attn_weights)
   ```

4. **PyTorch nn.TransformerEncoderLayer 内部结构**:
   ```python
   self.self_attn = MultiheadAttention(...)
   self.norm1 = LayerNorm(...)
   self.linear1 = Linear(...)
   self.dropout = Dropout(...)
   self.linear2 = Linear(...)
   self.norm2 = LayerNorm(...)

   def forward(x):
       # Pre-norm 架构
       x = x + self.dropout(self.self_attn(self.norm1(x))[0])
       x = x + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
       return x
   ```

**两者对应关系：**
- 代码生成使用 PyTorch 原生模块（最优）
- internalStructure 显示教育性的计算流程
- 这是**抽象层级不同**，不是错误

### 待完成功能 (按优先级)

#### 高优先级
1. ~~**代码生成器支持展开 internalStructure**~~ ✅ 已完成
   - 代码生成器现在可以展开复合节点
   - 实现了拓扑排序、变量追踪、多端口连接处理

2. **数据预处理配置面板**
   - Dataset 节点配置 (path, transform, split)
   - DataLoader 参数 (batch_size, shuffle, num_workers)
   - 自定义 transform 链

3. **迭代器 (Iterator) 选择/构建**
   - for 循环迭代器
   - while 循环迭代器
   - 自定义迭代逻辑

4. **Binding 完整闭环**
   - 数据图 → 模型图字段映射的完整验证
   - Binding 参与代码生成

#### 中优先级
5. **预设模板完善**
   - LeNet, ResNet, ViT, BERT 模板
   - 模板预览和分类

6. **后端 templates 目录**
   - 55+ 模块模板文件填充
   - 或完善 ast_core.py 作为唯一代码生成源

7. **梯度计算图可视化**
   - 完整的梯度流向可视化
   - 消失/爆炸检测高亮

#### 低优先级
8. **中间张量预览完善**
   - 完整的前向传播预览
   - 任意节点输出预览

9. **自定义组合节点编辑器**
   - 可视化编辑内部结构
   - 保存为独立模板

---

## 技术债务

1. 前后端各有一套代码生成逻辑 (codeGenerator.ts vs ast_core.py)
2. 数据流编译有前后端两套实现 (dataWorkflowCompiler.ts vs dataflow_compiler.py)
3. Mamba 节点依赖 `flowhamster.modules` 外部库
4. ~~代码生成器与 internalStructure 完全割裂~~ ✅ 已通过展开模式解决

---

## 架构建议

~~1. **方案A (当前)**: 代码生成器使用 PyTorch 原生模块~~
   - ~~internalStructure 只做教育展示~~
   - ~~优点：代码正确、优化~~
   - ~~缺点：图结构与代码不完全对应~~

~~2. **方案B (用户期望)**: 代码生成器展开 internalStructure~~ ✅ **已实现**

3. **方案C (折中)**: ~~添加"展开模式"开关~~
   - ~~用户可选择：优化模式（PyTorch原生）或透明模式（展开internalStructure）~~
   - 用户可选择：当前实现即为透明模式

**当前状态**: 方案B已实现，图结构与代码完全对应。

---

## SubModuleType 完整列表

```
conv1d, conv2d, conv3d, linear, embedding
relu, gelu, silu, sigmoid, tanh, leakyrelu
dropout, droppath, layernorm, batchnorm2d, groupnorm
maxpool2d, avgpool2d, adaptiveavgpool2d, globalavgpool
softmax, flatten, reshape
multiheadattention
selfattention, crossattention
transformerencoder, transformerdecoder
q_proj, k_proj, v_proj, out_proj
attn_score, attn_weight, attn_apply
residual_add, layer_norm
ffn_linear1, ffn_linear2, ffn_gelu, ffn_drop
add, mul, concat, matmul
```
