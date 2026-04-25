# FlowHamster 开发日志

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
