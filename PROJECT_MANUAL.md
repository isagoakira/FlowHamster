# FlowHamster 项目手册

> 类 ComfyUI 逻辑的 PyTorch 可视化建模工具
>
> 版本: 2.0.0 | 更新日期: 2026-04-22

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [前端架构](#3-前端架构)
4. [后端架构](#4-后端架构)
5. [核心功能详解](#5-核心功能详解)
6. [节点类型体系](#6-节点类型体系)
7. [数据流与代码生成](#7-数据流与代码生成)
8. [工作流文档](#8-工作流文档)
9. [开发指南](#9-开发指南)

---

## 1. 项目概述

### 1.1 项目定位

**FlowHamster** 是一个类 ComfyUI 逻辑的 PyTorch 可视化建模工具，允许用户通过图形化节点编辑生成 PyTorch 代码。

### 1.2 核心特性

| 特性 | 描述 | 状态 |
|------|------|------|
| 双图系统 | 分离模型架构与数据预处理管道 | ✅ 成熟 |
| 可视化编辑 | React Flow 拖拽式节点编辑 | ✅ 成熟 |
| 代码生成 | AST-based 前端/后端双引擎 | ✅ 成熟 |
| 子图打包 | 将选中节点打包为自定义复合模块 | ✅ 成熟 |
| 实时预览 | WebSocket 实时代码生成 | ✅ 成熟 |
| 执行与导出 | 后端执行训练代码，导出 .py / Jupyter Notebook | ✅ 成熟 |
| 模板系统 | 5 个预设模板 (ResNet/VGG/LeNet/MobileNet/ViT) | ✅ 成熟 |
| 工作流持久化 | JSON 导入/导出 | ✅ 成熟 |

### 1.3 技术栈

| 层级 | 技术 |
|------|------|
| 前端框架 | React 18 + TypeScript |
| 可视化框架 | React Flow |
| 状态管理 | Zustand |
| 后端框架 | FastAPI (Python) |
| 实时通信 | WebSocket |
| 构建工具 | Vite |

---

## 2. 项目结构

```
FlowHamster/
├── src/                          # React 前端
│   ├── App.tsx                   # 根组件 - 双画布切换
│   ├── main.tsx                  # 入口文件
│   ├── components/               # React 组件
│   │   ├── canvas/              # 画布组件
│   │   │   ├── Canvas.tsx       # 模型图画布 (ReactFlow)
│   │   │   ├── CompositeCanvas.tsx # 复合节点查看器
│   │   │   └── NodeEditPanel.tsx # 节点编辑面板
│   │   ├── nodes/                # 节点组件 (50+)
│   │   │   ├── BaseNode.tsx      # 通用基类
│   │   │   ├── Conv2dNode.tsx
│   │   │   ├── LinearNode.tsx
│   │   │   ├── SelfAttentionNode.tsx
│   │   │   ├── TransformerEncoderNode.tsx
│   │   │   ├── MambaNode.tsx
│   │   │   ├── PackagedGroupNode.tsx # 打包子图节点
│   │   │   ├── GroupNode.tsx
│   │   │   └── ...
│   │   ├── data/                 # 数据管道组件
│   │   │   └── DataCanvas.tsx
│   │   ├── toolbar/              # 工具栏
│   │   │   └── Toolbar.tsx       # 训练配置/Binding 面板
│   │   ├── sidebar/              # 侧边栏
│   │   │   └── Sidebar.tsx       # 节点选择面板
│   │   ├── codePreview/          # 代码预览
│   │   │   └── CodePreview.tsx   # 语法高亮 + Run 按钮
│   │   └── dialogs/              # 对话框
│   ├── hooks/                    # Zustand store hooks
│   │   ├── useGraphStore.ts      # 模型图状态 (800+ 行)
│   │   ├── useDataGraphStore.ts  # 数据图画布状态
│   │   ├── useAutoLayout.ts      # 自动布局
│   │   └── useTensorExecutor.ts  # 张量执行器
│   ├── stores/                   # Zustand 状态存储
│   │   ├── useWorkflowStore.ts   # 工作流持久化 (IndexedDB)
│   │   └── useTemplateStore.ts   # 模板加载状态
│   ├── utils/                    # 核心工具
│   │   ├── nodeRegistry.ts       # 模型节点注册表 (48+ 节点)
│   │   ├── dataNodeRegistry.ts   # 数据节点注册表 (80+ 节点)
│   │   ├── nodeComponentRegistry.ts # React 组件映射
│   │   ├── codeGenerator.ts      # 代码生成入口 (v4 模块化)
│   │   ├── codeEmitter.ts        # 代码发射器 (init/forward/loss)
│   │   ├── astBuilder.ts         # AST 构建器
│   │   ├── graphPruner.ts        # 图形剪枝
│   │   ├── nodeSignatures.ts     # 节点签名定义
│   │   ├── dataWorkflowCompiler.ts # 数据流编译器
│   │   ├── subgraphPackager.ts   # 子图打包/展开
│   │   └── customCompositeRegistry.ts # 自定义复合模块 (localStorage)
│   ├── types/                    # TypeScript 类型定义
│   │   └── graph.ts
│   ├── schema/                   # 工作流文档 Schema
│   │   └── workflowDocument.ts
│   └── constants/                # 常量
│       └── handleIds.ts
├── backend/                      # FastAPI 后端
│   ├── main.py                   # FastAPI 入口 (CORS 配置)
│   ├── routers/                  # API 路由
│   │   ├── generate.py           # 代码生成 /api/generate
│   │   ├── execute.py            # 执行 /api/execute/*
│   │   ├── websocket.py          # WebSocket /ws/code
│   │   ├── export.py             # 导出 /api/export
│   │   ├── templates.py          # 模板 /api/templates
│   │   └── workflows.py          # 工作流 CRUD /api/workflows
│   ├── services/                 # 业务服务
│   │   ├── ast_core.py           # AST 代码生成核心 (50KB+)
│   │   ├── code_gen_v2.py       # 代码生成器 v2
│   │   ├── dataflow_compiler.py  # 数据流编译
│   │   ├── gradient_analyzer.py  # 梯度分析
│   │   └── tensor_executor.py    # 张量执行
│   ├── schema/                   # Python Schema
│   │   └── workflow_document.py
│   ├── templates/                # 模板实现
│   │   ├── registry.py
│   │   ├── resnet.py
│   │   ├── vgg.py
│   │   ├── mobilenet.py
│   │   ├── lenet.py
│   │   └── vit.py
│   └── modules/
│       ├── datasets/
│       │   └── data_pipeline.py
│       └── mamba.py
├── workflows/                    # 工作流 JSON 文件
└── docs/                        # 设计文档
```

---

## 3. 前端架构

### 3.1 组件系统

#### 画布组件 (`src/components/canvas/`)

| 组件 | 功能 |
|------|------|
| `Canvas.tsx` | 模型图画布，使用 ReactFlow。处理拖放、节点选择、分支着色、梯度可视化、上下文菜单。默认节点: input → conv2d → relu → output |
| `CompositeCanvas.tsx` | 复合节点查看器 |
| `NodeEditPanel.tsx` | 内联参数编辑器 |

#### 节点组件 (`src/components/nodes/`)

50+ 节点类型，继承自 `BaseNode.tsx`（通用节点，包含左侧 target 和右侧 source handles）:

| 节点类型 | 文件 | 说明 |
|----------|------|------|
| 卷积/全连接 | `Conv2dNode.tsx`, `LinearNode.tsx` | Conv1d/2d/3d, Linear, Embedding |
| 注意力/Transformer | `SelfAttentionNode.tsx`, `TransformerEncoderNode.tsx` | 复杂复合节点，含完整内部结构 |
| 状态空间 | `MambaNode.tsx` | SSM 节点 |
| 打包子图 | `PackagedGroupNode.tsx` | 表示打包后的子图 |
| 自定义 | `CustomNode.tsx` | 用户粘贴 Python 代码 |

#### 代码预览组件 (`src/components/codePreview/CodePreview.tsx`)

- 语法高亮 Python 代码 (`react-syntax-highlighter`)
- 可调整大小面板 (200-800px)
- 本地生成 / WebSocket 实时预览切换
- Run 按钮: POST 到 `/api/execute/execute`

### 3.2 状态管理 (Zustand)

| Store | 文件 | 职责 |
|-------|------|------|
| `useGraphStore` | `src/hooks/useGraphStore.ts` | 模型图状态 (nodes, edges, selection, undo/redo, clipboard) |
| `useDataGraphStore` | `src/hooks/useDataGraphStore.ts` | 数据图画布状态 |
| `useWorkflowStore` | `src/stores/useWorkflowStore.ts` | 工作流持久化 (IndexedDB) |
| `useTemplateStore` | `src/stores/useTemplateStore.ts` | 模板加载状态 |
| `useSelectionStore` | - | 节点/边选择状态 |
| `useClipboardStore` | - | 剪贴板 |

#### useGraphStore 核心功能

```typescript
interface GraphState {
  // 图形数据
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
  rfSetNodes: Function  // ReactFlow setters injected by Canvas
  rfSetEdges: Function

  // 选择
  selectedNodeIds: string[]
  selectedEdgeIds: string[]

  // Workspace 模式
  workspaceMode: 'model' | 'data'

  // 训练配置
  trainingConfig: WorkflowTrainingConfig
  bindings: WorkflowBinding[]

  // 历史 (Undo/Redo) - 最多 50 个快照
  _history: GraphSnapshot[]
  _future: GraphSnapshot[]

  // 剪贴板
  clipboard: { nodes: FlowHamsterNode[], edges: FlowHamsterEdge[] } | null
  pasteCount: number

  // 功能开关
  features: FeatureToggles
}
```

**关键方法**:

| 方法 | 功能 |
|------|------|
| `addNode(node)` | 添加节点 |
| `removeNodes(ids)` | 删除节点 |
| `updateNode(id, changes)` | 更新节点 |
| `setEdges(edges)` | 设置边 |
| `undo()` / `redo()` | 撤销/重做 |
| `copySelection()` / `pasteClipboard()` | 复制/粘贴 |
| `packageSelection()` | 打包选中节点为复合模块 |
| `expandGroup()` / `collapseGroup()` | 展开/收起 |
| `unpackageGroup()` | 解包 |
| `onConnect()` | 自动创建隐式 concat 节点 |

### 3.3 代码生成系统

**管道流程**: Graph → Prune → AST Build → Code Emit → Python Module

| 文件 | 功能 |
|------|------|
| `codeGenerator.ts` | 主入口，协调整个生成流程 |
| `astBuilder.ts` | 图 → AST (拓扑排序、剪枝、branch 分配) |
| `codeEmitter.ts` | AST → Python 字符串 (init/forward/loss) |
| `graphPruner.ts` | 剪枝不可达节点 |
| `nodeSignatures.ts` | 节点签名定义 (inputs, outputs, category) |
| `subgraphPackager.ts` | 子图打包/展开 |
| `customCompositeRegistry.ts` | 自定义复合模块注册表 (localStorage) |

#### 代码生成流程

```
generateLocalCode()
        ↓
1. getExecutableGraph()     - 获取可达节点
2. compileDataWorkflow()    - 数据管道
3. buildAST()               - 拓扑排序 + 剪枝
4. collectModuleImports()   - 收集所需导入
5. genCompositeInit/Forward() - 复合节点
6. genInit/genForward()     - 普通节点
7. genTrainingConfigCode()  - 训练循环
8. genLossInit/genLossForward()
9. genEvaluationCode()
        ↓
组装完整 Python 模块
```

### 3.4 双图系统

| Graph | Store | 用途 |
|-------|-------|------|
| **modelGraph** | `useGraphStore` | 神经网络架构 (Conv2d, Linear, Attention 等) |
| **dataGraph** | `useDataGraphStore` | 数据加载与预处理管道 |

切换: `workspaceMode: 'model' | 'data'` 在 `useGraphStore` 中

---

## 4. 后端架构

### 4.1 FastAPI 入口 (`backend/main.py`)

```python
app = FastAPI()
app.include_router(generate.router, prefix="/api")
app.include_router(execute.router, prefix="/api")
app.include_router(websocket.router)
app.include_router(templates.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
```

### 4.2 API 路由

| 路由 | 方法 | 功能 |
|------|------|------|
| `/api/generate` | POST | 生成 Python 代码 |
| `/api/execute/execute` | POST | 执行训练代码 (子进程运行 python) |
| `/api/execute/forward` | POST | Forward pass 张量预览 |
| `/api/execute/gradients` | POST | 梯度分析 |
| `/api/export` | POST | 导出 .py 文件 |
| `/api/export-notebook` | POST | 导出 Jupyter Notebook |
| `/api/templates` | GET | 获取模板列表 |
| `/api/workflows` | GET/POST | 工作流列表/创建 |
| `/api/workflows/{id}` | GET/PUT/DELETE | 工作流 CRUD |
| `/ws/code` | WebSocket | 实时代码生成 |

### 4.3 代码执行流程

1. 前端 POST 代码到 `/api/execute/execute`
2. 后端写入临时文件 `tempfile.NamedTemporaryFile`
3. 运行 `python tmp_path` 子进程 (60s 超时)
4. 捕获 stdout + stderr，返回 `ExecuteResponse(success, output, error)`

### 4.4 数据流编译器 (`backend/services/dataflow_compiler.py`)

```python
compile_dataflow() → CompiledDataflow
```

生成 `FlowHamsterDataset(torch.utils.data.Dataset)` 类:
- `__init__` — 数据源初始化
- `__getitem__` — 数据预处理管道
- `build_demo_batch()` — 创建示例张量
- `resolve_bound_inputs()` — 映射数据字段到模型输入
- `select_primary_model_input()` — 选择主模型输入

---

## 5. 核心功能详解

### 5.1 子图打包系统 (Composite/Packaged Nodes)

用户可以将选中的节点打包成自定义复合模块:

| 操作 | 函数 | 说明 |
|------|------|------|
| 打包 | `packageSelection()` | 选中节点 → 创建黑盒复合模块 |
| 展开 | `expandPackage()` | 双击或点击按钮显示内部结构 |
| 收起 | `collapsePackage()` | 隐藏内部结构恢复黑盒 |
| 解包 | `unpackageGroup()` | 还原为普通节点 |

**自定义复合模块注册表**: `src/utils/customCompositeRegistry.ts` (localStorage 持久化)

### 5.2 隐式 Concat 节点

当多条边指向同一端口时:

1. `onConnect()` 检测到已有边指向该端口
2. 自动创建 `__iconcat__` 节点 (隐式 concat)
3. 现有边重定向到 `in_0`，新边到 `in_1`
4. Concat 节点输出 → 原目标
5. `mergeMode`: `'concat'` (默认), `'add'`, `'mul'`, `'stack'`

### 5.3 多输出分支着色

当 `features.multiOutput === true`:
- 从每个 Output 节点反向 BFS 分配 `branchIndex` (0, 1, 2...)
- 分支颜色: 蓝、绿、橙、粉、紫 (循环)
- 选中节点显示分支颜色 glow box-shadow
- 节点边框按 branch 着色

### 5.4 Data-Model 绑定系统

```
DataGraph Output ──→ Binding Panel ──→ Model Input
字段名匹配 + dtype 兼容性检查
```

---

## 6. 节点类型体系

### 6.1 模型节点 (48+ 种)

| 类别 | 数量 | 示例 |
|------|------|------|
| Input/Output | 2 | `input`, `output` |
| Convolution/Linear | 5 | `conv2d`, `conv1d`, `conv3d`, `linear`, `embedding` |
| Activation | 6 | `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `leakyrelu` |
| Normalization | 3 | `batchnorm2d`, `layernorm`, `groupnorm` |
| Pooling | 4 | `maxpool2d`, `avgpool2d`, `adaptiveavgpool2d`, `globalavgpool` |
| Attention/Transformer | 5+ | `selfattention`, `crossattention`, `multiheadattention`, `transformerencoder`, `transformerdecoder` |
| State Space | 1 | `mamba` |
| FFN/MLP | 2 | `ffn`, `mlp` |
| Tensor Operations | 10+ | `add`, `mul`, `concat`, `reshape`, `flatten`, `transpose`, `split`, `slice`, `squeeze`, `expand` |
| Regularization | 2 | `dropout`, `droppath` |
| Loss Functions | 4+ | `crossentropyloss`, `mseloss`, `focalloss`, `labelsmoothing` |
| Optimizers | 4 | `adam`, `adamw`, `sgd`, `rmsprop` |
| Schedulers | 3 | `steplr`, `cosineannealinglr`, `reducelronplateau` |
| Evaluation | 7 | `accuracy`, `f1`, `precision`, `recall`, `confusion_matrix`, `mean_iou`, `roc_auc` |
| Custom | 1 | `custom` (粘贴 Python 代码) |

### 6.2 数据节点 (80+ 种)

| 类别 | 数量 | 示例 |
|------|------|------|
| Sources | 5 | `folder_source`, `csv_source`, `jsonl_source`, `parquet_source`, `huggingface_source` |
| Readers | 4 | `read_image`, `read_video`, `read_audio`, `read_text` |
| Field Ops | 3 | `select_fields`, `rename_fields`, `filter` |
| Transforms | 8+ | `resize`, `crop`, `flip`, `rotate`, `pad`, `normalize`, `to_tensor`, `scale` |
| Augmentation | 15+ | `random_horizontal_flip`, `random_crop`, `color_jitter`, `gaussian_blur`, etc. |
| Compose | 1 | `compose` (管道分隔的 transforms) |
| Organization | 3 | `train_val_split`, `shuffle`, `concat` |
| Batch | 3 | `batch`, `collate`, `dataloader` |
| NLP | 4 | `tokenizer`, `truncate`, `add_special_tokens`, `random_mask` |
| Audio | 7 | `stft`, `istft`, `spectrogram`, `melspectrogram`, `mfcc`, `gammatone`, `pitch_extraction` |
| Tensor Ops | 10+ | `tensor_reshape`, `tensor_flatten`, `tensor_transpose`, `tensor_slice`, `tensor_stack`, etc. |
| Output | 1 | `dataset_output` |

---

## 7. 数据流与代码生成

### 7.1 用户交互流程

```
1. 加载应用 → App.tsx 渲染 Toolbar + Sidebar + Canvas + CodePreview

2. 添加节点 → 拖拽侧边栏 → onDrop() → useGraphStore.addNode() → ReactFlow setNodes()

3. 连接节点 → 拖拽 handle → onConnect() → 隐式 concat 检测 → addEdge()

4. 编辑参数 → 选择节点 → NodeEditPanel → updateNodeData() → pushHistory()

5. 生成代码 (本地):
   useEffect 在 CodePreview 中调用 getCachedOrGenerateCode()
   → 剪枝图 → 构建 AST → 发射 Python → 显示

6. 生成代码 (WebSocket):
   切换 WS 按钮 → connect() → /ws/code
   300ms debounce → 推送 JSON → 服务器返回代码

7. 运行代码 → 点击 Run → POST /api/execute/execute → 子进程 → stdout/error

8. 保存工作流 → useWorkflowStore.saveWorkflow() → PUT /api/workflows/{id}/document

9. 打包子图 → 选中 2+ 节点 → packageSelection() → 创建 PackagedGroupNode

10. 展开/收起包 → 右键包节点 → 上下文菜单 → expandPackage() / collapsePackage()
```

### 7.2 代码生成数据流

```
前端图 (nodes + edges)
        ↓
[graphPruner] pruneGraph() — 从 Input 到 Output 可达节点
        ↓
[astBuilder] buildAST() — 拓扑排序，branch 分配
        ↓
[codeEmitter] genInit + genForward — 每个 block 代码发射
        ↓
[codeGenerator] 组装完整 Python 模块
        ↓
[backend] ast_core.generate() — 后端备选生成
[dataflow_compiler] compile_dataflow() — 数据管道
        ↓
完整 Python: imports + nn.Module 类 + Dataset + main
```

---

## 8. 工作流文档

### 8.1 WorkflowDocument 结构

```typescript
WorkflowDocument {
  version: "2.0.0"
  metadata: {
    name: string
    description: string
    schemaVersion: string
    createdAt: string
    updatedAt: string
  }
  modelGraph: {
    kind: 'model'
    nodes: Node[]
    edges: Edge[]
    contract: { inputs: string[], outputs: string[] }
  }
  dataGraph: {
    kind: 'data'
    nodes: Node[]
    edges: Edge[]
    contract: { inputs: string[], outputs: string[] }
  }
  trainingConfig: WorkflowTrainingConfig {
    taskType: string
    loss: string
    optimizer: string
    scheduler: string
    metrics: string[]
    runtime: string
    checkpoint: string
  }
  bindings: WorkflowBinding[]  // data → model 字段映射
}
```

### 8.2 WorkflowBinding 结构

```typescript
WorkflowBinding {
  sourceGraph: 'model' | 'data'
  sourceKey: string      // 数据管道中的字段名
  target: 'model_input' | 'training_target'
  targetKey: string     // 模型输入名
}
```

---

## 9. 开发指南

### 9.1 开发命令

#### 前端
```bash
npm run dev          # 启动开发服务器 (http://localhost:5173)
npm run build        # TypeScript 编译 + Vite 构建
npm run preview      # 预览构建结果
npx tsc --noEmit     # TypeScript 类型检查
npm test             # 运行 vitest 测试
npm run test:run     # 运行测试 (单次)
```

#### 后端
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000  # 启动 FastAPI (http://localhost:8000)
```

### 9.2 CORS 配置

后端允许以下前端源:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://192.168.102.231:5173`

### 9.3 调试注意事项

- **代码生成调试**: 修改 `codeGenerator.ts` 后需清除 localStorage 缓存（生成的代码可能被缓存）
- **子图打包调试**: 打包/展开操作会修改 `internalStructure`，检查 `customClassId` 是否正确传递
- **测试**: 修改节点行为或代码生成后运行 `npm test` 验证回归

---

## 附录: 关键文件索引

| 文件路径 | 功能描述 |
|----------|----------|
| `src/utils/nodeRegistry.ts` | 模型节点注册表 (48+ 节点类型) |
| `src/utils/dataNodeRegistry.ts` | 数据节点注册表 (80+ 节点类型) |
| `src/utils/codeGenerator.ts` | 前端代码生成入口 (v4 模块化架构) |
| `src/utils/codeEmitter.ts` | 代码发射器 (init/forward/loss/evaluation) |
| `src/utils/astBuilder.ts` | AST 构建器 |
| `src/utils/subgraphPackager.ts` | 子图打包器 |
| `src/utils/customCompositeRegistry.ts` | 自定义复合模块注册表 |
| `src/hooks/useGraphStore.ts` | 模型图状态管理 (800+ 行) |
| `src/hooks/useDataGraphStore.ts` | 数据图画布状态管理 |
| `src/components/canvas/Canvas.tsx` | 模型图画布 |
| `src/components/codePreview/CodePreview.tsx` | 代码预览面板 |
| `backend/services/ast_core.py` | 后端 AST 代码生成核心 (50KB+) |
| `backend/services/dataflow_compiler.py` | 数据流编译器 |
| `backend/services/tensor_executor.py` | 张量执行器 |
| `backend/services/code_gen_v2.py` | 代码生成器 v2 |

---

> 📝 更新记录
> - 2026-04-22: 全面更新项目架构文档，基于代码探索结果
