# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

FlowHamster 是一个类 ComfyUI 逻辑的 PyTorch 可视化建模工具，允许用户通过图形化节点编辑生成 PyTorch 代码。

- **前端**: React 18 + TypeScript + React Flow + Zustand
- **后端**: FastAPI (Python) + WebSocket
- **核心功能**: 节点编辑 → 代码生成 → 模型执行/导出

## 开发命令

### 前端
```bash
npm run dev          # 启动开发服务器 (http://localhost:5173)
npm run build        # TypeScript 编译 + Vite 构建
npm run preview      # 预览构建结果
npx tsc --noEmit     # TypeScript 类型检查
npm test             # 运行 vitest 测试
npm run test:run     # 运行测试（单次）
```

### 后端
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000  # 启动 FastAPI (http://localhost:8000)
```

## 核心架构

### 双图系统

FlowHamster 使用两个独立的图来分离模型和数据流：

| Graph | 存储位置 | 用途 |
|-------|----------|------|
| **modelGraph** | `useGraphStore` | 神经网络架构（Conv2d, Linear, Attention 等） |
| **dataGraph** | `useDataGraphStore` | 数据加载与预处理管道 |

### 前端状态管理 (Zustand)

| Store | 位置 | 职责 |
|-------|------|------|
| `useGraphStore` | `src/hooks/` | 模型图状态 (nodes, edges, selection, undo/redo, clipboard) |
| `useDataGraphStore` | `src/hooks/` | 数据图状态 |
| `useWorkflowStore` | `src/stores/` | 工作流持久化 (IndexedDB) |
| `useTemplateStore` | `src/stores/` | 模板加载状态 |

### 子图打包系统 (Composite/Packaged Nodes)

用户可以将选中的节点打包成自定义复合模块：

- **打包** `packageSelection()`: 选中节点 → 创建黑盒复合模块
- **展开** `expandPackage()`: 双击或点击按钮显示内部结构
- **收起** `collapsePackage()`: 隐藏内部结构恢复黑盒
- **解包** `unpackageGroup()`: 还原为普通节点

自定义复合模块注册表: `src/utils/customCompositeRegistry.ts` (localStorage 持久化)

### 代码生成流程

```
用户操作 → Zustand Store → React Flow 渲染
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
前端代码生成           后端 API 调用
(src/utils/codeGenerator)  (/api/generate)
    ↓                       ↓
    └───────────┬───────────┘
                ↓
         代码输出/执行结果
```

**关键文件**:
- `src/utils/codeGenerator.ts` — 前端代码生成入口（v4 模块化架构）
- `src/utils/codeEmitter.ts` — 代码发射器（init/forward/loss/evaluation）
- `src/utils/astBuilder.ts` — AST 构建器
- `src/utils/dataWorkflowCompiler.ts` — 数据流编译器
- `backend/services/ast_core.py` — 后端 AST 代码生成核心

### 后端 API 路由
| 路由 | 方法 | 功能 |
|------|------|------|
| `/api/generate` | POST | 生成 Python 代码 |
| `/api/execute` | POST | 执行训练代码 |
| `/api/execute/forward` | POST | Forward pass 张量预览 |
| `/api/execute/gradients` | POST | 梯度分析 |
| `/api/export` | POST | 导出 .py 文件 |
| `/api/export-notebook` | POST | 导出 Jupyter Notebook |
| `/api/templates` | GET | 获取模板列表 |
| `/api/workflows` | GET/POST | 工作流列表/创建 |
| `/api/workflows/{id}` | GET/PUT/DELETE | 工作流 CRUD |
| `/ws` | WebSocket | 实时通信 |

### 节点类型体系

**模型节点** (48 种): 定义在 `src/utils/nodeRegistry.ts`
- Layers: Conv2d, Linear, LSTM, MultiheadAttention, TransformerEncoder
- Activations: ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU
- Normalization: BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm
- Training: SGD, Adam, AdamW, RMSprop, CosineAnnealing, StepLR

**数据节点** (60+ 种): 定义在 `src/utils/dataNodeRegistry.ts`
- 数据加载、预处理、增强

**组件映射**: `src/utils/nodeComponentRegistry.ts` — React 组件注册表

## 调试注意事项

- **代码生成调试**: 修改 `codeGenerator.ts` 后需清除 localStorage 缓存（生成的代码可能被缓存）
- **子图打包调试**: 打包/展开操作会修改 `internalStructure`，检查 `customClassId` 是否正确传递
- **测试**: 修改节点行为或代码生成后运行 `npm test` 验证回归

## CORS 配置

后端允许以下前端源:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://192.168.102.231:5173`
