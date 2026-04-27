# FlowHamster

A PyTorch visual modeling tool with ComfyUI-like logic. Build neural network models through graphical node editing and generate executable PyTorch code.

![FlowHamster](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Project Overview

FlowHamster is a visual PyTorch modeling tool that allows users to build neural network architectures through a drag-and-drop node-based interface and generate executable PyTorch code.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Dual Graph System** | Separates model architecture (modelGraph) from data preprocessing pipeline (dataGraph) |
| **Node Editor** | 48+ model nodes and 80+ data pipeline nodes |
| **Code Generation** | Graph → PyTorch nn.Module code via AST-based generator |
| **Subgraph Packaging** | Package selected nodes into reusable composite modules |
| **Real-time Preview** | WebSocket-powered live code generation and execution |

---

## Technical Architecture

### Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript + React Flow + Zustand |
| Backend | FastAPI (Python) + WebSocket |
| Desktop | Tauri |
| ML | PyTorch |

### Frontend Architecture

```
src/
├── components/           # React components
│   ├── canvas/          # Canvas (ReactFlow), NodeEditPanel
│   ├── nodes/           # 50+ node components
│   ├── data/            # Data pipeline canvas
│   ├── toolbar/         # Toolbar (training config, bindings)
│   ├── sidebar/         # Sidebar (node palette)
│   └── codePreview/     # Code preview panel
├── hooks/               # Zustand store hooks
│   ├── useGraphStore.ts # Model graph state (nodes, edges, undo/redo)
│   └── useDataGraphStore.ts
├── stores/              # State management
│   ├── useWorkflowStore.ts  # Workflow persistence
│   └── useTemplateStore.ts  # Template loading
└── utils/               # Core utilities
    ├── nodeRegistry.ts      # Model node definitions (48+)
    ├── dataNodeRegistry.ts  # Data node definitions (80+)
    ├── codeGenerator.ts     # Code generation orchestrator
    ├── codeEmitter.ts       # Code emitter (init/forward/loss)
    ├── astBuilder.ts        # AST builder
    ├── subgraphPackager.ts  # Subgraph packaging
    └── customCompositeRegistry.ts
```

**State Management (Zustand)**:

| Store | Responsibility |
|-------|----------------|
| `useGraphStore` | Model graph state (nodes, edges, selection, undo/redo, clipboard) |
| `useDataGraphStore` | Data pipeline graph state |
| `useWorkflowStore` | Workflow persistence (IndexedDB) |
| `useTemplateStore` | Template loading state |

### Backend Architecture

```
backend/
├── main.py              # FastAPI entry point
├── routers/             # API routes
│   ├── generate.py      # /api/generate
│   ├── execute.py       # /api/execute/*
│   ├── websocket.py     # /ws/code
│   ├── export.py        # /api/export
│   ├── templates.py     # /api/templates
│   └── workflows.py     # /api/workflows
├── services/            # Business logic
│   ├── ast_core.py      # AST code generation core
│   ├── code_gen_v2.py   # Code generator v2
│   ├── dataflow_compiler.py
│   ├── gradient_analyzer.py
│   └── tensor_executor.py
└── templates/           # Model templates (ResNet, VGG, etc.)
```

### Code Generation Pipeline

```
User Edit → [300ms debounce] → Local Generation OR WebSocket Push
                                              ↓
                                    getExecutableGraph()
                                              ↓
                                    pruneGraph() → reachable nodes
                                              ↓
                                    buildAST() → topological sort
                                              ↓
                                    For each block:
                                      - if composite: genCompositeInit/Forward
                                      - else: genInit + genForward
                                              ↓
                                    + Training Config + Loss + Evaluation
                                              ↓
                                    Full Python Module
```

---

## Features

### Model Nodes (48+ types)

| Category | Nodes |
|----------|-------|
| Input/Output | `input`, `output` |
| Convolution/Linear | `conv1d`, `conv2d`, `conv3d`, `linear`, `embedding` |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `leakyrelu` |
| Normalization | `batchnorm2d`, `layernorm`, `groupnorm` |
| Pooling | `maxpool2d`, `avgpool2d`, `adaptiveavgpool2d`, `globalavgpool` |
| Attention/Transformer | `selfattention`, `crossattention`, `multiheadattention`, `transformerencoder`, `transformerdecoder` |
| State Space | `mamba` |
| FFN/MLP | `ffn`, `mlp` |
| Tensor Operations | `add`, `mul`, `concat`, `reshape`, `flatten`, `transpose`, `split`, `slice`, `squeeze`, `expand` |
| Regularization | `dropout`, `droppath` |
| Loss Functions | `crossentropyloss`, `mseloss`, `focalloss`, `labelsmoothing` |
| Optimizers | `adam`, `adamw`, `sgd`, `rmsprop` |
| Schedulers | `steplr`, `cosineannealinglr`, `reducelronplateau` |
| Evaluation | `accuracy`, `f1`, `precision`, `recall`, `confusion_matrix`, `mean_iou`, `roc_auc` |

### Data Pipeline Nodes (80+ types)

| Category | Examples |
|----------|----------|
| Sources | `folder_source`, `csv_source`, `jsonl_source`, `parquet_source`, `huggingface_source` |
| Readers | `read_image`, `read_video`, `read_audio`, `read_text` |
| Transforms | `resize`, `crop`, `flip`, `rotate`, `pad`, `normalize`, `to_tensor`, `scale` |
| Augmentation | `random_horizontal_flip`, `random_crop`, `color_jitter`, `gaussian_blur` |
| Audio | `stft`, `spectrogram`, `melspectrogram`, `mfcc` |
| NLP | `tokenizer`, `truncate`, `add_special_tokens`, `random_mask` |
| Batch | `batch`, `collate`, `dataloader` |

### Key Features

- **Subgraph Packaging**: Package selected nodes into reusable composite modules
- **Implicit Concat**: Auto-creates concat nodes when multiple edges target the same port
- **Multi-Output Branch Coloring**: Visual branch differentiation for multi-output models
- **Data-Model Binding**: Bind data pipeline outputs to model inputs
- **Template Library**: Pre-built templates for ResNet, VGG, MobileNet, LeNet, ViT

---

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- PyTorch

### Installation

```bash
# Clone the repository
git clone https://github.com/isagoakira/FlowHamster.git
cd FlowHamster

# Install frontend dependencies
npm install

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### Development

**Start Frontend:**
```bash
npm run dev
# Access at http://localhost:5173
```

**Start Backend:**
```bash
npm run backend:dev
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

This launcher only watches [`backend/`](/Users/Zhuanz1/Desktop/file/FlowHamster/workspace/flowhamster_demo/backend) for reloads, which avoids high CPU usage from scanning the whole repo (especially `node_modules`).

**Production-style backend run (no file watching):**
```bash
npm run backend:start
```

### Build

```bash
npm run build        # TypeScript compile + Vite build
npm run preview      # Preview build
npx tsc --noEmit     # TypeScript type check
npm test             # Run vitest tests
```

---

## Project Structure

```
FlowHamster/
├── src/                          # React frontend
│   ├── App.tsx                   # Root component
│   ├── main.tsx                  # Entry point
│   ├── components/               # React components
│   │   ├── canvas/              # Canvas components
│   │   │   ├── Canvas.tsx       # Model graph canvas (ReactFlow)
│   │   │   ├── CompositeCanvas.tsx
│   │   │   └── NodeEditPanel.tsx
│   │   ├── nodes/                # 50+ node components
│   │   │   ├── BaseNode.tsx
│   │   │   ├── Conv2dNode.tsx
│   │   │   ├── SelfAttentionNode.tsx
│   │   │   └── ...
│   │   ├── data/                # Data pipeline components
│   │   ├── toolbar/             # Toolbar (training config)
│   │   ├── sidebar/             # Sidebar (node palette)
│   │   ├── codePreview/         # Code preview panel
│   │   └── dialogs/             # Dialogs
│   ├── hooks/                    # Zustand store hooks
│   │   ├── useGraphStore.ts     # Model graph state
│   │   ├── useDataGraphStore.ts # Data graph state
│   │   ├── useAutoLayout.ts
│   │   └── useTensorExecutor.ts
│   ├── stores/                   # State stores
│   │   ├── useWorkflowStore.ts  # Workflow persistence
│   │   └── useTemplateStore.ts  # Template loading
│   ├── utils/                    # Core utilities
│   │   ├── nodeRegistry.ts       # Model node definitions (48+)
│   │   ├── dataNodeRegistry.ts   # Data node definitions (80+)
│   │   ├── nodeComponentRegistry.ts
│   │   ├── codeGenerator.ts      # Code generation orchestrator
│   │   ├── codeEmitter.ts        # Code emitter
│   │   ├── astBuilder.ts         # AST builder
│   │   ├── graphPruner.ts        # Graph pruner
│   │   ├── subgraphPackager.ts   # Subgraph packaging
│   │   └── customCompositeRegistry.ts
│   ├── types/                    # TypeScript types
│   ├── schema/                   # Workflow document schema
│   └── constants/                # Constants
├── backend/                      # FastAPI backend
│   ├── main.py                   # Entry point
│   ├── routers/                  # API routes
│   │   ├── generate.py           # Code generation
│   │   ├── execute.py            # Execution
│   │   ├── websocket.py          # WebSocket
│   │   ├── export.py             # Export
│   │   ├── templates.py          # Templates
│   │   └── workflows.py          # Workflow CRUD
│   ├── services/                 # Business logic
│   │   ├── ast_core.py           # AST code generation
│   │   ├── code_gen_v2.py
│   │   ├── dataflow_compiler.py
│   │   ├── gradient_analyzer.py
│   │   └── tensor_executor.py
│   ├── schema/                   # Python schemas
│   └── templates/                # Model templates
├── workflows/                    # Workflow JSON files
├── docs/                         # Design documents
└── README.md
```

### Key Files

| File | Description |
|------|-------------|
| `src/utils/nodeRegistry.ts` | Model node registry (48+ node types) |
| `src/utils/dataNodeRegistry.ts` | Data node registry (80+ node types) |
| `src/utils/codeGenerator.ts` | Frontend code generation entry point |
| `src/utils/codeEmitter.ts` | Code emitter (init/forward/loss/evaluation) |
| `src/hooks/useGraphStore.ts` | Model graph state management |
| `backend/services/ast_core.py` | Backend AST code generation core |
| `backend/services/dataflow_compiler.py` | Data pipeline compiler |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate Python code from graph |
| `/api/execute/execute` | POST | Execute training code |
| `/api/execute/forward` | POST | Forward pass tensor preview |
| `/api/execute/gradients` | POST | Gradient analysis |
| `/api/export` | POST | Export as .py file |
| `/api/export-notebook` | POST | Export as Jupyter Notebook |
| `/api/templates` | GET | List available templates |
| `/api/workflows` | GET/POST | Workflow list/create |
| `/api/workflows/{id}` | GET/PUT/DELETE | Workflow CRUD |
| `/ws/code` | WebSocket | Real-time code generation |

---

## License

MIT
