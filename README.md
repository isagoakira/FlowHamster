# FlowHamster

A PyTorch visual modeling tool with ComfyUI-like logic. Build neural network models through graphical node editing and generate executable PyTorch code.

![FlowHamster](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript)

## Features

- **Visual Node Editor** - Drag-and-drop interface for building neural network architectures
- **Code Generation** - Export your graph to production-ready PyTorch code
- **Real-time Preview** - Execute forward pass and view tensor outputs
- **Training Support** - Configure and run training loops with gradient analysis
- **Template Library** - Pre-built templates for ResNet, VGG, MobileNet, LeNet, ViT
- **Dual Graph System** - Separate model architecture and data pipeline graphs

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript + React Flow + Zustand |
| Backend | FastAPI (Python) + WebSocket |
| Desktop | Tauri |
| ML | PyTorch |

## Getting Started

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
cd backend
PYTHONPATH=. python3 -m uvicorn main:app --reload --port 8000
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Build

```bash
npm run build        # Frontend build
npm run tauri:build  # Desktop app build
```

## Project Structure

```
FlowHamster/
├── src/                    # React frontend
│   ├── components/         # React components
│   │   ├── nodes/          # Model node components (48 types)
│   │   └── data/           # Data pipeline components
│   ├── hooks/              # React hooks (Zustand stores)
│   ├── stores/             # State management
│   └── utils/              # Code generators, registries
├── backend/                # FastAPI backend
│   ├── modules/            # Custom PyTorch modules (Mamba, etc.)
│   ├── routers/            # API routes
│   ├── services/           # AST code generation core
│   ├── schema/             # Data models
│   └── templates/          # Model templates
├── src-tauri/              # Tauri desktop app
└── workflows/              # Example workflows
```

## Node Types

### Model Nodes (48 types)
- **Layers**: Conv2d, Linear, LSTM, MultiheadAttention, TransformerEncoder/Decoder
- **Activations**: ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU
- **Normalization**: BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm
- **Pooling**: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- **Training**: SGD, Adam, AdamW, RMSprop, CosineAnnealing, StepLR
- **Losses**: MSELoss, CrossEntropyLoss
- **Special**: Mamba (SSM), FFN, MLP, DropPath, Embedding

### Data Nodes (60+ types)
Data pipeline components for loading, preprocessing, and transforming data.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate Python code from graph |
| `/api/execute` | POST | Execute training code |
| `/api/execute/forward` | POST | Forward pass tensor preview |
| `/api/execute/gradients` | POST | Gradient analysis |
| `/api/export` | POST | Export as .py file |
| `/api/templates` | GET | List available templates |
| `/api/workflows` | GET/POST | Workflow CRUD |
| `/ws` | WebSocket | Real-time communication |

## Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Node Editor │────▶│ Code Generator │────▶│ PyTorch Code │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  Tensor Exec  │     │   Execute   │
                    └──────────────┘     └─────────────┘
```

## License

MIT
