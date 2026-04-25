import { create } from 'zustand'

// =============================================================================
// Types — mirror the backend API response
// =============================================================================

export interface TemplateNode {
  id: string
  data: {
    nodeType: string
    label: string
    params: Record<string, unknown>
    position?: { x: number; y: number }
  }
}

export interface TemplateEdge {
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
}

export interface TemplateGraph {
  nodes: TemplateNode[]
  edges: TemplateEdge[]
}

export interface Template {
  id: string
  name: string
  description: string
  category: 'cv' | 'nlp' | 'gan' | 'other'
  emoji: string
  graph: TemplateGraph
}

export interface TemplateStore {
  templates: Template[]
  isLoading: boolean
  error: string | null
  loadTemplates: () => Promise<void>
}

// =============================================================================
// Local template graph definitions (fallback + source of truth for graphs)
// =============================================================================

function makeLayout(nodes: any[], numRows: number = 1): any[] {
  const result = []
  const nodesPerRow = Math.ceil(nodes.length / numRows)
  let row = 0
  let col = 0
  for (const node of nodes) {
    const x = 100 + col * 280
    const y = 80 + row * 120
    result.push({ ...node, position: { x, y } })
    col++
    if (col >= nodesPerRow) { col = 0; row++ }
  }
  return result
}

const LOCAL_TEMPLATES: Template[] = [
  {
    id: 'resnet18', name: 'ResNet-18', description: '18层残差网络，含4个残差阶段，适合图像分类', category: 'cv', emoji: '🔬',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d', label: 'Conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3, bias: false } } },
        { id: 'bn1', data: { nodeType: 'batchnorm2d', label: 'BatchNorm2d', params: { num_features: 64, eps: 1e-5, momentum: 0.1 } } },
        { id: 'relu', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'maxpool', data: { nodeType: 'maxpool2d', label: 'MaxPool2d', params: { kernel_size: 3, stride: 2, padding: 1 } } },
        { id: 'layer1_1', data: { nodeType: 'conv2d', label: 'BasicBlock1-1', params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'layer1_2', data: { nodeType: 'conv2d', label: 'BasicBlock1-2', params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'add1', data: { nodeType: 'add', label: 'Add', params: {} } },
        { id: 'layer2_1', data: { nodeType: 'conv2d', label: 'BasicBlock2-1', params: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'layer2_2', data: { nodeType: 'conv2d', label: 'BasicBlock2-2', params: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'down1', data: { nodeType: 'conv2d', label: 'Downsample', params: { in_channels: 64, out_channels: 128, kernel_size: 1, stride: 2 } } },
        { id: 'add2', data: { nodeType: 'add', label: 'Add', params: {} } },
        { id: 'gap', data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc', data: { nodeType: 'linear', label: 'Linear', params: { in_features: 512, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        { source: 'input', target: 'conv1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv1', target: 'bn1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn1', target: 'relu', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu', target: 'maxpool', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'maxpool', target: 'layer1_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'layer1_1', target: 'layer1_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'layer1_2', target: 'add1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'maxpool', target: 'add1', sourceHandle: 'result', targetHandle: 'b' },
        { source: 'add1', target: 'layer2_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'add1', target: 'down1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'layer2_1', target: 'layer2_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'layer2_2', target: 'add2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'down1', target: 'add2', sourceHandle: 'result', targetHandle: 'b' },
        { source: 'add2', target: 'gap', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'gap', target: 'fc', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },
  {
    id: 'vgg16', name: 'VGG-16', description: '16层VGG网络，3阶段卷积+2层全连接', category: 'cv', emoji: '🔷',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'conv1_1', data: { nodeType: 'conv2d', label: 'Conv 64-1', params: { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv1_2', data: { nodeType: 'conv2d', label: 'Conv 64-2', params: { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool1', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'conv2_1', data: { nodeType: 'conv2d', label: 'Conv 128-1', params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv2_2', data: { nodeType: 'conv2d', label: 'Conv 128-2', params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool2', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'conv3_1', data: { nodeType: 'conv2d', label: 'Conv 256-1', params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv3_2', data: { nodeType: 'conv2d', label: 'Conv 256-2', params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool3', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'conv4_1', data: { nodeType: 'conv2d', label: 'Conv 512-1', params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv4_2', data: { nodeType: 'conv2d', label: 'Conv 512-2', params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool4', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'gap', data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc1', data: { nodeType: 'linear', label: 'FC 4096', params: { in_features: 512, out_features: 4096, bias: true } } },
        { id: 'fc2', data: { nodeType: 'linear', label: 'FC 1000', params: { in_features: 4096, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        { source: 'input', target: 'conv1_1' }, { source: 'conv1_1', target: 'conv1_2' }, { source: 'conv1_2', target: 'pool1' },
        { source: 'pool1', target: 'conv2_1' }, { source: 'conv2_1', target: 'conv2_2' }, { source: 'conv2_2', target: 'pool2' },
        { source: 'pool2', target: 'conv3_1' }, { source: 'conv3_1', target: 'conv3_2' }, { source: 'conv3_2', target: 'pool3' },
        { source: 'pool3', target: 'conv4_1' }, { source: 'conv4_1', target: 'conv4_2' }, { source: 'conv4_2', target: 'pool4' },
        { source: 'pool4', target: 'gap' }, { source: 'gap', target: 'fc1' }, { source: 'fc1', target: 'fc2' }, { source: 'fc2', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: 'result', targetHandle: 'a' }))
    }
  },
  {
    id: 'lenet5', name: 'LeNet-5', description: '5层LeNet，适合MNIST/CIFAR入门', category: 'cv', emoji: '✏️',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '1, 32, 32', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d', label: 'Conv C6', params: { in_channels: 1, out_channels: 6, kernel_size: 5, padding: 2, bias: true } } },
        { id: 'relu1', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'pool1', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'conv2', data: { nodeType: 'conv2d', label: 'Conv C16', params: { in_channels: 6, out_channels: 16, kernel_size: 5, bias: true } } },
        { id: 'relu2', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'pool2', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'flatten', data: { nodeType: 'flatten', label: 'Flatten', params: { start_dim: 1 } } },
        { id: 'fc1', data: { nodeType: 'linear', label: 'FC 120', params: { in_features: 400, out_features: 120, bias: true } } },
        { id: 'relu3', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'fc2', data: { nodeType: 'linear', label: 'FC 84', params: { in_features: 120, out_features: 84, bias: true } } },
        { id: 'relu4', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'fc3', data: { nodeType: 'linear', label: 'FC 10', params: { in_features: 84, out_features: 10, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        { source: 'input', target: 'conv1' }, { source: 'conv1', target: 'relu1' }, { source: 'relu1', target: 'pool1' },
        { source: 'pool1', target: 'conv2' }, { source: 'conv2', target: 'relu2' }, { source: 'relu2', target: 'pool2' },
        { source: 'pool2', target: 'flatten' }, { source: 'flatten', target: 'fc1' }, { source: 'fc1', target: 'relu3' },
        { source: 'relu3', target: 'fc2' }, { source: 'fc2', target: 'relu4' }, { source: 'relu4', target: 'fc3' },
        { source: 'fc3', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: 'result', targetHandle: 'a' }))
    }
  },
  {
    id: 'vit', name: 'Vision Transformer', description: '视觉Transformer，Patch Embed + Transformer Encoder + CLS Token', category: 'cv', emoji: '🔮',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'patch_embed', data: { nodeType: 'conv2d', label: 'PatchEmbed', params: { in_channels: 3, out_channels: 768, kernel_size: 16, stride: 16, bias: false } } },
        { id: 'flatten', data: { nodeType: 'flatten', label: 'Flatten', params: { start_dim: 2, end_dim: 3 } } },
        { id: 'permute', data: { nodeType: 'permute', label: 'Permute', params: { dims: '0,2,1' } } },
        { id: 'cls_token', data: { nodeType: 'parameter', label: 'CLS Token', params: { shape: '1,1,768', trainable: true } } },
        { id: 'concat', data: { nodeType: 'concat', label: 'Concat CLS', params: { dim: 1 } } },
        { id: 'pos_embed', data: { nodeType: 'parameter', label: 'Pos Embed', params: { shape: '1,197,768', trainable: true } } },
        { id: 'add_pos', data: { nodeType: 'add', label: 'Add Pos', params: {} } },
        { id: 'dropout', data: { nodeType: 'dropout', label: 'Dropout', params: { p: 0.1 } } },
        { id: 'encoder', data: { nodeType: 'transformerencoder', label: 'TransformerEncoder', params: { embed_dim: 768, num_heads: 12, num_layers: 12, dim_feedforward: 3072, dropout: 0.1 } } },
        { id: 'norm_out', data: { nodeType: 'layernorm', label: 'LayerNorm', params: { normalized_shape: 768, eps: 1e-6 } } },
        { id: 'cls_slice', data: { nodeType: 'slice', label: 'Extract CLS', params: { start: 0, end: 1, dim: 1 } } },
        { id: 'squeeze', data: { nodeType: 'squeeze', label: 'Squeeze', params: { dim: 1 } } },
        { id: 'head', data: { nodeType: 'linear', label: 'Head', params: { in_features: 768, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        { source: 'input', target: 'patch_embed' }, { source: 'patch_embed', target: 'flatten' }, { source: 'flatten', target: 'permute' },
        { source: 'permute', target: 'concat', sourceHandle: 'result', targetHandle: 'in_0' },
        { source: 'cls_token', target: 'concat', sourceHandle: 'result', targetHandle: 'in_1' },
        { source: 'concat', target: 'add_pos' }, { source: 'pos_embed', target: 'add_pos' },
        { source: 'add_pos', target: 'dropout' }, { source: 'dropout', target: 'encoder' },
        { source: 'encoder', target: 'norm_out' }, { source: 'norm_out', target: 'cls_slice' },
        { source: 'cls_slice', target: 'squeeze' }, { source: 'squeeze', target: 'head' }, { source: 'head', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: e.sourceHandle || 'result', targetHandle: e.targetHandle || 'a' }))
    }
  },
  {
    id: 'mobilenetv2', name: 'MobileNetV2', description: '倒残差结构+深度可分离卷积，适合移动端部署', category: 'cv', emoji: '📱',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d', label: 'Conv 3→32', params: { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'bn1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 32 } } },
        { id: 'silu1', data: { nodeType: 'silu', label: 'SiLU', params: {} } },
        { id: 'dw1', data: { nodeType: 'conv2d', label: 'DW 32→32', params: { in_channels: 32, out_channels: 32, kernel_size: 3, stride: 1, padding: 1, groups: 32, bias: false } } },
        { id: 'bn_dw1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 32 } } },
        { id: 'pw1', data: { nodeType: 'conv2d', label: 'PW 32→16', params: { in_channels: 32, out_channels: 16, kernel_size: 1, bias: false } } },
        { id: 'bn_pw1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 16 } } },
        { id: 'exp2', data: { nodeType: 'conv2d', label: 'PW 16→96', params: { in_channels: 16, out_channels: 96, kernel_size: 1, bias: false } } },
        { id: 'bn_exp2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 96 } } },
        { id: 'dw2', data: { nodeType: 'conv2d', label: 'DW 96→96', params: { in_channels: 96, out_channels: 96, kernel_size: 3, stride: 2, padding: 1, groups: 96, bias: false } } },
        { id: 'bn_dw2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 96 } } },
        { id: 'pw2', data: { nodeType: 'conv2d', label: 'PW 96→24', params: { in_channels: 96, out_channels: 24, kernel_size: 1, bias: false } } },
        { id: 'bn_pw2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 24 } } },
        { id: 'conv_last', data: { nodeType: 'conv2d', label: 'Conv 24→1280', params: { in_channels: 24, out_channels: 1280, kernel_size: 1, bias: false } } },
        { id: 'bn_last', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 1280 } } },
        { id: 'silu_last', data: { nodeType: 'silu', label: 'SiLU', params: {} } },
        { id: 'gap', data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc', data: { nodeType: 'linear', label: 'FC', params: { in_features: 1280, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 5),
      edges: [
        { source: 'input', target: 'conv1' }, { source: 'conv1', target: 'bn1' }, { source: 'bn1', target: 'silu1' },
        { source: 'silu1', target: 'dw1' }, { source: 'dw1', target: 'bn_dw1' }, { source: 'bn_dw1', target: 'pw1' },
        { source: 'pw1', target: 'bn_pw1' }, { source: 'bn_pw1', target: 'exp2' }, { source: 'exp2', target: 'bn_exp2' },
        { source: 'bn_exp2', target: 'dw2' }, { source: 'dw2', target: 'bn_dw2' }, { source: 'bn_dw2', target: 'pw2' },
        { source: 'pw2', target: 'bn_pw2' }, { source: 'bn_pw2', target: 'conv_last' }, { source: 'conv_last', target: 'bn_last' },
        { source: 'bn_last', target: 'silu_last' }, { source: 'silu_last', target: 'gap' }, { source: 'gap', target: 'fc' },
        { source: 'fc', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: 'result', targetHandle: 'a' }))
    }
  },
  {
    id: 'mlp', name: 'MLP 分类器', description: '多层感知机，适合MNIST/CIFAR入门', category: 'cv', emoji: '⚡',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '784', dtype: 'float32' } } },
        { id: 'flatten', data: { nodeType: 'flatten', label: 'Flatten', params: { start_dim: 1 } } },
        { id: 'fc1', data: { nodeType: 'linear', label: 'FC 512', params: { in_features: 784, out_features: 512, bias: true } } },
        { id: 'relu1', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'drop1', data: { nodeType: 'dropout', label: 'Dropout', params: { p: 0.2 } } },
        { id: 'fc2', data: { nodeType: 'linear', label: 'FC 256', params: { in_features: 512, out_features: 256, bias: true } } },
        { id: 'relu2', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'drop2', data: { nodeType: 'dropout', label: 'Dropout', params: { p: 0.2 } } },
        { id: 'fc3', data: { nodeType: 'linear', label: 'FC 10', params: { in_features: 256, out_features: 10, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 3),
      edges: [
        { source: 'input', target: 'flatten' }, { source: 'flatten', target: 'fc1' }, { source: 'fc1', target: 'relu1' },
        { source: 'relu1', target: 'drop1' }, { source: 'drop1', target: 'fc2' }, { source: 'fc2', target: 'relu2' },
        { source: 'relu2', target: 'drop2' }, { source: 'drop2', target: 'fc3' }, { source: 'fc3', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: 'result', targetHandle: 'a' }))
    }
  },
  {
    id: 'unet', name: 'U-Net', description: '编码器-解码器结构，适合图像分割', category: 'cv', emoji: '🧠',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '1, 572, 572', dtype: 'float32' } } },
        { id: 'enc1_c1', data: { nodeType: 'conv2d', label: 'Enc1 Conv', params: { in_channels: 1, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc1_c2', data: { nodeType: 'conv2d', label: 'Enc1 Conv', params: { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool1', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'enc2_c1', data: { nodeType: 'conv2d', label: 'Enc2 Conv', params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc2_c2', data: { nodeType: 'conv2d', label: 'Enc2 Conv', params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool2', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        { id: 'bot_c1', data: { nodeType: 'conv2d', label: 'Bot Conv', params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'bot_c2', data: { nodeType: 'conv2d', label: 'Bot Conv', params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'up1', data: { nodeType: 'conv2d', label: 'UpConv', params: { in_channels: 256, out_channels: 128, kernel_size: 2, stride: 2, bias: false } } },
        { id: 'dec1_conv', data: { nodeType: 'conv2d', label: 'Dec1 Conv', params: { in_channels: 256, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'up2', data: { nodeType: 'conv2d', label: 'UpConv', params: { in_channels: 128, out_channels: 64, kernel_size: 2, stride: 2, bias: false } } },
        { id: 'dec2_conv', data: { nodeType: 'conv2d', label: 'Dec2 Conv', params: { in_channels: 128, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'final', data: { nodeType: 'conv2d', label: 'Final Conv', params: { in_channels: 64, out_channels: 2, kernel_size: 1, bias: false } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 5),
      edges: [
        { source: 'input', target: 'enc1_c1' }, { source: 'enc1_c1', target: 'enc1_c2' }, { source: 'enc1_c2', target: 'pool1' },
        { source: 'pool1', target: 'enc2_c1' }, { source: 'enc2_c1', target: 'enc2_c2' }, { source: 'enc2_c2', target: 'pool2' },
        { source: 'pool2', target: 'bot_c1' }, { source: 'bot_c1', target: 'bot_c2' }, { source: 'bot_c2', target: 'up1' },
        { source: 'up1', target: 'dec1_conv' }, { source: 'dec1_conv', target: 'up2' }, { source: 'up2', target: 'dec2_conv' },
        { source: 'dec2_conv', target: 'final' }, { source: 'final', target: 'output' }
      ].map(e => ({ ...e, sourceHandle: 'result', targetHandle: 'a' }))
    }
  }
]

// =============================================================================
// Merge backend metadata with local graph definitions
// =============================================================================

function mergeWithLocalGraphs(backendTemplates: Template[]): Template[] {
  const localMap = new Map(LOCAL_TEMPLATES.map(t => [t.id, t.graph]))
  return backendTemplates.map(backendTpl => ({
    ...backendTpl,
    graph: localMap.get(backendTpl.id) ?? { nodes: [], edges: [] },
  }))
}

// =============================================================================
// Store — fetches from backend API, merges with local graph definitions
// =============================================================================

export const useTemplateStore = create<TemplateStore>((set) => ({
  templates: [],
  isLoading: false,
  error: null,

  loadTemplates: async () => {
    set({ isLoading: true, error: null })
    try {
      const res = await fetch('http://localhost:8000/api/templates')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: Template[] = await res.json()
      set({ templates: mergeWithLocalGraphs(data), isLoading: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      console.warn('[useTemplateStore] Backend unavailable, using local templates:', message)
      set({ templates: LOCAL_TEMPLATES, error: message, isLoading: false })
    }
  },
}))

// Re-export local templates for use in templateRegistry.ts
export { LOCAL_TEMPLATES }
