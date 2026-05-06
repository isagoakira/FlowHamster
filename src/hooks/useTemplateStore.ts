import { create } from 'zustand'

// =============================================================================
// Types
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
// Helpers
// =============================================================================

function makeLayout(nodes: any[], numRows: number = 1): any[] {
  const result = []
  const nodesPerRow = Math.ceil(nodes.length / numRows)
  let row = 0, col = 0
  for (const node of nodes) {
    result.push({ ...node, position: { x: 100 + col * 280, y: 80 + row * 130 } })
    col++
    if (col >= nodesPerRow) { col = 0; row++ }
  }
  return result
}

// Edge helper: explicit handles for every connection
function E(src: string, tgt: string, sh = 'result', th = 'a'): TemplateEdge {
  return { source: src, target: tgt, sourceHandle: sh, targetHandle: th }
}

// =============================================================================
// Local template definitions — all handles explicitly specified
// =============================================================================

const LOCAL_TEMPLATES: Template[] = [

  // -------------------------------------------------------------------------
  // ResNet-18
  // -------------------------------------------------------------------------
  {
    id: 'resnet18', name: 'ResNet-18', description: '18层残差网络，含4个残差阶段，适合图像分类', category: 'cv', emoji: '🔬',
    graph: {
      nodes: makeLayout([
        { id: 'input',  data: { nodeType: 'input',       label: 'Input',         params: { shape: '3,224,224', dtype: 'float32' } } },
        { id: 'conv1',  data: { nodeType: 'conv2d',      label: 'Conv2d 7×7',    params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3, bias: false } } },
        { id: 'bn1',    data: { nodeType: 'batchnorm2d', label: 'BN',            params: { num_features: 64 } } },
        { id: 'relu',   data: { nodeType: 'relu',         label: 'ReLU',          params: {} } },
        { id: 'maxpool',data: { nodeType: 'maxpool2d',    label: 'MaxPool 3×3',  params: { kernel_size: 3, stride: 2, padding: 1 } } },
        { id: 'l1_1',   data: { nodeType: 'conv2d',      label: 'BasicBlock 1-1',params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'l1_2',   data: { nodeType: 'conv2d',      label: 'BasicBlock 1-2',params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'add1',   data: { nodeType: 'add',          label: 'Add',           params: {} } },
        { id: 'l2_1',   data: { nodeType: 'conv2d',      label: 'BasicBlock 2-1',params: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'l2_2',   data: { nodeType: 'conv2d',      label: 'BasicBlock 2-2',params: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'down1',  data: { nodeType: 'conv2d',      label: 'Downsample',    params: { in_channels: 64, out_channels: 128, kernel_size: 1, stride: 2 } } },
        { id: 'add2',   data: { nodeType: 'add',          label: 'Add',           params: {} } },
        { id: 'gap',    data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc',     data: { nodeType: 'linear',        label: 'Linear',        params: { in_features: 512, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output',        label: 'Output',        params: {} } },
      ], 4),
      edges: [
        E('input','conv1'), E('conv1','bn1'), E('bn1','relu'), E('relu','maxpool'),
        E('maxpool','l1_1'), E('l1_1','l1_2'), E('l1_2','add1'),
        E('maxpool','add1',  'result','b'),
        E('add1','l2_1'),    E('add1','down1'),
        E('l2_1','l2_2'), E('l2_2','add2'),
        E('down1','add2', 'result','b'),
        E('add2','gap'), E('gap','fc'), E('fc','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // VGG-16
  // -------------------------------------------------------------------------
  {
    id: 'vgg16', name: 'VGG-16', description: '16层VGG网络，3阶段卷积+2层全连接', category: 'cv', emoji: '🔷',
    graph: {
      nodes: makeLayout([
        { id: 'input',  data: { nodeType: 'input',       label: 'Input',         params: { shape: '3,224,224', dtype: 'float32' } } },
        { id: 'c1_1',   data: { nodeType: 'conv2d',      label: 'Conv 64-1',     params: { in_channels: 3,   out_channels: 64,  kernel_size: 3, padding: 1, bias: false } } },
        { id: 'c1_2',   data: { nodeType: 'conv2d',      label: 'Conv 64-2',     params: { in_channels: 64,  out_channels: 64,  kernel_size: 3, padding: 1, bias: false } } },
        { id: 'p1',     data: { nodeType: 'maxpool2d',   label: 'Pool',           params: { kernel_size: 2, stride: 2 } } },
        { id: 'c2_1',   data: { nodeType: 'conv2d',      label: 'Conv 128-1',    params: { in_channels: 64,  out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'c2_2',   data: { nodeType: 'conv2d',      label: 'Conv 128-2',    params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'p2',     data: { nodeType: 'maxpool2d',   label: 'Pool',           params: { kernel_size: 2, stride: 2 } } },
        { id: 'c3_1',   data: { nodeType: 'conv2d',      label: 'Conv 256-1',    params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'c3_2',   data: { nodeType: 'conv2d',      label: 'Conv 256-2',    params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'p3',     data: { nodeType: 'maxpool2d',   label: 'Pool',           params: { kernel_size: 2, stride: 2 } } },
        { id: 'c4_1',   data: { nodeType: 'conv2d',      label: 'Conv 512-1',    params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'c4_2',   data: { nodeType: 'conv2d',      label: 'Conv 512-2',    params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'p4',     data: { nodeType: 'maxpool2d',   label: 'Pool',           params: { kernel_size: 2, stride: 2 } } },
        { id: 'gap',    data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc1',    data: { nodeType: 'linear',        label: 'FC 4096',       params: { in_features: 512, out_features: 4096, bias: true } } },
        { id: 'fc2',    data: { nodeType: 'linear',        label: 'FC 1000',       params: { in_features: 4096, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output',        label: 'Output',        params: {} } },
      ], 4),
      edges: [
        E('input','c1_1'), E('c1_1','c1_2'), E('c1_2','p1'),
        E('p1','c2_1'),   E('c2_1','c2_2'), E('c2_2','p2'),
        E('p2','c3_1'),   E('c3_1','c3_2'), E('c3_2','p3'),
        E('p3','c4_1'),   E('c4_1','c4_2'), E('c4_2','p4'),
        E('p4','gap'),    E('gap','fc1'),   E('fc1','fc2'), E('fc2','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // LeNet-5
  // -------------------------------------------------------------------------
  {
    id: 'lenet5', name: 'LeNet-5', description: '5层LeNet，适合MNIST/CIFAR入门', category: 'cv', emoji: '✏️',
    graph: {
      nodes: makeLayout([
        { id: 'input',  data: { nodeType: 'input',       label: 'Input',   params: { shape: '1,32,32', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d',      label: 'Conv C6', params: { in_channels: 1,  out_channels: 6,  kernel_size: 5, padding: 2, bias: true } } },
        { id: 'relu1',  data: { nodeType: 'relu',         label: 'ReLU',    params: {} } },
        { id: 'pool1', data: { nodeType: 'maxpool2d',   label: 'Pool',    params: { kernel_size: 2, stride: 2 } } },
        { id: 'conv2', data: { nodeType: 'conv2d',      label: 'Conv C16',params: { in_channels: 6,  out_channels: 16, kernel_size: 5, bias: true } } },
        { id: 'relu2',  data: { nodeType: 'relu',         label: 'ReLU',    params: {} } },
        { id: 'pool2', data: { nodeType: 'maxpool2d',   label: 'Pool',    params: { kernel_size: 2, stride: 2 } } },
        { id: 'flat',   data: { nodeType: 'flatten',     label: 'Flatten', params: { start_dim: 1 } } },
        { id: 'fc1',   data: { nodeType: 'linear',       label: 'FC 120',  params: { in_features: 400,  out_features: 120, bias: true } } },
        { id: 'relu3',  data: { nodeType: 'relu',         label: 'ReLU',    params: {} } },
        { id: 'fc2',   data: { nodeType: 'linear',       label: 'FC 84',   params: { in_features: 120,  out_features: 84,  bias: true } } },
        { id: 'relu4',  data: { nodeType: 'relu',         label: 'ReLU',    params: {} } },
        { id: 'fc3',   data: { nodeType: 'linear',       label: 'FC 10',   params: { in_features: 84,   out_features: 10,  bias: true } } },
        { id: 'output',data: { nodeType: 'output',        label: 'Output', params: {} } },
      ], 4),
      edges: [
        E('input','conv1'), E('conv1','relu1'), E('relu1','pool1'),
        E('pool1','conv2'), E('conv2','relu2'), E('relu2','pool2'),
        E('pool2','flat'),  E('flat','fc1'),    E('fc1','relu3'),
        E('relu3','fc2'),   E('fc2','relu4'),   E('relu4','fc3'),
        E('fc3','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // Vision Transformer (ViT) — correct handles
  // -------------------------------------------------------------------------
  {
    id: 'vit', name: 'Vision Transformer', description: '视觉Transformer，Patch Embed + Transformer Encoder + CLS Token', category: 'cv', emoji: '🔮',
    graph: {
      nodes: makeLayout([
        { id: 'input',     data: { nodeType: 'input',             label: 'Input',             params: { shape: '3,224,224', dtype: 'float32' } } },
        { id: 'patch_emb',data: { nodeType: 'conv2d',              label: 'PatchEmbed',        params: { in_channels: 3, out_channels: 768, kernel_size: 16, stride: 16, bias: false } } },
        { id: 'flatten',  data: { nodeType: 'flatten',             label: 'Flatten',           params: { start_dim: 2, end_dim: 3 } } },
        { id: 'permute',  data: { nodeType: 'permute',            label: 'Permute',           params: { dims: '0,2,1' } } },
        { id: 'cls_tok',  data: { nodeType: 'parameter',           label: 'CLS Token',         params: { shape: '1,1,768', trainable: true } } },
        { id: 'concat',   data: { nodeType: 'concat',             label: 'Concat CLS',        params: { dim: 1 } } },
        { id: 'pos_emb',  data: { nodeType: 'parameter',          label: 'Pos Embed',         params: { shape: '1,197,768', trainable: true } } },
        { id: 'add_pos',  data: { nodeType: 'add',                label: 'Add Pos',           params: {} } },
        { id: 'dropout',  data: { nodeType: 'dropout',             label: 'Dropout',           params: { p: 0.1 } } },
        { id: 'encoder',  data: { nodeType: 'transformerencoder',  label: 'TransformerEncoder',params: { embed_dim: 768, num_heads: 12, num_layers: 12, dim_feedforward: 3072, dropout: 0.1 } } },
        { id: 'norm',     data: { nodeType: 'layernorm',           label: 'LayerNorm',         params: { normalized_shape: 768, eps: 1e-6 } } },
        { id: 'cls_slc',  data: { nodeType: 'slice',              label: 'Extract CLS',       params: { start: 0, end: 1, dim: 1 } } },
        { id: 'squeeze',  data: { nodeType: 'squeeze',             label: 'Squeeze',           params: { dim: 1 } } },
        { id: 'head',     data: { nodeType: 'linear',              label: 'Head',              params: { in_features: 768, out_features: 1000, bias: true } } },
        { id: 'output',   data: { nodeType: 'output',             label: 'Output',            params: {} } },
      ], 4),
      edges: [
        E('input','patch_emb'),
        E('patch_emb','flatten'),
        E('flatten','permute'),
        E('permute','concat', 'result', 'in_0'),
        E('cls_tok','concat', 'result', 'in_1'),
        E('concat','add_pos'),
        E('pos_emb','add_pos', 'result', 'b'),
        E('add_pos','dropout'),
        E('dropout','encoder', 'result', 'x'),
        E('encoder','norm'),
        E('norm','cls_slc'),
        E('cls_slc','squeeze'),
        E('squeeze','head'),
        E('head','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // MobileNetV2
  // -------------------------------------------------------------------------
  {
    id: 'mobilenetv2', name: 'MobileNetV2', description: '倒残差结构+深度可分离卷积，适合移动端部署', category: 'cv', emoji: '📱',
    graph: {
      nodes: makeLayout([
        { id: 'input',    data: { nodeType: 'input',          label: 'Input',           params: { shape: '3,224,224', dtype: 'float32' } } },
        { id: 'conv1',   data: { nodeType: 'conv2d',      label: 'Conv 3→32',      params: { in_channels: 3,   out_channels: 32,  kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'bn1',      data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 32 } } },
        { id: 'silu1',    data: { nodeType: 'silu',         label: 'SiLU',            params: {} } },
        { id: 'dw1',      data: { nodeType: 'conv2d',      label: 'DW 32→32',        params: { in_channels: 32,  out_channels: 32,  kernel_size: 3, stride: 1, padding: 1, groups: 32, bias: false } } },
        { id: 'bn_dw1',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 32 } } },
        { id: 'pw1',      data: { nodeType: 'conv2d',      label: 'PW 32→16',        params: { in_channels: 32,  out_channels: 16,  kernel_size: 1, bias: false } } },
        { id: 'bn_pw1',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 16 } } },
        { id: 'exp2',     data: { nodeType: 'conv2d',      label: 'PW 16→96',        params: { in_channels: 16,  out_channels: 96,  kernel_size: 1, bias: false } } },
        { id: 'bn_exp2',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 96 } } },
        { id: 'dw2',      data: { nodeType: 'conv2d',      label: 'DW 96→96',        params: { in_channels: 96,  out_channels: 96,  kernel_size: 3, stride: 2, padding: 1, groups: 96, bias: false } } },
        { id: 'bn_dw2',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 96 } } },
        { id: 'pw2',      data: { nodeType: 'conv2d',      label: 'PW 96→24',        params: { in_channels: 96,  out_channels: 24,  kernel_size: 1, bias: false } } },
        { id: 'bn_pw2',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 24 } } },
        { id: 'conv_last',data: { nodeType: 'conv2d',      label: 'Conv 24→1280',    params: { in_channels: 24,  out_channels: 1280, kernel_size: 1, bias: false } } },
        { id: 'bn_last',  data: { nodeType: 'batchnorm2d', label: 'BN',              params: { num_features: 1280 } } },
        { id: 'silu_last',data: { nodeType: 'silu',         label: 'SiLU',            params: {} } },
        { id: 'gap',      data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool',   params: {} } },
        { id: 'fc',      data: { nodeType: 'linear',        label: 'FC',              params: { in_features: 1280, out_features: 1000, bias: true } } },
        { id: 'output',   data: { nodeType: 'output',       label: 'Output',          params: {} } },
      ], 5),
      edges: [
        E('input','bn1'),    E('bn1','silu1'),
        E('silu1','dw1'),    E('dw1','bn_dw1'), E('bn_dw1','pw1'),
        E('pw1','bn_pw1'),  E('bn_pw1','exp2'),
        E('exp2','bn_exp2'),E('bn_exp2','dw2'),
        E('dw2','bn_dw2'),  E('bn_dw2','pw2'),
        E('pw2','bn_pw2'),  E('bn_pw2','conv_last'),
        E('conv_last','bn_last'), E('bn_last','silu_last'),
        E('silu_last','gap'),  E('gap','fc'), E('fc','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // MLP 分类器
  // -------------------------------------------------------------------------
  {
    id: 'mlp', name: 'MLP 分类器', description: '多层感知机，适合MNIST/CIFAR入门', category: 'cv', emoji: '⚡',
    graph: {
      nodes: makeLayout([
        { id: 'input',  data: { nodeType: 'input',   label: 'Input',    params: { shape: '784', dtype: 'float32' } } },
        { id: 'flat',   data: { nodeType: 'flatten',   label: 'Flatten',  params: { start_dim: 1 } } },
        { id: 'fc1',   data: { nodeType: 'linear',     label: 'FC 512',   params: { in_features: 784,  out_features: 512, bias: true } } },
        { id: 'relu1',  data: { nodeType: 'relu',      label: 'ReLU',     params: {} } },
        { id: 'drop1',  data: { nodeType: 'dropout',    label: 'Dropout',  params: { p: 0.2 } } },
        { id: 'fc2',   data: { nodeType: 'linear',     label: 'FC 256',   params: { in_features: 512,  out_features: 256, bias: true } } },
        { id: 'relu2',  data: { nodeType: 'relu',      label: 'ReLU',     params: {} } },
        { id: 'drop2',  data: { nodeType: 'dropout',    label: 'Dropout',  params: { p: 0.2 } } },
        { id: 'fc3',   data: { nodeType: 'linear',     label: 'FC 10',    params: { in_features: 256,  out_features: 10,  bias: true } } },
        { id: 'output', data: { nodeType: 'output',    label: 'Output',   params: {} } },
      ], 3),
      edges: [
        E('input','flat'),
        E('flat','fc1'),   E('fc1','relu1'), E('relu1','drop1'),
        E('drop1','fc2'), E('fc2','relu2'), E('relu2','drop2'),
        E('drop2','fc3'), E('fc3','output'),
      ]
    }
  },

  // -------------------------------------------------------------------------
  // U-Net — skip connections as Add nodes (not concat)
  // -------------------------------------------------------------------------
  {
    id: 'unet', name: 'U-Net', description: '编码器-解码器结构，适合图像分割', category: 'cv', emoji: '🧠',
    graph: {
      nodes: makeLayout([
        { id: 'input',       data: { nodeType: 'conv2d',    label: 'Enc1 Conv',     params: { in_channels: 1,   out_channels: 64,  kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc1_c2',     data: { nodeType: 'conv2d',    label: 'Enc1 Conv',     params: { in_channels: 64,  out_channels: 64,  kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool1',       data: { nodeType: 'maxpool2d', label: 'Pool',          params: { kernel_size: 2, stride: 2 } } },
        { id: 'enc2_c1',     data: { nodeType: 'conv2d',    label: 'Enc2 Conv',     params: { in_channels: 64,  out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc2_c2',     data: { nodeType: 'conv2d',    label: 'Enc2 Conv',     params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool2',       data: { nodeType: 'maxpool2d', label: 'Pool',          params: { kernel_size: 2, stride: 2 } } },
        { id: 'bot_c1',     data: { nodeType: 'conv2d',    label: 'Bot Conv',       params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'bot_c2',     data: { nodeType: 'conv2d',    label: 'Bot Conv',       params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'up1',         data: { nodeType: 'conv2d',    label: 'UpConv',         params: { in_channels: 256, out_channels: 128, kernel_size: 2, stride: 2, output_padding: 1 } } },
        { id: 'add1',        data: { nodeType: 'add',       label: 'Skip Add 1',     params: {} } },
        { id: 'dec1_conv',  data: { nodeType: 'conv2d',    label: 'Dec1 Conv',     params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'up2',         data: { nodeType: 'conv2d',    label: 'UpConv',         params: { in_channels: 128, out_channels: 64,  kernel_size: 2, stride: 2, output_padding: 1 } } },
        { id: 'add2',        data: { nodeType: 'add',       label: 'Skip Add 2',     params: {} } },
        { id: 'dec2_conv',   data: { nodeType: 'conv2d',    label: 'Dec2 Conv',     params: { in_channels: 64,  out_channels: 64,  kernel_size: 3, padding: 1, bias: false } } },
        { id: 'final',      data: { nodeType: 'conv2d',    label: 'Final Conv',    params: { in_channels: 64,  out_channels: 2,   kernel_size: 1, bias: false } } },
        { id: 'output',     data: { nodeType: 'output',     label: 'Output',         params: {} } },
      ], 5),
      edges: [
        E('input','enc1_c2'),    E('enc1_c2','pool1'),
        E('pool1','enc2_c1'),    E('enc2_c1','enc2_c2'), E('enc2_c2','pool2'),
        E('pool2','bot_c1'),     E('bot_c1','bot_c2'),
        E('bot_c2','up1'),
        E('up1','add1',  'result','a'),
        E('enc2_c2','add1', 'result','b'),
        E('add1','dec1_conv'),
        E('dec1_conv','up2'),
        E('up2','add2',  'result','a'),
        E('enc1_c2','add2', 'result','b'),
        E('add2','dec2_conv'),
        E('dec2_conv','final'), E('final','output'),
      ]
    }
  },
]

// =============================================================================
// Merge backend metadata with local graphs
// =============================================================================

// 后端模板ID → 前端本地模板ID 的确定性映射
const BACKEND_TO_LOCAL_ID: Record<string, string> = {
  'resnet':    'resnet18',
  'vgg':       'vgg16',
  'lenet':     'lenet5',
  'mobilenet': 'mobilenetv2',
}

function mergeWithLocalGraphs(backendTemplates: Template[]): Template[] {
  const localById = new Map(LOCAL_TEMPLATES.map(t => [t.id, t]))
  const localByName = new Map(LOCAL_TEMPLATES.map(t => [t.name, t]))

  return backendTemplates.map(backendTpl => {
    // 1. 确定映射后的本地ID
    const mappedId = BACKEND_TO_LOCAL_ID[backendTpl.id] ?? backendTpl.id
    // 2. 优先用映射后的ID查找，再按名称查找
    const local = localById.get(mappedId) ?? localById.get(backendTpl.id) ?? localByName.get(backendTpl.name)
    if (local) {
      return { ...backendTpl, id: mappedId !== backendTpl.id ? mappedId : backendTpl.id, graph: local.graph }
    }
    return { ...backendTpl, graph: { nodes: [], edges: [] } }
  })
}

// =============================================================================
// Store
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

export { LOCAL_TEMPLATES }