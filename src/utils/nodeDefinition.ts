/**
 * 统一节点类型定义 - Single Source of Truth
 *
 * 所有节点类型的定义都集中在这里，包括：
 * - 节点类型字符串
 * - 显示名称
 * - 参数定义
 * - Handle ID 定义
 * - 是否为复合节点
 * - 复合节点的内部结构
 */

import { HandleIds } from '../constants/handleIds'

// ============================================================
// 节点分类枚举
// ============================================================
export enum NodeCategory {
  INPUT_OUTPUT = 'io',
  CONVOLUTION = 'conv',
  LINEAR = 'linear',
  ACTIVATION = 'activation',
  NORMALIZATION = 'norm',
  POOLING = 'pool',
  ATTENTION = 'attention',
  TENSOR_OP = 'tensor',
  REGULARIZATION = 'reg',
  FFN = 'ffn',
  LOSS = 'loss',
  OPTIMIZER = 'optimizer',
  SCHEDULER = 'scheduler',
  EVALUATION = 'evaluation',
  CUSTOM = 'custom',
}

// ============================================================
// 参数定义
// ============================================================
export interface ParamDefinition {
  type: 'number' | 'string' | 'boolean' | 'select'
  default: number | string | boolean
  description?: string
  options?: { label: string; value: string | number }[]
  min?: number
  max?: number
}

// ============================================================
// Handle 定义
// ============================================================
export interface HandleDefinition {
  id: string
  position: 'left' | 'right' | 'top' | 'bottom'
  label?: string
}

// ============================================================
// 子模块定义（复合节点内部）
// ============================================================
export interface SubModuleDefinition {
  id: string
  type: string
  label: string
  params: Record<string, number | string | boolean>
  sourceInput?: string  // 外部输入映射，如 'x', 'tgt'
}

export interface InternalEdge {
  from: string
  fromPort?: string
  to: string
  toPort?: string
}

// ============================================================
// 节点定义
// ============================================================
export interface NodeDef {
  type: string
  label: string
  category: NodeCategory
  description: string
  params: Record<string, ParamDefinition>
  handles: {
    inputs: HandleDefinition[]
    outputs: HandleDefinition[]
  }
  isComposite: boolean
  internalStructure?: SubModuleDefinition[]
  internalEdges?: InternalEdge[]
  outputVar?: string
}

// ============================================================
// 节点定义映射表 - Single Source of Truth
// ============================================================
export const NODE_DEFINITIONS: Record<string, NodeDef> = {

  // ============================================================
  // Input / Output
  // ============================================================
  input: {
    type: 'input',
    label: 'Input',
    category: NodeCategory.INPUT_OUTPUT,
    description: '数据输入源',
    params: {
      shape: { type: 'string', default: '3,224,224', description: '输入形状' },
      dtype: { type: 'string', default: 'float32', description: '数据类型' },
    },
    handles: {
      inputs: [],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  output: {
    type: 'output',
    label: 'Output',
    category: NodeCategory.INPUT_OUTPUT,
    description: '模型输出 / 损失目标',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [],
    },
    isComposite: false,
  },

  // ============================================================
  // Convolution
  // ============================================================
  conv2d: {
    type: 'conv2d',
    label: 'Conv2d',
    category: NodeCategory.CONVOLUTION,
    description: '2D 卷积',
    params: {
      in_channels: { type: 'number', default: 3, description: '输入通道' },
      out_channels: { type: 'number', default: 64, description: '输出通道' },
      kernel_size: { type: 'number', default: 3, description: '卷积核大小' },
      stride: { type: 'number', default: 1, description: '步长' },
      padding: { type: 'number', default: 1, description: '填充' },
      bias: { type: 'boolean', default: false, description: '是否使用偏置' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  conv1d: {
    type: 'conv1d',
    label: 'Conv1d',
    category: NodeCategory.CONVOLUTION,
    description: '1D 卷积',
    params: {
      in_channels: { type: 'number', default: 64, description: '输入通道' },
      out_channels: { type: 'number', default: 128, description: '输出通道' },
      kernel_size: { type: 'number', default: 3, description: '卷积核大小' },
      stride: { type: 'number', default: 1, description: '步长' },
      padding: { type: 'number', default: 1, description: '填充' },
      bias: { type: 'boolean', default: false, description: '是否使用偏置' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  conv3d: {
    type: 'conv3d',
    label: 'Conv3d',
    category: NodeCategory.CONVOLUTION,
    description: '3D 卷积',
    params: {
      in_channels: { type: 'number', default: 3, description: '输入通道' },
      out_channels: { type: 'number', default: 64, description: '输出通道' },
      kernel_size: { type: 'number', default: 3, description: '卷积核大小' },
      stride: { type: 'number', default: 1, description: '步长' },
      padding: { type: 'number', default: 1, description: '填充' },
      bias: { type: 'boolean', default: false, description: '是否使用偏置' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Linear
  // ============================================================
  linear: {
    type: 'linear',
    label: 'Linear',
    category: NodeCategory.LINEAR,
    description: '全连接层',
    params: {
      in_features: { type: 'number', default: 512, description: '输入特征数' },
      out_features: { type: 'number', default: 256, description: '输出特征数' },
      bias: { type: 'boolean', default: true, description: '是否使用偏置' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  embedding: {
    type: 'embedding',
    label: 'Embedding',
    category: NodeCategory.LINEAR,
    description: 'Embedding 查找',
    params: {
      num_embeddings: { type: 'number', default: 50000, description: '词表大小' },
      embedding_dim: { type: 'number', default: 512, description: '嵌入维度' },
      padding_idx: { type: 'number', default: 0, description: 'padding 索引' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Activation
  // ============================================================
  relu: {
    type: 'relu',
    label: 'ReLU',
    category: NodeCategory.ACTIVATION,
    description: 'ReLU 激活函数',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  gelu: {
    type: 'gelu',
    label: 'GELU',
    category: NodeCategory.ACTIVATION,
    description: 'Gaussian Error Linear Unit',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  silu: {
    type: 'silu',
    label: 'SiLU / Swish',
    category: NodeCategory.ACTIVATION,
    description: 'Sigmoid Linear Unit',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  sigmoid: {
    type: 'sigmoid',
    label: 'Sigmoid',
    category: NodeCategory.ACTIVATION,
    description: 'Sigmoid 激活函数',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  tanh: {
    type: 'tanh',
    label: 'Tanh',
    category: NodeCategory.ACTIVATION,
    description: '双曲正切激活函数',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  leakyrelu: {
    type: 'leakyrelu',
    label: 'LeakyReLU',
    category: NodeCategory.ACTIVATION,
    description: 'Leaky ReLU',
    params: {
      negative_slope: { type: 'number', default: 0.01, description: '负斜率' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Normalization
  // ============================================================
  batchnorm2d: {
    type: 'batchnorm2d',
    label: 'BatchNorm2d',
    category: NodeCategory.NORMALIZATION,
    description: '批归一化',
    params: {
      num_features: { type: 'number', default: 64, description: '特征数' },
      eps: { type: 'number', default: 1e-5, description: 'epsilon' },
      momentum: { type: 'number', default: 0.1, description: '动量' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  layernorm: {
    type: 'layernorm',
    label: 'LayerNorm',
    category: NodeCategory.NORMALIZATION,
    description: '层归一化',
    params: {
      normalized_shape: { type: 'string', default: 'auto', description: '归一化形状' },
      eps: { type: 'number', default: 1e-5, description: 'epsilon' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  groupnorm: {
    type: 'groupnorm',
    label: 'GroupNorm',
    category: NodeCategory.NORMALIZATION,
    description: '组归一化',
    params: {
      num_groups: { type: 'number', default: 32, description: '组数' },
      num_channels: { type: 'number', default: 64, description: '通道数' },
      eps: { type: 'number', default: 1e-5, description: 'epsilon' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Pooling
  // ============================================================
  maxpool2d: {
    type: 'maxpool2d',
    label: 'MaxPool2d',
    category: NodeCategory.POOLING,
    description: '最大池化',
    params: {
      kernel_size: { type: 'number', default: 2, description: '池化核大小' },
      stride: { type: 'number', default: 2, description: '步长' },
      padding: { type: 'number', default: 0, description: '填充' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  avgpool2d: {
    type: 'avgpool2d',
    label: 'AvgPool2d',
    category: NodeCategory.POOLING,
    description: '平均池化',
    params: {
      kernel_size: { type: 'number', default: 2, description: '池化核大小' },
      stride: { type: 'number', default: 2, description: '步长' },
      padding: { type: 'number', default: 0, description: '填充' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  adaptiveavgpool2d: {
    type: 'adaptiveavgpool2d',
    label: 'AdaptiveAvgPool2d',
    category: NodeCategory.POOLING,
    description: '自适应平均池化',
    params: {
      output_size: { type: 'number', default: 1, description: '输出尺寸' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  globalavgpool: {
    type: 'globalavgpool',
    label: 'GlobalAvgPool',
    category: NodeCategory.POOLING,
    description: '全局平均池化',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Tensor Operations
  // ============================================================
  add: {
    type: 'add',
    label: 'Add',
    category: NodeCategory.TENSOR_OP,
    description: '逐元素加法',
    params: {},
    handles: {
      inputs: [
        { id: HandleIds.ADD_A, position: 'top', label: 'A' },
        { id: HandleIds.ADD_B, position: 'left', label: 'B' },
      ],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  mul: {
    type: 'mul',
    label: 'Multiply',
    category: NodeCategory.TENSOR_OP,
    description: '逐元素乘法',
    params: {},
    handles: {
      inputs: [
        { id: HandleIds.ADD_A, position: 'top', label: 'A' },
        { id: HandleIds.ADD_B, position: 'left', label: 'B' },
      ],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  concat: {
    type: 'concat',
    label: 'Concat',
    category: NodeCategory.TENSOR_OP,
    description: '张量拼接',
    params: {
      dim: { type: 'number', default: 1, description: '拼接维度' },
    },
    handles: {
      inputs: [
        { id: HandleIds.CONCAT_IN_0, position: 'left', label: 'in_0' },
        { id: HandleIds.CONCAT_IN_1, position: 'left', label: 'in_1' },
        { id: HandleIds.CONCAT_IN_2, position: 'left', label: 'in_2' },
      ],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  reshape: {
    type: 'reshape',
    label: 'Reshape',
    category: NodeCategory.TENSOR_OP,
    description: '张量重塑',
    params: {
      shape: { type: 'string', default: '-1', description: '目标形状' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  flatten: {
    type: 'flatten',
    label: 'Flatten',
    category: NodeCategory.TENSOR_OP,
    description: '张量展平',
    params: {
      start_dim: { type: 'number', default: 1, description: '起始维度' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  transpose: {
    type: 'transpose',
    label: 'Transpose',
    category: NodeCategory.TENSOR_OP,
    description: '张量转置',
    params: {
      dim0: { type: 'number', default: 0, description: '维度0' },
      dim1: { type: 'number', default: 1, description: '维度1' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  permute: {
    type: 'permute',
    label: 'Permute',
    category: NodeCategory.TENSOR_OP,
    description: '张量维度重排',
    params: {
      dims: { type: 'string', default: '0,2,1', description: '维度顺序' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  split: {
    type: 'split',
    label: 'Split',
    category: NodeCategory.TENSOR_OP,
    description: '张量分割',
    params: {
      split_size: { type: 'number', default: 32, description: '分割大小' },
      dim: { type: 'number', default: 0, description: '分割维度' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  slice: {
    type: 'slice',
    label: 'Slice',
    category: NodeCategory.TENSOR_OP,
    description: '张量切片',
    params: {
      start: { type: 'number', default: 0, description: '起始索引' },
      end: { type: 'number', default: -1, description: '结束索引' },
      step: { type: 'number', default: 1, description: '步长' },
      dim: { type: 'number', default: 1, description: '切片维度' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  squeeze: {
    type: 'squeeze',
    label: 'Squeeze',
    category: NodeCategory.TENSOR_OP,
    description: '移除尺寸为1的维度',
    params: {
      dim: { type: 'number', default: 0, description: '维度' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  expand: {
    type: 'expand',
    label: 'Expand',
    category: NodeCategory.TENSOR_OP,
    description: '扩展张量',
    params: {
      shape: { type: 'string', default: '-1', description: '目标形状' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  constant: {
    type: 'constant',
    label: 'Constant',
    category: NodeCategory.TENSOR_OP,
    description: '创建常量张量',
    params: {
      value: { type: 'string', default: '0', description: '值' },
      shape: { type: 'string', default: '1', description: '形状' },
      trainable: { type: 'boolean', default: false, description: '是否可学习' },
    },
    handles: {
      inputs: [],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  parameter: {
    type: 'parameter',
    label: 'Parameter',
    category: NodeCategory.TENSOR_OP,
    description: '创建可学习参数',
    params: {
      shape: { type: 'string', default: '1,1,768', description: '形状' },
      trainable: { type: 'boolean', default: true, description: '是否可学习' },
    },
    handles: {
      inputs: [],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Regularization
  // ============================================================
  dropout: {
    type: 'dropout',
    label: 'Dropout',
    category: NodeCategory.REGULARIZATION,
    description: 'Dropout 正则化',
    params: {
      p: { type: 'number', default: 0.5, description: '丢弃概率' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  droppath: {
    type: 'droppath',
    label: 'DropPath',
    category: NodeCategory.REGULARIZATION,
    description: '随机深度',
    params: {
      p: { type: 'number', default: 0.1, description: '丢弃概率' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Attention / Transformer
  // ============================================================
  selfattention: {
    type: 'selfattention',
    label: 'SelfAttention',
    category: NodeCategory.ATTENTION,
    description: '自注意力机制',
    params: {
      embed_dim: { type: 'number', default: 512, description: '嵌入维度' },
      num_heads: { type: 'number', default: 8, description: '注意力头数' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'q_proj', type: 'linear', label: 'Q_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'k_proj', type: 'linear', label: 'K_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'v_proj', type: 'linear', label: 'V_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'split_q', type: 'reshape', label: 'SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_q', type: 'transpose', label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
      { id: 'split_k', type: 'reshape', label: 'SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_k', type: 'transpose', label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
      { id: 'split_v', type: 'reshape', label: 'SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_v', type: 'transpose', label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
      { id: 'attn_score', type: 'attn_score', label: 'AttnScore', params: {} },
      { id: 'attn_weight', type: 'attn_weight', label: 'AttnWeight', params: { dim: -1 } },
      { id: 'attn_drop', type: 'dropout', label: 'AttnDrop', params: { p: '${dropout}' } },
      { id: 'attn_apply', type: 'attn_apply', label: 'AttnApply', params: {} },
      { id: 'transpose_o', type: 'transpose', label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
      { id: 'merge_heads', type: 'reshape', label: 'MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
      { id: 'out_proj', type: 'linear', label: 'Out_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
    ],
    internalEdges: [
      { from: 'q_proj', to: 'split_q' },
      { from: 'k_proj', to: 'split_k' },
      { from: 'v_proj', to: 'split_v' },
      { from: 'split_q', to: 'transpose_q' },
      { from: 'split_k', to: 'transpose_k' },
      { from: 'split_v', to: 'transpose_v' },
      { from: 'transpose_q', to: 'attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'transpose_k', to: 'attn_score', toPort: HandleIds.ATTN_K },
      { from: 'attn_score', to: 'attn_weight' },
      { from: 'attn_weight', to: 'attn_drop' },
      { from: 'attn_drop', to: 'attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'transpose_v', to: 'attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'attn_apply', to: 'transpose_o' },
      { from: 'transpose_o', to: 'merge_heads' },
      { from: 'merge_heads', to: 'out_proj' },
      { from: 'out_proj', to: 'drop' },
    ],
    outputVar: 'drop',
  },

  crossattention: {
    type: 'crossattention',
    label: 'CrossAttention',
    category: NodeCategory.ATTENTION,
    description: '交叉注意力机制',
    params: {
      query_dim: { type: 'number', default: 512, description: '查询维度' },
      kv_dim: { type: 'number', default: 512, description: '键值维度' },
      num_heads: { type: 'number', default: 8, description: '注意力头数' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
    },
    handles: {
      inputs: [
        { id: HandleIds.ENCODER_TGT, position: 'top', label: 'query' },
        { id: 'kv', position: 'left', label: 'kv' },
      ],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'q_proj', type: 'linear', label: 'Q_proj', params: { in_features: '${query_dim}', out_features: '${query_dim}' }, sourceInput: 'query' },
      { id: 'k_proj', type: 'linear', label: 'K_proj', params: { in_features: '${kv_dim}', out_features: '${query_dim}' }, sourceInput: 'kv' },
      { id: 'v_proj', type: 'linear', label: 'V_proj', params: { in_features: '${kv_dim}', out_features: '${query_dim}' }, sourceInput: 'kv' },
      { id: 'split_q', type: 'reshape', label: 'SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_q', type: 'transpose', label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
      { id: 'split_k', type: 'reshape', label: 'SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_k', type: 'transpose', label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
      { id: 'split_v', type: 'reshape', label: 'SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_v', type: 'transpose', label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
      { id: 'attn_score', type: 'attn_score', label: 'AttnScore', params: {} },
      { id: 'attn_weight', type: 'attn_weight', label: 'AttnWeight', params: { dim: -1 } },
      { id: 'attn_drop', type: 'dropout', label: 'AttnDrop', params: { p: '${dropout}' } },
      { id: 'attn_apply', type: 'attn_apply', label: 'AttnApply', params: {} },
      { id: 'transpose_o', type: 'transpose', label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
      { id: 'merge_heads', type: 'reshape', label: 'MergeHeads', params: { shape: '(-1, ${query_dim})' } },
      { id: 'out_proj', type: 'linear', label: 'Out_proj', params: { in_features: '${query_dim}', out_features: '${query_dim}' } },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
    ],
    internalEdges: [
      { from: 'q_proj', to: 'split_q' },
      { from: 'k_proj', to: 'split_k' },
      { from: 'v_proj', to: 'split_v' },
      { from: 'split_q', to: 'transpose_q' },
      { from: 'split_k', to: 'transpose_k' },
      { from: 'split_v', to: 'transpose_v' },
      { from: 'transpose_q', to: 'attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'transpose_k', to: 'attn_score', toPort: HandleIds.ATTN_K },
      { from: 'attn_score', to: 'attn_weight' },
      { from: 'attn_weight', to: 'attn_drop' },
      { from: 'attn_drop', to: 'attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'transpose_v', to: 'attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'attn_apply', to: 'transpose_o' },
      { from: 'transpose_o', to: 'merge_heads' },
      { from: 'merge_heads', to: 'out_proj' },
      { from: 'out_proj', to: 'drop' },
    ],
    outputVar: 'drop',
  },

  multiheadattention: {
    type: 'multiheadattention',
    label: 'MultiheadAttention',
    category: NodeCategory.ATTENTION,
    description: '多头注意力机制（完全展开实现）',
    params: {
      embed_dim: { type: 'number', default: 512, description: '嵌入维度' },
      num_heads: { type: 'number', default: 8, description: '注意力头数' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
      bias: { type: 'boolean', default: true, description: '是否使用偏置' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'q_proj', type: 'linear', label: 'Q_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'k_proj', type: 'linear', label: 'K_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'v_proj', type: 'linear', label: 'V_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'split_q', type: 'reshape', label: 'SplitQ', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_q', type: 'transpose', label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
      { id: 'split_k', type: 'reshape', label: 'SplitK', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_k', type: 'transpose', label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
      { id: 'split_v', type: 'reshape', label: 'SplitV', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
      { id: 'transpose_v', type: 'transpose', label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
      { id: 'attn_score', type: 'attn_score', label: 'AttnScore', params: {} },
      { id: 'attn_weight', type: 'attn_weight', label: 'AttnWeight', params: { dim: -1 } },
      { id: 'attn_drop', type: 'dropout', label: 'AttnDrop', params: { p: '${dropout}' } },
      { id: 'attn_apply', type: 'attn_apply', label: 'AttnApply', params: {} },
      { id: 'transpose_o', type: 'transpose', label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
      { id: 'merge_heads', type: 'reshape', label: 'MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
      { id: 'out_proj', type: 'linear', label: 'Out_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
    ],
    internalEdges: [
      { from: 'q_proj', to: 'split_q' },
      { from: 'k_proj', to: 'split_k' },
      { from: 'v_proj', to: 'split_v' },
      { from: 'split_q', to: 'transpose_q' },
      { from: 'split_k', to: 'transpose_k' },
      { from: 'split_v', to: 'transpose_v' },
      { from: 'transpose_q', to: 'attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'transpose_k', to: 'attn_score', toPort: HandleIds.ATTN_K },
      { from: 'attn_score', to: 'attn_weight' },
      { from: 'attn_weight', to: 'attn_drop' },
      { from: 'attn_drop', to: 'attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'transpose_v', to: 'attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'attn_apply', to: 'transpose_o' },
      { from: 'transpose_o', to: 'merge_heads' },
      { from: 'merge_heads', to: 'out_proj' },
      { from: 'out_proj', to: 'drop' },
    ],
    outputVar: 'drop',
  },

  transformerencoder: {
    type: 'transformerencoder',
    label: 'TransformerEncoder',
    category: NodeCategory.ATTENTION,
    description: 'Transformer 编码器（Pre-LN）',
    params: {
      embed_dim: { type: 'number', default: 512, description: '嵌入维度' },
      num_heads: { type: 'number', default: 8, description: '注意力头数' },
      num_layers: { type: 'number', default: 6, description: '层数' },
      dim_feedforward: { type: 'number', default: 2048, description: 'FFN 隐藏层维度' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'sa_q_proj', type: 'linear', label: 'SA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'sa_k_proj', type: 'linear', label: 'SA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'sa_v_proj', type: 'linear', label: 'SA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
      { id: 'sa_split_q', type: 'reshape', label: 'SA_SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'sa_transpose_q', type: 'transpose', label: 'SA_TransposeQ', params: { dim0: 1, dim1: 2 } },
      { id: 'sa_split_k', type: 'reshape', label: 'SA_SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'sa_transpose_k', type: 'transpose', label: 'SA_TransposeK', params: { dim0: 1, dim1: 2 } },
      { id: 'sa_split_v', type: 'reshape', label: 'SA_SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
      { id: 'sa_transpose_v', type: 'transpose', label: 'SA_TransposeV', params: { dim0: 1, dim1: 2 } },
      { id: 'sa_attn_score', type: 'attn_score', label: 'SA_Score', params: {} },
      { id: 'sa_attn_weight', type: 'attn_weight', label: 'SA_Weight', params: { dim: -1 } },
      { id: 'sa_attn_drop', type: 'dropout', label: 'SA_Drop', params: { p: '${dropout}' } },
      { id: 'sa_attn_apply', type: 'attn_apply', label: 'SA_Apply', params: {} },
      { id: 'sa_transpose_o', type: 'transpose', label: 'SA_TransposeO', params: { dim0: 1, dim1: 2 } },
      { id: 'sa_merge_heads', type: 'reshape', label: 'SA_MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
      { id: 'sa_out_proj', type: 'linear', label: 'SA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'sa_drop', type: 'dropout', label: 'SA_FinalDrop', params: { p: '${dropout}' } },
      { id: 'residual_add1', type: 'residual_add', label: 'ResAdd1', params: {} },
      { id: 'norm1', type: 'layernorm', label: 'Norm1', params: { normalized_shape: '${embed_dim}' } },
      { id: 'ffn_linear1', type: 'linear', label: 'FFN1', params: { in_features: '${embed_dim}', out_features: '${dim_feedforward}' } },
      { id: 'ffn_gelu', type: 'gelu', label: 'GELU', params: {} },
      { id: 'ffn_drop', type: 'dropout', label: 'FFN_Drop', params: { p: '${dropout}' } },
      { id: 'ffn_linear2', type: 'linear', label: 'FFN2', params: { in_features: '${dim_feedforward}', out_features: '${embed_dim}' } },
      { id: 'residual_add2', type: 'residual_add', label: 'ResAdd2', params: {} },
      { id: 'norm2', type: 'layernorm', label: 'Norm2', params: { normalized_shape: '${embed_dim}' } },
    ],
    internalEdges: [
      { from: 'sa_q_proj', to: 'sa_split_q' },
      { from: 'sa_split_q', to: 'sa_transpose_q' },
      { from: 'sa_k_proj', to: 'sa_split_k' },
      { from: 'sa_split_k', to: 'sa_transpose_k' },
      { from: 'sa_v_proj', to: 'sa_split_v' },
      { from: 'sa_split_v', to: 'sa_transpose_v' },
      { from: 'sa_transpose_q', to: 'sa_attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'sa_transpose_k', to: 'sa_attn_score', toPort: HandleIds.ATTN_K },
      { from: 'sa_attn_score', to: 'sa_attn_weight' },
      { from: 'sa_attn_weight', to: 'sa_attn_drop' },
      { from: 'sa_attn_drop', to: 'sa_attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'sa_transpose_v', to: 'sa_attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'sa_attn_apply', to: 'sa_transpose_o' },
      { from: 'sa_transpose_o', to: 'sa_merge_heads' },
      { from: 'sa_merge_heads', to: 'sa_out_proj' },
      { from: 'sa_out_proj', to: 'sa_drop' },
      { from: 'sa_drop', to: 'residual_add1', toPort: 'sublayer' },
      { from: 'residual_add1', to: 'norm1' },
      { from: 'norm1', to: 'ffn_linear1' },
      { from: 'ffn_linear1', to: 'ffn_gelu' },
      { from: 'ffn_gelu', to: 'ffn_drop' },
      { from: 'ffn_drop', to: 'ffn_linear2' },
      { from: 'norm1', to: 'residual_add2', toPort: 'x' },
      { from: 'ffn_linear2', to: 'residual_add2', toPort: 'sublayer' },
      { from: 'residual_add2', to: 'norm2' },
    ],
    outputVar: 'norm2',
  },

  transformerdecoder: {
    type: 'transformerdecoder',
    label: 'TransformerDecoder',
    category: NodeCategory.ATTENTION,
    description: 'Transformer 解码器（Pre-LN）',
    params: {
      embed_dim: { type: 'number', default: 512, description: '嵌入维度' },
      num_heads: { type: 'number', default: 8, description: '注意力头数' },
      num_layers: { type: 'number', default: 6, description: '层数' },
      dim_feedforward: { type: 'number', default: 2048, description: 'FFN 隐藏层维度' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
    },
    handles: {
      inputs: [
        { id: HandleIds.ENCODER_TGT, position: 'top', label: 'tgt' },
        { id: HandleIds.ENCODER_MEM, position: 'left', label: 'memory' },
      ],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'sa_q_proj', type: 'linear', label: 'SA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
      { id: 'sa_k_proj', type: 'linear', label: 'SA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
      { id: 'sa_v_proj', type: 'linear', label: 'SA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
      { id: 'sa_attn_score', type: 'attn_score', label: 'SA_Score', params: {} },
      { id: 'sa_attn_weight', type: 'attn_weight', label: 'SA_Weight', params: { dim: -1 } },
      { id: 'sa_attn_drop', type: 'dropout', label: 'SA_Drop', params: { p: '${dropout}' } },
      { id: 'sa_attn_apply', type: 'attn_apply', label: 'SA_Apply', params: {} },
      { id: 'sa_out_proj', type: 'linear', label: 'SA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'sa_drop', type: 'dropout', label: 'SA_FinalDrop', params: { p: '${dropout}' } },
      { id: 'res_add1', type: 'residual_add', label: 'ResAdd1', params: {} },
      { id: 'norm1', type: 'layernorm', label: 'Norm1', params: { normalized_shape: '${embed_dim}' } },
      { id: 'ca_q_proj', type: 'linear', label: 'CA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'ca_k_proj', type: 'linear', label: 'CA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'memory' },
      { id: 'ca_v_proj', type: 'linear', label: 'CA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'memory' },
      { id: 'ca_attn_score', type: 'attn_score', label: 'CA_Score', params: {} },
      { id: 'ca_attn_weight', type: 'attn_weight', label: 'CA_Weight', params: { dim: -1 } },
      { id: 'ca_attn_drop', type: 'dropout', label: 'CA_Drop', params: { p: '${dropout}' } },
      { id: 'ca_attn_apply', type: 'attn_apply', label: 'CA_Apply', params: {} },
      { id: 'ca_out_proj', type: 'linear', label: 'CA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
      { id: 'ca_drop', type: 'dropout', label: 'CA_FinalDrop', params: { p: '${dropout}' } },
      { id: 'res_add2', type: 'residual_add', label: 'ResAdd2', params: {} },
      { id: 'norm2', type: 'layernorm', label: 'Norm2', params: { normalized_shape: '${embed_dim}' } },
      { id: 'ffn_linear1', type: 'linear', label: 'FFN1', params: { in_features: '${embed_dim}', out_features: '${dim_feedforward}' } },
      { id: 'ffn_gelu', type: 'gelu', label: 'GELU', params: {} },
      { id: 'ffn_drop', type: 'dropout', label: 'FFN_Drop', params: { p: '${dropout}' } },
      { id: 'ffn_linear2', type: 'linear', label: 'FFN2', params: { in_features: '${dim_feedforward}', out_features: '${embed_dim}' } },
      { id: 'res_add3', type: 'residual_add', label: 'ResAdd3', params: {} },
      { id: 'norm3', type: 'layernorm', label: 'Norm3', params: { normalized_shape: '${embed_dim}' } },
    ],
    internalEdges: [
      { from: 'sa_q_proj', to: 'sa_attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'sa_k_proj', to: 'sa_attn_score', toPort: HandleIds.ATTN_K },
      { from: 'sa_v_proj', to: 'sa_attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'sa_attn_score', to: 'sa_attn_weight' },
      { from: 'sa_attn_weight', to: 'sa_attn_drop' },
      { from: 'sa_attn_drop', to: 'sa_attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'sa_attn_apply', to: 'sa_out_proj' },
      { from: 'sa_out_proj', to: 'sa_drop' },
      { from: 'sa_drop', to: 'res_add1', toPort: 'sublayer' },
      { from: 'res_add1', to: 'norm1' },
      { from: 'norm1', to: 'ca_q_proj' },
      { from: 'ca_q_proj', to: 'ca_attn_score', toPort: HandleIds.ATTN_Q },
      { from: 'ca_k_proj', to: 'ca_attn_score', toPort: HandleIds.ATTN_K },
      { from: 'ca_v_proj', to: 'ca_attn_apply', toPort: HandleIds.ATTN_V },
      { from: 'ca_attn_score', to: 'ca_attn_weight' },
      { from: 'ca_attn_weight', to: 'ca_attn_drop' },
      { from: 'ca_attn_drop', to: 'ca_attn_apply', toPort: HandleIds.ATTN_WEIGHT },
      { from: 'ca_attn_apply', to: 'ca_out_proj' },
      { from: 'ca_out_proj', to: 'ca_drop' },
      { from: 'ca_drop', to: 'res_add2', toPort: 'sublayer' },
      { from: 'res_add2', to: 'norm2' },
      { from: 'norm2', to: 'ffn_linear1' },
      { from: 'ffn_linear1', to: 'ffn_gelu' },
      { from: 'ffn_gelu', to: 'ffn_drop' },
      { from: 'ffn_drop', to: 'ffn_linear2' },
      { from: 'ffn_linear2', to: 'res_add3', toPort: 'sublayer' },
      { from: 'res_add3', to: 'norm3' },
    ],
    outputVar: 'norm3',
  },

  mamba: {
    type: 'mamba',
    label: 'Mamba-2',
    category: NodeCategory.ATTENTION,
    description: 'Mamba-2 状态空间模型 (基于选择性 SSM，论文 https://arxiv.org/abs/2405.xxxxx)',
    params: {
      d_model: { type: 'number', default: 512, description: '模型维度 D' },
      d_state: { type: 'number', default: 16, description: '状态维度 N (通常 16 或 64)' },
      d_conv: { type: 'number', default: 4, description: '卷积核大小' },
      expand: { type: 'number', default: 2, description: '扩展因子 (d_inner = expand * d_model)' },
      dt_rank: { type: 'number', default: 0, description: 'dt 投影维度 (0=auto)' },
      n_layers: { type: 'number', default: 1, description: 'Mamba 层数' },
      dropout: { type: 'number', default: 0.0, description: 'Dropout 概率' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'in_proj', type: 'linear', label: 'InputProj', sourceInput: 'x', params: { in_features: '${d_model}', out_features: '${d_model * expand}' } },
      { id: 'conv1d', type: 'conv1d', label: 'Conv1D', params: { in_channels: '${d_model * expand}', out_channels: '${d_model * expand}', kernel_size: '${d_conv}', padding: '${d_conv - 1}' } },
      { id: 'act', type: 'silu', label: 'SiLU', params: {} },
      { id: 'ssm', type: 'linear', label: 'SSM', params: { in_features: '${d_model * expand}', out_features: '${d_model * expand}' } },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
      { id: 'out_proj', type: 'linear', label: 'OutputProj', params: { in_features: '${d_model * expand}', out_features: '${d_model}' } },
    ],
    internalEdges: [
      { from: 'in_proj', to: 'conv1d' },
      { from: 'conv1d', to: 'act' },
      { from: 'act', to: 'ssm' },
      { from: 'ssm', to: 'drop' },
      { from: 'drop', to: 'out_proj' },
    ],
    outputVar: 'out_proj',
  },

  // ============================================================
  // FFN / MLP
  // ============================================================
  ffn: {
    type: 'ffn',
    label: 'FFN',
    category: NodeCategory.FFN,
    description: '前馈网络（Transformer 风格）',
    params: {
      dim: { type: 'number', default: 512, description: '输入维度' },
      hidden_dim: { type: 'number', default: 2048, description: '隐藏层维度' },
      dropout: { type: 'number', default: 0.1, description: 'dropout' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'w1', type: 'linear', label: 'W1', params: { in_features: '${dim}', out_features: '${hidden_dim}' }, sourceInput: 'x' },
      { id: 'act', type: 'gelu', label: 'GELU', params: {} },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
      { id: 'w2', type: 'linear', label: 'W2', params: { in_features: '${hidden_dim}', out_features: '${dim}' } },
    ],
    internalEdges: [
      { from: 'w1', to: 'act' },
      { from: 'act', to: 'drop' },
      { from: 'drop', to: 'w2' },
    ],
    outputVar: 'w2',
  },

  mlp: {
    type: 'mlp',
    label: 'MLP',
    category: NodeCategory.FFN,
    description: '多层感知机',
    params: {
      in_features: { type: 'number', default: 784, description: '输入特征数' },
      hidden_features: { type: 'number', default: 256, description: '隐藏层特征数' },
      out_features: { type: 'number', default: 10, description: '输出特征数' },
      dropout: { type: 'number', default: 0.0, description: 'dropout' },
    },
    handles: {
      inputs: [{ id: HandleIds.ENCODER_X, position: 'left', label: 'x' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: true,
    internalStructure: [
      { id: 'fc1', type: 'linear', label: 'FC1', params: { in_features: '${in_features}', out_features: '${hidden_features}' }, sourceInput: 'x' },
      { id: 'act', type: 'relu', label: 'ReLU', params: {} },
      { id: 'drop', type: 'dropout', label: 'Dropout', params: { p: '${dropout}' } },
      { id: 'fc2', type: 'linear', label: 'FC2', params: { in_features: '${hidden_features}', out_features: '${out_features}' } },
    ],
    internalEdges: [
      { from: 'fc1', to: 'act' },
      { from: 'act', to: 'drop' },
      { from: 'drop', to: 'fc2' },
    ],
    outputVar: 'fc2',
  },

  // ============================================================
  // Loss Functions
  // ============================================================
  crossentropyloss: {
    type: 'crossentropyloss',
    label: 'CrossEntropyLoss',
    category: NodeCategory.LOSS,
    description: '交叉熵损失',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  mseloss: {
    type: 'mseloss',
    label: 'MSELoss',
    category: NodeCategory.LOSS,
    description: '均方误差损失',
    params: {},
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  focalloss: {
    type: 'focalloss',
    label: 'FocalLoss',
    category: NodeCategory.LOSS,
    description: 'Focal 损失（用于不平衡分类）',
    params: {
      alpha: { type: 'number', default: 0.25, description: 'alpha' },
      gamma: { type: 'number', default: 2.0, description: 'gamma' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  labelsmoothing: {
    type: 'labelsmoothing',
    label: 'LabelSmoothing',
    category: NodeCategory.LOSS,
    description: '标签平滑损失',
    params: {
      smoothing: { type: 'number', default: 0.1, description: '平滑因子' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Optimizers
  // ============================================================
  adam: {
    type: 'adam',
    label: 'Adam',
    category: NodeCategory.OPTIMIZER,
    description: 'Adam 优化器',
    params: {
      lr: { type: 'number', default: 0.001, description: '学习率' },
      betas: { type: 'string', default: '0.9,0.999', description: 'beta 参数' },
      eps: { type: 'number', default: 1e-8, description: 'epsilon' },
      weight_decay: { type: 'number', default: 0, description: '权重衰减' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  adamw: {
    type: 'adamw',
    label: 'AdamW',
    category: NodeCategory.OPTIMIZER,
    description: 'Adam with Weight Decay',
    params: {
      lr: { type: 'number', default: 0.001, description: '学习率' },
      betas: { type: 'string', default: '0.9,0.999', description: 'beta 参数' },
      eps: { type: 'number', default: 1e-8, description: 'epsilon' },
      weight_decay: { type: 'number', default: 0.01, description: '权重衰减' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  sgd: {
    type: 'sgd',
    label: 'SGD',
    category: NodeCategory.OPTIMIZER,
    description: '随机梯度下降',
    params: {
      lr: { type: 'number', default: 0.01, description: '学习率' },
      momentum: { type: 'number', default: 0.9, description: '动量' },
      weight_decay: { type: 'number', default: 0, description: '权重衰减' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  rmsprop: {
    type: 'rmsprop',
    label: 'RMSprop',
    category: NodeCategory.OPTIMIZER,
    description: 'RMSprop 优化器',
    params: {
      lr: { type: 'number', default: 0.001, description: '学习率' },
      alpha: { type: 'number', default: 0.99, description: 'alpha' },
      eps: { type: 'number', default: 1e-8, description: 'epsilon' },
      weight_decay: { type: 'number', default: 0, description: '权重衰减' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Schedulers
  // ============================================================
  steplr: {
    type: 'steplr',
    label: 'StepLR',
    category: NodeCategory.SCHEDULER,
    description: '阶梯学习率衰减',
    params: {
      step_size: { type: 'number', default: 10, description: '步长' },
      gamma: { type: 'number', default: 0.1, description: '衰减因子' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  cosineannealinglr: {
    type: 'cosineannealinglr',
    label: 'CosineAnnealing',
    category: NodeCategory.SCHEDULER,
    description: '余弦退火学习率调度器',
    params: {
      T_max: { type: 'number', default: 50, description: '最大迭代次数' },
      eta_min: { type: 'number', default: 1e-6, description: '最小学习率' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  reducelronplateau: {
    type: 'reducelronplateau',
    label: 'ReduceLROnPlateau',
    category: NodeCategory.SCHEDULER,
    description: '指标 plateau 时降低学习率',
    params: {
      mode: { type: 'string', default: 'min', description: '模式' },
      factor: { type: 'number', default: 0.1, description: '衰减因子' },
      patience: { type: 'number', default: 5, description: '耐心值' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Evaluation
  // ============================================================
  accuracy: {
    type: 'accuracy',
    label: 'Accuracy',
    category: NodeCategory.EVALUATION,
    description: '分类准确率',
    params: {
      metric_type: { type: 'string', default: 'accuracy', description: '指标类型' },
      average: { type: 'string', default: 'macro', description: '平均方式' },
      top_k: { type: 'number', default: 1, description: 'top k' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Custom
  // ============================================================
  custom: {
    type: 'custom',
    label: 'CustomModule',
    category: NodeCategory.CUSTOM,
    description: '自定义 PyTorch 模块',
    params: {
      custom_code: { type: 'string', default: '', description: '自定义代码' },
    },
    handles: {
      inputs: [{ id: HandleIds.INPUT, position: 'left', label: 'input' }],
      outputs: [{ id: HandleIds.OUTPUT, position: 'right', label: 'output' }],
    },
    isComposite: false,
  },

  // ============================================================
  // Group
  // ============================================================
  group: {
    type: 'group',
    label: 'Group',
    category: NodeCategory.CUSTOM,
    description: '节点分组',
    params: {},
    handles: {
      inputs: [],
      outputs: [],
    },
    isComposite: false,
  },
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 获取节点定义
 */
export function getNodeDef(type: string): NodeDef | undefined {
  return NODE_DEFINITIONS[type]
}

/**
 * 检查节点类型是否存在
 */
export function nodeTypeExists(type: string): boolean {
  return type in NODE_DEFINITIONS
}

/**
 * 检查 Handle ID 是否有效
 */
export function isValidHandleId(handleId: string, nodeType: string): boolean {
  const def = NODE_DEFINITIONS[nodeType]
  if (!def) return false

  const validInputs = def.handles.inputs.map(h => h.id)
  const validOutputs = def.handles.outputs.map(h => h.id)

  return validInputs.includes(handleId) || validOutputs.includes(handleId)
}

/**
 * 获取所有节点类型列表
 */
export function getAllNodeTypes(): string[] {
  return Object.keys(NODE_DEFINITIONS)
}

/**
 * 按分类获取节点类型
 */
export function getNodeTypesByCategory(category: NodeCategory): string[] {
  return Object.entries(NODE_DEFINITIONS)
    .filter(([_, def]) => def.category === category)
    .map(([type, _]) => type)
}
