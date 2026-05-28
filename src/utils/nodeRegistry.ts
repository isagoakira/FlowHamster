import { NodeType } from '../types/graph'

/**
 * Sub-module type for composite nodes.
 * ALL types must be directly expandable to executable Python/PyTorch code.
 * No black-box modules like nn.MultiheadAttention - they must be expanded.
 */
export type SubModuleType =
  // === 基础层 ===
  | 'conv1d' | 'conv2d' | 'conv3d'
  | 'linear' | 'embedding'
  // === 归一化层 ===
  | 'layernorm' | 'batchnorm2d' | 'groupnorm' | 'instancenorm'
  // === 激活函数 ===
  | 'relu' | 'gelu' | 'silu' | 'sigmoid' | 'tanh' | 'leakyrelu'
  // === Dropout ===
  | 'dropout' | 'droppath'
  // === 池化层 ===
  | 'maxpool2d' | 'avgpool2d' | 'adaptiveavgpool2d' | 'globalavgpool'
  // === 张量操作 ===
  | 'softmax' | 'flatten' | 'reshape' | 'view'
  | 'transpose' | 'permute' | 'squeeze' | 'expand'
  | 'cat' | 'stack' | 'add' | 'mul' | 'matmul' | 'div' | 'sqrt'
  // === Attention 专用投影层 ===
  | 'q_proj' | 'k_proj' | 'v_proj' | 'out_proj'
  // === Attention 计算操作 ===
  | 'attn_score'       // QK^T / sqrt(d_k)
  | 'attn_weight'       // softmax(scores, dim)
  | 'attn_apply'       // attn_weights @ V
  // === 残差连接 ===
  | 'residual_add'     // x + sublayer_output
  // === 多头分割/合并 ===
  | 'split_heads'      // (B, N, C) -> (B, num_heads, N, head_dim)
  | 'merge_heads'       // (B, num_heads, N, head_dim) -> (B, N, C)
  // === FFN 子模块 ===
  | 'ffn_linear1' | 'ffn_linear2'
  | 'ffn_gelu' | 'ffn_drop'
  // === 特殊操作 ===
  | 'constant'         // 创建常量 (如 scale)
  | 'parameter'        // 可学习参数
  // F.scaled_dot_product_attention (当 PyTorch 可用时)
  | 'scaledotattn'

export interface SubModulePort {
  name: string
  description?: string
}

export interface SubModule {
  id: string
  type: SubModuleType
  label: string
  params: Record<string, number | string | boolean>
  // Input ports (for submodules that need multiple inputs like attention)
  inputs?: SubModulePort[]
  // Output port name
  output?: string
  // External input source: for submodules that receive composite node's external input
  // e.g., 'x' for SelfAttention (all QKV from same input), or 'query'/'kv' for CrossAttention
  sourceInput?: string
}

export interface InternalEdge {
  from: string           // source submodule id
  fromPort?: string      // source output port, defaults to 'result'
  to: string             // target submodule id
  toPort?: string        // target input port, defaults based on submodule type
}

export interface CompositeNodeDefinition extends NodeDefinition {
  isComposite: true
  internalStructure: SubModule[]
  internalEdges: InternalEdge[]
  outputVar: string
  // Meta information for visualization
  metaInfo?: {
    isStack?: boolean
    stackSize?: string  // template string like '${num_layers}'
    previewType?: 'single_layer' | 'full_stack' | 'abstract'
  }
}

export interface NodeCategory {
  label: string
  nodes: NodeDefinition[]
}

export interface NodeDefinition {
  type: NodeType
  label: string
  description: string
  defaultParams: Record<string, number | string | boolean>
  outputType: string
  // Optional fields for composite nodes
  isComposite?: boolean
  internalStructure?: SubModule[]
  internalEdges?: InternalEdge[]
  outputVar?: string
  // Meta information for composite node visualization
  metaInfo?: {
    isStack?: boolean
    stackSize?: string
    previewType?: 'single_layer' | 'full_stack' | 'abstract'
  }
}

export const NODE_REGISTRY: NodeCategory[] = [
  {
    label: 'Input / Output',
    nodes: [
      {
        type: 'input',
        label: 'Input',
        description: 'Data input source',
        defaultParams: { shape: '3,224,224', dtype: 'float32' },
        outputType: 'Tensor',
      },
      {
        type: 'output',
        label: 'Output',
        description: 'Model output / loss target',
        defaultParams: {},
        outputType: 'Tensor',
      },
    ],
  },
  {
    label: 'Convolution / Linear',
    nodes: [
      {
        type: 'conv2d',
        label: 'Conv2d',
        description: '2D convolution',
        defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false },
        outputType: 'Tensor[N,C,H,W]',
      },
      {
        type: 'conv1d',
        label: 'Conv1d',
        description: '1D convolution',
        defaultParams: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, bias: false },
        outputType: 'Tensor[N,C,L]',
      },
      {
        type: 'conv3d',
        label: 'Conv3d',
        description: '3D convolution',
        defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false },
        outputType: 'Tensor[N,C,D,H,W]',
      },
      {
        type: 'linear',
        label: 'Linear',
        description: 'Fully connected layer',
        defaultParams: { in_features: 512, out_features: 256, bias: true },
        outputType: 'Tensor[N,in_features]',
      },
      {
        type: 'embedding',
        label: 'Embedding',
        description: 'Embedding lookup',
        defaultParams: { num_embeddings: 50000, embedding_dim: 512, padding_idx: 0 },
        outputType: 'Tensor[N,L,embedding_dim]',
      },
    ],
  },
  {
    label: 'Activation Functions',
    nodes: [
      { type: 'relu', label: 'ReLU', description: 'ReLU activation', defaultParams: {}, outputType: 'Tensor' },
      { type: 'gelu', label: 'GELU', description: 'Gaussian Error Linear Unit', defaultParams: {}, outputType: 'Tensor' },
      { type: 'silu', label: 'SiLU / Swish', description: 'Sigmoid Linear Unit', defaultParams: {}, outputType: 'Tensor' },
      { type: 'sigmoid', label: 'Sigmoid', description: 'Sigmoid activation', defaultParams: {}, outputType: 'Tensor' },
      { type: 'tanh', label: 'Tanh', description: 'Hyperbolic tangent', defaultParams: {}, outputType: 'Tensor' },
      { type: 'leakyrelu', label: 'LeakyReLU', description: 'Leaky ReLU', defaultParams: { negative_slope: 0.01 }, outputType: 'Tensor' },
    ],
  },
  {
    label: 'Normalization',
    nodes: [
      { type: 'batchnorm2d', label: 'BatchNorm2d', description: 'Batch normalization', defaultParams: { num_features: 64, eps: 1e-5, momentum: 0.1 }, outputType: 'Tensor' },
      { type: 'layernorm', label: 'LayerNorm', description: 'Layer normalization', defaultParams: { normalized_shape: 'auto', eps: 1e-5 }, outputType: 'Tensor' },
      { type: 'groupnorm', label: 'GroupNorm', description: 'Group normalization', defaultParams: { num_groups: 32, num_channels: 64, eps: 1e-5 }, outputType: 'Tensor' },
    ],
  },
  {
    label: 'Pooling',
    nodes: [
      { type: 'maxpool2d', label: 'MaxPool2d', description: 'Max pooling', defaultParams: { kernel_size: 2, stride: 2, padding: 0 }, outputType: 'Tensor' },
      { type: 'avgpool2d', label: 'AvgPool2d', description: 'Average pooling', defaultParams: { kernel_size: 2, stride: 2, padding: 0 }, outputType: 'Tensor' },
      { type: 'adaptiveavgpool2d', label: 'AdaptiveAvgPool2d', description: 'Adaptive avg pooling', defaultParams: { output_size: 1 }, outputType: 'Tensor' },
      { type: 'globalavgpool', label: 'GlobalAvgPool', description: 'Global average pooling', defaultParams: {}, outputType: 'Tensor' },
    ],
  },
  {
    label: 'Attention / Transformer',
    nodes: [
      {
        type: 'selfattention',
        label: 'SelfAttention',
        description: 'Scaled dot-product self-attention: softmax(QK^T / sqrt(d)) * V',
        defaultParams: { embed_dim: 512, num_heads: 8, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: true as const,
        // Complete PyTorch implementation using only basic operations.
        // Forward: QKV_proj -> split_heads -> attn_score -> attn_weight -> attn_apply -> merge_heads -> out_proj -> drop
        internalStructure: [
          // QKV Projections (all receive external input 'x')
          { id: 'q_proj', type: 'linear' as const, label: 'Q_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'k_proj', type: 'linear' as const, label: 'K_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'v_proj', type: 'linear' as const, label: 'V_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          // Split into multi-head: (B, N, C) -> (B, num_heads, N, head_dim)
          // Use -1 for dynamic dimensions (B, N) since they depend on input shape
          { id: 'split_q', type: 'reshape' as const, label: 'SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_q', type: 'transpose' as const, label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
          { id: 'split_k', type: 'reshape' as const, label: 'SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_k', type: 'transpose' as const, label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
          { id: 'split_v', type: 'reshape' as const, label: 'SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_v', type: 'transpose' as const, label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
          // Attention computation: scores = QK^T / sqrt(d_k)
          { id: 'attn_score', type: 'attn_score' as const, label: 'AttnScore', params: {} },
          // Softmax
          { id: 'attn_weight', type: 'attn_weight' as const, label: 'AttnWeight', params: { dim: -1 } },
          // Dropout on attention weights
          { id: 'attn_drop', type: 'dropout' as const, label: 'AttnDrop', params: { p: '${dropout}' } },
          // Apply attention: attn_weights @ V
          { id: 'attn_apply', type: 'attn_apply' as const, label: 'AttnApply', params: {} },
          // Merge heads: (B, num_heads, N, head_dim) -> (B, N, C)
          { id: 'transpose_o', type: 'transpose' as const, label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
          { id: 'merge_heads', type: 'reshape' as const, label: 'MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
          // Output projection
          { id: 'out_proj', type: 'linear' as const, label: 'Out_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
          // Final dropout
          { id: 'drop', type: 'dropout' as const, label: 'Dropout', params: { p: '${dropout}' } },
        ],
        internalEdges: [
          // QKV projections
          { from: 'q_proj', to: 'split_q' },
          { from: 'k_proj', to: 'split_k' },
          { from: 'v_proj', to: 'split_v' },
          // Reshape and transpose for Q
          { from: 'split_q', to: 'transpose_q' },
          // Reshape and transpose for K
          { from: 'split_k', to: 'transpose_k' },
          // Reshape and transpose for V
          { from: 'split_v', to: 'transpose_v' },
          // Attention score: QK^T / sqrt(d_k)
          { from: 'transpose_q', to: 'attn_score', toPort: 'q' },
          { from: 'transpose_k', to: 'attn_score', toPort: 'k' },
          // Softmax
          { from: 'attn_score', to: 'attn_weight' },
          // Dropout on weights
          { from: 'attn_weight', to: 'attn_drop' },
          // Apply attention
          { from: 'attn_drop', to: 'attn_apply', toPort: 'weight' },
          { from: 'transpose_v', to: 'attn_apply', toPort: 'v' },
          // Merge heads
          { from: 'attn_apply', to: 'transpose_o' },
          { from: 'transpose_o', to: 'merge_heads' },
          // Output projection
          { from: 'merge_heads', to: 'out_proj' },
          { from: 'out_proj', to: 'drop' },
        ],
        outputVar: 'drop',
      },
      {
        type: 'crossattention',
        label: 'CrossAttention',
        description: 'Cross-attention: Q from target, K,V from source',
        defaultParams: { query_dim: 512, kv_dim: 512, num_heads: 8, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: true as const,
        // Complete PyTorch implementation: Q_proj(query) x K,V_proj(kv)
        internalStructure: [
          // Q projection (from query input)
          { id: 'q_proj', type: 'linear' as const, label: 'Q_proj', params: { in_features: '${query_dim}', out_features: '${query_dim}' }, sourceInput: 'query' },
          // KV projections (from kv input)
          { id: 'k_proj', type: 'linear' as const, label: 'K_proj', params: { in_features: '${kv_dim}', out_features: '${query_dim}' }, sourceInput: 'kv' },
          { id: 'v_proj', type: 'linear' as const, label: 'V_proj', params: { in_features: '${kv_dim}', out_features: '${query_dim}' }, sourceInput: 'kv' },
          // Split and transpose for multi-head
          // T = target sequence length (from query), S = source sequence length (from kv)
          { id: 'split_q', type: 'reshape' as const, label: 'SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_q', type: 'transpose' as const, label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
          { id: 'split_k', type: 'reshape' as const, label: 'SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_k', type: 'transpose' as const, label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
          { id: 'split_v', type: 'reshape' as const, label: 'SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_v', type: 'transpose' as const, label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
          // Attention
          { id: 'attn_score', type: 'attn_score' as const, label: 'AttnScore', params: {} },
          { id: 'attn_weight', type: 'attn_weight' as const, label: 'AttnWeight', params: { dim: -1 } },
          { id: 'attn_drop', type: 'dropout' as const, label: 'AttnDrop', params: { p: '${dropout}' } },
          { id: 'attn_apply', type: 'attn_apply' as const, label: 'AttnApply', params: {} },
          { id: 'transpose_o', type: 'transpose' as const, label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
          { id: 'merge_heads', type: 'reshape' as const, label: 'MergeHeads', params: { shape: '(-1, ${query_dim})' } },
          { id: 'out_proj', type: 'linear' as const, label: 'Out_proj', params: { in_features: '${query_dim}', out_features: '${query_dim}' } },
          { id: 'drop', type: 'dropout' as const, label: 'Dropout', params: { p: '${dropout}' } },
        ],
        internalEdges: [
          { from: 'q_proj', to: 'split_q' },
          { from: 'k_proj', to: 'split_k' },
          { from: 'v_proj', to: 'split_v' },
          { from: 'split_q', to: 'transpose_q' },
          { from: 'split_k', to: 'transpose_k' },
          { from: 'split_v', to: 'transpose_v' },
          { from: 'transpose_q', to: 'attn_score', toPort: 'q' },
          { from: 'transpose_k', to: 'attn_score', toPort: 'k' },
          { from: 'attn_score', to: 'attn_weight' },
          { from: 'attn_weight', to: 'attn_drop' },
          { from: 'attn_drop', to: 'attn_apply', toPort: 'weight' },
          { from: 'transpose_v', to: 'attn_apply', toPort: 'v' },
          { from: 'attn_apply', to: 'transpose_o' },
          { from: 'transpose_o', to: 'merge_heads' },
          { from: 'merge_heads', to: 'out_proj' },
          { from: 'out_proj', to: 'drop' },
        ],
        outputVar: 'drop',
      },
      {
        type: 'multiheadattention',
        label: 'MultiheadAttention',
        description: 'Multi-head attention (fully expanded implementation)',
        defaultParams: { embed_dim: 512, num_heads: 8, dropout: 0.1, bias: true },
        outputType: 'Tensor',
        isComposite: true as const,
        // Same as SelfAttention - Q, K, V all from same input
        internalStructure: [
          { id: 'q_proj', type: 'linear' as const, label: 'Q_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'k_proj', type: 'linear' as const, label: 'K_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'v_proj', type: 'linear' as const, label: 'V_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'split_q', type: 'reshape' as const, label: 'SplitQ', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_q', type: 'transpose' as const, label: 'TransposeQ', params: { dim0: 1, dim1: 2 } },
          { id: 'split_k', type: 'reshape' as const, label: 'SplitK', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_k', type: 'transpose' as const, label: 'TransposeK', params: { dim0: 1, dim1: 2 } },
          { id: 'split_v', type: 'reshape' as const, label: 'SplitV', params: { shape: '(B, N, ${num_heads}, ${head_dim})' } },
          { id: 'transpose_v', type: 'transpose' as const, label: 'TransposeV', params: { dim0: 1, dim1: 2 } },
          { id: 'attn_score', type: 'attn_score' as const, label: 'AttnScore', params: {} },
          { id: 'attn_weight', type: 'attn_weight' as const, label: 'AttnWeight', params: { dim: -1 } },
          { id: 'attn_drop', type: 'dropout' as const, label: 'AttnDrop', params: { p: '${dropout}' } },
          { id: 'attn_apply', type: 'attn_apply' as const, label: 'AttnApply', params: {} },
          { id: 'transpose_o', type: 'transpose' as const, label: 'TransposeO', params: { dim0: 1, dim1: 2 } },
          { id: 'merge_heads', type: 'reshape' as const, label: 'MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
          { id: 'out_proj', type: 'linear' as const, label: 'Out_proj', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
          { id: 'drop', type: 'dropout' as const, label: 'Dropout', params: { p: '${dropout}' } },
        ],
        internalEdges: [
          { from: 'q_proj', to: 'split_q' },
          { from: 'k_proj', to: 'split_k' },
          { from: 'v_proj', to: 'split_v' },
          { from: 'split_q', to: 'transpose_q' },
          { from: 'split_k', to: 'transpose_k' },
          { from: 'split_v', to: 'transpose_v' },
          { from: 'transpose_q', to: 'attn_score', toPort: 'q' },
          { from: 'transpose_k', to: 'attn_score', toPort: 'k' },
          { from: 'attn_score', to: 'attn_weight' },
          { from: 'attn_weight', to: 'attn_drop' },
          { from: 'attn_drop', to: 'attn_apply', toPort: 'weight' },
          { from: 'transpose_v', to: 'attn_apply', toPort: 'v' },
          { from: 'attn_apply', to: 'transpose_o' },
          { from: 'transpose_o', to: 'merge_heads' },
          { from: 'merge_heads', to: 'out_proj' },
          { from: 'out_proj', to: 'drop' },
        ],
        outputVar: 'drop',
      },
      {
        type: 'transformerencoder',
        label: 'TransformerEncoder',
        description: 'Transformer encoder (Pre-LN): x + SelfAttn(x) + FFN(x + SelfAttn(x))',
        defaultParams: { embed_dim: 512, num_heads: 8, num_layers: 6, dim_feedforward: 2048, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: true as const,
        // PyTorch TransformerEncoderLayer structure (Pre-LN variant):
        // x → self_attn → attn_drop → residual_add1(x, attn_out) → norm1
        // → ffn_linear1 → gelu → ffn_drop → ffn_linear2 → residual_add2(norm1, ffn_out) → norm2 → output
        // NOTE: self_attn is expanded inline (see SelfAttention.internalStructure)
        internalStructure: [
          // Self-Attention: QKV projections + attention computation (all receive external input 'x')
          { id: 'sa_q_proj', type: 'linear' as const, label: 'SA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'sa_k_proj', type: 'linear' as const, label: 'SA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          { id: 'sa_v_proj', type: 'linear' as const, label: 'SA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'x' },
          // Split into multi-head: (B, N, C) -> (B, N, num_heads, head_dim)
          { id: 'sa_split_q', type: 'reshape' as const, label: 'SA_SplitQ', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'sa_transpose_q', type: 'transpose' as const, label: 'SA_TransposeQ', params: { dim0: 1, dim1: 2 } },
          { id: 'sa_split_k', type: 'reshape' as const, label: 'SA_SplitK', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'sa_transpose_k', type: 'transpose' as const, label: 'SA_TransposeK', params: { dim0: 1, dim1: 2 } },
          { id: 'sa_split_v', type: 'reshape' as const, label: 'SA_SplitV', params: { shape: '(-1, ${num_heads}, ${head_dim})' } },
          { id: 'sa_transpose_v', type: 'transpose' as const, label: 'SA_TransposeV', params: { dim0: 1, dim1: 2 } },
          // Attention computation
          { id: 'sa_attn_score', type: 'attn_score' as const, label: 'SA_Score', params: {} },
          { id: 'sa_attn_weight', type: 'attn_weight' as const, label: 'SA_Weight', params: { dim: -1 } },
          { id: 'sa_attn_drop', type: 'dropout' as const, label: 'SA_Drop', params: { p: '${dropout}' } },
          { id: 'sa_attn_apply', type: 'attn_apply' as const, label: 'SA_Apply', params: {} },
          // Merge heads: (B, num_heads, N, head_dim) -> (B, N, C)
          { id: 'sa_transpose_o', type: 'transpose' as const, label: 'SA_TransposeO', params: { dim0: 1, dim1: 2 } },
          { id: 'sa_merge_heads', type: 'reshape' as const, label: 'SA_MergeHeads', params: { shape: '(-1, ${embed_dim})' } },
          { id: 'sa_out_proj', type: 'linear' as const, label: 'SA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
          { id: 'sa_drop', type: 'dropout' as const, label: 'SA_FinalDrop', params: { p: '${dropout}' } },
          // First residual add: x + attn_output
          { id: 'residual_add1', type: 'residual_add' as const, label: 'ResAdd1', params: {} },
          // First norm
          { id: 'norm1', type: 'layernorm' as const, label: 'Norm1', params: { normalized_shape: '${embed_dim}' } },
          // FFN
          { id: 'ffn_linear1', type: 'linear' as const, label: 'FFN1', params: { in_features: '${embed_dim}', out_features: '${dim_feedforward}' } },
          { id: 'ffn_gelu', type: 'gelu' as const, label: 'GELU', params: {} },
          { id: 'ffn_drop', type: 'dropout' as const, label: 'FFN_Drop', params: { p: '${dropout}' } },
          { id: 'ffn_linear2', type: 'linear' as const, label: 'FFN2', params: { in_features: '${dim_feedforward}', out_features: '${embed_dim}' } },
          // Second residual add
          { id: 'residual_add2', type: 'residual_add' as const, label: 'ResAdd2', params: {} },
          // Final norm
          { id: 'norm2', type: 'layernorm' as const, label: 'Norm2', params: { normalized_shape: '${embed_dim}' } },
        ],
        internalEdges: [
          // Self-Attention QKV with split/merge
          { from: 'sa_q_proj', to: 'sa_split_q' },
          { from: 'sa_split_q', to: 'sa_transpose_q' },
          { from: 'sa_k_proj', to: 'sa_split_k' },
          { from: 'sa_split_k', to: 'sa_transpose_k' },
          { from: 'sa_v_proj', to: 'sa_split_v' },
          { from: 'sa_split_v', to: 'sa_transpose_v' },
          { from: 'sa_transpose_q', to: 'sa_attn_score', toPort: 'q' },
          { from: 'sa_transpose_k', to: 'sa_attn_score', toPort: 'k' },
          { from: 'sa_attn_score', to: 'sa_attn_weight' },
          { from: 'sa_attn_weight', to: 'sa_attn_drop' },
          { from: 'sa_attn_drop', to: 'sa_attn_apply', toPort: 'weight' },
          { from: 'sa_transpose_v', to: 'sa_attn_apply', toPort: 'v' },
          { from: 'sa_attn_apply', to: 'sa_transpose_o' },
          { from: 'sa_transpose_o', to: 'sa_merge_heads' },
          { from: 'sa_merge_heads', to: 'sa_out_proj' },
          { from: 'sa_out_proj', to: 'sa_drop' },
          // First residual: original input x + attn_output → norm1
          { from: 'sa_drop', to: 'residual_add1', toPort: 'sublayer' },
          { from: 'residual_add1', to: 'norm1' },
          // FFN
          { from: 'norm1', to: 'ffn_linear1' },
          { from: 'ffn_linear1', to: 'ffn_gelu' },
          { from: 'ffn_gelu', to: 'ffn_drop' },
          { from: 'ffn_drop', to: 'ffn_linear2' },
          // Second residual: norm1 + ffn_out
          { from: 'norm1', to: 'residual_add2', toPort: 'x' },
          { from: 'ffn_linear2', to: 'residual_add2', toPort: 'sublayer' },
          { from: 'residual_add2', to: 'norm2' },
        ],
        outputVar: 'norm2',
      },
      {
        type: 'transformerdecoder',
        label: 'TransformerDecoder',
        description: 'Transformer decoder (Pre-LN): self-attn + cross-attn + FFN',
        defaultParams: { embed_dim: 512, num_heads: 8, num_layers: 6, dim_feedforward: 2048, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: true as const,
        // TransformerDecoderLayer (Pre-LN):
        // 1. self_attn: tgt → Q, tgt → K, tgt → V → attention → out_proj → drop → residual_add1(tgt, attn_out) → norm1
        // 2. cross_attn: norm1 → Q, memory → K, memory → V → attention → out_proj → drop → residual_add2(norm1, cross_out) → norm2
        // 3. FFN: norm2 → ffn_linear1 → gelu → ffn_drop → ffn_linear2 → residual_add3(norm2, ffn_out) → norm3 → output
        internalStructure: [
          // Self-Attention QKV (all receive external input 'tgt')
          { id: 'sa_q_proj', type: 'linear' as const, label: 'SA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
          { id: 'sa_k_proj', type: 'linear' as const, label: 'SA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
          { id: 'sa_v_proj', type: 'linear' as const, label: 'SA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'tgt' },
          { id: 'sa_attn_score', type: 'attn_score' as const, label: 'SA_Score', params: {} },
          { id: 'sa_attn_weight', type: 'attn_weight' as const, label: 'SA_Weight', params: { dim: -1 } },
          { id: 'sa_attn_drop', type: 'dropout' as const, label: 'SA_Drop', params: { p: '${dropout}' } },
          { id: 'sa_attn_apply', type: 'attn_apply' as const, label: 'SA_Apply', params: {} },
          { id: 'sa_out_proj', type: 'linear' as const, label: 'SA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
          { id: 'sa_drop', type: 'dropout' as const, label: 'SA_FinalDrop', params: { p: '${dropout}' } },
          { id: 'res_add1', type: 'residual_add' as const, label: 'ResAdd1', params: {} },
          { id: 'norm1', type: 'layernorm' as const, label: 'Norm1', params: { normalized_shape: '${embed_dim}' } },
          // Cross-Attention QKV (Q from norm1, K and V from external 'memory')
          { id: 'ca_q_proj', type: 'linear' as const, label: 'CA_Q', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },  // Q from norm1 (has edge)
          { id: 'ca_k_proj', type: 'linear' as const, label: 'CA_K', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'memory' },
          { id: 'ca_v_proj', type: 'linear' as const, label: 'CA_V', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' }, sourceInput: 'memory' },
          { id: 'ca_attn_score', type: 'attn_score' as const, label: 'CA_Score', params: {} },
          { id: 'ca_attn_weight', type: 'attn_weight' as const, label: 'CA_Weight', params: { dim: -1 } },
          { id: 'ca_attn_drop', type: 'dropout' as const, label: 'CA_Drop', params: { p: '${dropout}' } },
          { id: 'ca_attn_apply', type: 'attn_apply' as const, label: 'CA_Apply', params: {} },
          { id: 'ca_out_proj', type: 'linear' as const, label: 'CA_Out', params: { in_features: '${embed_dim}', out_features: '${embed_dim}' } },
          { id: 'ca_drop', type: 'dropout' as const, label: 'CA_FinalDrop', params: { p: '${dropout}' } },
          { id: 'res_add2', type: 'residual_add' as const, label: 'ResAdd2', params: {} },
          { id: 'norm2', type: 'layernorm' as const, label: 'Norm2', params: { normalized_shape: '${embed_dim}' } },
          // FFN
          { id: 'ffn_linear1', type: 'linear' as const, label: 'FFN1', params: { in_features: '${embed_dim}', out_features: '${dim_feedforward}' } },
          { id: 'ffn_gelu', type: 'gelu' as const, label: 'GELU', params: {} },
          { id: 'ffn_drop', type: 'dropout' as const, label: 'FFN_Drop', params: { p: '${dropout}' } },
          { id: 'ffn_linear2', type: 'linear' as const, label: 'FFN2', params: { in_features: '${dim_feedforward}', out_features: '${embed_dim}' } },
          { id: 'res_add3', type: 'residual_add' as const, label: 'ResAdd3', params: {} },
          { id: 'norm3', type: 'layernorm' as const, label: 'Norm3', params: { normalized_shape: '${embed_dim}' } },
        ],
        internalEdges: [
          // Self-Attention
          { from: 'sa_q_proj', to: 'sa_attn_score', toPort: 'q' },
          { from: 'sa_k_proj', to: 'sa_attn_score', toPort: 'k' },
          { from: 'sa_v_proj', to: 'sa_attn_apply', toPort: 'v' },
          { from: 'sa_attn_score', to: 'sa_attn_weight' },
          { from: 'sa_attn_weight', to: 'sa_attn_drop' },
          { from: 'sa_attn_drop', to: 'sa_attn_apply', toPort: 'weight' },
          { from: 'sa_attn_apply', to: 'sa_out_proj' },
          { from: 'sa_out_proj', to: 'sa_drop' },
          { from: 'sa_drop', to: 'res_add1', toPort: 'sublayer' },
          { from: 'res_add1', to: 'norm1' },
          // Cross-Attention
          { from: 'norm1', to: 'ca_q_proj' },
          { from: 'ca_q_proj', to: 'ca_attn_score', toPort: 'q' },
          { from: 'ca_k_proj', to: 'ca_attn_score', toPort: 'k' },
          { from: 'ca_v_proj', to: 'ca_attn_apply', toPort: 'v' },
          { from: 'ca_attn_score', to: 'ca_attn_weight' },
          { from: 'ca_attn_weight', to: 'ca_attn_drop' },
          { from: 'ca_attn_drop', to: 'ca_attn_apply', toPort: 'weight' },
          { from: 'ca_attn_apply', to: 'ca_out_proj' },
          { from: 'ca_out_proj', to: 'ca_drop' },
          { from: 'ca_drop', to: 'res_add2', toPort: 'sublayer' },
          { from: 'res_add2', to: 'norm2' },
          // FFN
          { from: 'norm2', to: 'ffn_linear1' },
          { from: 'ffn_linear1', to: 'ffn_gelu' },
          { from: 'ffn_gelu', to: 'ffn_drop' },
          { from: 'ffn_drop', to: 'ffn_linear2' },
          { from: 'ffn_linear2', to: 'res_add3', toPort: 'sublayer' },
          { from: 'res_add3', to: 'norm3' },
        ],
        outputVar: 'norm3',
      },
    ],
  },
  {
    label: 'State Space / RNN',
    nodes: [
      {
        type: 'mamba',
        label: 'Mamba',
        description: 'Mamba state space model (simplified)',
        defaultParams: { d_model: 512, d_state: 16, d_conv: 4, expand: 2, dt_rank: 0, n_layers: 1, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: false as const,
      },
    ],
  },
  {
    label: 'FFN / MLP',
    nodes: [
      {
        type: 'ffn',
        label: 'FFN',
        description: 'Feed-forward network (Transformer style)',
        defaultParams: { dim: 512, hidden_dim: 2048, dropout: 0.1 },
        outputType: 'Tensor',
        isComposite: true as const,
        // Standard FFN: Linear(d_model, d_ff) → GELU() → Dropout → Linear(d_ff, d_model)
        internalStructure: [
          { id: 'w1', type: 'linear' as const, label: 'W1', params: { in_features: '${dim}', out_features: '${hidden_dim}' }, sourceInput: 'x' },
          { id: 'act', type: 'gelu' as const, label: 'GELU', params: {} },
          { id: 'drop', type: 'dropout' as const, label: 'Dropout', params: { p: '${dropout}' } },
          { id: 'w2', type: 'linear' as const, label: 'W2', params: { in_features: '${hidden_dim}', out_features: '${dim}' } },
        ],
        internalEdges: [
          { from: 'w1', to: 'act' },
          { from: 'act', to: 'drop' },
          { from: 'drop', to: 'w2' },
        ],
        outputVar: 'w2',
      },
      {
        type: 'mlp',
        label: 'MLP',
        description: 'Multi-layer perceptron',
        defaultParams: { in_features: 784, hidden_features: 256, out_features: 10, dropout: 0.0 },
        outputType: 'Tensor',
        isComposite: true as const,
        // Standard MLP: Linear → ReLU → Dropout(optional) → Linear
        internalStructure: [
          { id: 'fc1', type: 'linear' as const, label: 'FC1', params: { in_features: '${in_features}', out_features: '${hidden_features}' }, sourceInput: 'x' },
          { id: 'act', type: 'relu' as const, label: 'ReLU', params: {} },
          { id: 'drop', type: 'dropout' as const, label: 'Dropout', params: { p: '${dropout}' } },
          { id: 'fc2', type: 'linear' as const, label: 'FC2', params: { in_features: '${hidden_features}', out_features: '${out_features}' } },
        ],
        internalEdges: [
          { from: 'fc1', to: 'act' },
          { from: 'act', to: 'drop' },
          { from: 'drop', to: 'fc2' },
        ],
        outputVar: 'fc2',
      },
    ],
  },
  {
    label: 'Tensor Operations',
    nodes: [
      { type: 'constant', label: 'Constant', description: 'Create a constant tensor', defaultParams: { value: '0', shape: '1', trainable: false }, outputType: 'Tensor' },
      { type: 'parameter', label: 'Parameter', description: 'Create a learnable parameter', defaultParams: { shape: '1,1,768', trainable: true }, outputType: 'Tensor' },
      { type: 'concat', label: 'Concat', description: 'Concatenate tensors', defaultParams: { dim: 1 }, outputType: 'Tensor' },
      { type: 'add', label: 'Add', description: 'Element-wise addition', defaultParams: {}, outputType: 'Tensor' },
      { type: 'mul', label: 'Multiply', description: 'Element-wise multiplication', defaultParams: {}, outputType: 'Tensor' },
      { type: 'reshape', label: 'Reshape', description: 'Reshape tensor', defaultParams: { shape: '-1' }, outputType: 'Tensor' },
      { type: 'flatten', label: 'Flatten', description: 'Flatten tensor', defaultParams: { start_dim: 1 }, outputType: 'Tensor' },
      { type: 'transpose', label: 'Transpose', description: 'Transpose tensor dimensions', defaultParams: { dim0: 0, dim1: 1 }, outputType: 'Tensor' },
      { type: 'permute', label: 'Permute', description: 'Permute tensor dimensions', defaultParams: { dims: '0,2,1' }, outputType: 'Tensor' },
      { type: 'split', label: 'Split', description: 'Split tensor', defaultParams: { split_size: 32, dim: 0 }, outputType: 'Tensor' },
      { type: 'slice', label: 'Slice', description: 'Slice tensor', defaultParams: { start: 0, end: -1 }, outputType: 'Tensor' },
      { type: 'squeeze', label: 'Squeeze', description: 'Remove dimensions of size 1', defaultParams: { dim: 0 }, outputType: 'Tensor' },
      { type: 'expand', label: 'Expand', description: 'Expand tensor to new shape', defaultParams: { shape: '-1' }, outputType: 'Tensor' },
    ],
  },
  {
    label: 'Regularization',
    nodes: [
      { type: 'dropout', label: 'Dropout', description: 'Dropout regularization', defaultParams: { p: 0.5 }, outputType: 'Tensor' },
      { type: 'droppath', label: 'DropPath', description: 'Stochastic depth', defaultParams: { p: 0.1 }, outputType: 'Tensor' },
    ],
  },
  {
    label: 'Loss Functions',
    nodes: [
      { type: 'crossentropyloss', label: 'CrossEntropyLoss', description: 'Cross-entropy loss', defaultParams: {}, outputType: 'Scalar' },
      { type: 'mseloss', label: 'MSELoss', description: 'Mean squared error', defaultParams: {}, outputType: 'Scalar' },
      { type: 'focalloss', label: 'FocalLoss', description: 'Focal loss for imbalanced classification', defaultParams: { alpha: 0.25, gamma: 2.0 }, outputType: 'Scalar' },
      { type: 'labelsmoothing', label: 'LabelSmoothing', description: 'Label smoothing loss', defaultParams: { smoothing: 0.1 }, outputType: 'Scalar' },
    ],
  },
  {
    label: 'Optimizers',
    nodes: [
      { type: 'adam', label: 'Adam', description: 'Adam optimizer', defaultParams: { lr: 0.001, betas: '0.9,0.999', eps: 1e-8, weight_decay: 0 }, outputType: 'Optimizer' },
      { type: 'adamw', label: 'AdamW', description: 'Adam with weight decay', defaultParams: { lr: 0.001, betas: '0.9,0.999', eps: 1e-8, weight_decay: 0.01 }, outputType: 'Optimizer' },
      { type: 'sgd', label: 'SGD', description: 'Stochastic gradient descent', defaultParams: { lr: 0.01, momentum: 0.9, weight_decay: 0 }, outputType: 'Optimizer' },
      { type: 'rmsprop', label: 'RMSprop', description: 'RMSprop optimizer', defaultParams: { lr: 0.001, alpha: 0.99, eps: 1e-8, weight_decay: 0 }, outputType: 'Optimizer' },
    ],
  },
  {
    label: 'Schedulers',
    nodes: [
      { type: 'steplr', label: 'StepLR', description: 'Step learning rate decay', defaultParams: { step_size: 10, gamma: 0.1 }, outputType: 'Scheduler' },
      { type: 'cosineannealinglr', label: 'CosineAnnealing', description: 'Cosine annealing scheduler', defaultParams: { T_max: 50, eta_min: 1e-6 }, outputType: 'Scheduler' },
      { type: 'reducelronplateau', label: 'ReduceLROnPlateau', description: 'Reduce LR on plateau', defaultParams: { mode: 'min', factor: 0.1, patience: 5 }, outputType: 'Scheduler' },
    ],
  },
  {
    label: 'Evaluation',
    nodes: [
      { type: 'accuracy', label: 'Accuracy', description: 'Classification accuracy score', defaultParams: { metric_type: 'accuracy', average: 'macro', top_k: 1 }, outputType: 'Scalar' },
      { type: 'f1', label: 'F1 Score', description: 'F1 score (binary or macro/micro/weighted)', defaultParams: { metric_type: 'f1', average: 'macro' }, outputType: 'Scalar' },
      { type: 'precision', label: 'Precision', description: 'Classification precision score', defaultParams: { metric_type: 'precision', average: 'macro' }, outputType: 'Scalar' },
      { type: 'recall', label: 'Recall', description: 'Classification recall score', defaultParams: { metric_type: 'recall', average: 'macro' }, outputType: 'Scalar' },
      { type: 'confusion_matrix', label: 'Confusion Matrix', description: 'Confusion matrix for classification', defaultParams: { metric_type: 'confusion_matrix' }, outputType: 'Matrix' },
      { type: 'mean_iou', label: 'Mean IoU', description: 'Mean Intersection-over-Union for segmentation', defaultParams: { metric_type: 'mean_iou' }, outputType: 'Scalar' },
      { type: 'roc_auc', label: 'ROC AUC', description: 'Area under ROC curve score', defaultParams: { metric_type: 'roc_auc' }, outputType: 'Scalar' },
    ],
  },
  {
    label: 'Custom',
    nodes: [
      { type: 'custom', label: 'CustomModule', description: 'Custom PyTorch module - paste your Python code to import', defaultParams: { custom_code: '' }, outputType: 'Tensor' },
    ],
  },
]

// Import getNodeDef from nodeDefinition for unified node definitions
import { getNodeDef } from './nodeDefinition'

/**
 * Get composite node definition by type.
 * First checks nodeDefinition.ts (new unified), then falls back to NODE_REGISTRY (old).
 */
export function getCompositeNodeDef(opType: string): CompositeNodeDefinition | undefined {
  // First try nodeDefinition.ts (new unified definitions)
  const nodeDef = getNodeDef(opType)
  if (nodeDef && nodeDef.isComposite && nodeDef.internalStructure) {
    // Convert to old format for compatibility
    return {
      type: nodeDef.type,
      label: nodeDef.label,
      description: nodeDef.description,
      defaultParams: Object.fromEntries(
        Object.entries(nodeDef.params).map(([k, v]) => [k, (v as any).default ?? v.default])
      ),
      outputType: 'Tensor',
      isComposite: true,
      internalStructure: nodeDef.internalStructure.map(sm => ({
        id: sm.id,
        type: sm.type,
        label: sm.label,
        params: sm.params,
        sourceInput: sm.sourceInput,
      })) as SubModule[],
      internalEdges: nodeDef.internalEdges as InternalEdge[],
      outputVar: nodeDef.outputVar,
    } as CompositeNodeDefinition
  }

  // Fall back to NODE_REGISTRY (old definitions)
  for (const category of NODE_REGISTRY) {
    for (const node of category.nodes) {
      if (node.type === opType && node.isComposite) {
        return node as CompositeNodeDefinition
      }
    }
  }
  return undefined
}
