export interface Template {
  id: string
  name: string
  description: string
  category: 'cv' | 'nlp' | 'gan' | 'other'
  emoji: string
  graph: {
    nodes: any[]
    edges: any[]
  }
}

// 辅助函数：生成从左到右的水平布局
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
    if (col >= nodesPerRow) {
      col = 0
      row++
    }
  }
  return result
}

export const TEMPLATES: Template[] = [
  // ============================================================================
  // ResNet-18 - 残差网络
  // ============================================================================
  {
    id: 'resnet18',
    name: 'ResNet-18',
    description: '18层残差网络，含4个残差阶段，适合图像分类',
    category: 'cv',
    emoji: '🔬',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d', label: 'Conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3, bias: false } } },
        { id: 'bn1', data: { nodeType: 'batchnorm2d', label: 'BatchNorm2d', params: { num_features: 64, eps: 1e-5, momentum: 0.1 } } },
        { id: 'relu', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'maxpool', data: { nodeType: 'maxpool2d', label: 'MaxPool2d', params: { kernel_size: 3, stride: 2, padding: 1 } } },
        // Stage 1
        { id: 'layer1_1', data: { nodeType: 'conv2d', label: 'BasicBlock1-1', params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'layer1_2', data: { nodeType: 'conv2d', label: 'BasicBlock1-2', params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'add1', data: { nodeType: 'add', label: 'Add', params: {} } },
        // Stage 2
        { id: 'layer2_1', data: { nodeType: 'conv2d', label: 'BasicBlock2-1', params: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'layer2_2', data: { nodeType: 'conv2d', label: 'BasicBlock2-2', params: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, bias: false } } },
        { id: 'down1', data: { nodeType: 'conv2d', label: 'Downsample', params: { in_channels: 64, out_channels: 128, kernel_size: 1, stride: 2 } } },
        { id: 'add2', data: { nodeType: 'add', label: 'Add', params: {} } },
        // GAP + FC
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

  // ============================================================================
  // VGG-16
  // ============================================================================
  {
    id: 'vgg16',
    name: 'VGG-16',
    description: '16层VGG网络，3阶段卷积+2层全连接',
    category: 'cv',
    emoji: '🔷',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        // Block 1
        { id: 'conv1_1', data: { nodeType: 'conv2d', label: 'Conv 64-1', params: { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv1_2', data: { nodeType: 'conv2d', label: 'Conv 64-2', params: { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool1', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // Block 2
        { id: 'conv2_1', data: { nodeType: 'conv2d', label: 'Conv 128-1', params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv2_2', data: { nodeType: 'conv2d', label: 'Conv 128-2', params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool2', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // Block 3
        { id: 'conv3_1', data: { nodeType: 'conv2d', label: 'Conv 256-1', params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv3_2', data: { nodeType: 'conv2d', label: 'Conv 256-2', params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool3', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // Block 4
        { id: 'conv4_1', data: { nodeType: 'conv2d', label: 'Conv 512-1', params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'conv4_2', data: { nodeType: 'conv2d', label: 'Conv 512-2', params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'pool4', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // GAP + FC
        { id: 'gap', data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc1', data: { nodeType: 'linear', label: 'FC 4096', params: { in_features: 512, out_features: 4096, bias: true } } },
        { id: 'fc2', data: { nodeType: 'linear', label: 'FC 1000', params: { in_features: 4096, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        { source: 'input', target: 'conv1_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv1_1', target: 'conv1_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv1_2', target: 'pool1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool1', target: 'conv2_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv2_1', target: 'conv2_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv2_2', target: 'pool2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool2', target: 'conv3_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv3_1', target: 'conv3_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv3_2', target: 'pool3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool3', target: 'conv4_1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv4_1', target: 'conv4_2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv4_2', target: 'pool4', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool4', target: 'gap', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'gap', target: 'fc1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc1', target: 'fc2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc2', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },

  // ============================================================================
  // LeNet-5
  // ============================================================================
  {
    id: 'lenet5',
    name: 'LeNet-5',
    description: '5层LeNet，适合MNIST/CIFAR入门',
    category: 'cv',
    emoji: '✏️',
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
        { source: 'input', target: 'conv1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv1', target: 'relu1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu1', target: 'pool1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool1', target: 'conv2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv2', target: 'relu2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu2', target: 'pool2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool2', target: 'flatten', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'flatten', target: 'fc1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc1', target: 'relu3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu3', target: 'fc2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc2', target: 'relu4', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu4', target: 'fc3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc3', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },

  // ============================================================================
  // Vision Transformer
  // Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  // ============================================================================
  {
    id: 'vit',
    name: 'Vision Transformer',
    description: '视觉Transformer，Patch Embed + Transformer Encoder + CLS Token',
    category: 'cv',
    emoji: '🔮',
    graph: {
      nodes: makeLayout([
        // Input: 3x224x224
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        // Patch Embedding: Conv2d(3, 768, 16, stride=16) -> (B, 768, 14, 14)
        { id: 'patch_embed', data: { nodeType: 'conv2d', label: 'PatchEmbed', params: { in_channels: 3, out_channels: 768, kernel_size: 16, stride: 16, bias: false } } },
        // Flatten: (B, 768, 14, 14) -> (B, 768, 196)
        { id: 'flatten', data: { nodeType: 'flatten', label: 'Flatten', params: { start_dim: 2, end_dim: 3 } } },
        // Permute: (B, 768, 196) -> (B, 196, 768)
        { id: 'permute', data: { nodeType: 'permute', label: 'Permute', params: { dims: '0,2,1' } } },
        // CLS token (learnable parameter)
        { id: 'cls_token', data: { nodeType: 'parameter', label: 'CLS Token', params: { shape: '1,1,768', trainable: true } } },
        // Concatenate: [CLS; patches] -> (B, 197, 768)
        { id: 'concat', data: { nodeType: 'concat', label: 'Concat CLS', params: { dim: 1 } } },
        // Position embedding (learnable parameter)
        { id: 'pos_embed', data: { nodeType: 'parameter', label: 'Pos Embed', params: { shape: '1,197,768', trainable: true } } },
        // Add position embedding: (B, 197, 768) + (B, 197, 768)
        { id: 'add_pos', data: { nodeType: 'add', label: 'Add Pos', params: {} } },
        // Dropout
        { id: 'dropout', data: { nodeType: 'dropout', label: 'Dropout', params: { p: 0.1 } } },
        // Transformer Encoder (12 layers)
        { id: 'encoder', data: { nodeType: 'transformerencoder', label: 'TransformerEncoder', params: { embed_dim: 768, num_heads: 12, num_layers: 12, dim_feedforward: 3072, dropout: 0.1 } } },
        // Final LayerNorm
        { id: 'norm_out', data: { nodeType: 'layernorm', label: 'LayerNorm', params: { normalized_shape: 768, eps: 1e-6 } } },
        // Extract CLS token: (B, 197, 768) -> (B, 1, 768)
        { id: 'cls_slice', data: { nodeType: 'slice', label: 'Extract CLS', params: { start: 0, end: 1, dim: 1 } } },
        // Squeeze: (B, 1, 768) -> (B, 768)
        { id: 'squeeze', data: { nodeType: 'squeeze', label: 'Squeeze', params: { dim: 1 } } },
        // Classification head
        { id: 'head', data: { nodeType: 'linear', label: 'Head', params: { in_features: 768, out_features: 1000, bias: true } } },
        // Output
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 4),
      edges: [
        // Patch embedding pipeline
        { source: 'input', target: 'patch_embed', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'patch_embed', target: 'flatten', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'flatten', target: 'permute', sourceHandle: 'result', targetHandle: 'a' },
        // Concat: patches (in_0) + CLS token (in_1)
        { source: 'permute', target: 'concat', sourceHandle: 'result', targetHandle: 'in_0' },
        { source: 'cls_token', target: 'concat', sourceHandle: 'result', targetHandle: 'in_1' },
        // Add position embedding
        { source: 'concat', target: 'add_pos', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pos_embed', target: 'add_pos', sourceHandle: 'result', targetHandle: 'b' },
        // Dropout
        { source: 'add_pos', target: 'dropout', sourceHandle: 'result', targetHandle: 'a' },
        // Transformer Encoder
        { source: 'dropout', target: 'encoder', sourceHandle: 'result', targetHandle: 'x' },
        // Final LayerNorm
        { source: 'encoder', target: 'norm_out', sourceHandle: 'result', targetHandle: 'a' },
        // Extract CLS token for classification
        { source: 'norm_out', target: 'cls_slice', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'cls_slice', target: 'squeeze', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'squeeze', target: 'head', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'head', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },

  // ============================================================================
  // MobileNetV2
  // ============================================================================
  {
    id: 'mobilenetv2',
    name: 'MobileNetV2',
    description: '倒残差结构+深度可分离卷积，适合移动端部署',
    category: 'cv',
    emoji: '📱',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '3, 224, 224', dtype: 'float32' } } },
        { id: 'conv1', data: { nodeType: 'conv2d', label: 'Conv 3→32', params: { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, bias: false } } },
        { id: 'bn1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 32 } } },
        { id: 'silu1', data: { nodeType: 'silu', label: 'SiLU', params: {} } },
        // Inverted Residual 1
        { id: 'dw1', data: { nodeType: 'conv2d', label: 'DW 32→32', params: { in_channels: 32, out_channels: 32, kernel_size: 3, stride: 1, padding: 1, groups: 32, bias: false } } },
        { id: 'bn_dw1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 32 } } },
        { id: 'pw1', data: { nodeType: 'conv2d', label: 'PW 32→16', params: { in_channels: 32, out_channels: 16, kernel_size: 1, bias: false } } },
        { id: 'bn_pw1', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 16 } } },
        // Inverted Residual 2
        { id: 'exp2', data: { nodeType: 'conv2d', label: 'PW 16→96', params: { in_channels: 16, out_channels: 96, kernel_size: 1, bias: false } } },
        { id: 'bn_exp2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 96 } } },
        { id: 'dw2', data: { nodeType: 'conv2d', label: 'DW 96→96', params: { in_channels: 96, out_channels: 96, kernel_size: 3, stride: 2, padding: 1, groups: 96, bias: false } } },
        { id: 'bn_dw2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 96 } } },
        { id: 'pw2', data: { nodeType: 'conv2d', label: 'PW 96→24', params: { in_channels: 96, out_channels: 24, kernel_size: 1, bias: false } } },
        { id: 'bn_pw2', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 24 } } },
        // Output
        { id: 'exp3', data: { nodeType: 'conv2d', label: 'PW 24→144', params: { in_channels: 24, out_channels: 144, kernel_size: 1, bias: false } } },
        { id: 'bn_exp3', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 144 } } },
        { id: 'dw3', data: { nodeType: 'conv2d', label: 'DW 144→144', params: { in_channels: 144, out_channels: 144, kernel_size: 3, stride: 1, padding: 1, groups: 144, bias: false } } },
        { id: 'bn_dw3', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 144 } } },
        { id: 'pw3', data: { nodeType: 'conv2d', label: 'PW 144→24', params: { in_channels: 144, out_channels: 24, kernel_size: 1, bias: false } } },
        { id: 'bn_pw3', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 24 } } },
        // Final
        { id: 'conv_last', data: { nodeType: 'conv2d', label: 'Conv 24→1280', params: { in_channels: 24, out_channels: 1280, kernel_size: 1, bias: false } } },
        { id: 'bn_last', data: { nodeType: 'batchnorm2d', label: 'BN', params: { num_features: 1280 } } },
        { id: 'silu_last', data: { nodeType: 'silu', label: 'SiLU', params: {} } },
        { id: 'gap', data: { nodeType: 'globalavgpool', label: 'GlobalAvgPool', params: {} } },
        { id: 'fc', data: { nodeType: 'linear', label: 'FC', params: { in_features: 1280, out_features: 1000, bias: true } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 6),
      edges: [
        { source: 'input', target: 'conv1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv1', target: 'bn1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn1', target: 'silu1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'silu1', target: 'dw1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'dw1', target: 'bn_dw1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_dw1', target: 'pw1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pw1', target: 'bn_pw1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_pw1', target: 'exp2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'exp2', target: 'bn_exp2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_exp2', target: 'silu1', sourceHandle: 'result', targetHandle: 'b' },
        { source: 'silu1', target: 'dw2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'dw2', target: 'bn_dw2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_dw2', target: 'pw2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pw2', target: 'bn_pw2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_pw2', target: 'exp3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'exp3', target: 'bn_exp3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_exp3', target: 'dw3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'dw3', target: 'bn_dw3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_dw3', target: 'pw3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pw3', target: 'bn_pw3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_pw3', target: 'conv_last', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'conv_last', target: 'bn_last', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bn_last', target: 'silu_last', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'silu_last', target: 'gap', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'gap', target: 'fc', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },

  // ============================================================================
  // MLP 分类器
  // ============================================================================
  {
    id: 'mlp',
    name: 'MLP 分类器',
    description: '多层感知机，适合MNIST/CIFAR入门',
    category: 'cv',
    emoji: '⚡',
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
        { source: 'input', target: 'flatten', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'flatten', target: 'fc1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc1', target: 'relu1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu1', target: 'drop1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'drop1', target: 'fc2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc2', target: 'relu2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'relu2', target: 'drop2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'drop2', target: 'fc3', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'fc3', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  },

  // ============================================================================
  // U-Net
  // ============================================================================
  {
    id: 'unet',
    name: 'U-Net',
    description: '编码器-解码器结构，适合图像分割',
    category: 'cv',
    emoji: '🧠',
    graph: {
      nodes: makeLayout([
        { id: 'input', data: { nodeType: 'input', label: 'Input', params: { shape: '1, 572, 572', dtype: 'float32' } } },
        // Encoder
        { id: 'enc1_c1', data: { nodeType: 'conv2d', label: 'Enc1 Conv', params: { in_channels: 1, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc1_relu', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'enc1_c2', data: { nodeType: 'conv2d', label: 'Enc1 Conv', params: { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc1_out', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'pool1', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // Encoder 2
        { id: 'enc2_c1', data: { nodeType: 'conv2d', label: 'Enc2 Conv', params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc2_relu', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'enc2_c2', data: { nodeType: 'conv2d', label: 'Enc2 Conv', params: { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'enc2_out', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'pool2', data: { nodeType: 'maxpool2d', label: 'Pool', params: { kernel_size: 2, stride: 2 } } },
        // Bottleneck
        { id: 'bot_c1', data: { nodeType: 'conv2d', label: 'Bot Conv', params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'bot_relu', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        { id: 'bot_c2', data: { nodeType: 'conv2d', label: 'Bot Conv', params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'bot_out', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        // Decoder 1
        { id: 'up1', data: { nodeType: 'conv2d', label: 'UpConv', params: { in_channels: 256, out_channels: 128, kernel_size: 2, stride: 2, bias: false } } },
        { id: 'dec1_conv', data: { nodeType: 'conv2d', label: 'Dec1 Conv', params: { in_channels: 256, out_channels: 128, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'dec1_out', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        // Decoder 2
        { id: 'up2', data: { nodeType: 'conv2d', label: 'UpConv', params: { in_channels: 128, out_channels: 64, kernel_size: 2, stride: 2, bias: false } } },
        { id: 'dec2_conv', data: { nodeType: 'conv2d', label: 'Dec2 Conv', params: { in_channels: 128, out_channels: 64, kernel_size: 3, padding: 1, bias: false } } },
        { id: 'dec2_out', data: { nodeType: 'relu', label: 'ReLU', params: {} } },
        // Output
        { id: 'final', data: { nodeType: 'conv2d', label: 'Final Conv', params: { in_channels: 64, out_channels: 2, kernel_size: 1, bias: false } } },
        { id: 'output', data: { nodeType: 'output', label: 'Output', params: {} } }
      ], 5),
      edges: [
        // Encoder
        { source: 'input', target: 'enc1_c1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc1_c1', target: 'enc1_relu', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc1_relu', target: 'enc1_c2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc1_c2', target: 'enc1_out', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc1_out', target: 'pool1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'pool1', target: 'enc2_c1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc2_c1', target: 'enc2_relu', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc2_relu', target: 'enc2_c2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc2_c2', target: 'enc2_out', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'enc2_out', target: 'pool2', sourceHandle: 'result', targetHandle: 'a' },
        // Bottleneck
        { source: 'pool2', target: 'bot_c1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bot_c1', target: 'bot_relu', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bot_relu', target: 'bot_c2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'bot_c2', target: 'bot_out', sourceHandle: 'result', targetHandle: 'a' },
        // Decoder 1
        { source: 'bot_out', target: 'up1', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'up1', target: 'dec1_conv', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'dec1_conv', target: 'dec1_out', sourceHandle: 'result', targetHandle: 'a' },
        // Decoder 2
        { source: 'dec1_out', target: 'up2', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'up2', target: 'dec2_conv', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'dec2_conv', target: 'dec2_out', sourceHandle: 'result', targetHandle: 'a' },
        // Final
        { source: 'dec2_out', target: 'final', sourceHandle: 'result', targetHandle: 'a' },
        { source: 'final', target: 'output', sourceHandle: 'result', targetHandle: 'a' }
      ]
    }
  }
]
