/**
 * Module Registry
 *
 * Manages module imports for code generation.
 * Maps node types to their PyTorch module classes.
 */

import { NodeBlock } from './astBuilder'

export type ModuleImportSource = 'torch.nn' | 'torch' | 'flowhamster.modules' | null

export interface ModuleImportInfo {
  source: ModuleImportSource
  className: string
}

const MODULE_IMPORT_MAP: Record<string, ModuleImportInfo> = {
  // torch.nn 内置模块
  conv2d: { source: 'torch.nn', className: 'Conv2d' },
  conv1d: { source: 'torch.nn', className: 'Conv1d' },
  conv3d: { source: 'torch.nn', className: 'Conv3d' },
  linear: { source: 'torch.nn', className: 'Linear' },
  relu: { source: 'torch.nn', className: 'ReLU' },
  gelu: { source: 'torch.nn', className: 'GELU' },
  silu: { source: 'torch.nn', className: 'SiLU' },
  sigmoid: { source: 'torch.nn', className: 'Sigmoid' },
  tanh: { source: 'torch.nn', className: 'Tanh' },
  leakyrelu: { source: 'torch.nn', className: 'LeakyReLU' },
  maxpool2d: { source: 'torch.nn', className: 'MaxPool2d' },
  avgpool2d: { source: 'torch.nn', className: 'AvgPool2d' },
  adaptiveavgpool2d: { source: 'torch.nn', className: 'AdaptiveAvgPool2d' },
  globalavgpool: { source: 'torch.nn', className: 'AdaptiveAvgPool2d' },
  batchnorm2d: { source: 'torch.nn', className: 'BatchNorm2d' },
  layernorm: { source: 'torch.nn', className: 'LayerNorm' },
  groupnorm: { source: 'torch.nn', className: 'GroupNorm' },
  dropout: { source: 'torch.nn', className: 'Dropout' },
  softmax: { source: 'torch.nn', className: 'Softmax' },
  flatten: { source: 'torch.nn', className: 'Flatten' },
  embedding: { source: 'torch.nn', className: 'Embedding' },
  instnorm: { source: 'torch.nn', className: 'InstanceNorm2d' },
  lstm: { source: 'torch.nn', className: 'LSTM' },
  multiheadattention: { source: 'torch.nn', className: 'MultiheadAttention' },
  // Transformer 模块 (在 torch.nn 中)
  transformerencoder: { source: 'torch.nn', className: 'TransformerEncoder' },
  transformerdecoder: { source: 'torch.nn', className: 'TransformerDecoder' },
  // 需要自定义实现的模块
  mamba: { source: 'flowhamster.modules', className: 'Mamba' },
  // ffn 和 mlp 使用 nn.Sequential 组合，不需要额外 import
  ffn: { source: null, className: '' },
  mlp: { source: null, className: '' },
  // Tensor 操作不需要 import
  add: { source: null, className: '' },
  mul: { source: null, className: '' },
  concat: { source: null, className: '' },
  reshape: { source: null, className: '' },
  transpose: { source: null, className: '' },
  split: { source: null, className: '' },
  slice: { source: null, className: '' },
  // crossattention 和 selfattention 不是标准 nn 模块
  selfattention: { source: 'torch.nn', className: 'MultiheadAttention' },
  crossattention: { source: 'torch.nn', className: 'CrossAttention' },
}

/**
 * Get import info for a node type
 */
export function getModuleImportInfo(opType: string): ModuleImportInfo | undefined {
  return MODULE_IMPORT_MAP[opType]
}

/**
 * Collect all custom module imports needed by blocks
 */
export function collectModuleImports(blocks: NodeBlock[]): Set<string> {
  const customModules = new Set<string>()

  for (const block of blocks) {
    if (block.category !== 'module') continue
    const info = MODULE_IMPORT_MAP[block.opType]
    if (!info) continue

    if (info.source === 'flowhamster.modules') {
      customModules.add(info.className)
    }
  }

  return customModules
}

/**
 * Generate module import statements
 */
export function genModuleImports(customModules: Set<string>): string {
  const lines: string[] = ['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F']

  // 添加自定义模块 import
  if (customModules.size > 0) {
    const moduleList = Array.from(customModules).join(', ')
    lines.push(`from flowhamster.modules import ${moduleList}`)
  }

  return lines.join('\n')
}

/**
 * Check if a node type requires a module import
 */
export function requiresModuleImport(opType: string): boolean {
  const info = MODULE_IMPORT_MAP[opType]
  return info?.source !== null && info?.source !== undefined
}
