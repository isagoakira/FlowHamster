/**
 * Node Signatures Registry
 *
 * Defines input/output signatures for all node types.
 * This is the first step in code generation.
 */

export type NodeCategory = 'io' | 'module' | 'operation' | 'training' | 'evaluation'

export interface PortSig { type: string }
export interface NodeSig {
  inputs: Record<string, PortSig>
  outputs: Record<string, string>
  category: NodeCategory
}

const SIGNATURES: Record<string, NodeSig> = {}

function register(
  opType: string,
  inputs: Record<string, PortSig>,
  outputs: Record<string, string>,
  category: NodeCategory
) {
  SIGNATURES[opType] = { inputs, outputs, category }
}

// IO
register('input',    {},                              { result: 'tensor' }, 'io')
register('output',   { x: { type: 'tensor' } },       {},                    'io')

// Modules
register('conv2d',   { in_channels: { type: 'int' }, out_channels: { type: 'int' } }, { result: 'tensor' }, 'module')
register('conv1d',   { in_channels: { type: 'int' }, out_channels: { type: 'int' } }, { result: 'tensor' }, 'module')
register('conv3d',   { in_channels: { type: 'int' }, out_channels: { type: 'int' } }, { result: 'tensor' }, 'module')
register('linear',   { in_features: { type: 'int' }, out_features: { type: 'int' } }, { result: 'tensor' }, 'module')
register('embedding', {},                             { result: 'tensor' }, 'module')
register('relu',     {},                              { result: 'tensor' }, 'module')
register('gelu',     {},                              { result: 'tensor' }, 'module')
register('silu',     {},                              { result: 'tensor' }, 'module')
register('sigmoid',  {},                              { result: 'tensor' }, 'module')
register('tanh',     {},                              { result: 'tensor' }, 'module')
register('leakyrelu',{},                              { result: 'tensor' }, 'module')
register('maxpool2d',{},                              { result: 'tensor' }, 'module')
register('avgpool2d',{},                              { result: 'tensor' }, 'module')
register('adaptiveavgpool2d',{},                      { result: 'tensor' }, 'module')
register('globalavgpool',{},                          { result: 'tensor' }, 'module')
register('batchnorm2d',{},                            { result: 'tensor' }, 'module')
register('layernorm',{},                              { result: 'tensor' }, 'module')
register('groupnorm',{},                              { result: 'tensor' }, 'module')
register('dropout',  {},                              { result: 'tensor' }, 'module')
register('droppath', {},                              { result: 'tensor' }, 'module')
register('softmax',  {},                              { result: 'tensor' }, 'module')
register('flatten',  {},                              { result: 'tensor' }, 'module')
register('selfattention',{},                          { result: 'tensor' }, 'module')
register('crossattention',{ q: { type: 'tensor' }, kv: { type: 'tensor' } }, { result: 'tensor' }, 'module')
register('multiheadattention',{},                     { result: 'tensor' }, 'module')
register('transformerencoder',{},                    { result: 'tensor' }, 'module')
register('transformerdecoder',{},                    { result: 'tensor' }, 'module')
register('mamba',   {},                              { result: 'tensor' }, 'module')
register('ffn',      {},                              { result: 'tensor' }, 'module')
register('mlp',      {},                              { result: 'tensor' }, 'module')

// Operations
register('add',      { a: { type: 'tensor' }, b: { type: 'tensor' } }, { result: 'tensor' }, 'operation')
register('mul',      { a: { type: 'tensor' }, b: { type: 'tensor' } }, { result: 'tensor' }, 'operation')
register('concat',   {},                              { result: 'tensor' }, 'operation')
register('__iconcat__', {},                           { result: 'tensor' }, 'operation')  // 隐式 concat（多边汇聚）
register('reshape',  {},                              { result: 'tensor' }, 'operation')
register('transpose',{},                              { result: 'tensor' }, 'operation')
register('split',    {},                              { result: 'tensor' }, 'operation')
register('slice',    {},                              { result: 'tensor' }, 'operation')
register('permute',  {},                              { result: 'tensor' }, 'operation')
register('squeeze',  {},                              { result: 'tensor' }, 'operation')
register('expand',   {},                              { result: 'tensor' }, 'operation')
register('constant', {},                              { result: 'tensor' }, 'operation')
register('parameter',{},                             { result: 'tensor' }, 'operation')

// Training 组件（不参与 nn.Module 定义）
register('adam',     {},                              { result: 'tensor' }, 'training')
register('adamw',    {},                              { result: 'tensor' }, 'training')
register('sgd',      {},                              { result: 'tensor' }, 'training')
register('rmsprop',  {},                              { result: 'tensor' }, 'training')
register('crossentropyloss',{},                      { result: 'tensor' }, 'training')
register('mseloss',  {},                              { result: 'tensor' }, 'training')
register('focalloss',{},                             { result: 'tensor' }, 'training')
register('labelsmoothing',{},                        { result: 'tensor' }, 'training')
register('cosineannealinglr',{},                     { result: 'tensor' }, 'training')
register('steplr',   {},                              { result: 'tensor' }, 'training')
register('reducelronplateau',{},                     { result: 'tensor' }, 'training')

// Evaluation
register('accuracy', {},                               { result: 'scalar' }, 'evaluation')
register('f1',       {},                               { result: 'scalar' }, 'evaluation')
register('precision', {},                              { result: 'scalar' }, 'evaluation')
register('recall',   {},                               { result: 'scalar' }, 'evaluation')
register('confusion_matrix', {},                     { result: 'matrix' }, 'evaluation')
register('mean_iou', {},                              { result: 'scalar' }, 'evaluation')
register('roc_auc',  {},                              { result: 'scalar' }, 'evaluation')

/**
 * Get signature for a node type
 */
export function getSignature(opType: string): NodeSig | undefined {
  return SIGNATURES[opType]
}

/**
 * Get category for a node type
 */
export function getNodeCategory(opType: string): NodeCategory {
  return SIGNATURES[opType]?.category ?? 'module'
}

/**
 * Check if a node type is a training component
 */
export function isTrainingNode(opType: string): boolean {
  return SIGNATURES[opType]?.category === 'training'
}

/**
 * Check if a node type is an evaluation component
 */
export function isEvaluationNode(opType: string): boolean {
  return SIGNATURES[opType]?.category === 'evaluation'
}

/**
 * Get all registered node types
 */
export function getRegisteredNodeTypes(): string[] {
  return Object.keys(SIGNATURES)
}

export { SIGNATURES }
