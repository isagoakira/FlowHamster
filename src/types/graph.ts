import { Node, Edge } from 'reactflow'

export type CanonicalNodeType =
  | 'group'
  | 'input'
  | 'output'
  | 'conv1d'
  | 'conv2d'
  | 'conv3d'
  | 'linear'
  | 'embedding'
  | 'relu'
  | 'gelu'
  | 'silu'
  | 'sigmoid'
  | 'tanh'
  | 'leakyrelu'
  | 'batchnorm2d'
  | 'layernorm'
  | 'groupnorm'
  | 'maxpool2d'
  | 'avgpool2d'
  | 'adaptiveavgpool2d'
  | 'globalavgpool'
  | 'selfattention'
  | 'crossattention'
  | 'multiheadattention'
  | 'transformerencoder'
  | 'transformerdecoder'
  | 'mamba'
  | 'ffn'
  | 'mlp'
  | 'concat'
  | 'add'
  | 'mul'
  | 'reshape'
  | 'flatten'
  | 'transpose'
  | 'permute'
  | 'split'
  | 'slice'
  | 'squeeze'
  | 'expand'
  | 'constant'
  | 'parameter'
  | 'dropout'
  | 'droppath'
  | 'softmax'
  | 'crossentropyloss'
  | 'mseloss'
  | 'focalloss'
  | 'labelsmoothing'
  | 'adam'
  | 'adamw'
  | 'sgd'
  | 'rmsprop'
  | 'steplr'
  | 'cosineannealinglr'
  | 'reducelronplateau'
  | 'accuracy'
  | 'f1'
  | 'precision'
  | 'recall'
  | 'confusion_matrix'
  | 'mean_iou'
  | 'roc_auc'
  | 'custom'

export type LegacyNodeType =
  | 'adaptiveavgpool'
  | 'avgpool'
  | 'batchnorm'
  | 'cosineannealing'
  | 'maxpool'
  | 'reduceonplateau'
  | 'splitnode'
  | 'slicenode'
  | 'transposenode'

export type NodeType = CanonicalNodeType | LegacyNodeType

// Conv2d params
export interface Conv2dParams {
  in_channels: number
  out_channels: number
  kernel_size: number
  stride: number
  padding: number
  bias: boolean
}

// Input/Output params
export interface InputParams {
  shape: string // e.g. "3,224,224"
  dtype: string  // e.g. "float32"
}

// Linear params
export interface LinearParams {
  in_features: number
  out_features: number
  bias: boolean
}

// Node data payload
export interface NodeData {
  nodeType: NodeType
  label: string
  params: Record<string, number | string | boolean>
  [key: string]: unknown
}

// Composite node base - shared by both predefined and custom composites
export interface CompositeNodeBase extends NodeData {
  isComposite: true
  internalStructure: SubModuleData[]
  internalEdges: InternalEdgeData[]
  outputVar: string
}

// Predefined composite node (like TransformerEncoder, SelfAttention)
export interface PredefinedCompositeNodeData extends CompositeNodeBase {
  isPredefined: true
  compositeType: string // e.g., 'transformerencoder', 'selfattention'
}

// Custom composite class node (user-created from packaged group)
export interface CustomCompositeNodeData extends CompositeNodeBase {
  isCustomComposite: true
  customClassId: string
  customClassRegistryId?: string
  originClassId?: string
  // 外部接口（自动推断）
  inputs: GroupPort[]
  outputs: GroupPort[]
  // UI 状态
  isExpanded: boolean
  childNodeIds: string[]
  internalEdgeIds: string[]
  // 边界边信息（保留原始 edge ID）
  boundaryEdges?: BoundaryEdge[]
}

export interface BoundaryEdge {
  originalEdgeId: string
  direction: 'input' | 'output'
  internalNodeId: string
  internalHandle: string
  groupHandleId?: string
  source: string
  target: string
  sourceHandle?: string | null
  targetHandle?: string | null
}

// Group port for external connections
export interface GroupPort {
  id: string
  handleId: string
  label: string
  nodeId: string // 内部连接的节点 ID
  edgeId?: string // 原始边 ID，用于恢复
}

export interface SubModuleData {
  id: string
  type: string
  label: string
  params: Record<string, number | string | boolean>
  position?: { x: number; y: number }  // 原始位置，解包时恢复
  customClassId?: string  // For custom composite nodes inside another package
  data?: NodeData
}

export interface InternalEdgeData {
  id?: string
  from: string
  to: string
  fromHandle?: string
  toHandle?: string
}

export type FlowHamsterNode = Node<NodeData>
export type FlowHamsterEdge = Edge
