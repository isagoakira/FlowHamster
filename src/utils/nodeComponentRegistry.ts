/**
 * 节点组件注册表
 *
 * 将节点类型映射到对应的 React 组件
 * 使用 Object.freeze() 冻结防止运行时修改
 */

import { NodeTypes } from 'reactflow'

// 导入所有节点组件
import InputNode from '../components/nodes/InputNode'
import OutputNode from '../components/nodes/OutputNode'
import Conv2dNode from '../components/nodes/Conv2dNode'
import Conv1dNode from '../components/nodes/Conv1dNode'
import Conv3dNode from '../components/nodes/Conv3dNode'
import ReLUNode from '../components/nodes/ReLUNode'
import GELUNode from '../components/nodes/GELUNode'
import SiLUNode from '../components/nodes/SiLUNode'
import SigmoidNode from '../components/nodes/SigmoidNode'
import TanhNode from '../components/nodes/TanhNode'
import LeakyReLUNode from '../components/nodes/LeakyReLUNode'
import BatchNormNode from '../components/nodes/BatchNormNode'
import LayerNormNode from '../components/nodes/LayerNormNode'
import GroupNormNode from '../components/nodes/GroupNormNode'
import MaxPoolNode from '../components/nodes/MaxPoolNode'
import AvgPoolNode from '../components/nodes/AvgPoolNode'
import AdaptiveAvgPoolNode from '../components/nodes/AdaptiveAvgPoolNode'
import GlobalAvgPoolNode from '../components/nodes/GlobalAvgPoolNode'
import LinearNode from '../components/nodes/LinearNode'
import EmbeddingNode from '../components/nodes/EmbeddingNode'
import FFNNode from '../components/nodes/FFNNode'
import MLPNode from '../components/nodes/MLPNode'
import DropoutNode from '../components/nodes/DropoutNode'
import DropPathNode from '../components/nodes/DropPathNode'
import AddNode from '../components/nodes/AddNode'
import MulNode from '../components/nodes/MulNode'
import ConcatNode from '../components/nodes/ConcatNode'
import ReshapeNode from '../components/nodes/ReshapeNode'
import FlattenNode from '../components/nodes/FlattenNode'
import TransposeNode from '../components/nodes/TransposeNode'
import SplitNode from '../components/nodes/SplitNode'
import SliceNode from '../components/nodes/SliceNode'
import PermuteNode from '../components/nodes/PermuteNode'
import SqueezeNode from '../components/nodes/SqueezeNode'
import ExpandNode from '../components/nodes/ExpandNode'
import ConstantNode from '../components/nodes/ConstantNode'
import ParameterNode from '../components/nodes/ParameterNode'
import SelfAttentionNode from '../components/nodes/SelfAttentionNode'
import CrossAttentionNode from '../components/nodes/CrossAttentionNode'
import MultiheadAttentionNode from '../components/nodes/MultiheadAttentionNode'
import TransformerEncoderNode from '../components/nodes/TransformerEncoderNode'
import TransformerDecoderNode from '../components/nodes/TransformerDecoderNode'
import MambaNode from '../components/nodes/MambaNode'
import CrossEntropyLossNode from '../components/nodes/CrossEntropyLossNode'
import MSELossNode from '../components/nodes/MSELossNode'
import AdamNode from '../components/nodes/AdamNode'
import AdamWNode from '../components/nodes/AdamWNode'
import SGDNode from '../components/nodes/SGDNode'
import RMSpropNode from '../components/nodes/RMSpropNode'
import StepLRNode from '../components/nodes/StepLRNode'
import CosineAnnealingNode from '../components/nodes/CosineAnnealingNode'
import ReduceLROnPlateauNode from '../components/nodes/ReduceLROnPlateauNode'
import EvaluationNode from '../components/nodes/EvaluationNode'
import GroupNode from '../components/nodes/GroupNode'
import PackagedGroupNode from '../components/nodes/PackagedGroupNode'

// 节点组件映射表
const NODE_COMPONENT_MAP: Record<string, React.ComponentType<any>> = {
  // Input/Output
  input: InputNode,
  output: OutputNode,

  // Convolution
  conv2d: Conv2dNode,
  conv1d: Conv1dNode,
  conv3d: Conv3dNode,

  // Activation
  relu: ReLUNode,
  gelu: GELUNode,
  silu: SiLUNode,
  sigmoid: SigmoidNode,
  tanh: TanhNode,
  leakyrelu: LeakyReLUNode,

  // Normalization
  batchnorm2d: BatchNormNode,
  layernorm: LayerNormNode,
  groupnorm: GroupNormNode,

  // Pooling
  maxpool2d: MaxPoolNode,
  avgpool2d: AvgPoolNode,
  adaptiveavgpool2d: AdaptiveAvgPoolNode,
  globalavgpool: GlobalAvgPoolNode,

  // Linear
  linear: LinearNode,
  embedding: EmbeddingNode,

  // FFN/MLP
  ffn: FFNNode,
  mlp: MLPNode,

  // Regularization
  dropout: DropoutNode,
  droppath: DropPathNode,

  // Tensor Operations
  add: AddNode,
  mul: MulNode,
  concat: ConcatNode,
  reshape: ReshapeNode,
  flatten: FlattenNode,
  transpose: TransposeNode,
  split: SplitNode,
  slice: SliceNode,
  permute: PermuteNode,
  squeeze: SqueezeNode,
  expand: ExpandNode,
  constant: ConstantNode,
  parameter: ParameterNode,

  // Attention
  selfattention: SelfAttentionNode,
  crossattention: CrossAttentionNode,
  multiheadattention: MultiheadAttentionNode,
  transformerencoder: TransformerEncoderNode,
  transformerdecoder: TransformerDecoderNode,
  mamba: MambaNode,

  // Loss
  crossentropyloss: CrossEntropyLossNode,
  mseloss: MSELossNode,
  focalloss: CrossEntropyLossNode,
  labelsmoothing: CrossEntropyLossNode,

  // Optimizer
  adam: AdamNode,
  adamw: AdamWNode,
  sgd: SGDNode,
  rmsprop: RMSpropNode,

  // Scheduler
  steplr: StepLRNode,
  cosineannealinglr: CosineAnnealingNode,
  reducelronplateau: ReduceLROnPlateauNode,

  // Evaluation
  accuracy: EvaluationNode,
  f1: EvaluationNode,
  precision: EvaluationNode,
  recall: EvaluationNode,
  confusion_matrix: EvaluationNode,
  mean_iou: EvaluationNode,
  roc_auc: EvaluationNode,

  // Custom
  custom: PackagedGroupNode,
  group: GroupNode,
}

// 冻结映射表，防止运行时修改
const FROZEN_NODE_COMPONENT_MAP = Object.freeze(NODE_COMPONENT_MAP)

/**
 * 获取节点类型对应的 React Flow 组件名
 * 例如: 'transformerencoder' -> 'transformerencoderNode'
 */
export function getComponentName(nodeType: string): string {
  return `${nodeType}Node`
}

/**
 * 获取节点组件
 * 例如: 'transformerencoder' -> TransformerEncoderNode
 */
export function getNodeComponent(nodeType: string): React.ComponentType<any> | undefined {
  return FROZEN_NODE_COMPONENT_MAP[nodeType]
}

/**
 * 生成 React Flow 的 nodeTypes 对象
 */
export function getReactFlowNodeTypes(): NodeTypes {
  const result: NodeTypes = {}

  for (const [nodeType, Component] of Object.entries(FROZEN_NODE_COMPONENT_MAP)) {
    result[getComponentName(nodeType)] = Component
  }

  return Object.freeze(result)
}

/**
 * 检查节点类型是否有对应的组件
 */
export function hasNodeComponent(nodeType: string): boolean {
  return nodeType in FROZEN_NODE_COMPONENT_MAP
}

/**
 * 获取所有注册的节点类型
 */
export function getRegisteredNodeTypes(): string[] {
  return Object.keys(FROZEN_NODE_COMPONENT_MAP)
}

// 导出冻结的映射表
export { FROZEN_NODE_COMPONENT_MAP }
