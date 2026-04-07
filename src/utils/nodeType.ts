import { NodeData, NodeType } from '../types/graph'
import { getAllNodeTypes, nodeTypeExists } from './nodeDefinition'
import { getComponentName } from './nodeComponentRegistry'

// 节点类型别名映射
const NODE_TYPE_ALIASES: Record<string, NodeType> = {
  adaptiveavgpool: 'adaptiveavgpool2d',
  avgpool: 'avgpool2d',
  batchnorm: 'batchnorm2d',
  cosineannealing: 'cosineannealinglr',
  maxpool: 'maxpool2d',
  reduceonplateau: 'reducelronplateau',
  slicenode: 'slice',
  splitnode: 'split',
  transpose: 'transpose',
  transposenode: 'transpose',
}

/**
 * 规范化节点类型（处理别名）
 */
export function normalizeNodeType(nodeType: string): NodeType {
  // 先查别名
  if (nodeType in NODE_TYPE_ALIASES) {
    return NODE_TYPE_ALIASES[nodeType]
  }
  // 再检查是否已经是有效类型
  if (nodeTypeExists(nodeType)) {
    return nodeType as NodeType
  }
  // 返回原值（可能是 legacy 类型）
  return nodeType as NodeType
}

/**
 * 获取节点对应的 React Flow 组件名
 * 例如: 'transformerencoder' -> 'transformerencoderNode'
 */
export function getNodeComponentType(nodeType: string): string {
  const normalizedType = normalizeNodeType(nodeType)
  // group 类型特殊处理
  if (normalizedType === 'group') {
    return 'group'
  }
  return getComponentName(normalizedType)
}

/**
 * 规范化节点数据
 */
export function normalizeNodeData<T extends NodeData>(data: T): T {
  return {
    ...data,
    nodeType: normalizeNodeType(data.nodeType),
  }
}

/**
 * 检查节点类型是否有效
 */
export function isValidNodeType(nodeType: string): boolean {
  return nodeTypeExists(nodeType) || nodeType in NODE_TYPE_ALIASES
}

/**
 * 获取所有有效的节点类型（包括别名）
 */
export function getAllValidNodeTypes(): string[] {
  const aliases = Object.keys(NODE_TYPE_ALIASES)
  const defined = getAllNodeTypes()
  return [...new Set([...aliases, ...defined])]
}
