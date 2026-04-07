import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'

export const GROUP_NODE_TYPE = 'group'
export const GROUP_PADDING = 32
export const GROUP_MIN_WIDTH = 260
export const GROUP_MIN_HEIGHT = 180
export const DEFAULT_NODE_WIDTH = 180
export const DEFAULT_NODE_HEIGHT = 80

function toNumber(value: unknown, fallback: number) {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string') {
    const parsed = Number(value.replace('px', '').trim())
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

export function isGroupNode(node: Pick<FlowHamsterNode, 'type' | 'data'> | null | undefined): boolean {
  return Boolean(node) && (node?.type === GROUP_NODE_TYPE || node?.data?.nodeType === GROUP_NODE_TYPE)
}

export function getNodeDimensions(node: FlowHamsterNode) {
  return {
    width: toNumber(node.width ?? node.style?.width, DEFAULT_NODE_WIDTH),
    height: toNumber(node.height ?? node.style?.height, DEFAULT_NODE_HEIGHT),
  }
}

export function getAbsoluteNodePosition(
  node: FlowHamsterNode,
  nodeMap: Map<string, FlowHamsterNode>
) {
  let x = node.position.x
  let y = node.position.y
  let parentId = node.parentNode ?? node.parentId

  while (parentId) {
    const parentNode = nodeMap.get(parentId)
    if (!parentNode) break
    x += parentNode.position.x
    y += parentNode.position.y
    parentId = parentNode.parentNode ?? parentNode.parentId
  }

  return { x, y }
}

export function getDescendantNodeIds(nodes: FlowHamsterNode[], parentIds: string[]): string[] {
  const descendants = new Set<string>()
  const queue = [...parentIds]

  while (queue.length > 0) {
    const parentId = queue.shift()!
    for (const node of nodes) {
      const nodeParentId = node.parentNode ?? node.parentId
      if (nodeParentId === parentId && !descendants.has(node.id)) {
        descendants.add(node.id)
        queue.push(node.id)
      }
    }
  }

  return [...descendants]
}

export function getExecutableGraph(nodes: FlowHamsterNode[], edges: FlowHamsterEdge[]) {
  const executableNodes = nodes.filter((node) => !isGroupNode(node))
  const executableNodeIds = new Set(executableNodes.map((node) => node.id))
  const executableEdges = edges.filter(
    (edge) => executableNodeIds.has(edge.source) && executableNodeIds.has(edge.target)
  )

  return { nodes: executableNodes, edges: executableEdges }
}
