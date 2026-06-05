import dagre from '@dagrejs/dagre'
import { Node, Edge } from 'reactflow'
import { getNodeDimensions } from './graphStructure'

export type LayoutDirection = 'LR' | 'TB'

export interface LayoutEngineOptions {
  direction?: LayoutDirection
  nodeSep?: number
  rankSep?: number
  marginX?: number
  marginY?: number
  align?: 'UL' | 'UR' | 'DL' | 'DR'
}

export interface LayoutBounds {
  width: number
  height: number
  minX: number
  minY: number
}

export interface LayoutEngineResult<TNodeData = unknown, TEdgeData = unknown> {
  nodes: Node<TNodeData>[]
  edges: Edge<TEdgeData>[]
  bounds: LayoutBounds
  durationMs: number
}

const DEFAULT_OPTIONS: Required<LayoutEngineOptions> = {
  direction: 'LR',
  nodeSep: 60,
  rankSep: 100,
  marginX: 20,
  marginY: 20,
  align: 'UL',
}

/**
 * Compute a layered graph layout using the Dagre algorithm.
 *
 * - Only top-level nodes (no parentNode) are passed to Dagre.
 * - Child nodes retain their existing positions.
 * - Node dimensions are read from node.width/height or node.style.width/height,
 *   falling back to DEFAULT_NODE_WIDTH / DEFAULT_NODE_HEIGHT.
 * - Returns bounding box and execution time for performance monitoring.
 */
export function computeLayout<TNodeData = unknown, TEdgeData = unknown>(
  nodes: Node<TNodeData>[],
  edges: Edge<TEdgeData>[],
  options: LayoutEngineOptions = {}
): LayoutEngineResult<TNodeData, TEdgeData> {
  const start = performance.now()
  const opts = { ...DEFAULT_OPTIONS, ...options }

  if (nodes.length === 0) {
    return {
      nodes: [],
      edges,
      bounds: { width: 0, height: 0, minX: 0, minY: 0 },
      durationMs: 0,
    }
  }

  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({
    rankdir: opts.direction,
    nodesep: opts.nodeSep,
    ranksep: opts.rankSep,
    marginx: opts.marginX,
    marginy: opts.marginY,
    align: opts.align,
  })

  // Only layout top-level nodes (no parentNode)
  const layoutableNodes = nodes.filter((node) => !node.parentNode)
  const layoutableNodeIds = new Set(layoutableNodes.map((node) => node.id))

  layoutableNodes.forEach((node) => {
    const { width, height } = getNodeDimensions(node as Node<unknown>)
    g.setNode(node.id, { width, height })
  })

  edges.forEach((edge) => {
    if (layoutableNodeIds.has(edge.source) && layoutableNodeIds.has(edge.target)) {
      g.setEdge(edge.source, edge.target)
    }
  })

  dagre.layout(g)

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  const layoutedNodes = nodes.map((node) => {
    if (!layoutableNodeIds.has(node.id)) return node

    const dagreNode = g.node(node.id)
    const x = dagreNode.x - dagreNode.width / 2
    const y = dagreNode.y - dagreNode.height / 2

    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x + dagreNode.width)
    maxY = Math.max(maxY, y + dagreNode.height)

    return {
      ...node,
      position: { x, y },
    }
  })

  const bounds: LayoutBounds =
    minX === Infinity
      ? { width: 0, height: 0, minX: 0, minY: 0 }
      : {
          width: maxX - minX,
          height: maxY - minY,
          minX,
          minY,
        }

  const durationMs = performance.now() - start

  return {
    nodes: layoutedNodes,
    edges,
    bounds,
    durationMs,
  }
}

/**
 * Check whether any two nodes overlap in the given array.
 * Used for layout validation in tests.
 */
export function hasOverlappingNodes<T>(nodes: Node<T>[]): boolean {
  const rects = nodes
    .filter((n) => !n.parentNode)
    .map((n) => {
      const { width, height } = getNodeDimensions(n as Node<unknown>)
      return {
        id: n.id,
        left: n.position.x,
        top: n.position.y,
        right: n.position.x + width,
        bottom: n.position.y + height,
      }
    })

  for (let i = 0; i < rects.length; i++) {
    for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i]
      const b = rects[j]
      if (a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top) {
        return true
      }
    }
  }
  return false
}
