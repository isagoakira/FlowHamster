import { useCallback } from 'react'
import { Node } from 'reactflow'
import dagre from '@dagrejs/dagre'
import { GROUP_NODE_TYPE } from '../utils/graphStructure'

const NODE_WIDTH = 180
const NODE_HEIGHT = 80

export function useAutoLayout() {
  const layout = useCallback((nodes: Node[], edges: any[]) => {
    if (nodes.length === 0) return { nodes, edges }

    const g = new dagre.graphlib.Graph()
    g.setDefaultEdgeLabel(() => ({}))
    g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 100 })

    const layoutableNodes = nodes.filter((node) => !node.parentNode)
    const layoutableNodeIds = new Set(layoutableNodes.map((node) => node.id))

    layoutableNodes.forEach((node) => {
      const width = node.type === GROUP_NODE_TYPE ? Number(node.style?.width ?? 260) : NODE_WIDTH
      const height = node.type === GROUP_NODE_TYPE ? Number(node.style?.height ?? 180) : NODE_HEIGHT
      g.setNode(node.id, { width, height })
    })

    edges.forEach((edge) => {
      if (layoutableNodeIds.has(edge.source) && layoutableNodeIds.has(edge.target)) {
        g.setEdge(edge.source, edge.target)
      }
    })

    dagre.layout(g)

    const layoutedNodes = nodes.map((node) => {
      if (!layoutableNodeIds.has(node.id)) return node
      const { x, y } = g.node(node.id)
      return {
        ...node,
        position: { x, y },
      }
    })

    return { nodes: layoutedNodes, edges }
  }, [])

  return { layout }
}
