/**
 * 模板格式适配器
 *
 * 将后端返回的模板格式转换为前端格式
 */

import { TemplateSchema, TemplateNode, TemplateEdge } from '../../schema/template'
import { FlowHamsterNode, FlowHamsterEdge, NodeType } from '../../types/graph'
import { getNodeComponentType, normalizeNodeData } from '../nodeType'

export interface AdaptedTemplate {
  id: string
  name: string
  description: string
  category: string
  emoji: string
  graph: {
    nodes: FlowHamsterNode[]
    edges: FlowHamsterEdge[]
  }
  backend_code?: string
}

/**
 * 将后端模板节点转换为前端节点
 */
function adaptNode(node: TemplateNode): FlowHamsterNode {
  return {
    id: node.id,
    type: getNodeComponentType(node.type) as NodeType,
    position: node.position || { x: 0, y: 0 },
    data: normalizeNodeData({
      nodeType: node.type as NodeType,
      label: node.label || node.type,
      params: node.params || {},
    }),
  }
}

/**
 * 将后端模板边转换为前端边
 */
function adaptEdge(edge: TemplateEdge): FlowHamsterEdge {
  return {
    id: edge.id || `e_${edge.source}_${edge.target}`,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle || 'result',
    targetHandle: edge.targetHandle || 'a',
  }
}

/**
 * 转换后端模板为前端格式
 */
export function adaptBackendTemplate(template: TemplateSchema): AdaptedTemplate {
  const adaptedNodes = template.graph.nodes.map((node) => adaptNode(node))
  const adaptedEdges = template.graph.edges.map((edge) => adaptEdge(edge))

  return {
    id: template.id,
    name: template.name,
    description: template.description,
    category: template.category,
    emoji: template.emoji,
    graph: {
      nodes: adaptedNodes,
      edges: adaptedEdges,
    },
    backend_code: template.backend_code?.python,
  }
}

/**
 * 转换节点列表（用于 makeLayout）
 */
export function adaptNodesWithLayout(
  nodes: TemplateNode[],
  layoutFn: (nodes: any[], numRows?: number) => any[]
): FlowHamsterNode[] {
  // 先转换节点
  const adaptedNodes = nodes.map((node) => adaptNode(node))

  // 应用布局
  const laidOutNodes = layoutFn(adaptedNodes)

  return laidOutNodes
}
