/**
 * 模板验证器
 *
 * 验证模板中的节点类型和 Handle ID 是否合法
 */

import { nodeTypeExists, isValidHandleId } from './nodeDefinition'

export interface ValidationError {
  type: 'unknown_node_type' | 'invalid_handle' | 'missing_node' | 'invalid_edge'
  message: string
  nodeId?: string
  edgeId?: string
}

export interface ValidationResult {
  valid: boolean
  errors: ValidationError[]
}

/**
 * 验证模板中的节点类型是否都有效
 */
export function validateNodeTypes(template: { nodes: any[] }): ValidationError[] {
  const errors: ValidationError[] = []

  for (const node of template.nodes) {
    const nodeType = node.data?.nodeType
    if (!nodeType) {
      errors.push({
        type: 'unknown_node_type',
        message: `Node ${node.id} is missing nodeType`,
        nodeId: node.id,
      })
      continue
    }

    if (!nodeTypeExists(nodeType)) {
      errors.push({
        type: 'unknown_node_type',
        message: `Unknown node type: ${nodeType} in node ${node.id}`,
        nodeId: node.id,
      })
    }
  }

  return errors
}

/**
 * 验证模板中的边连接是否有效
 */
export function validateEdges(template: { nodes: any[]; edges: any[] }): ValidationError[] {
  const errors: ValidationError[] = []
  const nodeIds = new Set(template.nodes.map(n => n.id))

  for (const edge of template.edges) {
    // 检查源节点是否存在
    if (!nodeIds.has(edge.source)) {
      errors.push({
        type: 'missing_node',
        message: `Edge ${edge.id} references unknown source node: ${edge.source}`,
        edgeId: edge.id,
      })
    }

    // 检查目标节点是否存在
    if (!nodeIds.has(edge.target)) {
      errors.push({
        type: 'missing_node',
        message: `Edge ${edge.id} references unknown target node: ${edge.target}`,
        edgeId: edge.id,
      })
      continue
    }

    // 获取目标节点
    const targetNode = template.nodes.find(n => n.id === edge.target)
    if (!targetNode) continue

    const targetNodeType = targetNode.data?.nodeType
    if (!targetNodeType) continue

    // 检查 targetHandle 是否有效
    const targetHandle = edge.targetHandle || 'a' // 默认是 'a'
    if (!isValidHandleId(targetHandle, targetNodeType)) {
      errors.push({
        type: 'invalid_handle',
        message: `Invalid targetHandle '${targetHandle}' for node type '${targetNodeType}' in edge ${edge.id}`,
        edgeId: edge.id,
        nodeId: edge.target,
      })
    }
  }

  return errors
}

/**
 * 完整验证模板
 */
export function validateTemplate(template: { nodes: any[]; edges: any[] }): ValidationResult {
  const errors: ValidationError[] = []

  // 验证节点类型
  errors.push(...validateNodeTypes(template))

  // 验证边连接
  errors.push(...validateEdges(template))

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * 验证并返回人类可读的错误信息
 */
export function validateTemplateWithMessages(template: { nodes: any[]; edges: any[] }): string {
  const result = validateTemplate(template)

  if (result.valid) {
    return 'Template is valid'
  }

  const messages = result.errors.map(e => e.message)
  return `Template validation failed:\n- ${messages.join('\n- ')}`
}
