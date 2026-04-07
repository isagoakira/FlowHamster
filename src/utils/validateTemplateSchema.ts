/**
 * 模板 Schema 验证器
 *
 * 验证从后端加载的模板是否符合 Schema
 */

import { TemplateSchema, TemplateValidationResult, ValidationError } from '../schema/template'
import { nodeTypeExists, isValidHandleId } from './nodeDefinition'

/**
 * 验证节点 ID 是否唯一
 */
function validateUniqueNodeIds(nodes: any[]): ValidationError[] {
  const errors: ValidationError[] = []
  const ids = new Set<string>()

  for (const node of nodes) {
    if (ids.has(node.id)) {
      errors.push({
        field: `nodes[${nodes.indexOf(node)}].id`,
        message: `Duplicate node ID: ${node.id}`,
      })
    }
    ids.add(node.id)
  }

  return errors
}

/**
 * 验证边连接的节点是否存在
 */
function validateEdgeConnections(nodes: any[], edges: any[]): ValidationError[] {
  const errors: ValidationError[] = []
  const nodeIds = new Set(nodes.map((n) => n.id))

  for (const edge of edges) {
    if (!nodeIds.has(edge.source)) {
      errors.push({
        field: `edges[${edges.indexOf(edge)}].source`,
        message: `Edge references unknown source node: ${edge.source}`,
      })
    }
    if (!nodeIds.has(edge.target)) {
      errors.push({
        field: `edges[${edges.indexOf(edge)}].target`,
        message: `Edge references unknown target node: ${edge.target}`,
      })
    }
  }

  return errors
}

/**
 * 验证节点类型是否有效
 */
function validateNodeTypes(nodes: any[]): ValidationError[] {
  const errors: ValidationError[] = []

  for (const node of nodes) {
    const nodeType = node.type
    if (!nodeType) {
      errors.push({
        field: `nodes[${nodes.indexOf(node)}].type`,
        message: `Node ${node.id} is missing type`,
      })
      continue
    }

    if (!nodeTypeExists(nodeType)) {
      errors.push({
        field: `nodes[${nodes.indexOf(node)}].type`,
        message: `Unknown node type: ${nodeType} in node ${node.id}`,
      })
    }
  }

  return errors
}

/**
 * 验证边的 Handle ID 是否有效
 */
function validateHandleIds(nodes: any[], edges: any[]): ValidationError[] {
  const errors: ValidationError[] = []
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))

  for (const edge of edges) {
    const targetNode = nodeMap.get(edge.target)
    if (!targetNode) continue

    const targetHandle = edge.targetHandle || 'a'
    const nodeType = targetNode.type

    if (nodeType && !isValidHandleId(targetHandle, nodeType)) {
      errors.push({
        field: `edges[${edges.indexOf(edge)}].targetHandle`,
        message: `Invalid targetHandle '${targetHandle}' for node type '${nodeType}'`,
      })
    }
  }

  return errors
}

/**
 * 验证模板 Schema
 */
export function validateTemplateSchema(template: TemplateSchema): TemplateValidationResult {
  const errors: ValidationError[] = []

  // 基本结构检查
  if (!template.id) {
    errors.push({ field: 'id', message: 'Template is missing id' })
  }
  if (!template.name) {
    errors.push({ field: 'name', message: 'Template is missing name' })
  }
  if (!template.graph) {
    errors.push({ field: 'graph', message: 'Template is missing graph' })
    return { valid: false, errors }
  }

  const { nodes = [], edges = [] } = template.graph

  // 验证节点 ID 唯一性
  errors.push(...validateUniqueNodeIds(nodes))

  // 验证边连接的节点存在
  errors.push(...validateEdgeConnections(nodes, edges))

  // 验证节点类型
  errors.push(...validateNodeTypes(nodes))

  // 验证 Handle ID
  errors.push(...validateHandleIds(nodes, edges))

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * 验证并返回人类可读的错误信息
 */
export function validateTemplateWithMessages(template: any): string {
  const result = validateTemplateSchema(template)

  if (result.valid) {
    return 'Template is valid'
  }

  const messages = result.errors.map((e) => `${e.field}: ${e.message}`)
  return `Template validation failed:\n- ${messages.join('\n- ')}`
}
