/**
 * 图工具函数
 *
 * 提取公共的图操作逻辑，避免代码重复
 */

import { Edge } from 'reactflow'

/**
 * 构建反向邻接表
 */
export function buildReverseAdj(nodes: string[], edges: Edge[]): Record<string, string[]> {
  const reverseAdj: Record<string, string[]> = {}
  for (const node of nodes) {
    reverseAdj[node] = []
  }
  for (const edge of edges) {
    if (reverseAdj[edge.target]) {
      reverseAdj[edge.target].push(edge.source)
    }
  }
  return reverseAdj
}

/**
 * 从节点出发的 BFS 遍历
 */
export function bfsFromNodes(
  startIds: string[],
  adj: Record<string, string[]>,
  visited: Set<string> = new Set()
): Set<string> {
  const queue = [...startIds]
  while (queue.length > 0) {
    const current = queue.shift()!
    if (visited.has(current)) continue
    visited.add(current)
    for (const neighbor of adj[current] || []) {
      if (!visited.has(neighbor)) {
        queue.push(neighbor)
      }
    }
  }
  return visited
}

/**
 * Kahn算法拓扑排序
 */
export function topologicalSort(
  nodeIds: string[],
  getIncomingEdges: (nodeId: string) => string[]
): string[] {
  const inDegree: Record<string, number> = {}
  const adj: Record<string, string[]> = {}

  for (const id of nodeIds) {
    inDegree[id] = 0
    adj[id] = []
  }

  // 构建邻接表和入度
  for (const id of nodeIds) {
    const sources = getIncomingEdges(id)
    inDegree[id] = sources.length
    for (const src of sources) {
      if (adj[src]) {
        adj[src].push(id)
      }
    }
  }

  // Kahn's algorithm
  const queue: string[] = nodeIds.filter((id) => inDegree[id] === 0)
  const sorted: string[] = []

  while (queue.length > 0) {
    const current = queue.shift()!
    sorted.push(current)
    for (const neighbor of adj[current] || []) {
      inDegree[neighbor]--
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor)
      }
    }
  }

  return sorted
}

/**
 * 生成唯一 ID
 */
export function makeGraphId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

/**
 * 安全获取数组元素
 */
export function safeArrayGet<T>(arr: T[], index: number, fallback: T): T {
  return index >= 0 && index < arr.length ? arr[index] : fallback
}

/**
 * 检查对象是否定义
 */
export function isDefined<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined
}
