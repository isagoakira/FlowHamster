/**
 * AST Builder
 *
 * Converts a pruned graph into an Abstract Syntax Tree (AST) of blocks.
 * Each block represents a node with its inputs, outputs, and metadata.
 */

import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { normalizeNodeType } from './nodeType'
import { getSignature, getNodeCategory, NodeCategory } from './nodeSignatures'
import { pruneGraph, FeatureToggles } from './graphPruner'

export interface NodeBlock {
  nodeId: string
  opType: string
  category: NodeCategory
  fields: Record<string, any>
  inputs: Record<string, string | string[] | null>
  outputVar: string
  instanceName: string
  branchIndex: number  // From which Output branch (0,1,2...)
}

function safeId(id: string): string {
  return id.replace(/[^A-Za-z0-9_]/g, '_').replace(/^(\d)/, '_$1')
}

function getSourceRefs(input: string | string[] | null | undefined): string[] {
  if (typeof input === 'string') return [input]
  if (Array.isArray(input)) return input.filter((item): item is string => typeof item === 'string')
  return []
}

function getSourceNodeId(input: string | string[] | null | undefined): string | null {
  const firstRef = getSourceRefs(input)[0]
  if (!firstRef) return null
  return firstRef.split(':')[1]?.split('/')[0] ?? null
}

function normalizeNameToken(value: string): string {
  return safeId(value)
    .replace(/^_+|_+$/g, '')
    .replace(/__+/g, '_')
    .toLowerCase()
}

function getBlockNameToken(block: NodeBlock): string {
  if (block.opType === '__iconcat__') {
    return `merge_${normalizeNameToken(String(block.fields?.mergeMode || 'concat'))}`
  }
  return normalizeNameToken(block.opType)
}

export interface BuildASTResult {
  blocks: NodeBlock[]
  branchAssignments: Record<string, number>
  outputBlocks: NodeBlock[]
}

/**
 * Build AST from graph nodes and edges
 */
export function buildAST(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  features?: FeatureToggles
): BuildASTResult {
  const { nodes: prunedNodes, edges: prunedEdges, branchAssignments } = pruneGraph(nodes, edges, features)
  const blockMap: Record<string, NodeBlock> = {}

  // Step 1: Create blocks for each node
  for (const n of prunedNodes) {
    const opType = normalizeNodeType(n.data.nodeType as string)
    const sig = getSignature(opType)
    const category = sig?.category ?? getNodeCategory(opType)
    const fields: Record<string, any> = { ...n.data.params, label: n.data.label }
    // For custom composite nodes, include customClassId and internalStructure in fields
    if (n.data.nodeType === 'custom' && (n.data as any).customClassId) {
      fields.customClassId = (n.data as any).customClassId
      // Include internalStructure for recursive custom class discovery
      if ((n.data as any).internalStructure) {
        fields.internalStructure = (n.data as any).internalStructure
      }
    }
    blockMap[n.id] = {
      nodeId: n.id, opType, category, fields,
      inputs: {}, outputVar: '',
      instanceName: '',
      branchIndex: branchAssignments[n.id] ?? 0,
    }
  }

  // Step 2: Fill in input sources from edges
  // Track port edge counts to detect when multiple edges connect to same port
  const portEdgeCounts: Record<string, number> = {}  // key = "nodeId|port"
  const portFirstRef: Record<string, string> = {}     // key = "nodeId|port" → first ref

  for (const edge of prunedEdges) {
    const tgt = blockMap[edge.target]
    if (!tgt) continue
    const srcHandle = edge.sourceHandle || 'result'
    const tgtHandle = edge.targetHandle || 'a'
    const portKey = `${edge.target}|${tgtHandle}`
    const ref = `node:${edge.source}/${srcHandle}`

    if (tgt.opType === 'concat' || tgt.opType === '__iconcat__') {
      const ci = tgt.inputs as Record<string, string[]>
      if (!(tgtHandle in ci)) ci[tgtHandle] = []
      ci[tgtHandle].push(ref)
    } else {
      const port = tgt.opType === 'output' ? 'x' : tgtHandle
      const existing = (tgt.inputs as Record<string, string | null>)[port]

      if (existing && existing !== ref) {
        if (
          (tgt.opType === 'add' || tgt.opType === 'mul') &&
          port === 'a' &&
          !(tgt.inputs as Record<string, string | null>).b
        ) {
          ;(tgt.inputs as Record<string, string | null>).b = ref
          portEdgeCounts[`${edge.target}|b`] = 1
          continue
        }
        // Multiple edges to same port → create implicit concat node
        const icatId = `__icat_${edge.target}_${port}`
        if (!blockMap[icatId]) {
          const mergeMode = (edge as any).data?.mergeMode || 'concat'
          blockMap[icatId] = {
            nodeId: icatId,
            opType: '__iconcat__',
            category: 'operation',
            fields: { dim: 1, mergeMode, _implicit: true },
            inputs: { in_0: [], in_1: [] },
            outputVar: '',
            instanceName: '',
            branchIndex: branchAssignments[edge.target] ?? 0,
          }
          // Move existing reference to in_0
          const oldRef = portFirstRef[portKey] || existing
          ;(blockMap[icatId].inputs as Record<string, string[]>).in_0.push(oldRef)
        }
        // Add current edge to in_1
        const icatBlock = blockMap[icatId]
        ;(icatBlock.inputs as Record<string, string[]>).in_1.push(ref)
        // Update target's port to point to implicit concat
        ;(tgt.inputs as Record<string, string | null>)[port] = `node:${icatId}/result`
      } else if (!existing) {
        (tgt.inputs as Record<string, string | null>)[port] = ref
        portFirstRef[portKey] = ref
      }
      portEdgeCounts[portKey] = (portEdgeCounts[portKey] || 0) + 1
    }
  }

  // Step 3: Topological sort (Kahn's algorithm)
  const inDegree: Record<string, number> = {}
  const adj: Record<string, string[]> = {}
  for (const id of Object.keys(blockMap)) { inDegree[id] = 0; adj[id] = [] }

  for (const block of Object.values(blockMap)) {
    if (block.opType === 'concat' || block.opType === '__iconcat__') {
      const allSources = new Set<string>()
      const ci = block.inputs as Record<string, string[]>
      for (const srcList of Object.values(ci)) {
        if (Array.isArray(srcList)) {
          for (const r of srcList) {
            const srcId = getSourceNodeId(r)
            if (srcId && srcId in inDegree) allSources.add(srcId)
          }
        }
      }
      for (const srcId of allSources) { inDegree[block.nodeId]++; adj[srcId].push(block.nodeId) }
    } else {
      const ni = block.inputs as Record<string, string | null>
      for (const srcRef of Object.values(ni)) {
        const srcId = getSourceNodeId(srcRef)
        if (srcId) {
          if (srcId in inDegree) { inDegree[block.nodeId]++; adj[srcId].push(block.nodeId) }
        }
      }
    }
  }

  const sortedIds: string[] = []
  const q: string[] = Object.keys(blockMap).filter(id => inDegree[id] === 0)
  while (q.length > 0) {
    const id = q.shift()!
    sortedIds.push(id)
    for (const nb of adj[id]) { inDegree[nb]--; if (inDegree[nb] === 0) q.push(nb) }
  }

  // Step 4: Assign output variable names and instance names
  const used = new Set<string>()
  const nameTokenCounter: Record<string, number> = {}

  for (const id of sortedIds) {
    const block = blockMap[id]
    const nameToken = getBlockNameToken(block)
    nameTokenCounter[nameToken] = (nameTokenCounter[nameToken] || 0) + 1
    const readableName = `${nameToken}_${nameTokenCounter[nameToken]}`

    // 对于 custom 类型，使用 label 作为实例名（确保有效的变量名）
    if (block.opType === 'custom' && block.fields.label) {
      const safeLabel = safeId(String(block.fields.label))
      block.instanceName = `x_${safeLabel}`
      block.outputVar = `x_${safeLabel}`
    } else {
      block.instanceName = `x_${readableName}`
      block.outputVar = `x_${readableName}`
    }

    let suffix = 1
    while (used.has(block.outputVar)) {
      suffix += 1
      block.instanceName = `x_${readableName}_${suffix}`
      block.outputVar = `x_${readableName}_${suffix}`
    }
    used.add(block.outputVar)
  }

  const blocks = sortedIds.map(id => blockMap[id])
  const outputBlocks = blocks.filter(b => b.opType === 'output')

  return { blocks, branchAssignments, outputBlocks }
}

export { getSourceNodeId, getSourceRefs }
