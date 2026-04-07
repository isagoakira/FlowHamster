/**
 * Graph Pruner
 *
 * Prunes the graph to only include nodes reachable from inputs
 * and necessary for outputs. Also assigns branch IDs for multi-output support.
 */

import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { normalizeNodeType } from './nodeType'
import { getSignature } from './nodeSignatures'

export interface PrunedGraph {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
  branchAssignments: Record<string, number>  // nodeId -> branchIndex (0,1,2...)
}

export interface FeatureToggles {
  tensorPreview?: boolean
  gradientViz?: boolean
  multiOutput?: boolean
  evaluationNodes?: boolean
}

/**
 * Prune the graph to only include nodes reachable from inputs and needed for outputs.
 */
export function pruneGraph(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  features?: FeatureToggles
): PrunedGraph {
  const nodeIds = new Set(nodes.map(n => n.id))

  // Filter out training nodes (they don't go in nn.Module)
  const nonTrainingIds = new Set(
    nodes.filter(n => {
      const opType = normalizeNodeType(n.data.nodeType as string)
      const sig = getSignature(opType)
      if (sig?.category === 'training') return false
      // When evaluationNodes feature is off, exclude evaluation nodes
      if (sig?.category === 'evaluation' && !features?.evaluationNodes) return false
      return true
    }).map(n => n.id)
  )

  // Get all Output nodes
  const outputNodes = nodes.filter(n => normalizeNodeType(n.data.nodeType as string) === 'output')

  // Build reverse adjacency list (target -> sources)
  const reverseAdj: Record<string, string[]> = {}
  for (const n of nodes) reverseAdj[n.id] = []
  for (const e of edges) {
    if (reverseAdj[e.target]) reverseAdj[e.target].push(e.source)
  }

  // branchAssignments[nodeId] = branchIndex
  const branchAssignments: Record<string, number> = {}

  // Reverse BFS from each Output node to mark reachable nodes
  outputNodes.forEach((outNode, outIdx) => {
    const reachable = new Set<string>()
    const queue = [outNode.id]
    while (queue.length > 0) {
      const nid = queue.shift()!
      if (reachable.has(nid)) continue
      reachable.add(nid)
      branchAssignments[nid] = outIdx
      for (const src of reverseAdj[nid] || []) {
        if (nonTrainingIds.has(src) && nodeIds.has(src)) queue.push(src)
      }
    }
  })

  // Build forward adjacency list
  const inputIds = new Set(
    nodes.filter(n => normalizeNodeType(n.data.nodeType as string) === 'input').map(n => n.id)
  )
  const forwardAdj: Record<string, string[]> = {}
  for (const n of nodes) forwardAdj[n.id] = []
  for (const e of edges) {
    if (forwardAdj[e.source]) forwardAdj[e.source].push(e.target)
  }

  // Forward BFS from all Input nodes
  const reachableFromInput = new Set<string>()
  let queue = [...inputIds]
  while (queue.length > 0) {
    const nid = queue.shift()!
    if (reachableFromInput.has(nid)) continue
    reachableFromInput.add(nid)
    for (const tgt of forwardAdj[nid] || []) {
      if (nonTrainingIds.has(tgt) && nodeIds.has(tgt)) queue.push(tgt)
    }
  }

  // Intersection: nodes that are both reachable from inputs AND needed for outputs
  const validIds = new Set<string>()
  for (const id of Object.keys(branchAssignments)) {
    if (reachableFromInput.has(id) || inputIds.has(id)) validIds.add(id)
  }
  // Always include input nodes
  for (const id of inputIds) validIds.add(id)

  const validNodes = nodes.filter(n => validIds.has(n.id))
  const validEdges = edges.filter(e => validIds.has(e.source) && validIds.has(e.target))

  return { nodes: validNodes, edges: validEdges, branchAssignments }
}
