import { useMemo } from 'react'
import { useGraphStore } from './useGraphStore'
import { useDataGraphStore } from './useDataGraphStore'
import { AgentChatContext } from '../types/agentChat'

const MAX_CONTEXT_NODES = 50
const MAX_CONTEXT_EDGES = 100
const MAX_SELECTED_NODES = 10
const MAX_ERROR_LOGS = 5
const MAX_PARAMS_LENGTH = 200

function truncateParams(params: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(params)) {
    const str = JSON.stringify(value)
    if (str.length > MAX_PARAMS_LENGTH) {
      result[key] = str.slice(0, MAX_PARAMS_LENGTH) + '...'
    } else {
      result[key] = value
    }
  }
  return result
}

function summarizeGraph(nodes: any[], edges: any[]) {
  const summaryNodes = nodes.slice(0, MAX_CONTEXT_NODES).map((n) => ({
    id: n.id,
    type: n.data?.nodeType || n.type,
    label: n.data?.label || n.id,
    params: truncateParams(n.data?.params || {}),
  }))

  const summaryEdges = edges.slice(0, MAX_CONTEXT_EDGES).map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
  }))

  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    nodes: summaryNodes,
    edges: summaryEdges,
    truncated: nodes.length > MAX_CONTEXT_NODES || edges.length > MAX_CONTEXT_EDGES,
  }
}

/**
 * Collect recent error logs from the window error buffer (if any) or console.
 * In a real app this could come from a dedicated error logging service.
 */
function collectRecentErrors(): string[] {
  const errors: string[] = []
  // Check for any globally stored errors
  const globalErrors = (window as any).__flowhamster_recentErrors as string[] | undefined
  if (globalErrors && Array.isArray(globalErrors)) {
    errors.push(...globalErrors.slice(-MAX_ERROR_LOGS))
  }
  return errors
}

export function useAgentContext(): { context: AgentChatContext; mode: 'model' | 'data' } {
  const workspaceMode = useGraphStore((s) => s.workspaceMode)
  const modelNodes = useGraphStore((s) => s.nodes)
  const modelEdges = useGraphStore((s) => s.edges)
  const selectedNodeIds = useGraphStore((s) => s.selectedNodeIds)
  const dataNodes = useDataGraphStore((s) => s.nodes)
  const dataEdges = useDataGraphStore((s) => s.edges)

  const selectedNodes = useMemo(() => {
    const nodeMap = new Map(modelNodes.map((n) => [n.id, n]))
    return selectedNodeIds
      .slice(0, MAX_SELECTED_NODES)
      .map((id) => {
        const n = nodeMap.get(id)
        if (!n) return null
        return {
          id: n.id,
          type: n.data?.nodeType || n.type,
          label: n.data?.label || n.id,
          params: truncateParams(n.data?.params || {}),
        }
      })
      .filter(Boolean) as AgentChatContext['selectedNodes']
  }, [modelNodes, selectedNodeIds])

  const modelSummary = useMemo(() => summarizeGraph(modelNodes, modelEdges), [modelNodes, modelEdges])
  const dataSummary = useMemo(() => summarizeGraph(dataNodes, dataEdges), [dataNodes, dataEdges])

  const context: AgentChatContext = useMemo(() => {
    return {
      modelGraph: {
        nodeCount: modelSummary.nodeCount,
        edgeCount: modelSummary.edgeCount,
        nodes: modelSummary.nodes,
        edges: modelSummary.edges,
      },
      dataGraph: {
        nodeCount: dataSummary.nodeCount,
        edgeCount: dataSummary.edgeCount,
        nodes: dataSummary.nodes,
        edges: dataSummary.edges,
      },
      selectedNodes,
      recentErrors: collectRecentErrors(),
    }
  }, [modelSummary, dataSummary, selectedNodes])

  return { context, mode: workspaceMode }
}
