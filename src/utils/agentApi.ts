/**
 * Agent Chat API Client
 *
 * Handles communication with the backend agent orchestrator.
 * Backend contract:
 *   Request:  { message, session_id, mode, graph_context }
 *   Response: { success, message?, tool_calls?, requires_confirmation?, confirmation_token?, error? }
 */
import { API_BASE_URL } from './runtimeConfig'
import { AgentChatContext, AgentToolCall } from '../types/agentChat'

export interface AgentChatRequest {
  message: string
  session_id: string
  mode: 'model' | 'data'
  graph_context: AgentChatContext
}

export interface AgentChatResponse {
  success: boolean
  message?: string
  tool_calls?: AgentToolCall[]
  requires_confirmation?: boolean
  confirmation_token?: string
  error?: string
}

export interface AgentConfirmRequest {
  session_id: string
  confirmation_token: string
  action_results?: Array<{ tool_call_id: string; approved: boolean }>
}

export interface AgentConfirmResponse {
  success: boolean
  message?: string
  error?: string
}

export async function sendAgentMessage(
  message: string,
  sessionId: string,
  mode: 'model' | 'data',
  graphContext: AgentChatContext
): Promise<AgentChatResponse> {
  const res = await fetch(`${API_BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      mode,
      graph_context: graphContext,
    } as AgentChatRequest),
  })

  if (!res.ok) {
    const text = await res.text()
    return { success: false, error: `HTTP ${res.status}: ${text}` }
  }

  return res.json()
}

export async function sendAgentConfirmation(
  sessionId: string,
  confirmationToken: string,
  actionResults: Array<{ tool_call_id: string; approved: boolean }>
): Promise<AgentConfirmResponse> {
  const res = await fetch(`${API_BASE_URL}/agent/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      confirmation_token: confirmationToken,
      action_results: actionResults,
    } as AgentConfirmRequest),
  })

  if (!res.ok) {
    const text = await res.text()
    return { success: false, error: `HTTP ${res.status}: ${text}` }
  }

  return res.json()
}

/**
 * Execute a confirmed agent tool call against the graph store.
 * Returns a descriptive string of what was done.
 */
export function executeAgentAction(
  action: AgentToolCall,
  storeActions: {
    addNode: (data: any, position: { x: number; y: number }) => void
    removeNode: (id: string) => void
    updateNodeParams: (id: string, params: Record<string, unknown>) => void
    updateNodeLabel: (id: string, label: string) => void
    setNodes: (nodes: any[]) => void
    setEdges: (edges: any[]) => void
    packageSelection: () => void
    unpackageGroup: (id: string) => void
    pushHistory?: () => void
  }
): string {
  // Ensure history is pushed before mutation so undo works
  storeActions.pushHistory?.()

  switch (action.type) {
    case 'add_node': {
      const { nodeType, label, params, position } = action.params as any
      storeActions.addNode(
        { nodeType, label, params: params || {} },
        position || { x: 250, y: 250 }
      )
      return `Added node "${label}" (${nodeType})`
    }
    case 'remove_node': {
      const { nodeId } = action.params as any
      storeActions.removeNode(nodeId)
      return `Removed node ${nodeId}`
    }
    case 'update_node': {
      const { nodeId, params, label } = action.params as any
      if (params) storeActions.updateNodeParams(nodeId, params)
      if (label) storeActions.updateNodeLabel(nodeId, label)
      return `Updated node ${nodeId}`
    }
    case 'set_nodes': {
      const { nodes } = action.params as any
      storeActions.setNodes(nodes)
      return `Replaced all nodes (${nodes.length} nodes)`
    }
    case 'set_edges': {
      const { edges } = action.params as any
      storeActions.setEdges(edges)
      return `Replaced all edges (${edges.length} edges)`
    }
    case 'package_selection': {
      storeActions.packageSelection()
      return 'Packaged selected nodes'
    }
    case 'unpackage_group': {
      const { nodeId } = action.params as any
      storeActions.unpackageGroup(nodeId)
      return `Unpacked group ${nodeId}`
    }
    default:
      return `Executed action: ${action.description}`
  }
}

/**
 * Build a human-readable diff preview for a tool call.
 */
export function buildActionDiffPreview(action: AgentToolCall): string {
  if (action.diffPreview) return action.diffPreview
  switch (action.type) {
    case 'add_node': {
      const { nodeType, label, params } = action.params as any
      return `+ Add node: ${label} (${nodeType})\n  Params: ${JSON.stringify(params || {}, null, 2)}`
    }
    case 'remove_node': {
      const { nodeId, label } = action.params as any
      return `- Remove node: ${label || nodeId}`
    }
    case 'update_node': {
      const { nodeId, params, label } = action.params as any
      let diff = `~ Update node: ${nodeId}\n`
      if (label) diff += `  Label: ${label}\n`
      if (params) diff += `  Params: ${JSON.stringify(params, null, 2)}`
      return diff
    }
    case 'package_selection':
      return `📦 Package selected nodes into composite module`
    case 'unpackage_group': {
      const { nodeId } = action.params as any
      return `📦 Unpack group: ${nodeId}`
    }
    default:
      return action.description
  }
}
