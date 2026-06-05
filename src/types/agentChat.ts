// Agent Chat types

export type MessageRole = 'user' | 'agent' | 'system'

export interface AgentChatMessage {
  id: string
  role: MessageRole
  content: string
  timestamp: number
  // Optional node references extracted from agent response
  referencedNodeIds?: string[]
  // Pending tool calls from agent
  pendingActions?: AgentToolCall[]
  // If this message requires confirmation
  requiresConfirmation?: boolean
  confirmationToken?: string
}

export interface AgentToolCall {
  id: string
  type: 'add_node' | 'remove_node' | 'update_node' | 'add_edge' | 'remove_edge' | 'update_edge' | 'set_nodes' | 'set_edges' | 'package_selection' | 'unpackage_group' | 'other'
  description: string
  params: Record<string, unknown>
  // Serialized diff for confirmation UI
  diffPreview?: string
}

export interface GraphContextSummary {
  nodeCount: number
  edgeCount: number
  nodes: Array<{ id: string; type: string; label: string; params?: Record<string, unknown> }>
  edges: Array<{ id: string; source: string; target: string }>
}

export interface AgentChatContext {
  modelGraph: GraphContextSummary
  dataGraph: GraphContextSummary
  selectedNodes: Array<{ id: string; type: string; label: string; params?: Record<string, unknown> }>
  recentErrors: string[]
}

export interface AgentChatState {
  isOpen: boolean
  messages: AgentChatMessage[]
  inputValue: string
  isLoading: boolean
  pendingActions: AgentToolCall[]
  error: string | null
  sessionId: string | null

  // Actions
  toggleOpen: () => void
  setOpen: (open: boolean) => void
  setInputValue: (value: string) => void
  addMessage: (message: Omit<AgentChatMessage, 'id' | 'timestamp'>) => string
  updateMessage: (id: string, updates: Partial<AgentChatMessage>) => void
  setLoading: (loading: boolean) => void
  setPendingActions: (actions: AgentToolCall[]) => void
  confirmAction: (actionId: string) => void
  rejectAction: (actionId: string) => void
  clearPendingActions: () => void
  setError: (error: string | null) => void
  clearMessages: () => void
  setSessionId: (sessionId: string | null) => void
  ensureSession: () => string
}
