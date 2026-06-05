import { create } from 'zustand'
import { AgentChatState, AgentChatMessage } from '../types/agentChat'

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
}

function generateSessionId(): string {
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

export const useAgentChatStore = create<AgentChatState>((set, get) => ({
  isOpen: false,
  messages: [],
  inputValue: '',
  isLoading: false,
  pendingActions: [],
  error: null,
  sessionId: null,

  toggleOpen: () => set((state) => ({ isOpen: !state.isOpen })),
  setOpen: (open) => set({ isOpen: open }),

  setInputValue: (value) => set({ inputValue: value }),

  addMessage: (message) => {
    const newMessage: AgentChatMessage = {
      ...message,
      id: generateMessageId(),
      timestamp: Date.now(),
    }
    set((state) => ({
      messages: [...state.messages, newMessage],
      error: null,
    }))
    return newMessage.id
  },

  updateMessage: (id, updates) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    }))
  },

  setLoading: (loading) => set({ isLoading: loading }),

  setPendingActions: (actions) => set({ pendingActions: actions }),

  confirmAction: (actionId) => {
    set((state) => ({
      pendingActions: state.pendingActions.filter((a) => a.id !== actionId),
    }))
  },

  rejectAction: (actionId) => {
    set((state) => ({
      pendingActions: state.pendingActions.filter((a) => a.id !== actionId),
    }))
  },

  clearPendingActions: () => set({ pendingActions: [] }),

  setError: (error) => set({ error, isLoading: false }),

  clearMessages: () => set({ messages: [], pendingActions: [], error: null }),

  setSessionId: (sessionId) => set({ sessionId }),

  ensureSession: () => {
    let { sessionId } = get()
    if (!sessionId) {
      sessionId = generateSessionId()
      set({ sessionId })
    }
    return sessionId
  },
}))
