import { useRef, useEffect, useCallback, useState } from 'react'
import { useAgentChatStore } from '../../stores/useAgentChatStore'
import { useAgentContext } from '../../hooks/useAgentContext'
import { useGraphStore } from '../../hooks/useGraphStore'
import { sendAgentMessage, sendAgentConfirmation, executeAgentAction, buildActionDiffPreview } from '../../utils/agentApi'
import { MessageBubble } from './MessageBubble'
import {
  chatPanelStyle,
  chatHeaderStyle,
  chatTitleStyle,
  chatMessagesStyle,
  chatInputAreaStyle,
  chatInputStyle,
  chatSendBtnStyle,
} from './agentChatStyles'

export function AgentChatPanel() {
  const {
    isOpen,
    messages,
    inputValue,
    isLoading,
    pendingActions,
    error,
    sessionId,
    setOpen,
    setInputValue,
    addMessage,
    updateMessage,
    setLoading,
    setPendingActions,
    confirmAction,
    rejectAction,
    clearPendingActions,
    setError,
    ensureSession,
  } = useAgentChatStore()

  const { context, mode } = useAgentContext()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [localPending, setLocalPending] = useState<typeof pendingActions>([])

  const graphStore = useGraphStore()

  // Merge store pending with any message-level pending actions
  useEffect(() => {
    setLocalPending(pendingActions)
  }, [pendingActions])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, localPending])

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  const handleSend = useCallback(async () => {
    const text = inputValue.trim()
    if (!text || isLoading) return

    // Ensure session exists
    const currentSessionId = ensureSession()

    // Add user message
    addMessage({ role: 'user', content: text })
    setInputValue('')
    setLoading(true)
    clearPendingActions()

    try {
      const response = await sendAgentMessage(text, currentSessionId, mode, context)

      if (!response.success || response.error) {
        setError(response.error || 'Agent request failed')
        addMessage({
          role: 'system',
          content: `Error: ${response.error || 'Unknown error'}`,
        })
        return
      }

      const agentMessageId = addMessage({
        role: 'agent',
        content: response.message || '',
        requiresConfirmation: response.requires_confirmation,
        confirmationToken: response.confirmation_token,
      })

      if (response.tool_calls && response.tool_calls.length > 0) {
        // Enrich diffs
        const enriched = response.tool_calls.map((tc) => ({
          ...tc,
          diffPreview: tc.diffPreview || buildActionDiffPreview(tc),
        }))
        setPendingActions(enriched)
        updateMessage(agentMessageId, { pendingActions: enriched })
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Network error'
      setError(msg)
      addMessage({ role: 'system', content: `Error: ${msg}` })
    } finally {
      setLoading(false)
    }
  }, [inputValue, isLoading, mode, context, ensureSession, addMessage, setInputValue, setLoading, clearPendingActions, setError, setPendingActions, updateMessage])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  const handleConfirmAction = useCallback(
    async (actionId: string) => {
      const action = localPending.find((a) => a.id === actionId)
      if (!action) return

      // Find confirmation token from the most recent agent message
      const lastAgentMessage = [...messages].reverse().find(
        (m) => m.role === 'agent' && m.confirmationToken
      )

      try {
        // Execute the local action first
        const result = executeAgentAction(action, {
          addNode: graphStore.addNode,
          removeNode: graphStore.removeNode,
          updateNodeParams: graphStore.updateNodeParams,
          updateNodeLabel: graphStore.updateNodeLabel,
          setNodes: graphStore.setNodes,
          setEdges: graphStore.setEdges,
          packageSelection: graphStore.packageSelection,
          unpackageGroup: graphStore.unpackageGroup,
          pushHistory: graphStore.pushHistory,
        })

        addMessage({ role: 'system', content: `Executed: ${result}` })

        // If backend expects confirmation token, send it back
        if (lastAgentMessage?.confirmationToken && sessionId) {
          const confirmRes = await sendAgentConfirmation(sessionId, lastAgentMessage.confirmationToken, [
            { tool_call_id: actionId, approved: true },
          ])
          if (!confirmRes.success && confirmRes.error) {
            addMessage({ role: 'system', content: `Backend confirm error: ${confirmRes.error}` })
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Execution failed'
        addMessage({ role: 'system', content: `Execution failed: ${msg}` })
      }

      confirmAction(actionId)
    },
    [localPending, messages, sessionId, graphStore, confirmAction, addMessage]
  )

  const handleRejectAction = useCallback(
    async (actionId: string) => {
      // Find confirmation token from the most recent agent message
      const lastAgentMessage = [...messages].reverse().find(
        (m) => m.role === 'agent' && m.confirmationToken
      )

      if (lastAgentMessage?.confirmationToken && sessionId) {
        try {
          const confirmRes = await sendAgentConfirmation(sessionId, lastAgentMessage.confirmationToken, [
            { tool_call_id: actionId, approved: false },
          ])
          if (!confirmRes.success && confirmRes.error) {
            addMessage({ role: 'system', content: `Backend reject error: ${confirmRes.error}` })
          }
        } catch {
          // Network error on reject is non-critical
        }
      }

      rejectAction(actionId)
      addMessage({ role: 'system', content: 'Action rejected. No changes were made.' })
    },
    [messages, sessionId, rejectAction, addMessage]
  )

  if (!isOpen) return null

  return (
    <div style={chatPanelStyle}>
      <div style={chatHeaderStyle}>
        <div style={chatTitleStyle}>
          <span>🤖</span>
          <span>FlowHamster Agent</span>
          {sessionId && (
            <span style={{ fontSize: '10px', color: '#555', marginLeft: '6px' }}>
              {sessionId.slice(0, 12)}…
            </span>
          )}
        </div>
        <button
          onClick={() => setOpen(false)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#888',
            fontSize: '16px',
            cursor: 'pointer',
            lineHeight: 1,
          }}
          title="Close"
        >
          ×
        </button>
      </div>

      <div style={chatMessagesStyle}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#555', fontSize: '12px', marginTop: '20px' }}>
            Ask me anything about your model or data pipeline.
            <br />
            I can add nodes, modify parameters, and more.
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            pendingActions={msg.pendingActions}
            onConfirmAction={handleConfirmAction}
            onRejectAction={handleRejectAction}
          />
        ))}

        {isLoading && (
          <div style={{ ...chatMessagesStyle, alignSelf: 'flex-start', color: '#888', fontSize: '12px' }}>
            Agent is thinking…
          </div>
        )}

        {error && (
          <div style={{ ...chatMessagesStyle, alignSelf: 'center', color: '#ff9999', fontSize: '12px' }}>
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div style={chatInputAreaStyle}>
        <textarea
          ref={inputRef}
          style={chatInputStyle}
          placeholder="Type a message…"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          style={{
            ...chatSendBtnStyle,
            opacity: isLoading || !inputValue.trim() ? 0.5 : 1,
            cursor: isLoading || !inputValue.trim() ? 'not-allowed' : 'pointer',
          }}
          onClick={handleSend}
          disabled={isLoading || !inputValue.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default AgentChatPanel
