/**
 * Unit tests for useAgentChatStore
 */
import { describe, it, expect, beforeEach } from 'vitest'
import { act } from 'react'
import { useAgentChatStore } from './useAgentChatStore'

describe('useAgentChatStore', () => {
  beforeEach(() => {
    act(() => {
      useAgentChatStore.setState({
        isOpen: false,
        messages: [],
        inputValue: '',
        isLoading: false,
        pendingActions: [],
        error: null,
        sessionId: null,
      })
    })
  })

  it('should toggle open state', () => {
    expect(useAgentChatStore.getState().isOpen).toBe(false)
    act(() => {
      useAgentChatStore.getState().toggleOpen()
    })
    expect(useAgentChatStore.getState().isOpen).toBe(true)
    act(() => {
      useAgentChatStore.getState().toggleOpen()
    })
    expect(useAgentChatStore.getState().isOpen).toBe(false)
  })

  it('should add messages with generated id and timestamp', () => {
    act(() => {
      useAgentChatStore.getState().addMessage({ role: 'user', content: 'Hello' })
    })
    const messages = useAgentChatStore.getState().messages
    expect(messages).toHaveLength(1)
    expect(messages[0].role).toBe('user')
    expect(messages[0].content).toBe('Hello')
    expect(messages[0].id).toBeDefined()
    expect(messages[0].timestamp).toBeGreaterThan(0)
  })

  it('should update a message by id', () => {
    let id: string = ''
    act(() => {
      id = useAgentChatStore.getState().addMessage({ role: 'agent', content: 'Initial' })
    })
    act(() => {
      useAgentChatStore.getState().updateMessage(id, { content: 'Updated' })
    })
    const msg = useAgentChatStore.getState().messages.find((m) => m.id === id)
    expect(msg?.content).toBe('Updated')
  })

  it('should set loading state', () => {
    act(() => {
      useAgentChatStore.getState().setLoading(true)
    })
    expect(useAgentChatStore.getState().isLoading).toBe(true)
    act(() => {
      useAgentChatStore.getState().setLoading(false)
    })
    expect(useAgentChatStore.getState().isLoading).toBe(false)
  })

  it('should manage pending actions', () => {
    const actions = [
      { id: 'a1', type: 'add_node' as const, description: 'Add ReLU', params: {} },
      { id: 'a2', type: 'remove_node' as const, description: 'Remove Linear', params: {} },
    ]
    act(() => {
      useAgentChatStore.getState().setPendingActions(actions)
    })
    expect(useAgentChatStore.getState().pendingActions).toHaveLength(2)

    act(() => {
      useAgentChatStore.getState().confirmAction('a1')
    })
    expect(useAgentChatStore.getState().pendingActions).toHaveLength(1)
    expect(useAgentChatStore.getState().pendingActions[0].id).toBe('a2')

    act(() => {
      useAgentChatStore.getState().rejectAction('a2')
    })
    expect(useAgentChatStore.getState().pendingActions).toHaveLength(0)
  })

  it('should clear all messages and pending actions', () => {
    act(() => {
      useAgentChatStore.getState().addMessage({ role: 'user', content: 'Hi' })
      useAgentChatStore.getState().setPendingActions([{ id: 'a1', type: 'add_node' as const, description: '', params: {} }])
    })
    act(() => {
      useAgentChatStore.getState().clearMessages()
    })
    expect(useAgentChatStore.getState().messages).toHaveLength(0)
    expect(useAgentChatStore.getState().pendingActions).toHaveLength(0)
  })

  it('should set and clear error', () => {
    act(() => {
      useAgentChatStore.getState().setError('Something went wrong')
    })
    expect(useAgentChatStore.getState().error).toBe('Something went wrong')
    expect(useAgentChatStore.getState().isLoading).toBe(false)
  })

  describe('session management', () => {
    it('should generate session id on ensureSession', () => {
      expect(useAgentChatStore.getState().sessionId).toBeNull()
      let id: string = ''
      act(() => {
        id = useAgentChatStore.getState().ensureSession()
      })
      expect(id).toBeDefined()
      expect(id.startsWith('sess_')).toBe(true)
      expect(useAgentChatStore.getState().sessionId).toBe(id)
    })

    it('should reuse existing session id', () => {
      act(() => {
        useAgentChatStore.getState().setSessionId('existing_session')
      })
      let id: string = ''
      act(() => {
        id = useAgentChatStore.getState().ensureSession()
      })
      expect(id).toBe('existing_session')
    })

    it('should set session id explicitly', () => {
      act(() => {
        useAgentChatStore.getState().setSessionId('my_session')
      })
      expect(useAgentChatStore.getState().sessionId).toBe('my_session')
    })
  })
})
