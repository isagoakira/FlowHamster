/**
 * Unit tests for agentApi utilities
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { executeAgentAction, buildActionDiffPreview, sendAgentMessage, sendAgentConfirmation } from './agentApi'
import { AgentToolCall, AgentChatContext } from '../types/agentChat'

// Mock runtimeConfig
vi.mock('./runtimeConfig', () => ({
  API_BASE_URL: 'http://localhost:8000/api',
}))

const mockStoreActions = {
  addNode: vi.fn(),
  removeNode: vi.fn(),
  updateNodeParams: vi.fn(),
  updateNodeLabel: vi.fn(),
  setNodes: vi.fn(),
  setEdges: vi.fn(),
  packageSelection: vi.fn(),
  unpackageGroup: vi.fn(),
  pushHistory: vi.fn(),
}

const mockContext: AgentChatContext = {
  modelGraph: { nodeCount: 2, edgeCount: 1, nodes: [], edges: [] },
  dataGraph: { nodeCount: 0, edgeCount: 0, nodes: [], edges: [] },
  selectedNodes: [],
  recentErrors: [],
}

describe('agentApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('sendAgentMessage', () => {
    it('should send correct backend contract payload', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        json: async () => ({ success: true, message: 'Hello' }),
      } as any)

      const res = await sendAgentMessage('test message', 'sess_123', 'model', mockContext)

      expect(fetchSpy).toHaveBeenCalledWith(
        'http://localhost:8000/api/agent/chat',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: expect.any(String),
        })
      )

      const body = JSON.parse(fetchSpy.mock.calls[0][1]!.body as string)
      expect(body).toEqual({
        message: 'test message',
        session_id: 'sess_123',
        mode: 'model',
        graph_context: mockContext,
      })

      expect(res.success).toBe(true)
      fetchSpy.mockRestore()
    })

    it('should return error on non-ok response', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      } as any)

      const res = await sendAgentMessage('test', 'sess_1', 'data', mockContext)
      expect(res.success).toBe(false)
      expect(res.error).toContain('500')
      fetchSpy.mockRestore()
    })
  })

  describe('sendAgentConfirmation', () => {
    it('should send confirmation token with action results', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      } as any)

      const res = await sendAgentConfirmation('sess_1', 'token_abc', [
        { tool_call_id: 'tc1', approved: true },
      ])

      const body = JSON.parse(fetchSpy.mock.calls[0][1]!.body as string)
      expect(body).toEqual({
        session_id: 'sess_1',
        confirmation_token: 'token_abc',
        action_results: [{ tool_call_id: 'tc1', approved: true }],
      })

      expect(res.success).toBe(true)
      fetchSpy.mockRestore()
    })
  })

  describe('executeAgentAction', () => {
    it('should call pushHistory before any mutation', () => {
      const action: AgentToolCall = {
        id: 'a1',
        type: 'add_node',
        description: 'Add ReLU',
        params: { nodeType: 'relu', label: 'ReLU', params: {}, position: { x: 100, y: 100 } },
      }
      executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.pushHistory).toHaveBeenCalled()
    })

    it('should execute add_node action', () => {
      const action: AgentToolCall = {
        id: 'a1',
        type: 'add_node',
        description: 'Add ReLU',
        params: { nodeType: 'relu', label: 'ReLU', params: {}, position: { x: 100, y: 100 } },
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.addNode).toHaveBeenCalledWith(
        { nodeType: 'relu', label: 'ReLU', params: {} },
        { x: 100, y: 100 }
      )
      expect(result).toContain('Added node')
    })

    it('should execute remove_node action', () => {
      const action: AgentToolCall = {
        id: 'a2',
        type: 'remove_node',
        description: 'Remove node',
        params: { nodeId: 'node_1' },
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.removeNode).toHaveBeenCalledWith('node_1')
      expect(result).toContain('Removed')
    })

    it('should execute update_node action', () => {
      const action: AgentToolCall = {
        id: 'a3',
        type: 'update_node',
        description: 'Update node',
        params: { nodeId: 'node_1', params: { out_features: 128 } },
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.updateNodeParams).toHaveBeenCalledWith('node_1', { out_features: 128 })
      expect(result).toContain('Updated')
    })

    it('should execute package_selection action', () => {
      const action: AgentToolCall = {
        id: 'a4',
        type: 'package_selection',
        description: 'Package nodes',
        params: {},
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.packageSelection).toHaveBeenCalled()
      expect(result).toContain('Packaged')
    })

    it('should execute unpackage_group action', () => {
      const action: AgentToolCall = {
        id: 'a5',
        type: 'unpackage_group',
        description: 'Unpackage',
        params: { nodeId: 'pkg_1' },
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(mockStoreActions.unpackageGroup).toHaveBeenCalledWith('pkg_1')
      expect(result).toContain('Unpacked')
    })

    it('should fallback for unknown action type', () => {
      const action: AgentToolCall = {
        id: 'a6',
        type: 'other',
        description: 'Custom action',
        params: {},
      }
      const result = executeAgentAction(action, mockStoreActions)
      expect(result).toContain('Executed action')
    })
  })

  describe('buildActionDiffPreview', () => {
    it('should show diff for add_node', () => {
      const action: AgentToolCall = {
        id: 'a1',
        type: 'add_node',
        description: 'Add ReLU',
        params: { nodeType: 'relu', label: 'ReLU', params: {} },
      }
      const diff = buildActionDiffPreview(action)
      expect(diff).toContain('Add node')
      expect(diff).toContain('ReLU')
    })

    it('should show diff for remove_node', () => {
      const action: AgentToolCall = {
        id: 'a2',
        type: 'remove_node',
        description: 'Remove node',
        params: { nodeId: 'node_1', label: 'OldNode' },
      }
      const diff = buildActionDiffPreview(action)
      expect(diff).toContain('Remove node')
      expect(diff).toContain('OldNode')
    })

    it('should show diff for update_node', () => {
      const action: AgentToolCall = {
        id: 'a3',
        type: 'update_node',
        description: 'Update',
        params: { nodeId: 'node_1', label: 'NewLabel', params: { out_features: 64 } },
      }
      const diff = buildActionDiffPreview(action)
      expect(diff).toContain('Update node')
      expect(diff).toContain('NewLabel')
    })

    it('should use diffPreview if provided', () => {
      const action: AgentToolCall = {
        id: 'a4',
        type: 'other',
        description: 'Custom',
        params: {},
        diffPreview: 'Custom diff',
      }
      expect(buildActionDiffPreview(action)).toBe('Custom diff')
    })
  })
})
