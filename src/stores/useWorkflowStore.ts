/**
 * 工作流持久化管理
 *
 * 负责工作流的创建、加载、保存等操作
 * 完整项目数据（nodes, edges, trainingConfig, dataGraph, bindings）存储在后端
 */

import { create } from 'zustand'
import { API_BASE_URL } from '../utils/runtimeConfig'
import { WorkflowDocument } from '../schema/workflowDocument'

export interface WorkflowInfo {
  id: string
  name: string
  description: string
  created_at: number
  updated_at: number
}

interface WorkflowState {
  // 当前打开的工作流
  currentWorkflowId: string | null
  currentWorkflowName: string | null
  isDirty: boolean  // 是否有未保存的更改

  // 工作流列表
  workflows: WorkflowInfo[]
  isLoading: boolean

  // 完整项目数据（用于保存）
  _cachedDocument: WorkflowDocument | null

  // Actions
  fetchWorkflows: () => Promise<void>
  createWorkflow: (name: string, description?: string) => Promise<WorkflowInfo | null>
  openWorkflow: (id: string) => Promise<WorkflowDocument | null>
  saveWorkflow: (document: WorkflowDocument) => Promise<boolean>
  deleteWorkflow: (id: string) => Promise<boolean>
  setDirty: (dirty: boolean) => void
  clearCurrentWorkflow: () => void
}

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  currentWorkflowId: null,
  currentWorkflowName: null,
  isDirty: false,
  workflows: [],
  isLoading: false,
  _cachedDocument: null,

  fetchWorkflows: async () => {
    set({ isLoading: true })
    try {
      const res = await fetch(`${API_BASE_URL}/workflows`)
      if (res.ok) {
        const data = await res.json()
        set({ workflows: data, isLoading: false })
      } else {
        set({ isLoading: false })
      }
    } catch {
      set({ isLoading: false })
    }
  },

  createWorkflow: async (name, description = '') => {
    try {
      const res = await fetch(`${API_BASE_URL}/workflows`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      })
      if (res.ok) {
        const workflow = await res.json()
        set((state) => ({
          workflows: [...state.workflows, workflow],
          currentWorkflowId: workflow.id,
          currentWorkflowName: workflow.name,
          isDirty: false,
          _cachedDocument: null,
        }))
        return workflow
      }
    } catch {
      console.error('Failed to create workflow')
    }
    return null
  },

  openWorkflow: async (id) => {
    try {
      const res = await fetch(`${API_BASE_URL}/workflows/${id}/document`)
      if (res.ok) {
        const document = await res.json()
        set({
          currentWorkflowId: id,
          isDirty: false,
          _cachedDocument: document,
        })
        // 获取工作流名称
        const metaRes = await fetch(`${API_BASE_URL}/workflows/${id}`)
        if (metaRes.ok) {
          const meta = await metaRes.json()
          set({ currentWorkflowName: meta.name })
        }
        return document
      }
    } catch {
      console.error('Failed to open workflow')
    }
    return null
  },

  saveWorkflow: async (document) => {
    const { currentWorkflowId } = get()
    if (!currentWorkflowId) return false

    try {
      const res = await fetch(`${API_BASE_URL}/workflows/${currentWorkflowId}/document`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(document),
      })
      if (res.ok) {
        set({ isDirty: false, _cachedDocument: document })
        return true
      }
    } catch {
      console.error('Failed to save workflow')
    }
    return false
  },

  deleteWorkflow: async (id) => {
    try {
      const res = await fetch(`${API_BASE_URL}/workflows/${id}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        set((state) => ({
          workflows: state.workflows.filter((w) => w.id !== id),
          currentWorkflowId: state.currentWorkflowId === id ? null : state.currentWorkflowId,
          currentWorkflowName: state.currentWorkflowId === id ? null : state.currentWorkflowName,
          _cachedDocument: state.currentWorkflowId === id ? null : state._cachedDocument,
        }))
        return true
      }
    } catch {
      console.error('Failed to delete workflow')
    }
    return false
  },

  setDirty: (dirty) => set({ isDirty: dirty }),

  clearCurrentWorkflow: () => set({
    currentWorkflowId: null,
    currentWorkflowName: null,
    isDirty: false,
    _cachedDocument: null,
  }),
}))
