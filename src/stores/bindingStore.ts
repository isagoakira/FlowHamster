/**
 * Binding Store - Dedicated state management for data-model bindings
 *
 * Manages WorkflowBinding[] for connecting data graph outputs to model inputs
 * and training targets. Provides add/remove/update operations with validation.
 */

import { create } from 'zustand'
import { WorkflowBinding, WorkflowGraphKind, WorkflowBindingTarget } from '../schema/workflowDocument'

interface BindingState {
  // Current bindings array
  bindings: WorkflowBinding[]

  // Actions
  addBinding: (binding: Omit<WorkflowBinding, 'id'>) => string
  removeBinding: (id: string) => void
  updateBinding: (id: string, updates: Partial<Omit<WorkflowBinding, 'id'>>) => void
  setBindings: (bindings: WorkflowBinding[]) => void
  clearBindings: () => void

  // Query helpers
  getBindingsByTarget: (target: WorkflowBindingTarget) => WorkflowBinding[]
  getBindingForTarget: (target: WorkflowBindingTarget, targetKey: string) => WorkflowBinding | undefined
  hasBinding: (target: WorkflowBindingTarget, targetKey: string) => boolean

  // Loss binding helpers
  getLossBindings: () => WorkflowBinding[]
  getLossBindingForNode: (lossNodeId: string) => WorkflowBinding | undefined
  addLossBinding: (lossNodeId: string, sourceKey: string) => string
  removeLossBinding: (lossNodeId: string) => void
}

function generateBindingId(sourceGraph: WorkflowGraphKind, sourceKey: string, target: WorkflowBindingTarget, targetKey: string): string {
  return `${sourceGraph}:${sourceKey}->${target}:${targetKey}`
}

export const useBindingStore = create<BindingState>((set, get) => ({
  bindings: [],

  addBinding: (binding) => {
    const id = generateBindingId(binding.sourceGraph, binding.sourceKey, binding.target, binding.targetKey)

    // Check if binding already exists
    const exists = get().bindings.some(
      (b) => b.target === binding.target && b.targetKey === binding.targetKey
    )
    if (exists) {
      // Update existing binding instead
      get().updateBinding(id, binding)
      return id
    }

    const newBinding: WorkflowBinding = {
      ...binding,
      id,
    }

    set((state) => ({
      bindings: [...state.bindings, newBinding],
    }))

    return id
  },

  removeBinding: (id) => {
    set((state) => ({
      bindings: state.bindings.filter((b) => b.id !== id),
    }))
  },

  updateBinding: (id, updates) => {
    set((state) => ({
      bindings: state.bindings.map((b) =>
        b.id === id ? { ...b, ...updates } : b
      ),
    }))
  },

  setBindings: (bindings) => {
    set({ bindings })
  },

  clearBindings: () => {
    set({ bindings: [] })
  },

  getBindingsByTarget: (target) => {
    return get().bindings.filter((b) => b.target === target)
  },

  getBindingForTarget: (target, targetKey) => {
    return get().bindings.find(
      (b) => b.target === target && b.targetKey === targetKey
    )
  },

  hasBinding: (target, targetKey) => {
    return get().bindings.some(
      (b) => b.target === target && b.targetKey === targetKey
    )
  },

  getLossBindings: () => {
    return get().bindings.filter((b) => b.target === 'loss_target')
  },

  getLossBindingForNode: (lossNodeId) => {
    return get().bindings.find(
      (b) => b.target === 'loss_target' && b.targetKey === lossNodeId
    )
  },

  addLossBinding: (lossNodeId, sourceKey) => {
    const existing = get().bindings.find(
      (b) => b.target === 'loss_target' && b.targetKey === lossNodeId
    )
    if (existing) {
      get().updateBinding(existing.id, { sourceKey })
      return existing.id
    }
    return get().addBinding({
      sourceGraph: 'data',
      sourceKey,
      target: 'loss_target',
      targetKey: lossNodeId,
    })
  },

  removeLossBinding: (lossNodeId) => {
    const binding = get().bindings.find(
      (b) => b.target === 'loss_target' && b.targetKey === lossNodeId
    )
    if (binding) {
      get().removeBinding(binding.id)
    }
  },
}))

/**
 * Sync bindings between bindingStore and graphStore
 * Call this when loading/saving workflows
 */
export function syncBindingsToGraphStore(bindings: WorkflowBinding[]): void {
  const { setBindings } = useGraphStore.getState()
  setBindings(bindings)
}

// Import this at the bottom to avoid circular dependency
import { useGraphStore } from '../hooks/useGraphStore'
