import { create } from 'zustand'

interface SelectionState {
  selectedNodeIds: string[]
  selectedEdgeIds: string[]
  // Actions
  setSelection: (nodeIds: string[], edgeIds: string[]) => void
  clearSelection: () => void
  selectNodes: (nodeIds: string[]) => void
  addToSelection: (nodeIds: string[]) => void
  removeFromSelection: (nodeIds: string[]) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedNodeIds: [],
  selectedEdgeIds: [],

  setSelection: (nodeIds, edgeIds) => {
    set({ selectedNodeIds: nodeIds, selectedEdgeIds: edgeIds })
  },

  clearSelection: () => set({ selectedNodeIds: [], selectedEdgeIds: [] }),

  selectNodes: (nodeIds) => {
    set({ selectedNodeIds: nodeIds, selectedEdgeIds: [] })
  },

  addToSelection: (nodeIds) => {
    set((state) => ({
      selectedNodeIds: [...new Set([...state.selectedNodeIds, ...nodeIds])],
    }))
  },

  removeFromSelection: (nodeIds) => {
    const nodeIdSet = new Set(nodeIds)
    set((state) => ({
      selectedNodeIds: state.selectedNodeIds.filter((id) => !nodeIdSet.has(id)),
    }))
  },
}))
