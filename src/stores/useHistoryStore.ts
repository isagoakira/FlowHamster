import { create } from 'zustand'
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { useGraphStore } from '../hooks/useGraphStore'
import { useSelectionStore } from './useSelectionStore'

// Snapshot of graph state for history
interface GraphSnapshot {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
}

interface HistoryStoreState {
  _history: GraphSnapshot[]
  _future: GraphSnapshot[]
  _isUndoRedo: boolean
  maxHistory: number
  // Actions
  pushHistory: () => void
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
  clearHistory: () => void
}

function filterValidSelections(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  selectedNodeIds: string[],
  selectedEdgeIds: string[]
) {
  const nodeIds = new Set(nodes.map((node) => node.id))
  const edgeIds = new Set(edges.map((edge) => edge.id))

  return {
    selectedNodeIds: selectedNodeIds.filter((id) => nodeIds.has(id)),
    selectedEdgeIds: selectedEdgeIds.filter((id) => edgeIds.has(id)),
  }
}

export const useHistoryStore = create<HistoryStoreState>((set, get) => ({
  _history: [],
  _future: [],
  _isUndoRedo: false,
  maxHistory: 50,

  pushHistory: () => {
    if (get()._isUndoRedo) return

    const currentGraph = useGraphStore.getState()
    const snapshot: GraphSnapshot = {
      nodes: JSON.parse(JSON.stringify(currentGraph.nodes)),
      edges: JSON.parse(JSON.stringify(currentGraph.edges)),
    }

    set((state) => {
      const newHistory = [...state._history, snapshot]
      if (newHistory.length > state.maxHistory) newHistory.shift()
      return { _history: newHistory, _future: [] }
    })
  },

  undo: () => {
    const { _history } = get()
    if (_history.length === 0) return

    const currentGraph = useGraphStore.getState()
    const current: GraphSnapshot = {
      nodes: [...currentGraph.nodes],
      edges: [...currentGraph.edges],
    }

    const prev = _history[_history.length - 1]
    const selectionState = useSelectionStore.getState()
    const selection = filterValidSelections(
      prev.nodes,
      prev.edges,
      selectionState.selectedNodeIds,
      selectionState.selectedEdgeIds
    )

    set((state) => ({
      _history: state._history.slice(0, -1),
      _future: [...state._future, current],
      _isUndoRedo: true,
    }))

    // Apply the previous state
    useGraphStore.getState()._internalSetNodes(
      JSON.parse(JSON.stringify(prev.nodes)),
      JSON.parse(JSON.stringify(prev.edges))
    )

    // Update selection if valid
    if (selection.selectedNodeIds.length > 0 || selection.selectedEdgeIds.length > 0) {
      useSelectionStore.getState().setSelection(selection.selectedNodeIds, selection.selectedEdgeIds)
    }

    set({ _isUndoRedo: false })
  },

  redo: () => {
    const { _future } = get()
    if (_future.length === 0) return

    const currentGraph = useGraphStore.getState()
    const current: GraphSnapshot = {
      nodes: [...currentGraph.nodes],
      edges: [...currentGraph.edges],
    }

    const next = _future[_future.length - 1]
    const selectionState = useSelectionStore.getState()
    const selection = filterValidSelections(
      next.nodes,
      next.edges,
      selectionState.selectedNodeIds,
      selectionState.selectedEdgeIds
    )

    set((state) => ({
      _history: [...state._history, current],
      _future: state._future.slice(0, -1),
      _isUndoRedo: true,
    }))

    // Apply the next state
    useGraphStore.getState()._internalSetNodes(
      JSON.parse(JSON.stringify(next.nodes)),
      JSON.parse(JSON.stringify(next.edges))
    )

    // Update selection if valid
    if (selection.selectedNodeIds.length > 0 || selection.selectedEdgeIds.length > 0) {
      useSelectionStore.getState().setSelection(selection.selectedNodeIds, selection.selectedEdgeIds)
    }

    set({ _isUndoRedo: false })
  },

  canUndo: () => get()._history.length > 0,
  canRedo: () => get()._future.length > 0,

  clearHistory: () => set({ _history: [], _future: [] }),
}))
