import { create } from 'zustand'
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { getAbsoluteNodePosition, getDescendantNodeIds } from '../utils/graphStructure'
import { makeGraphId } from '../utils/graphUtils'
import { useGraphStore } from '../hooks/useGraphStore'
import { useSelectionStore } from './useSelectionStore'
import { useHistoryStore } from './useHistoryStore'

interface ClipboardState {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
}

interface ClipboardStore {
  clipboard: ClipboardState | null
  pasteCount: number
  // Actions
  copySelection: () => void
  cutSelection: () => void
  pasteClipboard: () => void
  deleteSelection: () => void
}

function syncGroupMetadata(nodes: FlowHamsterNode[], edges: FlowHamsterEdge[]) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]))

  return nodes.map((node) => {
    const descendantIds = getDescendantNodeIds(nodes, [node.id])
    const descendantSet = new Set(descendantIds)
    const childNodeIds = descendantIds.filter((childId) => {
      const childNode = nodeMap.get(childId)
      return childNode && !node.data?.isGroup
    })
    const internalEdgeIds = edges
      .filter((edge) => descendantSet.has(edge.source) && descendantSet.has(edge.target))
      .map((edge) => edge.id)

    return {
      ...node,
      data: {
        ...node.data,
        childNodeIds,
        internalEdgeIds,
      },
    }
  }) as FlowHamsterNode[]
}

function expandSelectedNodeIds(nodes: FlowHamsterNode[], selectedNodeIds: string[]) {
  const expandedNodeIds = new Set(selectedNodeIds)
  getDescendantNodeIds(nodes, selectedNodeIds).forEach((id) => expandedNodeIds.add(id))
  return [...expandedNodeIds]
}

export const useClipboardStore = create<ClipboardStore>((set, get) => ({
  clipboard: null,
  pasteCount: 0,

  copySelection: () => {
    const graphState = useGraphStore.getState()
    const selectionState = useSelectionStore.getState()
    const expandedNodeIds = expandSelectedNodeIds(graphState.nodes, selectionState.selectedNodeIds)

    if (expandedNodeIds.length === 0) return

    const selectedNodeIdSet = new Set(expandedNodeIds)
    const nodeMap = new Map(graphState.nodes.map((node) => [node.id, node]))
    const clipboardNodes = graphState.nodes
      .filter((node) => selectedNodeIdSet.has(node.id))
      .map((node) => {
        const parentNodeId = node.parentNode ?? node.parentId
        const keepParent = parentNodeId ? selectedNodeIdSet.has(parentNodeId) : false

        if (keepParent) {
          return { ...node, selected: false }
        }

        const absolutePosition = getAbsoluteNodePosition(node, nodeMap)
        return {
          ...node,
          position: absolutePosition,
          parentNode: undefined,
          parentId: undefined,
          extent: undefined,
          selected: false,
        }
      }) as FlowHamsterNode[]

    const clipboardEdges = graphState.edges
      .filter((edge) => selectedNodeIdSet.has(edge.source) && selectedNodeIdSet.has(edge.target))
      .map((edge) => ({
        ...edge,
        selected: false,
      })) as FlowHamsterEdge[]

    set({
      clipboard: {
        nodes: syncGroupMetadata(clipboardNodes, clipboardEdges),
        edges: clipboardEdges,
      },
      pasteCount: 0,
    })
  },

  cutSelection: () => {
    // Copy first (does NOT push history), then delete (pushes history once)
    get().copySelection()
    get().deleteSelection()
  },

  pasteClipboard: () => {
    const state = get()
    const graphState = useGraphStore.getState()
    if (!state.clipboard || state.clipboard.nodes.length === 0) return

    const nodeIdMap = new Map<string, string>()
    const edgeIdMap = new Map<string, string>()
    const pasteOffset = 40 * state.pasteCount
    const clipboardNodeIds = new Set(state.clipboard.nodes.map((node) => node.id))

    state.clipboard.nodes.forEach((node) => {
      nodeIdMap.set(node.id, makeGraphId(node.data.nodeType || 'node'))
    })
    state.clipboard.edges.forEach((edge) => {
      edgeIdMap.set(edge.id, makeGraphId('edge'))
    })

    const newEdges = state.clipboard.edges.map((edge) => ({
      ...edge,
      id: edgeIdMap.get(edge.id)!,
      source: nodeIdMap.get(edge.source)!,
      target: nodeIdMap.get(edge.target)!,
      selected: true,
    })) as FlowHamsterEdge[]

    const newNodes = state.clipboard.nodes.map((node) => {
      const parentNodeId = node.parentNode ?? node.parentId
      const keepParent = parentNodeId ? clipboardNodeIds.has(parentNodeId) : false
      const translatedNode = {
        ...node,
        id: nodeIdMap.get(node.id)!,
        position: keepParent
          ? node.position
          : { x: node.position.x + pasteOffset, y: node.position.y + pasteOffset },
        parentNode: keepParent && parentNodeId ? nodeIdMap.get(parentNodeId)! : undefined,
        parentId: undefined,
        extent: keepParent ? node.extent : undefined,
        selected: true,
        data: {
          ...node.data,
          childNodeIds: Array.isArray(node.data.childNodeIds)
            ? node.data.childNodeIds.map((childId) => nodeIdMap.get(childId) ?? childId)
            : node.data.childNodeIds,
          internalEdgeIds: Array.isArray(node.data.internalEdgeIds)
            ? node.data.internalEdgeIds.map((edgeId) => edgeIdMap.get(edgeId) ?? edgeId)
            : node.data.internalEdgeIds,
          boundaryEdges: Array.isArray(node.data.boundaryEdges)
            ? node.data.boundaryEdges.map((edge: any) => ({
                ...edge,
                originalEdgeId: edgeIdMap.get(edge.originalEdgeId) ?? edge.originalEdgeId,
                source: nodeIdMap.get(edge.source) ?? edge.source,
                target: nodeIdMap.get(edge.target) ?? edge.target,
              }))
            : node.data.boundaryEdges,
        },
      }
      return translatedNode
    }) as FlowHamsterNode[]

    const mergedEdges = [...graphState.edges, ...newEdges]
    const mergedNodes = syncGroupMetadata([...graphState.nodes, ...newNodes], mergedEdges)

    // Record history before changing
    useHistoryStore.getState().pushHistory()

    graphState._internalSetNodes(mergedNodes, mergedEdges)
    useSelectionStore.getState().setSelection(
      newNodes.map((node) => node.id),
      newEdges.map((edge) => edge.id)
    )

    set({ pasteCount: state.pasteCount + 1 })
  },

  deleteSelection: () => {
    const graphState = useGraphStore.getState()
    const selectionState = useSelectionStore.getState()
    const expandedNodeIds = expandSelectedNodeIds(graphState.nodes, selectionState.selectedNodeIds)
    const selectedNodeIdSet = new Set(expandedNodeIds)
    const selectedEdgeIdSet = new Set(selectionState.selectedEdgeIds)

    if (selectedNodeIdSet.size === 0 && selectedEdgeIdSet.size === 0) return

    const newEdges = graphState.edges.filter(
      (edge) =>
        !selectedEdgeIdSet.has(edge.id) &&
        !selectedNodeIdSet.has(edge.source) &&
        !selectedNodeIdSet.has(edge.target)
    ) as FlowHamsterEdge[]
    const newNodes = graphState.nodes.filter((node) => !selectedNodeIdSet.has(node.id)) as FlowHamsterNode[]

    // Record history before changing
    useHistoryStore.getState().pushHistory()

    graphState._internalSetNodes(newNodes, newEdges)
    useSelectionStore.getState().clearSelection()
  },
}))
