import { create } from 'zustand'
import { addEdge, applyNodeChanges, applyEdgeChanges, Connection, NodeChange, EdgeChange } from 'reactflow'
import { FlowHamsterNode, FlowHamsterEdge, NodeData, CustomCompositeNodeData } from '../types/graph'
import {
  getAbsoluteNodePosition,
  getDescendantNodeIds,
  isGroupNode,
} from '../utils/graphStructure'
import { getNodeComponentType, normalizeNodeData } from '../utils/nodeType'
import { validateTemplateWithMessages } from '../utils/templateValidator'
import { adaptBackendTemplate } from '../utils/adapters/templateAdapter'
import { packageNodes, expandPackage, collapsePackage, fullyUnpackageGroup } from '../utils/subgraphPackager'
import { registerCustomClass, updateCustomClass, getAllCustomClasses } from '../utils/customCompositeRegistry'
import { SubModule, InternalEdge } from '../utils/nodeRegistry'
import { WorkflowBinding, WorkflowTrainingConfig } from '../schema/workflowDocument'
import { createDefaultTrainingConfig } from '../utils/workflowDocument'
import { makeGraphId } from '../utils/graphUtils'

export type MergeMode = 'concat' | 'add' | 'mul' | 'stack'
export type WorkspaceMode = 'model' | 'data'

export interface FeatureToggles {
  tensorPreview: boolean      // 节点输出张量预览
  gradientViz: boolean       // 梯度可视化分析
  multiOutput: boolean        // 多任务/多输出分支着色
  evaluationNodes: boolean    // 评估节点 (Loss / Optimizer / Scheduler 分组)
}

interface HistoryState {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
}

interface ClipboardState {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
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

function expandSelectedNodeIds(nodes: FlowHamsterNode[], selectedNodeIds: string[]) {
  const expandedNodeIds = new Set(selectedNodeIds)
  getDescendantNodeIds(nodes, selectedNodeIds).forEach((id) => expandedNodeIds.add(id))
  return [...expandedNodeIds]
}

function syncGroupMetadata(nodes: FlowHamsterNode[], edges: FlowHamsterEdge[]) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]))

  return nodes.map((node) => {
    if (!isGroupNode(node)) return node

    const descendantIds = getDescendantNodeIds(nodes, [node.id])
    const descendantSet = new Set(descendantIds)
    const childNodeIds = descendantIds.filter((childId) => {
      const childNode = nodeMap.get(childId)
      return childNode && !isGroupNode(childNode)
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

interface GraphState {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
  selectedNodeIds: string[]
  selectedEdgeIds: string[]
  clipboard: ClipboardState | null
  pasteCount: number
  features: FeatureToggles
  trainingConfig: WorkflowTrainingConfig
  bindings: WorkflowBinding[]
  workspaceMode: WorkspaceMode
  // React Flow setters — injected by Canvas so store can drive RF state directly
  rfSetNodes: ((nodes: FlowHamsterNode[]) => void) | null
  rfSetEdges: ((edges: FlowHamsterEdge[]) => void) | null
  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  addNode: (data: NodeData, position: { x: number; y: number }) => void
  removeNode: (id: string) => void
  updateNodeData: (id: string, data: Partial<NodeData>) => void
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  updateNodeLabel: (id: string, label: string) => void
  setNodes: (nodes: FlowHamsterNode[]) => void
  setEdges: (edges: FlowHamsterEdge[]) => void
  setSelection: (nodeIds: string[], edgeIds: string[]) => void
  clearSelection: () => void
  packageSelection: () => void
  expandGroup: (groupId: string) => void
  collapseGroup: (groupId: string) => void
  unpackageGroup: (groupId: string) => void
  renamePackage: (groupId: string, newName: string) => void
  renamePackageClass: (groupId: string, newClassName: string) => void
  copySelection: () => void
  pasteClipboard: () => void
  deleteSelection: () => void
  setMergeMode: (nodeId: string, mode: MergeMode) => void
  // Register React Flow setters (called by Canvas on mount)
  registerRfSetters: (setNodes: (n: FlowHamsterNode[]) => void, setEdges: (e: FlowHamsterEdge[]) => void) => void
  // Load a template into the canvas (replaces current graph)
  loadTemplate: (template: any) => void
  // Load a template from backend by ID
  loadTemplateFromBackend: (templateId: string) => Promise<boolean>
  // Feature toggles
  toggleFeature: (key: keyof FeatureToggles) => void
  setTrainingConfig: (trainingConfig: WorkflowTrainingConfig) => void
  resetTrainingConfig: () => void
  setBindings: (bindings: WorkflowBinding[]) => void
  setWorkspaceMode: (mode: WorkspaceMode) => void
  // Package Viewer Dialog state (managed at Canvas level to avoid ReactFlow conflicts)
  packageViewerOpen: boolean
  packageViewerData: CustomCompositeNodeData | null
  packageViewerNodeId: string | null
  openPackageViewer: (nodeId: string) => void
  closePackageViewer: () => void
  // Undo/Redo
  _history: HistoryState[]
  _future: HistoryState[]
  _isUndoRedo: boolean
  maxHistory: number
  pushHistory: () => void
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
  clearHistory: () => void
  // Internal setter for history/clipboard operations (doesn't trigger history)
  _internalSetNodes: (nodes: FlowHamsterNode[], edges?: FlowHamsterEdge[]) => void
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeIds: [],
  selectedEdgeIds: [],
  clipboard: null,
  pasteCount: 0,
  features: {
    tensorPreview: false,
    gradientViz: false,
    multiOutput: false,
    evaluationNodes: false,
  },
  trainingConfig: createDefaultTrainingConfig(),
  bindings: [],
  workspaceMode: 'model',
  rfSetNodes: null,
  rfSetEdges: null,
  _history: [],
  _future: [],
  _isUndoRedo: false,
  maxHistory: 50,
  // Package Viewer Dialog state
  packageViewerOpen: false,
  packageViewerData: null,
  packageViewerNodeId: null,

  registerRfSetters: (setNodes, setEdges) => set({ rfSetNodes: setNodes, rfSetEdges: setEdges }),

  toggleFeature: (key) => {
    set((state) => ({
      features: { ...state.features, [key]: !state.features[key] },
    }))
  },

  setTrainingConfig: (trainingConfig) => set({ trainingConfig }),

  resetTrainingConfig: () => set({ trainingConfig: createDefaultTrainingConfig() }),

  setBindings: (bindings) => set({ bindings }),

  setWorkspaceMode: (workspaceMode) => set({ workspaceMode }),

  openPackageViewer: (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId)
    if (node && (node.data as CustomCompositeNodeData)?.isCustomComposite) {
      set({
        packageViewerOpen: true,
        packageViewerData: node.data as CustomCompositeNodeData,
        packageViewerNodeId: nodeId,
      })
    }
  },

  closePackageViewer: () => {
    set({
      packageViewerOpen: false,
      packageViewerData: null,
      packageViewerNodeId: null,
    })
  },

  pushHistory: () => {
    if (get()._isUndoRedo) return
    const snapshot: HistoryState = {
      nodes: structuredClone(get().nodes),
      edges: structuredClone(get().edges),
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
    set({ _isUndoRedo: true })
    const current: HistoryState = {
      nodes: [...get().nodes],
      edges: [...get().edges],
    }
    const prev = _history[_history.length - 1]
    const selection = filterValidSelections(
      prev.nodes as FlowHamsterNode[],
      prev.edges as FlowHamsterEdge[],
      get().selectedNodeIds,
      get().selectedEdgeIds
    )
    set((state) => ({
      _history: state._history.slice(0, -1),
      _future: [...state._future, current],
      nodes: structuredClone(prev.nodes),
      edges: structuredClone(prev.edges),
      ...selection,
      _isUndoRedo: false,
    }))
    get().rfSetNodes?.(get().nodes)
    get().rfSetEdges?.(get().edges)
  },

  redo: () => {
    const { _future } = get()
    if (_future.length === 0) return
    set({ _isUndoRedo: true })
    const current: HistoryState = {
      nodes: [...get().nodes],
      edges: [...get().edges],
    }
    const next = _future[_future.length - 1]
    const selection = filterValidSelections(
      next.nodes as FlowHamsterNode[],
      next.edges as FlowHamsterEdge[],
      get().selectedNodeIds,
      get().selectedEdgeIds
    )
    set((state) => ({
      _history: [...state._history, current],
      _future: state._future.slice(0, -1),
      nodes: structuredClone(next.nodes),
      edges: structuredClone(next.edges),
      ...selection,
      _isUndoRedo: false,
    }))
    get().rfSetNodes?.(get().nodes)
    get().rfSetEdges?.(get().edges)
  },

  canUndo: () => get()._history.length > 0,
  canRedo: () => get()._future.length > 0,

  clearHistory: () => set({ _history: [], _future: [] }),

  onNodesChange: (changes) => {
    set((state) => {
      const newNodes = applyNodeChanges(changes, state.nodes) as FlowHamsterNode[]
      state.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  onEdgesChange: (changes) => {
    set((state) => {
      const newEdges = applyEdgeChanges(changes, state.edges) as FlowHamsterEdge[]
      state.rfSetEdges?.(newEdges)
      return { edges: newEdges }
    })
  },

  onConnect: (connection) => {
    if (!connection.source || !connection.target) return
    get().pushHistory()
    set((state) => {
      const sourceId = connection.source as string
      const targetId = connection.target as string
      const targetHandle = connection.targetHandle || 'a'
      const existingEdgeToPort = state.edges.find(
        (e) => e.target === targetId && e.targetHandle === targetHandle
      )

      if (existingEdgeToPort) {
        const mergeMode: MergeMode = (existingEdgeToPort.data as any)?.mergeMode || 'concat'
        const icatId = `__icat_${targetId}_${targetHandle}`
        const icatExists = state.nodes.some((n) => n.id === icatId)

        if (!icatExists) {
          const icatNode: FlowHamsterNode = {
            id: icatId,
            type: 'concatNode',
            position: { x: 350, y: 200 },
            data: { nodeType: 'concat', label: 'Concat', params: { dim: 1, _implicit: true, mergeMode } },
          }
          const redirectedEdges = state.edges.map((e) => {
            if (e.id === existingEdgeToPort.id) {
              return { ...e, target: icatId, targetHandle: 'in_0' }
            }
            return e
          })
          const newEdgeFromIcat: FlowHamsterEdge = {
            id: `e_icat_${Date.now()}`,
            source: icatId,
            sourceHandle: 'result',
            target: targetId,
            targetHandle,
            type: 'smoothstep',
            animated: true,
            data: { mergeMode },
          }
          const newConnEdge: FlowHamsterEdge = {
            id: `e_${Date.now()}`,
            source: sourceId,
            sourceHandle: connection.sourceHandle ?? 'result',
            target: icatId,
            targetHandle: 'in_1',
            type: 'smoothstep',
            animated: true,
            data: { mergeMode },
          }
          const allEdges = [...redirectedEdges, newConnEdge, newEdgeFromIcat]
          state.rfSetNodes?.([...state.nodes, icatNode])
          state.rfSetEdges?.(allEdges)
          return { nodes: [...state.nodes, icatNode], edges: allEdges }
        } else {
          const usedPorts = state.edges
            .filter((e) => e.target === icatId)
            .map((e) => e.targetHandle)
            .filter(Boolean)
          const nextPort = `in_${usedPorts.length}`
          const newConnEdge: FlowHamsterEdge = {
            id: `e_${Date.now()}`,
            source: sourceId,
            sourceHandle: connection.sourceHandle ?? 'result',
            target: icatId,
            targetHandle: nextPort,
            type: 'smoothstep',
            animated: true,
            data: { mergeMode },
          }
          const allEdges = [...state.edges, newConnEdge]
          state.rfSetEdges?.(allEdges)
          return { edges: allEdges }
        }
      }

      const newEdges = addEdge(
        { ...connection, type: 'smoothstep', animated: true, data: { mergeMode: 'concat' as MergeMode } },
        state.edges
      ) as FlowHamsterEdge[]
      state.rfSetEdges?.(newEdges)
      return { edges: newEdges }
    })
  },

  addNode: (data, position) => {
    const id = `node_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    const normalizedData = normalizeNodeData(data)
    const newNode: FlowHamsterNode = { id, type: getNodeComponentType(normalizedData.nodeType), position, data: normalizedData }
    get().pushHistory()
    set((state) => {
      const newNodes = [...state.nodes, newNode]
      state.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  removeNode: (id) => {
    get().pushHistory()
    set((state) => {
      const removedNodeIds = new Set([id, ...getDescendantNodeIds(state.nodes, [id])])
      const newEdges = state.edges.filter(
        (e) => !removedNodeIds.has(e.source) && !removedNodeIds.has(e.target)
      ) as FlowHamsterEdge[]
      const newNodes = syncGroupMetadata(
        state.nodes.filter((n) => !removedNodeIds.has(n.id)) as FlowHamsterNode[],
        newEdges
      )
      const selection = filterValidSelections(
        newNodes,
        newEdges,
        state.selectedNodeIds,
        state.selectedEdgeIds
      )
      state.rfSetNodes?.(newNodes)
      state.rfSetEdges?.(newEdges)
      return { nodes: newNodes, edges: newEdges, ...selection }
    })
  },

  updateNodeData: (id, data) => {
    get().pushHistory()
    set((state) => {
      const newNodes = state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ) as FlowHamsterNode[]
      state.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  updateNodeParams: (id, params) => {
    get().pushHistory()
    set((state) => {
      const newNodes = state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } } : n
      ) as FlowHamsterNode[]
      state.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  updateNodeLabel: (id, label) => {
    get().pushHistory()
    set((state) => {
      const newNodes = state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, label } } : n
      ) as FlowHamsterNode[]
      state.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  setNodes: (nodes) => {
    set((state) => {
      const selection = filterValidSelections(
        nodes,
        state.edges,
        state.selectedNodeIds,
        state.selectedEdgeIds
      )
      state.rfSetNodes?.(nodes)
      return { nodes, ...selection }
    })
  },

  setEdges: (edges) => {
    set((state) => {
      const selection = filterValidSelections(
        state.nodes,
        edges,
        state.selectedNodeIds,
        state.selectedEdgeIds
      )
      state.rfSetEdges?.(edges)
      return { edges, ...selection }
    })
  },

  setSelection: (nodeIds, edgeIds) => {
    const selection = filterValidSelections(get().nodes, get().edges, nodeIds, edgeIds)
    set(selection)
  },

  clearSelection: () => set({ selectedNodeIds: [], selectedEdgeIds: [] }),

  packageSelection: () => {
    const state = get()
    const selectedNodes = state.nodes.filter(
      (node) => state.selectedNodeIds.includes(node.id) &&
        !isGroupNode(node) &&
        !(node.data as any)?.isCustomComposite  // exclude already-packaged composites
    )

    if (selectedNodes.length < 2) return

    // 计算位置（选中节点的中心）
    let sumX = 0, sumY = 0
    for (const node of selectedNodes) {
      sumX += node.position.x
      sumY += node.position.y
    }
    const centerPos = {
      x: sumX / selectedNodes.length,
      y: sumY / selectedNodes.length,
    }

    // 打包节点
    const pkgNode = packageNodes(
      state.nodes,
      state.edges,
      selectedNodes.map((n) => n.id),
      centerPos
    )

    // 注册为自定义类（用于代码生成）- 如果已存在同名类则复用
    const existingClasses = getAllCustomClasses()
    const existingClass = existingClasses.find(c => c.name === pkgNode.data.customClassId)

    if (!existingClass) {
      // 只有不存在同名类时才注册
      const internalStructure: SubModule[] = pkgNode.data.internalStructure.map((sm) => ({
        id: sm.id,
        type: sm.type as any,
        label: sm.label,
        params: sm.params,
      }))
      const internalEdges: InternalEdge[] = pkgNode.data.internalEdges.map((ie) => ({
        from: ie.from,
        to: ie.to,
      }))

      registerCustomClass(
        pkgNode.data.customClassId, // name
        'custom', // baseType
        'other', // category
        '📦', // emoji
        'Custom packaged module', // description
        internalStructure,
        internalEdges,
        pkgNode.data.outputVar
      )
    }

    // 移除被打包的节点（保留它们的数据在 pkgNode 中）
    const childNodeIds = new Set(selectedNodes.map((n) => n.id))
    const filteredNodes = state.nodes.filter((n) => !childNodeIds.has(n.id))

    // 获取边界边信息用于重定向
    const boundaryEdgesData = pkgNode.data.boundaryEdges as any[] || []

    // 移除内部边（两端都在组内的边），并重定向外部边到包节点
    const internalEdgeIds = new Set(pkgNode.data.internalEdgeIds)
    const filteredEdges = state.edges
      .filter(
        (e) => !internalEdgeIds.has(e.id) &&
               !(childNodeIds.has(e.source) && childNodeIds.has(e.target))
      )
      .map((e) => {
        // 输入边：external -> child，重定向到 external -> package
        if (!childNodeIds.has(e.source) && childNodeIds.has(e.target)) {
          const boundaryEdge = boundaryEdgesData.find(
            (be: any) => be.originalEdgeId === e.id
          )
          return {
            ...e,
            target: pkgNode.id,
            targetHandle: boundaryEdge?.internalHandle || e.targetHandle,
          }
        }
        // 输出边：child -> external，重定向到 package -> external
        if (childNodeIds.has(e.source) && !childNodeIds.has(e.target)) {
          const boundaryEdge = boundaryEdgesData.find(
            (be: any) => be.originalEdgeId === e.id
          )
          return {
            ...e,
            source: pkgNode.id,
            sourceHandle: boundaryEdge?.internalHandle || e.sourceHandle,
          }
        }
        return e
      })

    get().pushHistory()
    const newNodes = [...filteredNodes, pkgNode]

    set((state) => {
      state.rfSetNodes?.(newNodes)
      state.rfSetEdges?.(filteredEdges)
      return {
        nodes: newNodes,
        edges: filteredEdges,
        selectedNodeIds: [pkgNode.id],
        selectedEdgeIds: [],
      }
    })
  },

  expandGroup: (groupId: string) => {
    const state = get()
    const groupNode = state.nodes.find(
      (n) => n.id === groupId && (n.data as any).isCustomComposite === true
    )
    if (!groupNode) return

    const { nodes: newNodes, edges: newEdges } = expandPackage(
      groupNode as any,
      state.nodes,
      state.edges
    )

    get().pushHistory()
    set((store) => {
      store.rfSetNodes?.(newNodes)
      store.rfSetEdges?.(newEdges)
      return {
        nodes: newNodes,
        edges: newEdges,
      }
    })
  },

  collapseGroup: (groupId: string) => {
    const state = get()
    const { nodes: newNodes, edges: newEdges } = collapsePackage(
      groupId,
      state.nodes,
      state.edges
    )

    get().pushHistory()
    set((store) => {
      store.rfSetNodes?.(newNodes)
      store.rfSetEdges?.(newEdges)
      return {
        nodes: newNodes,
        edges: newEdges,
        selectedNodeIds: [groupId],
      }
    })
  },

  unpackageGroup: (groupId: string) => {
    const state = get()
    const node = state.nodes.find(
      (n) => n.id === groupId && (n.data as CustomCompositeNodeData)?.isCustomComposite
    )
    if (!node) return

    const { customClassId } = node.data as CustomCompositeNodeData

    // Count how many instances of this class exist in the graph
    const instanceCount = state.nodes.filter(
      (n) => (n.data as CustomCompositeNodeData)?.isCustomComposite &&
             (n.data as CustomCompositeNodeData).customClassId === customClassId
    ).length

    get().pushHistory()

    if (instanceCount === 1) {
      // Single instance: fully dissolve into individual nodes
      const { nodes: newNodes, edges: newEdges } = fullyUnpackageGroup(
        groupId,
        state.nodes,
        state.edges
      )
      set((store) => {
        store.rfSetNodes?.(newNodes)
        store.rfSetEdges?.(newEdges)
        return { nodes: newNodes, edges: newEdges, selectedNodeIds: [] }
      })
    } else {
      // Multiple instances: expand only this instance in-place
      const { nodes: newNodes, edges: newEdges } = expandPackage(
        node as any,
        state.nodes,
        state.edges
      )
      set((store) => {
        store.rfSetNodes?.(newNodes)
        store.rfSetEdges?.(newEdges)
        return { nodes: newNodes, edges: newEdges, selectedNodeIds: [groupId] }
      })
    }
  },

  renamePackage: (groupId: string, newName: string) => {
    const state = get()
    const newNodes = state.nodes.map((n) => {
      if (n.id === groupId && (n.data as CustomCompositeNodeData)?.isCustomComposite) {
        const newData = {
          ...n.data,
          label: newName,  // 只更新显示名/实例名
        }
        return { ...n, data: newData }
      }
      return n
    })
    set((store) => {
      store.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  renamePackageClass: (groupId: string, newClassName: string) => {
    const state = get()
    const node = state.nodes.find((n) => n.id === groupId && (n.data as CustomCompositeNodeData)?.isCustomComposite)
    if (!node) return

    const oldClassName = (node.data as CustomCompositeNodeData).customClassId
    if (oldClassName === newClassName) return

    // Update class name in the registry
    const allClasses = getAllCustomClasses()
    const classToRename = allClasses.find(c => c.name === oldClassName)
    if (classToRename) {
      updateCustomClass(classToRename.id, { name: newClassName })
    }

    // Update all instances: keep node IDs stable (critical!), only update class ref and label
    const newNodes = state.nodes.map((n) => {
      if ((n.data as CustomCompositeNodeData)?.isCustomComposite &&
          (n.data as CustomCompositeNodeData).customClassId === oldClassName) {
        // node.id is NOT changed — edges reference it by id, must stay stable
        return {
          ...n,
          data: {
            ...n.data,
            customClassId: newClassName,
            label: newClassName,
          },
        }
      }
      return n
    })
    set((store) => {
      store.rfSetNodes?.(newNodes)
      return { nodes: newNodes }
    })
  },

  copySelection: () => {
    const state = get()
    const expandedNodeIds = expandSelectedNodeIds(state.nodes, state.selectedNodeIds)

    if (expandedNodeIds.length === 0) return

    const selectedNodeIdSet = new Set(expandedNodeIds)
    const nodeMap = new Map(state.nodes.map((node) => [node.id, node]))
    const clipboardNodes = state.nodes
      .filter((node) => selectedNodeIdSet.has(node.id))
      .map((node) => {
        const parentNodeId = node.parentNode ?? node.parentId
        const keepParent = parentNodeId ? selectedNodeIdSet.has(parentNodeId) : false

        if (keepParent) {
          return {
            ...node,
            selected: false,
          }
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

    const clipboardEdges = state.edges
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

  pasteClipboard: () => {
    const state = get()
    if (!state.clipboard || state.clipboard.nodes.length === 0) return

    const nodeIdMap = new Map<string, string>()
    const edgeIdMap = new Map<string, string>()
    const pasteOffset = 40 * (state.pasteCount + 1)
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
          : {
              x: node.position.x + pasteOffset,
              y: node.position.y + pasteOffset,
            },
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
        },
      }

      return translatedNode
    }) as FlowHamsterNode[]

    const mergedEdges = [...state.edges, ...newEdges]
    const mergedNodes = syncGroupMetadata([...state.nodes, ...newNodes], mergedEdges)

    get().pushHistory()
    set((store) => {
      store.rfSetNodes?.(mergedNodes)
      store.rfSetEdges?.(mergedEdges)
      return {
        nodes: mergedNodes,
        edges: mergedEdges,
        selectedNodeIds: newNodes.map((node) => node.id),
        selectedEdgeIds: newEdges.map((edge) => edge.id),
        pasteCount: state.pasteCount + 1,
      }
    })
  },

  deleteSelection: () => {
    const state = get()
    const expandedNodeIds = expandSelectedNodeIds(state.nodes, state.selectedNodeIds)
    const selectedNodeIdSet = new Set(expandedNodeIds)
    const selectedEdgeIdSet = new Set(state.selectedEdgeIds)

    if (selectedNodeIdSet.size === 0 && selectedEdgeIdSet.size === 0) return

    const newEdges = state.edges.filter(
      (edge) =>
        !selectedEdgeIdSet.has(edge.id) &&
        !selectedNodeIdSet.has(edge.source) &&
        !selectedNodeIdSet.has(edge.target)
    ) as FlowHamsterEdge[]
    const newNodes = syncGroupMetadata(
      state.nodes.filter((node) => !selectedNodeIdSet.has(node.id)) as FlowHamsterNode[],
      newEdges
    )

    get().pushHistory()
    set((store) => {
      store.rfSetNodes?.(newNodes)
      store.rfSetEdges?.(newEdges)
      return {
        nodes: newNodes,
        edges: newEdges,
        selectedNodeIds: [],
        selectedEdgeIds: [],
      }
    })
  },

  setMergeMode: (nodeId, mode) => {
    get().pushHistory()
    set((state) => {
      const newEdges = state.edges.map((e) =>
        e.target === nodeId ? { ...e, data: { ...e.data, mergeMode: mode } } : e
      )
      state.rfSetEdges?.(newEdges)
      return { edges: newEdges }
    })
  },

  loadTemplate: (template) => {
    // 验证模板
    const validationMessage = validateTemplateWithMessages(template.graph)
    if (!validationMessage.startsWith('Template is valid')) {
      console.warn('[FlowHamster] Template validation warnings:', validationMessage)
    }

    get().pushHistory()
    set((state) => {
      const newNodes = template.graph.nodes.map((n: any) => ({
        ...n,
        type: getNodeComponentType(n.data.nodeType),
        position: n.position || { x: 0, y: 0 },
        data: normalizeNodeData(n.data),
      })) as FlowHamsterNode[]
      const newEdges = template.graph.edges as FlowHamsterEdge[]
      const selection = filterValidSelections(newNodes, newEdges, [], [])
      state.rfSetNodes?.(newNodes)
      state.rfSetEdges?.(newEdges)
      return { nodes: newNodes, edges: newEdges, ...selection }
    })
  },

  loadTemplateFromBackend: async (templateId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/templates/${templateId}`)

      if (!response.ok) {
        console.error(`[FlowHamster] Failed to load template ${templateId}: ${response.status}`)
        return false
      }

      const backendTemplate = await response.json()

      // 验证模板
      const validationResult = validateTemplateWithMessages(backendTemplate)
      if (!validationResult.startsWith('Template is valid')) {
        console.warn(`[FlowHamster] Template validation warnings for ${templateId}:`, validationResult)
      }

      // 转换为前端格式
      const adaptedTemplate = adaptBackendTemplate(backendTemplate)

      // 加载到画布
      get().loadTemplate(adaptedTemplate)

      console.log(`[FlowHamster] Loaded template ${templateId} from backend`)
      return true
    } catch (error) {
      console.error(`[FlowHamster] Error loading template ${templateId}:`, error)
      return false
    }
  },

  _internalSetNodes: (nodes, edges) => {
    set((state) => {
      state.rfSetNodes?.(nodes)
      if (edges !== undefined) {
        state.rfSetEdges?.(edges)
        return { nodes, edges }
      }
      return { nodes }
    })
  },
}))
