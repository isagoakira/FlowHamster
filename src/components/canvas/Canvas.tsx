import { useCallback, DragEvent, useState, useEffect, useMemo, useRef } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  NodeTypes,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
  Edge,
  Node,
  OnSelectionChangeParams,
  useReactFlow,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { useGraphStore, MergeMode } from '../../hooks/useGraphStore'
import { useClipboardStore } from '../../stores/useClipboardStore'
import { NodeData, FlowHamsterNode, CustomCompositeNodeData } from '../../types/graph'
import NodeEditPanel from './NodeEditPanel'
import { normalizeNodeData, normalizeNodeType } from '../../utils/nodeType'
import { getReactFlowNodeTypes } from '../../utils/nodeComponentRegistry'
import PackageViewerDialog from '../nodes/PackageViewerDialog'

// 分支颜色表（多 Output 时使用）
const BRANCH_COLORS = [
  '#4488ff', // 蓝色 — branch 0
  '#44cc88', // 绿色 — branch 1
  '#ffaa44', // 橙色 — branch 2
  '#ff6688', // 粉色 — branch 3
  '#aa66ff', // 紫色 — branch 4
]

/**
 * 计算图的"结构键"——仅包含拓扑结构，不含节点位置。
 * 用于 useMemo 依赖，避免拖拽时因 position 变化触发重算。
 */
function computeStructuralKey(nodes: Node<NodeData>[], edges: Edge[]): string {
  const nodeKey = nodes
    .map(n => `${n.id}:${JSON.stringify(n.data?.params ?? {})}`)
    .sort()
    .join('|')
  const edgeKey = edges
    .map(e => `${e.source}:${e.target}:${e.sourceHandle ?? ''}:${e.targetHandle ?? ''}`)
    .sort()
    .join('|')
  return `${nodeKey}#${edgeKey}`
}

/**
 * 计算节点的分支归属（反向 BFS，从每个 Output 往上）
 * 仅当 features.multiOutput === true 时使用
 */
function computeBranchAssignments(
  nodes: Node<NodeData>[],
  edges: Edge[]
): Record<string, number> {
  const nodeIds = new Set(nodes.map(n => n.id))
  const outputNodes = nodes.filter(n => (n.data.nodeType as string) === 'output')
  const reverseAdj: Record<string, string[]> = {}
  for (const n of nodes) reverseAdj[n.id] = []
  for (const e of edges) {
    if (reverseAdj[e.target]) reverseAdj[e.target].push(e.source)
  }

  const assignments: Record<string, number> = {}
  outputNodes.forEach((outNode, outIdx) => {
    const queue = [outNode.id]
    while (queue.length > 0) {
      const nid = queue.shift()!
      if (nid in assignments) continue
      assignments[nid] = outIdx
      for (const src of reverseAdj[nid] || []) {
        if (nodeIds.has(src)) queue.push(src)
      }
    }
  })
  return assignments
}

// 使用注册表生成 nodeTypes
const nodeTypes: NodeTypes = getReactFlowNodeTypes()

const defaultNodes = [
  { id: 'input_1', type: 'inputNode', position: { x: 80, y: 200 }, data: { nodeType: 'input', label: 'Input', params: { shape: '3,224,224', dtype: 'float32' } } as NodeData },
  { id: 'conv2d_1', type: 'conv2dNode', position: { x: 320, y: 180 }, data: { nodeType: 'conv2d', label: 'Conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false } } as NodeData },
  { id: 'relu_1', type: 'reluNode', position: { x: 560, y: 180 }, data: { nodeType: 'relu', label: 'ReLU', params: {} } as NodeData },
  { id: 'output_1', type: 'outputNode', position: { x: 800, y: 200 }, data: { nodeType: 'output', label: 'Output', params: {} } as NodeData },
]

const defaultEdges = [
  { id: 'e1-2', source: 'input_1', target: 'conv2d_1', type: 'smoothstep', animated: true },
  { id: 'e2-3', source: 'conv2d_1', target: 'relu_1', type: 'smoothstep', animated: true },
  { id: 'e3-4', source: 'relu_1', target: 'output_1', type: 'smoothstep', animated: true },
]

// Gradient edge style helpers
const getEdgeStyle = (gradient_strength: number): React.CSSProperties => {
  if (gradient_strength === 0) return { stroke: '#888', strokeDasharray: '5,5', strokeWidth: 1 }
  if (gradient_strength < 0.2) return { stroke: '#ef4444', strokeWidth: 1 }
  if (gradient_strength < 0.5) return { stroke: '#f59e0b', strokeWidth: 2 }
  if (gradient_strength < 0.8) return { stroke: '#22c55e', strokeWidth: 3 }
  return { stroke: '#16a34a', strokeWidth: 4 }
}

// Gradient legend overlay
function GradientLegend() {
  return (
    <div style={{
      position: 'absolute', bottom: 12, right: 12,
      background: '#111d', border: '1px solid #333',
      borderRadius: '8px', padding: '10px 14px',
      fontSize: '11px', color: '#ccc',
      zIndex: 10, minWidth: '160px',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, color: '#a0c0ff', fontSize: '11px' }}>Gradient Flow</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div style={{ width: 24, height: 3, background: '#16a34a', borderRadius: 2 }} />
        <span style={{ color: '#aaa' }}>Strong (0.8–1.0)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div style={{ width: 24, height: 2, background: '#22c55e', borderRadius: 2 }} />
        <span style={{ color: '#aaa' }}>Medium (0.5–0.8)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div style={{ width: 24, height: 2, background: '#f59e0b', borderRadius: 2 }} />
        <span style={{ color: '#aaa' }}>Weak (0.2–0.5)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div style={{ width: 24, height: 1, background: '#ef4444', borderRadius: 2 }} />
        <span style={{ color: '#aaa' }}>Very weak (0–0.2)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{ width: 24, height: 1, background: '#888', borderRadius: 2, borderBottom: '1px dashed #888' }} />
        <span style={{ color: '#aaa' }}>Zero (dropout)</span>
      </div>
    </div>
  )
}

const MERGE_MODES: { value: MergeMode; label: string; desc: string }[] = [
  { value: 'concat', label: 'Concat', desc: 'torch.cat([a, b], dim=N)' },
  { value: 'add', label: 'Add', desc: 'a + b (残差连接)' },
  { value: 'mul', label: 'Multiply', desc: 'a * b (门控)' },
  { value: 'stack', label: 'Stack', desc: 'torch.stack([a, b], dim=N)' },
]

function MergeModeEditor({ edge, onClose }: { edge: Edge; onClose: () => void }) {
  const setMergeMode = useGraphStore((s) => s.setMergeMode)
  const currentMode: MergeMode = (edge.data?.mergeMode as MergeMode) || 'concat'

  const handleModeChange = (mode: MergeMode) => {
    setMergeMode(edge.target, mode)
  }

  return (
    <div
      style={{
        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        background: '#1e1e1e', border: '1px solid #444', borderRadius: '10px',
        padding: '16px', zIndex: 1000, minWidth: '280px', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div style={{ fontSize: '12px', fontWeight: 700, color: '#a0c0ff', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        合并方式 — Merge Mode
      </div>
      <div style={{ fontSize: '11px', color: '#666', marginBottom: '10px' }}>
        边: {edge.source} → {edge.target}
      </div>
      {MERGE_MODES.map((m) => (
        <div
          key={m.value}
          onClick={() => handleModeChange(m.value)}
          style={{
            padding: '8px 10px', borderRadius: '6px', cursor: 'pointer', marginBottom: '4px',
            background: currentMode === m.value ? '#1a3a5a' : 'transparent',
            border: currentMode === m.value ? '1px solid #4488ff' : '1px solid transparent',
          }}
        >
          <div style={{ fontSize: '12px', fontWeight: 600, color: currentMode === m.value ? '#88bbff' : '#ccc' }}>{m.label}</div>
          <div style={{ fontSize: '10px', color: '#666', marginTop: 2 }}>{m.desc}</div>
        </div>
      ))}
      <button
        onClick={onClose}
        style={{ marginTop: '8px', width: '100%', background: '#333', border: '1px solid #555', borderRadius: '6px', color: '#ccc', padding: '6px', cursor: 'pointer', fontSize: '12px' }}
      >关闭</button>
    </div>
  )
}

function FlowCanvas() {
  const features = useGraphStore((s) => s.features)
  const registerRfSetters = useGraphStore((s) => s.registerRfSetters)
  const connectNodes = useGraphStore((s) => s.onConnect)
  const setSelection = useGraphStore((s) => s.setSelection)
  const clearSelection = useGraphStore((s) => s.clearSelection)
  const { screenToFlowPosition, fitView, zoomTo } = useReactFlow()
  const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null)
  const [editingNode, setEditingNode] = useState<FlowHamsterNode | null>(null)
  // Context menu state
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null)
  const nodeWasDraggedRef = useRef(false)
  const unpackageGroup = useGraphStore((s) => s.unpackageGroup)
  const renamePackage = useGraphStore((s) => s.renamePackage)
  const renamePackageClass = useGraphStore((s) => s.renamePackageClass)
  // Package viewer state
  const packageViewerOpen = useGraphStore((s) => s.packageViewerOpen)
  const packageViewerData = useGraphStore((s) => s.packageViewerData)
  const closePackageViewer = useGraphStore((s) => s.closePackageViewer)
  const openPackageViewer = useGraphStore((s) => s.openPackageViewer)
  const [gradientData, setGradientData] = useState<{
    edges: Array<{ source: string; target: string; gradient_strength: number; direction: string }>
    nodes: Array<{ id: string; gradient_score: number; risk: string }>
  } | null>(null)

  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes as any)
  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges as any)

  // 分支着色：仅当 multiOutput 功能开启时计算
  // 用结构键而非 nodes 数组本身——拖拽时 position 变化不会触发重算
  const structuralKey = useMemo(
    () => computeStructuralKey(nodes as Node<NodeData>[], edges),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [edges] // 边结构决定分支，只依赖 edges（结构稳定时不需要重新计算）
  )
  const branchAssignments = useMemo(
    () => features.multiOutput ? computeBranchAssignments(nodes as Node<NodeData>[], edges) : {},
    [structuralKey, features.multiOutput]
  )

  // 多选高亮：选中的节点添加发光边框（只有多选时才高亮）
  const selectedNodeIds = useGraphStore((s) => s.selectedNodeIds)
  const isMultiSelect = selectedNodeIds.length > 1
  useEffect(() => {
    // 只有多选时才应用高亮，单选不清高亮
    if (!isMultiSelect) {
      // 清除所有高亮
      setNodes((nds) =>
        nds.map((n: Node<NodeData>) => {
          // 移除高亮样式
          const { boxShadow: _, border: __, ...restStyle } = n.style || {}
          return {
            ...n,
            style: restStyle,
          }
        }) as any
      )
      return
    }
    // 多选时应用高亮
    setNodes((nds) =>
      nds.map((n: Node<NodeData>) => {
        const isSelected = selectedNodeIds.includes(n.id)
        const branchIdx = branchAssignments[n.id]
        const branchColor = branchIdx !== undefined ? BRANCH_COLORS[branchIdx % BRANCH_COLORS.length] : undefined
        return {
          ...n,
          style: {
            ...n.style,
            ...(isSelected ? {
              boxShadow: '0 0 12px rgba(99, 102, 241, 0.8)',
              border: `2px solid ${branchColor || '#6366f1'}`,
            } : {
              boxShadow: n.style?.boxShadow || 'none',
              border: branchColor ? `1px solid ${branchColor}` : n.style?.border,
            }),
          },
        }
      }) as any
    )
  }, [selectedNodeIds, branchAssignments, isMultiSelect])

  // 给节点注入分支颜色（用于 BaseNode / 自定义节点边框）
  useEffect(() => {
    if (!features.multiOutput) return
    setNodes((nds) =>
      nds.map((n: Node<NodeData>) => {
        const branchIdx = branchAssignments[n.id]
        const color = branchIdx !== undefined ? BRANCH_COLORS[branchIdx % BRANCH_COLORS.length] : '#333'
        return {
          ...n,
          style: { ...n.style, borderColor: color },
        }
      }) as any
    )
  }, [branchAssignments, features.multiOutput])

  // 同步 ReactFlow 状态到 store (用户拖拽节点时)
  useEffect(() => {
    useGraphStore.setState({ nodes: nodes as any })
  }, [nodes])

  useEffect(() => {
    useGraphStore.setState({ edges: edges as any })
  }, [edges])

  // 注册 ReactFlow setters 到 store
  useEffect(() => {
    registerRfSetters(setNodes as any, setEdges as any)
  }, [registerRfSetters, setNodes, setEdges])

  // Keyboard shortcuts: Ctrl+C/V/X, Delete, F=fit, 0=reset zoom
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input field
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) return
      
      const isCtrlOrMeta = e.ctrlKey || e.metaKey
      
      if (isCtrlOrMeta && e.key === 'c') {
        e.preventDefault()
        useClipboardStore.getState().copySelection()
      } else if (isCtrlOrMeta && e.key === 'v') {
        e.preventDefault()
        useClipboardStore.getState().pasteClipboard()
      } else if (isCtrlOrMeta && e.key === 'x') {
        e.preventDefault()
        useClipboardStore.getState().cutSelection()
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault()
        fitView({ duration: 300 })
      } else if (e.key === '0' || (isCtrlOrMeta && e.key === '0')) {
        e.preventDefault()
        zoomTo(1)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [fitView, zoomTo])

  // Gradient visualization callbacks (set by Toolbar)
  useEffect(() => {
    ;(window as any).__flowhamster_setGradient = (data: typeof gradientData) => {
      setGradientData(data)
    }
    ;(window as any).__flowhamster_exitGradient = () => {
      setGradientData(null)
    }
    return () => {
      delete (window as any).__flowhamster_setGradient
      delete (window as any).__flowhamster_exitGradient
    }
  }, [])

  const onConnect = useCallback(
    (connection: Connection) => {
      connectNodes(connection)
    },
    [connectNodes]
  )

  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      const ids = new Set(deleted.map((n) => n.id))
      setEdges((eds) => eds.filter((e) => !ids.has(e.source) && !ids.has(e.target)) as any)
    },
    [setEdges]
  )

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      const ids = new Set(deleted.map((d) => d.id))
      setEdges((eds) => eds.filter((e) => !ids.has(e.id)) as any)
      if (selectedEdge && ids.has(selectedEdge.id)) setSelectedEdge(null)
    },
    [setEdges, selectedEdge]
  )

  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    setSelectedEdge(edge)
  }, [])

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    if (nodeWasDraggedRef.current) {
      nodeWasDraggedRef.current = false
      return
    }
    if (event.ctrlKey || event.metaKey || event.shiftKey) {
      setEditingNode(null)
      return
    }
    setSelectedEdge(null)
    setEditingNode(node as unknown as FlowHamsterNode)
  }, [])

  const onNodeDragStart = useCallback(() => {
    nodeWasDraggedRef.current = false
  }, [])

  const onNodeDrag = useCallback(() => {
    nodeWasDraggedRef.current = true
  }, [])

  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes, edges: selectedEdges }: OnSelectionChangeParams) => {
      setSelection(
        selectedNodes.map((node) => node.id),
        selectedEdges.map((edge) => edge.id)
      )
      if (selectedNodes.length !== 1) {
        setEditingNode(null)
      }
    },
    [setSelection]
  )

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault()
      const data = event.dataTransfer.getData('application/flowhamster')
      if (!data) return
      const {
        nodeType,
        label,
        defaultParams,
        customClassRegistryId,
        customClassId,
        originClassId,
      } = JSON.parse(data) as {
        nodeType: string
        label: string
        defaultParams: Record<string, number | string | boolean>
        customClassRegistryId?: string
        customClassId?: string
        originClassId?: string
      }
      const normalizedType = normalizeNodeType(nodeType)
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY })
      useGraphStore.getState().addNode(
        normalizeNodeData({
          nodeType: normalizedType,
          label,
          params: { ...defaultParams },
          customClassRegistryId,
          customClassId,
          originClassId,
        } as NodeData),
        position
      )
    },
    [screenToFlowPosition]
  )

  // Build styled edges when gradient mode is active
  const styledEdges = gradientData
    ? edges.map((e) => {
        const gd = gradientData.edges.find(
          (g) => g.source === e.source && g.target === e.target
        )
        const strength = gd?.gradient_strength ?? 0
        const style = getEdgeStyle(strength)
        return {
          ...e,
          animated: false,
          style: {
            ...(e.style as Record<string, string | number>),
            ...style,
          },
          label: (
            <div style={{
              background: 'white',
              border: '1px solid #ccc',
              borderRadius: 4,
              padding: '2px 6px',
              fontSize: 10,
              fontFamily: 'monospace',
              color: '#333',
            }}>
              {strength.toFixed(2)}
            </div>
          ),
          labelStyle: {},
          labelBgStyle: {},
        }
      })
    : edges

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={styledEdges as any}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        onEdgeClick={onEdgeClick}
        onNodeClick={onNodeClick}
        onNodeDragStart={onNodeDragStart}
        onNodeDrag={onNodeDrag}
        onSelectionChange={onSelectionChange}
        onPaneClick={() => {
          clearSelection()
          setSelectedEdge(null)
          setEditingNode(null)
          setContextMenu(null)
        }}
        onNodeContextMenu={(event, node) => {
          event.preventDefault()
          const nodeData = node.data as any
          if (nodeData?.isCustomComposite) {
            setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id })
          }
        }}
        nodeTypes={nodeTypes}
        fitView
        deleteKeyCode="Delete"
        multiSelectionKeyCode={['Control', 'Meta']}
        style={{ background: '#0a0a0a' }}
        defaultEdgeOptions={{ type: 'smoothstep', animated: !gradientData }}
      >
        <Background color="#222" gap={20} />
        <Controls style={{ background: '#1a1a1a', border: '1px solid #333' }} />
        <MiniMap nodeColor="#334" maskColor="rgba(0,0,0,0.8)" style={{ background: '#111', border: '1px solid #333' }} />
      </ReactFlow>

      {selectedEdge && (
        <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setSelectedEdge(null)}>
          <MergeModeEditor edge={selectedEdge} onClose={() => setSelectedEdge(null)} />
        </div>
      )}

      {gradientData && <GradientLegend />}

      {editingNode && (
        <NodeEditPanel
          node={editingNode}
          onClose={() => setEditingNode(null)}
        />
      )}

      {/* Package Viewer Dialog */}
      <PackageViewerDialog
        isOpen={packageViewerOpen}
        mode={useGraphStore((s) => s.packageViewerMode)}
        packageData={packageViewerData}
        onClose={closePackageViewer}
        onUnpackage={() => {
          const nodeId = useGraphStore.getState().packageViewerNodeId
          if (nodeId) {
            unpackageGroup(nodeId)
          }
          closePackageViewer()
        }}
      />

      {/* Context Menu for Custom Composite Nodes */}
      {contextMenu && (
	          <div
	            style={{
            position: 'fixed',
            top: contextMenu.y,
            left: contextMenu.x,
            background: '#1e1e1e',
            border: '1px solid #444',
            borderRadius: '8px',
            padding: '4px 0',
            zIndex: 1000,
            minWidth: '160px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
          }}
	          onClick={(e) => e.stopPropagation()}
	        >
	          <div
	            style={{
	              padding: '8px 16px',
	              cursor: 'pointer',
	              color: '#22d3ee',
	              fontSize: '13px',
	              display: 'flex',
	              alignItems: 'center',
	              gap: '8px',
	            }}
	            onClick={() => {
	              const node = useGraphStore.getState().nodes.find((n) => n.id === contextMenu.nodeId)
	              const currentName = node?.data.label || 'Module'
	              const newName = window.prompt('请输入新的实例名称:', currentName)
	              if (newName?.trim() && newName.trim() !== currentName) {
	                renamePackage(contextMenu.nodeId, newName.trim())
	              }
	              setContextMenu(null)
	            }}
	            onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
	            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
	          >
	            ✏️ 重命名实例
	          </div>
	          <div
	            style={{
	              padding: '8px 16px',
	              cursor: 'pointer',
	              color: '#a78bfa',
	              fontSize: '13px',
	              display: 'flex',
	              alignItems: 'center',
	              gap: '8px',
	            }}
	            onClick={() => {
	              const node = useGraphStore.getState().nodes.find((n) => n.id === contextMenu.nodeId)
	              const currentClassName = (node?.data as CustomCompositeNodeData | undefined)?.customClassId || 'Module_1'
	              const newClassName = window.prompt('请输入新的类名称:', currentClassName)
	              if (newClassName?.trim() && newClassName.trim() !== currentClassName) {
	                renamePackageClass(contextMenu.nodeId, newClassName.trim())
	              }
	              setContextMenu(null)
	            }}
	            onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
	            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
	          >
	            🏷️ 重命名类
	          </div>
	          <div
	            style={{              padding: '8px 16px',
              cursor: 'pointer',
              color: '#f97316',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
            onClick={() => {
              openPackageViewer(contextMenu.nodeId, 'edit')
              setContextMenu(null)
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
	            ✨️ 编辑类定义
	          </div>
	          <div
	            style={{
	              padding: '8px 16px',
              cursor: 'pointer',
              color: '#ccc',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
            onClick={() => {
              unpackageGroup(contextMenu.nodeId)
              setContextMenu(null)
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            📦 解包为普通节点
          </div>
          <div
            style={{
              padding: '8px 16px',
              cursor: 'pointer',
              color: '#888',
              fontSize: '12px',
              borderTop: '1px solid #333',
              marginTop: '4px',
              paddingTop: '12px',
            }}
            onClick={() => setContextMenu(null)}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            取消
          </div>
        </div>
      )}
    </div>
  )
}

export default function Canvas() {
  return (
    <ReactFlowProvider>
      <FlowCanvas />
    </ReactFlowProvider>
  )
}
