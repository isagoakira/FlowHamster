/**
 * PackageViewerDialog - 弹窗查看/编辑打包模块的内部结构
 */

import { memo, useMemo, useEffect, useState, useCallback } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  ReactFlowProvider,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { CustomCompositeNodeData, FlowHamsterNode, FlowHamsterEdge } from '../../types/graph'
import { getNodeComponentType } from '../../utils/nodeType'
import { getReactFlowNodeTypes } from '../../utils/nodeComponentRegistry'
import { computeTopologicalLayout } from '../../utils/subgraphPackager'
import { updateCustomClass, registerCustomClass, getCustomClass } from '../../utils/customCompositeRegistry'

// 获取 ReactFlow 节点类型映射
const nodeTypes = getReactFlowNodeTypes()

// 弹窗样式
const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0, 0, 0, 0.85)',
  zIndex: 2000,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

const panelStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: 12,
  width: '90vw',
  maxWidth: 1200,
  height: '85vh',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 16px 64px rgba(0, 0, 0, 0.8)',
  overflow: 'hidden',
}

const headerStyle: React.CSSProperties = {
  padding: '16px 20px',
  borderBottom: '1px solid #333',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  background: '#222',
}

const contentStyle: React.CSSProperties = {
  flex: 1,
  overflow: 'hidden',
}

interface PackageViewerDialogProps {
  isOpen: boolean
  mode: 'view' | 'edit'
  packageData: CustomCompositeNodeData | null
  onClose: () => void
  onUnpackage: () => void
}

// 输入占位符节点组件
const InputPlaceholder = ({ data }: { data: { label: string } }) => (
  <div style={{
    background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(14, 116, 144, 0.25))',
    border: '2px solid #22d3ee',
    borderRadius: 8,
    padding: '10px 14px',
    minWidth: 90,
    textAlign: 'center',
  }}>
    <div style={{ fontSize: 9, color: '#22d3ee', marginBottom: 3, fontWeight: 600 }}>输入</div>
    <div style={{ fontSize: 11, color: '#e0e0e0', fontWeight: 500 }}>{data.label}</div>
    <Handle
      type="source"
      position={Position.Right}
      id="output"
      style={{
        background: '#22d3ee',
        border: '2px solid #0e7490',
        width: 9,
        height: 9,
        top: '50%',
        transform: 'translateY(-50%)',
      }}
    />
  </div>
)

// 输出占位符节点组件
const OutputPlaceholder = ({ data }: { data: { label: string } }) => (
  <div style={{
    background: 'linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(21, 128, 61, 0.25))',
    border: '2px solid #4ade80',
    borderRadius: 8,
    padding: '10px 14px',
    minWidth: 90,
    textAlign: 'center',
  }}>
    <Handle
      type="target"
      position={Position.Left}
      id="input"
      style={{
        background: '#4ade80',
        border: '2px solid #15803d',
        width: 9,
        height: 9,
        top: '50%',
        transform: 'translateY(-50%)',
      }}
    />
    <div style={{ fontSize: 9, color: '#4ade80', marginBottom: 3, fontWeight: 600 }}>输出</div>
    <div style={{ fontSize: 11, color: '#e0e0e0', fontWeight: 500 }}>{data.label}</div>
  </div>
)

// 内部 ReactFlow 组件 (view mode only - no editing internal structure)
function InternalFlowView({
  internalNodes,
  internalEdges,
}: {
  internalNodes: Node[]
  internalEdges: Edge[]
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState(internalNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(internalEdges)

  useEffect(() => {
    setNodes(internalNodes)
    setEdges(internalEdges)
  }, [internalNodes, internalEdges, setNodes, setEdges])

  const mergedNodeTypes = useMemo(() => ({
    ...nodeTypes,
    inputPlaceholder: InputPlaceholder,
    outputPlaceholder: OutputPlaceholder,
  }), [])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={mergedNodeTypes}
      fitView
      style={{ background: '#0a0a0a' }}
      defaultEdgeOptions={{ type: 'smoothstep', animated: true }}
    >
      <Background color="#222" gap={20} />
      <Controls style={{ background: '#1a1a1a', border: '1px solid #333' }} />
      <MiniMap
        nodeColor="#334"
        maskColor="rgba(0,0,0,0.8)"
        style={{ background: '#111', border: '1px solid #333' }}
      />
    </ReactFlow>
  )
}

// 内部 ReactFlow 组件 (edit mode - nodes/editing enabled)
function InternalFlowEdit({
  internalNodes,
  internalEdges,
  editedParams,
  onNodeSelect,
  selectedNodeId,
  onParamChange,
}: {
  internalNodes: Node[]
  internalEdges: Edge[]
  editedParams: Record<string, Record<string, any>>
  onNodeSelect: (nodeId: string | null) => void
  selectedNodeId: string | null
  onParamChange: (nodeId: string, param: string, value: any) => void
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState(internalNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(internalEdges)

  useEffect(() => {
    setNodes(internalNodes)
    setEdges(internalEdges)
  }, [internalNodes, internalEdges, setNodes, setEdges])

  const mergedNodeTypes = useMemo(() => ({
    ...nodeTypes,
    inputPlaceholder: InputPlaceholder,
    outputPlaceholder: OutputPlaceholder,
  }), [])

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <div style={{ flex: 1 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={(_, node) => onNodeSelect(node.id)}
          nodeTypes={mergedNodeTypes}
          fitView
          style={{ background: '#0a0a0a' }}
          defaultEdgeOptions={{ type: 'smoothstep', animated: true }}
        >
          <Background color="#222" gap={20} />
          <Controls style={{ background: '#1a1a1a', border: '1px solid #333' }} />
          <MiniMap
            nodeColor="#334"
            maskColor="rgba(0,0,0,0.8)"
            style={{ background: '#111', border: '1px solid #333' }}
          />
        </ReactFlow>
      </div>
      {/* Parameter editor sidebar */}
      {selectedNodeId && (
        <div style={{
          width: 280,
          background: '#1a1a1a',
          borderLeft: '1px solid #333',
          padding: 16,
          overflowY: 'auto',
        }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#e0e0e0', marginBottom: 12 }}>
            📝 节点参数
          </div>
          {(() => {
            const node = nodes.find(n => n.id === selectedNodeId)
            if (!node) return null
            const params = editedParams[selectedNodeId] || node.data?.params || {}
            return (
              <div>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>
                  类型: {node.data?.nodeType || 'unknown'}
                </div>
                {Object.entries(params).map(([key, value]) => (
                  <div key={key} style={{ marginBottom: 10 }}>
                    <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>
                      {key}
                    </label>
                    <input
                      type={typeof value === 'number' ? 'number' : 'text'}
                      value={value}
                      onChange={(e) => {
                        const num = typeof value === 'number' ? parseFloat(e.target.value) : e.target.value
                        onParamChange(selectedNodeId, key, isNaN(num as number) ? e.target.value : num)
                      }}
                      style={{
                        width: '100%',
                        padding: '6px 8px',
                        background: '#2a2a2a',
                        border: '1px solid #444',
                        borderRadius: 4,
                        color: '#e0e0e0',
                        fontSize: 12,
                      }}
                    />
                  </div>
                ))}
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}

// Derive class dialog component
function DeriveClassDialog({
  onSave,
  onCancel,
}: {
  onSave: (name: string, emoji: string, category: 'cv' | 'nlp' | 'gan' | 'other', description: string) => void
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [emoji, setEmoji] = useState('🔧')
  const [category, setCategory] = useState<'cv' | 'nlp' | 'gan' | 'other'>('other')
  const [description, setDescription] = useState('')

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.8)',
      zIndex: 2100,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={{
        background: '#1a1a1a',
        border: '1px solid #444',
        borderRadius: 12,
        padding: 24,
        width: 400,
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      }}>
        <div style={{ fontWeight: 700, fontSize: 16, color: '#e0e0e0', marginBottom: 20 }}>
          📋 保存为派生类
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 6 }}>类名称</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="MyDerivedClass"
            style={{
              width: '100%',
              padding: '8px 12px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#e0e0e0',
              fontSize: 13,
            }}
          />
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 6 }}>分类</label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value as any)}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#e0e0e0',
              fontSize: 13,
            }}
          >
            <option value="cv">Computer Vision</option>
            <option value="nlp">NLP</option>
            <option value="gan">GAN</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 6 }}>图标</label>
            <input
              type="text"
              value={emoji}
              onChange={(e) => setEmoji(e.target.value)}
              placeholder="🔧"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: 6,
                color: '#e0e0e0',
                fontSize: 13,
              }}
            />
          </div>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 6 }}>描述</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="描述这个派生类的功能..."
            rows={3}
            style={{
              width: '100%',
              padding: '8px 12px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#e0e0e0',
              fontSize: 13,
              resize: 'vertical',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
          <button
            onClick={onCancel}
            style={{
              padding: '8px 16px',
              background: '#333',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#ccc',
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            取消
          </button>
          <button
            onClick={() => {
              if (name.trim()) {
                onSave(name.trim(), emoji, category, description.trim())
              }
            }}
            disabled={!name.trim()}
            style={{
              padding: '8px 16px',
              background: name.trim() ? '#4c1d95' : '#2a2a2a',
              border: '1px solid #6b21a8',
              borderRadius: 6,
              color: name.trim() ? '#e9d5ff' : '#666',
              fontSize: 13,
              cursor: name.trim() ? 'pointer' : 'not-allowed',
            }}
          >
            💾 保存派生类
          </button>
        </div>
      </div>
    </div>
  )
}

export default memo(function PackageViewerDialog({
  isOpen,
  mode,
  packageData,
  onClose,
  onUnpackage,
}: PackageViewerDialogProps) {
  // Edit mode state
  const [editMode, setEditMode] = useState(mode === 'edit')
  const [editedParams, setEditedParams] = useState<Record<string, Record<string, any>>>({})
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [showDeriveDialog, setShowDeriveDialog] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  // Sync edit mode when prop changes
  useEffect(() => {
    setEditMode(mode === 'edit')
  }, [mode])

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (isOpen && packageData) {
      // Initialize editedParams from current internalStructure
      const initial: Record<string, Record<string, any>> = {}
      packageData.internalStructure.forEach(sub => {
        initial[sub.id] = { ...sub.params }
      })
      setEditedParams(initial)
      setSelectedNodeId(null)
      setHasChanges(false)
    }
  }, [isOpen, packageData])

  // Build internal nodes/edges from packageData
  const { internalNodes, allEdges } = useMemo(() => {
    if (!packageData) {
      return { internalNodes: [], allEdges: [] }
    }

    const { internalStructure, internalEdges: storedEdges, position, inputs, outputs } = packageData

    // Compute topological layout
    const nodeObjects = internalStructure.map((sub) => ({
      id: sub.id,
      position: { x: 0, y: 0 },
    })) as FlowHamsterNode[]

    const edgeObjects = storedEdges.map((e) => ({
      source: e.from,
      target: e.to,
    })) as FlowHamsterEdge[]

    const layout = computeTopologicalLayout(nodeObjects, edgeObjects)

    const NODE_WIDTH = 200
    const NODE_HEIGHT = 80
    const LAYER_GAP = 200
    const NODE_GAP = 40
    const PORT_HEIGHT = 50
    const PORT_GAP = 10

    const baseX = (position as { x: number })?.x || 0
    const baseY = (position as { y: number })?.y || 0

    const inputCount = (inputs || []).length || 1
    const outputCount = (outputs || []).length || 1
    const portAreaHeight = Math.max(inputCount, outputCount) * (PORT_HEIGHT + PORT_GAP)
    const internalNodeCount = internalStructure.length
    const internalAreaHeight = internalNodeCount * (NODE_HEIGHT + NODE_GAP)
    const totalHeight = Math.max(portAreaHeight, internalAreaHeight)
    const centerOffsetY = (totalHeight - internalAreaHeight) / 2

    const INPUT_X = 30
    const INTERNAL_X = INPUT_X + 120
    const OUTPUT_X = INTERNAL_X + 200 + (LAYER_GAP * 2)

    const inputNodes: Node[] = (inputs || []).map((input, idx) => ({
      id: `__input_${input.id}`,
      type: 'inputPlaceholder',
      position: {
        x: baseX + INPUT_X,
        y: baseY + centerOffsetY + idx * (PORT_HEIGHT + PORT_GAP),
      },
      data: { label: input.label || `In ${idx + 1}` },
      draggable: false,
    }))

    const outputNodes: Node[] = (outputs || []).map((output, idx) => ({
      id: `__output_${output.id}`,
      type: 'outputPlaceholder',
      position: {
        x: baseX + OUTPUT_X,
        y: baseY + centerOffsetY + idx * (PORT_HEIGHT + PORT_GAP),
      },
      data: { label: output.label || `Out ${idx + 1}` },
      draggable: false,
    }))

    const nodes: Node[] = internalStructure.map((sub) => {
      const layoutInfo = layout.get(sub.id) || { layer: 0, order: 0 }
      const nodeType = getNodeComponentType(sub.type)
      const orderOffsetY = (totalHeight - internalAreaHeight) / 2
      const params = editedParams[sub.id] || sub.params

      return {
        id: sub.id,
        type: nodeType,
        position: {
          x: baseX + INTERNAL_X + layoutInfo.layer * LAYER_GAP,
          y: baseY + orderOffsetY + layoutInfo.order * (NODE_HEIGHT + NODE_GAP),
        },
        data: {
          nodeType: sub.type,
          label: sub.label,
          params,
        },
        style: { width: NODE_WIDTH, height: NODE_HEIGHT },
        selected: selectedNodeId === sub.id,
      }
    })

    const internalEdges: Edge[] = storedEdges.map((edge, idx) => ({
      id: `dialog_edge_${idx}`,
      source: edge.from,
      target: edge.to,
      sourceHandle: edge.fromHandle,
      targetHandle: edge.toHandle,
      type: 'smoothstep',
      animated: true,
    }))

    const inputEdges: Edge[] = (inputs || []).map((input, idx) => ({
      id: `dialog_input_edge_${idx}`,
      source: `__input_${input.id}`,
      sourceHandle: 'output',
      target: input.nodeId,
      targetHandle: input.handleId,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#22d3ee', strokeWidth: 2 },
    }))

    const outputEdges: Edge[] = (outputs || []).map((output, idx) => ({
      id: `dialog_output_edge_${idx}`,
      source: output.nodeId,
      sourceHandle: output.handleId,
      target: `__output_${output.id}`,
      targetHandle: 'input',
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#4ade80', strokeWidth: 2 },
    }))

    return {
      internalNodes: [...inputNodes, ...nodes, ...outputNodes],
      allEdges: [...inputEdges, ...internalEdges, ...outputEdges],
    }
  }, [packageData, editedParams, selectedNodeId])

  const handleParamChange = useCallback((nodeId: string, param: string, value: any) => {
    setEditedParams(prev => ({
      ...prev,
      [nodeId]: { ...prev[nodeId], [param]: value },
    }))
    setHasChanges(true)
  }, [])

  const handleSaveChanges = useCallback(() => {
    if (!packageData) return

    const registryId = packageData.customClassRegistryId
    if (!registryId) return

    // Build updated internalStructure with new params
    const updatedStructure = packageData.internalStructure.map(sub => ({
      ...sub,
      params: editedParams[sub.id] || sub.params,
    }))

    updateCustomClass(registryId, { internalStructure: updatedStructure })
    setHasChanges(false)
    setEditMode(false)
  }, [packageData, editedParams])

  const handleCancelEdit = useCallback(() => {
    // Reset to original params
    if (packageData) {
      const initial: Record<string, Record<string, any>> = {}
      packageData.internalStructure.forEach(sub => {
        initial[sub.id] = { ...sub.params }
      })
      setEditedParams(initial)
    }
    setSelectedNodeId(null)
    setHasChanges(false)
    if (mode === 'edit') {
      onClose()
    } else {
      setEditMode(false)
    }
  }, [packageData, mode, onClose])

  const handleDeriveClass = useCallback((name: string, emoji: string, category: 'cv' | 'nlp' | 'gan' | 'other', description: string) => {
    if (!packageData) return

    // Build updated internalStructure with current edited params
    const updatedStructure = packageData.internalStructure.map(sub => ({
      ...sub,
      params: editedParams[sub.id] || sub.params,
    }))

    // Get source class info
    const registryId = packageData.customClassRegistryId
    const sourceClass = registryId ? getCustomClass(registryId) : undefined

    // Determine baseType from internal structure
    const baseType = updatedStructure.length > 0 ? (updatedStructure[0].type || 'mlp') : 'mlp'

    registerCustomClass(
      name,
      baseType,
      category,
      emoji,
      description,
      updatedStructure,
      packageData.internalEdges,
      packageData.outputVar || 'output',
      {
        parentClassId: sourceClass?.id,
        originClassId: sourceClass?.originClassId ?? sourceClass?.id,
      }
    )

    setShowDeriveDialog(false)
    setEditMode(false)
    onClose()
  }, [packageData, editedParams, onClose])

  if (!isOpen || !packageData) {
    return null
  }

  return (
    <div style={overlayStyle} onClick={onClose}>
      {showDeriveDialog && (
        <DeriveClassDialog
          onSave={handleDeriveClass}
          onCancel={() => setShowDeriveDialog(false)}
        />
      )}
      <div style={panelStyle} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={headerStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 6,
                background: 'linear-gradient(135deg, rgba(88, 28, 135, 0.5), rgba(30, 41, 59, 0.7))',
                border: '1px solid rgba(168, 85, 247, 0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 16,
              }}
            >
              📦
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#e0e0e0' }}>
                {packageData.label || 'Package'} - {editMode ? '编辑模式' : '内部结构'}
              </div>
              <div style={{ fontSize: 11, color: '#666' }}>
                子模块 {packageData.childNodeIds?.length || 0} | 输入 {packageData.inputs?.length || 0} | 输出 {packageData.outputs?.length || 0}
                {hasChanges && <span style={{ color: '#f97316' }}> (有未保存的更改)</span>}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {!editMode ? (
              <>
                {mode === 'edit' && (
                  <button
                    onClick={() => setEditMode(true)}
                    style={{
                      padding: '8px 16px',
                      background: '#7c2d12',
                      border: '1px solid #9a3412',
                      borderRadius: 6,
                      color: '#fed7aa',
                      fontSize: 12,
                      cursor: 'pointer',
                    }}
                  >
                    ✏️ 编辑类
                  </button>
                )}
                <button
                  onClick={onUnpackage}
                  style={{
                    padding: '8px 16px',
                    background: '#2a1a2e',
                    border: '1px solid #5c3a6e',
                    borderRadius: 6,
                    color: '#d8b4fe',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  📤 解包
                </button>
                <button
                  onClick={onClose}
                  style={{
                    padding: '8px 16px',
                    background: '#333',
                    border: '1px solid #444',
                    borderRadius: 6,
                    color: '#ccc',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  ✕ 关闭
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => setShowDeriveDialog(true)}
                  style={{
                    padding: '8px 16px',
                    background: '#1a2a4a',
                    border: '1px solid #2563eb',
                    borderRadius: 6,
                    color: '#93c5fd',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  📋 另存为派生类
                </button>
                <button
                  onClick={handleSaveChanges}
                  disabled={!hasChanges}
                  style={{
                    padding: '8px 16px',
                    background: hasChanges ? '#14532d' : '#1a1a1a',
                    border: '1px solid #166534',
                    borderRadius: 6,
                    color: hasChanges ? '#bbf7d0' : '#666',
                    fontSize: 12,
                    cursor: hasChanges ? 'pointer' : 'not-allowed',
                  }}
                >
                  💾 保存修改
                </button>
                <button
                  onClick={handleCancelEdit}
                  style={{
                    padding: '8px 16px',
                    background: '#333',
                    border: '1px solid #444',
                    borderRadius: 6,
                    color: '#ccc',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  ❌ 取消
                </button>
              </>
            )}
          </div>
        </div>

        {/* Content */}
        <div style={contentStyle}>
          <ReactFlowProvider>
            {editMode ? (
              <InternalFlowEdit
                internalNodes={internalNodes}
                internalEdges={allEdges}
                editedParams={editedParams}
                onNodeSelect={setSelectedNodeId}
                selectedNodeId={selectedNodeId}
                onParamChange={handleParamChange}
              />
            ) : (
              <InternalFlowView
                internalNodes={internalNodes}
                internalEdges={allEdges}
              />
            )}
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  )
})