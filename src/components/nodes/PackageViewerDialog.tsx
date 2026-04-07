/**
 * PackageViewerDialog - 弹窗查看打包模块的内部结构
 */

import { memo, useMemo, useEffect } from 'react'
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

// 内部 ReactFlow 组件
function InternalFlow({
  internalNodes,
  internalEdges,
}: {
  internalNodes: Node[]
  internalEdges: Edge[]
}) {
  // 使用 ReactFlow 的 hooks 来管理状态
  const [nodes, setNodes, onNodesChange] = useNodesState(internalNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(internalEdges)

  // 当 internalNodes/internalEdges 变化时更新本地状态
  useEffect(() => {
    setNodes(internalNodes)
    setEdges(internalEdges)
  }, [internalNodes, internalEdges, setNodes, setEdges])

  // 动态合并 nodeTypes
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

export default memo(function PackageViewerDialog({
  isOpen,
  packageData,
  onClose,
  onUnpackage,
}: PackageViewerDialogProps) {
  // 从 packageData 构建内部节点、边界节点和边
  const { internalNodes, allEdges } = useMemo(() => {
    if (!packageData) {
      return { internalNodes: [], allEdges: [] }
    }

    const { internalStructure, internalEdges: storedEdges, position, inputs, outputs } = packageData

    // 计算拓扑布局
    const nodeObjects = internalStructure.map((sub) => ({
      id: sub.id,
      position: { x: 0, y: 0 },
    })) as FlowHamsterNode[]

    const edgeObjects = storedEdges.map((e) => ({
      source: e.from,
      target: e.to,
    })) as FlowHamsterEdge[]

    const layout = computeTopologicalLayout(nodeObjects, edgeObjects)

    // 构建参数
    const NODE_WIDTH = 200
    const NODE_HEIGHT = 80
    const LAYER_GAP = 200
    const NODE_GAP = 40
    const PORT_HEIGHT = 50  // 输入/输出占位符的高度
    const PORT_GAP = 10     // 输入/输出之间的间距

    const baseX = (position as { x: number })?.x || 0
    const baseY = (position as { y: number })?.y || 0

    // 计算输入/输出区域的高度
    const inputCount = (inputs || []).length || 1
    const outputCount = (outputs || []).length || 1
    const portAreaHeight = Math.max(inputCount, outputCount) * (PORT_HEIGHT + PORT_GAP)

    // 内部节点区域居中
    const internalNodeCount = internalStructure.length
    const internalAreaHeight = internalNodeCount * (NODE_HEIGHT + NODE_GAP)

    // 总高度取较大值，确保所有内容都能显示
    const totalHeight = Math.max(portAreaHeight, internalAreaHeight)

    // 居中偏移
    const centerOffsetY = (totalHeight - internalAreaHeight) / 2

    // 输入X位置（最左侧）
    const INPUT_X = 30
    // 内部节点X位置（中间）
    const INTERNAL_X = INPUT_X + 120
    // 输出X位置（最右侧）
    const OUTPUT_X = INTERNAL_X + 200 + (LAYER_GAP * 2)

    // 构建输入边界节点（垂直居中排列）
    const inputNodes: Node[] = (inputs || []).map((input, idx) => ({
      id: `__input_${input.id}`,
      type: 'inputPlaceholder',
      position: {
        x: baseX + INPUT_X,
        y: baseY + centerOffsetY + idx * (PORT_HEIGHT + PORT_GAP),
      },
      data: {
        label: input.label || `In ${idx + 1}`,
      },
      draggable: false,
    }))

    // 构建输出边界节点（垂直居中排列）
    const outputNodes: Node[] = (outputs || []).map((output, idx) => ({
      id: `__output_${output.id}`,
      type: 'outputPlaceholder',
      position: {
        x: baseX + OUTPUT_X,
        y: baseY + centerOffsetY + idx * (PORT_HEIGHT + PORT_GAP),
      },
      data: {
        label: output.label || `Out ${idx + 1}`,
      },
      draggable: false,
    }))

    // 构建内部节点（基于拓扑排序的层级排列）
    const nodes: Node[] = internalStructure.map((sub) => {
      const layoutInfo = layout.get(sub.id) || { layer: 0, order: 0 }
      const nodeType = getNodeComponentType(sub.type)

      // 内部节点也居中
      const orderOffsetY = (totalHeight - internalAreaHeight) / 2

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
          params: { ...sub.params },
        },
        style: { width: NODE_WIDTH, height: NODE_HEIGHT },
      }
    })

    // 构建内部边
    const internalEdges: Edge[] = storedEdges.map((edge, idx) => ({
      id: `dialog_edge_${idx}`,
      source: edge.from,
      target: edge.to,
      sourceHandle: edge.fromHandle,
      targetHandle: edge.toHandle,
      type: 'smoothstep',
      animated: true,
    }))

    // 构建从输入到内部节点的边
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

    // 构建从内部节点到输出的边
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
  }, [packageData])

  if (!isOpen || !packageData) {
    return null
  }

  return (
    <div style={overlayStyle} onClick={onClose}>
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
                {packageData.label || 'Package'} - 内部结构
              </div>
              <div style={{ fontSize: 11, color: '#666' }}>
                子模块 {packageData.childNodeIds?.length || 0} | 输入 {packageData.inputs?.length || 0} | 输出 {packageData.outputs?.length || 0}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
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
          </div>
        </div>

        {/* Content - ReactFlow canvas */}
        <div style={contentStyle}>
          <ReactFlowProvider>
            <InternalFlow
              internalNodes={internalNodes}
              internalEdges={allEdges}
            />
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  )
})
