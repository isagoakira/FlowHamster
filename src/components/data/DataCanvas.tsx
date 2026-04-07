import { DragEvent, useCallback } from 'react'
import ReactFlow, { Background, Controls, MiniMap, NodeTypes, ReactFlowProvider, useReactFlow } from 'reactflow'
import 'reactflow/dist/style.css'
import { useDataGraphStore } from '../../hooks/useDataGraphStore'
import { DataNodeData } from '../../types/dataGraph'
import DataPipelineNode from './DataPipelineNode'

const nodeTypes: NodeTypes = {
  dataPipelineNode: DataPipelineNode,
}

function DataCanvasInner() {
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)
  const onNodesChange = useDataGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useDataGraphStore((s) => s.onEdgesChange)
  const onConnect = useDataGraphStore((s) => s.onConnect)
  const addNode = useDataGraphStore((s) => s.addNode)
  const { screenToFlowPosition } = useReactFlow()

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    const payload = event.dataTransfer.getData('application/flowhamster-data')
    if (!payload) return

    const parsed = JSON.parse(payload) as { nodeType: string; label: string; defaultParams: Record<string, string | number | boolean>; fieldOrder?: string[] }
    const position = screenToFlowPosition({ x: event.clientX, y: event.clientY })
    addNode({
      nodeType: parsed.nodeType as DataNodeData['nodeType'],
      label: parsed.label,
      params: { ...parsed.defaultParams },
      fieldOrder: parsed.fieldOrder,
    }, position)
  }, [addNode, screenToFlowPosition])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 500, background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '8px 12px', color: '#7e8ca1', fontSize: '11px' }}>
        Data Graph — 独立于模型图，仅声明数据流与输出契约
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  )
}

export default function DataCanvas() {
  return (
    <ReactFlowProvider>
      <DataCanvasInner />
    </ReactFlowProvider>
  )
}
