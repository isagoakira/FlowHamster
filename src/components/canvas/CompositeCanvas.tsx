/**
 * CompositeCanvas - Mini React Flow canvas for viewing composite node internals
 * Displays sub-modules and their connections within a composite node
 */

import { memo, useCallback, useMemo, useEffect, useRef } from 'react'
import ReactFlow, {
  Background,
  Node,
  Edge,
  Handle,
  Position,
  NodeProps,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  useReactFlow,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { SubModule, InternalEdge } from '../../utils/nodeRegistry'

// Submodule node style
const subModuleNodeStyle = {
  background: '#1a1a1a',
  border: '1px solid #444',
  borderRadius: 8,
  padding: '8px 12px',
  minWidth: 100,
  color: '#e0e0e0',
  fontSize: 11,
  position: 'relative' as const,
}

// Get color for submodule type
function getSubModuleColor(type: string): string {
  const colors: Record<string, string> = {
    linear: '#88cc44',
    conv1d: '#44aaff',
    conv2d: '#4488ff',
    conv3d: '#aa44ff',
    relu: '#ff6644',
    gelu: '#ff8844',
    silu: '#ffaa44',
    sigmoid: '#ffcc44',
    tanh: '#44ddff',
    leakyrelu: '#ff6644',
    dropout: '#ff8844',
    dropout1d: '#ff8844',
    droppath: '#ffaa66',
    layernorm: '#44ffaa',
    batchnorm2d: '#44ffaa',
    groupnorm: '#44ffcc',
    maxpool2d: '#aa44ff',
    avgpool2d: '#aa66ff',
    adaptiveavgpool2d: '#aa88ff',
    globalavgpool: '#aa99ff',
    softmax: '#ff44ff',
    flatten: '#888888',
    reshape: '#888888',
    multiheadattention: '#44ff88',
    embedding: '#cc44ff',
    add: '#ffdd44',       // Yellow for residual add
    q_proj: '#44ddff',    // Cyan for Q projection
    k_proj: '#44ffdd',    // Teal for K projection
    v_proj: '#dd44ff',    // Magenta for V projection
    out_proj: '#88ddaa',   // Light green for output projection
    attn: '#ffff44',       // Yellow for attention
  }
  return colors[type] || '#666666'
}

// Simple submodule node component
const SubModuleNode = memo(({ data }: NodeProps<{ subModule: SubModule; isSelected: boolean }>) => {
  const color = getSubModuleColor(data.subModule.type)

  return (
    <div
      style={{
        ...subModuleNodeStyle,
        borderColor: data.isSelected ? '#4488ff' : color,
        boxShadow: data.isSelected ? '0 0 8px rgba(68, 136, 255, 0.5)' : 'none',
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: color, width: 6, height: 6, border: 'none' }}
      />
      <div style={{ fontWeight: 600, fontSize: 10, color, marginBottom: 2, textTransform: 'uppercase' }}>
        {data.subModule.label}
      </div>
      <div style={{ fontSize: 9, color: '#666' }}>
        {data.subModule.type}
      </div>
      <div style={{ fontSize: 8, color: '#555', marginTop: 2 }}>
        {Object.entries(data.subModule.params)
          .slice(0, 2)
          .map(([k, v]) => `${k}=${v}`)
          .join(', ')}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: color, width: 6, height: 6, border: 'none' }}
      />
    </div>
  )
})
SubModuleNode.displayName = 'SubModuleNode'

interface CompositeCanvasProps {
  subModules: SubModule[]
  internalEdges: InternalEdge[]
  selectedSubModuleId: string | null
  onSelectSubModule: (id: string | null) => void
}

// Layout nodes in a horizontal flow
function layoutNodes(subModules: SubModule[], internalEdges: InternalEdge[]): { nodes: Node[]; edges: Edge[] } {
  if (subModules.length === 0) {
    return { nodes: [], edges: [] }
  }

  // Build adjacency list for topological sort
  const adj: Record<string, string[]> = {}
  const inDegree: Record<string, number> = {}

  subModules.forEach(s => {
    adj[s.id] = []
    inDegree[s.id] = 0
  })

  internalEdges.forEach(e => {
    if (adj[e.from]) {
      adj[e.from].push(e.to)
      inDegree[e.to] = (inDegree[e.to] || 0) + 1
    }
  })

  // Topological sort to determine layers
  const layers: string[][] = []
  const visited = new Set<string>()

  while (visited.size < subModules.length) {
    const layer: string[] = []
    subModules.forEach(s => {
      if (!visited.has(s.id) && (inDegree[s.id] || 0) === 0) {
        layer.push(s.id)
      }
    })

    if (layer.length === 0) {
      // Handle cycles or disconnected nodes by adding remaining nodes
      subModules.forEach(s => {
        if (!visited.has(s.id)) layer.push(s.id)
      })
    }

    layer.forEach(id => visited.add(id))
    layers.push(layer)

    // Update in-degrees
    layer.forEach(id => {
      adj[id]?.forEach(next => {
        if (inDegree[next] !== undefined) {
          inDegree[next]--
        }
      })
    })
  }

  // Position nodes in a vertical stack layout (each layer as a row)
  const nodeSpacingX = 200
  const nodeSpacingY = 100
  const startX = 100
  const startY = 50

  const nodes: Node[] = subModules.map((s) => {
    let layerIndex = 0
    let positionInLayer = 0

    for (let l = 0; l < layers.length; l++) {
      const idx = layers[l].indexOf(s.id)
      if (idx !== -1) {
        layerIndex = l
        positionInLayer = idx
        break
      }
    }

    return {
      id: s.id,
      type: 'subModuleNode',
      position: {
        x: startX + layerIndex * nodeSpacingX,
        y: startY + positionInLayer * nodeSpacingY,
      },
      data: { subModule: s, isSelected: false },
    }
  })

  // Create edges
  const edges: Edge[] = internalEdges.map((e) => ({
    id: `e-${e.from}-${e.to}`,
    source: e.from,
    target: e.to,
    type: 'smoothstep',
    animated: false,
    style: { stroke: '#555', strokeWidth: 2 },
  }))

  return { nodes, edges }
}

// Inner canvas component that uses useReactFlow
const CompositeCanvasInner = memo(function CompositeCanvasInner({
  subModules,
  internalEdges,
  selectedSubModuleId,
  onSelectSubModule,
}: CompositeCanvasProps) {
  const { fitView } = useReactFlow()
  const fitViewCalled = useRef(false)

  // Compute layouted nodes and edges
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => layoutNodes(subModules, internalEdges),
    [subModules, internalEdges]
  )

  // Use useNodesState to manage nodes with React Flow
  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes)
  const [edges, , onEdgesChange] = useEdgesState(layoutedEdges)

  // Sync nodes when layoutedNodes change (important!)
  useEffect(() => {
    setNodes(layoutedNodes)
  }, [layoutedNodes, setNodes])

  // Update selection state
  useEffect(() => {
    setNodes(nds =>
      nds.map(n => ({
        ...n,
        data: { ...n.data, isSelected: n.id === selectedSubModuleId },
      }))
    )
  }, [selectedSubModuleId, setNodes])

  // Fit view on mount and when nodes change
  useEffect(() => {
    if (layoutedNodes.length > 0 && !fitViewCalled.current) {
      fitViewCalled.current = true
      setTimeout(() => fitView({ padding: 0.2 }), 100)
    }
  }, [fitView, layoutedNodes.length])

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      onSelectSubModule(node.id === selectedSubModuleId ? null : node.id)
    },
    [selectedSubModuleId, onSelectSubModule]
  )

  const nodeTypes = useMemo(
    () => ({
      subModuleNode: SubModuleNode,
    }),
    []
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      panOnDrag={true}
      zoomOnScroll={true}
      minZoom={0.3}
      maxZoom={2}
      style={{ background: '#111' }}
    >
      <Background color="#222" gap={20} />
    </ReactFlow>
  )
})

// Wrapper with ReactFlowProvider
export default memo(function CompositeCanvas(props: CompositeCanvasProps) {
  return (
    <div style={{ width: '100%', height: 300, border: '1px solid #333', borderRadius: 8, overflow: 'hidden' }}>
      <ReactFlowProvider>
        <CompositeCanvasInner {...props} />
      </ReactFlowProvider>
    </div>
  )
})
