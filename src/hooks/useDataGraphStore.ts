import { create } from 'zustand'
import { addEdge, applyEdgeChanges, applyNodeChanges, Connection, EdgeChange, NodeChange } from 'reactflow'
import { DataFlowEdge, DataFlowNode, DataNodeData } from '../types/dataGraph'

function makeDataNodeId(prefix: string) {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
}

interface DataGraphState {
  nodes: DataFlowNode[]
  edges: DataFlowEdge[]
  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  addNode: (data: DataNodeData, position: { x: number; y: number }) => void
  removeNode: (id: string) => void
  updateNodeData: (id: string, data: Partial<DataNodeData>) => void
  setNodes: (nodes: DataFlowNode[]) => void
  setEdges: (edges: DataFlowEdge[]) => void
  loadGraph: (nodes: DataFlowNode[], edges: DataFlowEdge[]) => void
  // Cross-graph highlighting
  highlightedNodeIds: string[]
  setHighlightedNodes: (nodeIds: string[]) => void
}

export const useDataGraphStore = create<DataGraphState>((set) => ({
  nodes: [],
  edges: [],
  highlightedNodeIds: [],
  onNodesChange: (changes) => set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) as DataFlowNode[] })),
  onEdgesChange: (changes) => set((state) => ({ edges: applyEdgeChanges(changes, state.edges) as DataFlowEdge[] })),
  onConnect: (connection) => set((state) => ({ edges: addEdge({ ...connection, type: 'smoothstep', animated: true }, state.edges) as DataFlowEdge[] })),
  addNode: (data, position) => {
    const id = makeDataNodeId(data.nodeType)
    const node: DataFlowNode = {
      id,
      type: 'dataPipelineNode',
      position,
      data,
    }
    set((state) => ({ nodes: [...state.nodes, node] }))
  },
  removeNode: (id) => set((state) => ({
    nodes: state.nodes.filter((node) => node.id !== id),
    edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
  })),
  updateNodeData: (id, data) => set((state) => ({
    nodes: state.nodes.map((node) => (node.id === id ? { ...node, data: { ...node.data, ...data } } : node)) as DataFlowNode[],
  })),
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  loadGraph: (nodes, edges) => set({ nodes, edges }),
  setHighlightedNodes: (highlightedNodeIds) => set({ highlightedNodeIds }),
}))
