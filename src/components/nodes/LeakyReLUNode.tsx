import { useGraphStore } from '../../hooks/useGraphStore'
import { BASE_NODE_STYLE, NODE_COLORS, HANDLE_TARGET_STYLE, HANDLE_SOURCE_STYLE, DELETE_BUTTON_STYLE } from './nodeStyles'
import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'

const rs = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px', gap: '8px' }
const ls = { color: '#888', fontSize: '11px', flex: '0 0 80px' }
const is_ = { background: '#111', border: '1px solid #333', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '12px', width: '72px', textAlign: 'right' as const }
const LeakyReLUNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)
  const set = (k: string, v: string) => {
    const n = Number(v)
    updateNodeData(id, { params: { ...data.params, [k]: isNaN(n) ? v : n } })
  }
  const p = data.params
  return (
    <div style={{ ...BASE_NODE_STYLE, borderColor: h ? '#a0c0ff' : '#333', boxShadow: h ? '0 4px 16px rgba(0,0,0,0.5)' : BASE_NODE_STYLE.boxShadow }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={DELETE_BUTTON_STYLE} title="Delete node">x</button>}
      <Handle type="target" position={Position.Left} style={HANDLE_TARGET_STYLE} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: NODE_COLORS.layer, marginBottom: '6px', textTransform: 'uppercase' }}>LeakyReLU</div>
      <div key="negative_slope" style={rs}><span style={ls}>negative_slope</span><input className="nodrag nowheel" style={is_} value={String(p.negative_slope ?? "")} onPointerDown={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()} onWheel={(e) => e.stopPropagation()} onChange={e => set("negative_slope", e.target.value)} type="number" /></div>      <Handle type="source" position={Position.Right} style={HANDLE_SOURCE_STYLE} />
    </div>
  )
})
LeakyReLUNode.displayName = 'LeakyReLUNode'
export default LeakyReLUNode
