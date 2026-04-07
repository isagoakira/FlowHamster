import { useGraphStore } from '../../hooks/useGraphStore'
import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '180px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }
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
    <div style={{ ...ns, borderColor: h ? '#a0c0ff' : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }} title="Delete node">x</button>}
      <Handle type="target" position={Position.Left} style={{ background: '#4488ff', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#a0c0ff', marginBottom: '6px', textTransform: 'uppercase' }}>LeakyReLU</div>
      <div key="negative_slope" style={rs}><span style={ls}>negative_slope</span><input style={is_} value={String(p.negative_slope ?? "")} onChange={e => set("negative_slope", e.target.value)} type="number" /></div>      <Handle type="source" position={Position.Right} style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
LeakyReLUNode.displayName = 'LeakyReLUNode'
export default LeakyReLUNode
