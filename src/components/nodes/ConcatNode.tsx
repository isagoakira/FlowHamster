import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '140px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }

const ConcatNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  return (
    <div style={{ ...ns, borderColor: h ? '#a0c0ff' : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }} title="Delete node">×</button>}

      {/* Multiple target handles for concat inputs */}
      <Handle type="target" position={Position.Left} id="in_0" style={{ background: '#4488ff', width: 8, height: 8, border: 'none', top: '20%' }} />
      <Handle type="target" position={Position.Left} id="in_1" style={{ background: '#4488ff', width: 8, height: 8, border: 'none', top: '50%' }} />
      <Handle type="target" position={Position.Left} id="in_2" style={{ background: '#4488ff', width: 8, height: 8, border: 'none', top: '80%' }} />

      <div style={{ fontWeight: 600, fontSize: '12px', color: '#a0c0ff', marginBottom: '4px', textTransform: 'uppercase' }}>{data.label}</div>
      <div style={{ fontSize: '11px', color: '#666' }}>Concat</div>

      <Handle type="source" position={Position.Right} style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
ConcatNode.displayName = 'ConcatNode'
export default ConcatNode
