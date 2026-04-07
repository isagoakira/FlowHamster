import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '160px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }

const MLPNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  return (
    <div style={{ ...ns, borderColor: h ? '#a0c0ff' : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }} title="Delete node">×</button>}
      <Handle type="target" position={Position.Left} id="x" style={{ background: '#4488ff', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#a0c0ff', marginBottom: '4px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
        {data.label}
        <span style={{ fontSize: 10, color: '#888', fontWeight: 400, background: '#333', padding: '1px 5px', borderRadius: 3 }} title="Composite node - click to view internal structure">🔍</span>
      </div>
      <div style={{ fontSize: '11px', color: '#666', marginBottom: 3 }}>MLP</div>
      <div style={{ fontSize: '9px', color: '#555', display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        {Object.entries(data.params || {}).map(([k, v]) => (
          <span key={k} style={{ background: '#2a2a2a', padding: '1px 4px', borderRadius: 2 }}>{k}={String(v).slice(0, 8)}</span>
        ))}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
MLPNode.displayName = 'MLPNode'
export default MLPNode
