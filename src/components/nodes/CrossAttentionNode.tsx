import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '160px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }
const hlabel = { fontSize: '9px', color: '#aaa', textAlign: 'center' as const, marginTop: 2 }

const CrossAttentionNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  return (
    <div
      style={{ ...ns, borderColor: h ? '#8844ff' : '#333' }}
      onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
    >
      {h && (
        <button
          onClick={(e) => { e.stopPropagation(); removeNode(id) }}
          style={{
            position: 'absolute', top: 4, right: 4,
            background: '#ff4444', border: 'none', borderRadius: '50%',
            width: 18, height: 18, cursor: 'pointer', color: '#fff',
            fontSize: 11, lineHeight: 1, padding: 0,
          }}
          title="Delete node"
        >×</button>
      )}

      {/* Query handle (top) */}
      <div style={{ position: 'relative', marginBottom: 6 }}>
        <Handle type="target" position={Position.Top} id="q" style={{ background: '#8844ff', width: 8, height: 8, border: 'none', left: '40%' }} />
        <div style={hlabel}>Q (query)</div>
      </div>

      {/* Key-Value handle (left) */}
      <Handle type="target" position={Position.Left} id="kv" style={{ background: '#ff8844', width: 8, height: 8, border: 'none', top: '65%' }} />
      <div style={{ position: 'absolute', left: 4, top: '62%', fontSize: '9px', color: '#ff8844' }}>K/V</div>

      {/* Main body */}
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#cc88ff', marginBottom: '4px', textTransform: 'uppercase', paddingLeft: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
        {data.label}
        <span style={{ fontSize: 9, color: '#888', fontWeight: 400, background: '#333', padding: '1px 5px', borderRadius: 3 }} title="Composite node - click to view internal structure">🔍</span>
      </div>
      <div style={{ fontSize: '11px', color: '#666', paddingLeft: 12 }}>CrossAttention</div>
      <div style={{ fontSize: '9px', color: '#555', paddingLeft: 12, display: 'flex', flexWrap: 'wrap', gap: 2, marginTop: 4 }}>
        {Object.entries(data.params || {}).map(([k, v]) => (
          <span key={k} style={{ background: '#2a2a2a', padding: '1px 4px', borderRadius: 2 }}>{k}={String(v).slice(0, 6)}</span>
        ))}
      </div>

      {/* Output handle (right) */}
      <Handle type="source" position={Position.Right} style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
CrossAttentionNode.displayName = 'CrossAttentionNode'
export default CrossAttentionNode
