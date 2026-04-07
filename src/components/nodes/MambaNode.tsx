import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '160px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }

const MambaNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  const d_model = data.params?.d_model ?? 512
  const d_state = data.params?.d_state ?? 16
  const d_conv = data.params?.d_conv ?? 4
  const expand = data.params?.expand ?? 2
  const n_layers = data.params?.n_layers ?? 1

  return (
    <div style={{ ...ns, borderColor: h ? '#66ccff' : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }} title="Delete node">×</button>}
      <Handle type="target" position={Position.Left} id="x" style={{ background: '#4488ff', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#66ccff', marginBottom: '4px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
        {data.label}
        <span style={{ fontSize: 10, color: '#888', fontWeight: 400, background: '#2a3a4a', padding: '1px 5px', borderRadius: 3 }}>SSM</span>
      </div>
      <div style={{ fontSize: '10px', color: '#666', marginBottom: 6, borderBottom: '1px solid #333', paddingBottom: 4 }}>
        Mamba-2 Selective SSM
      </div>
      <div style={{ fontSize: '10px', color: '#888', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#555' }}>D:</span>
          <span style={{ color: '#aaa' }}>{d_model}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#555' }}>N:</span>
          <span style={{ color: '#aaa' }}>{d_state}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#555' }}>conv:</span>
          <span style={{ color: '#aaa' }}>{d_conv}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#555' }}>expand:</span>
          <span style={{ color: '#aaa' }}>{expand}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#555' }}>layers:</span>
          <span style={{ color: '#aaa' }}>{n_layers}</span>
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="output" style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
MambaNode.displayName = 'MambaNode'
export default MambaNode
