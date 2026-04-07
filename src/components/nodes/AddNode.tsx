import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const ns = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '140px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }
const hlabel = { fontSize: '9px', color: '#aaa', textAlign: 'center' as const, position: 'absolute' as const, left: 0, right: 0 }

const AddNode = memo((props: NodeProps<NodeData>) => {
  const { id } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  return (
    <div
      style={{ ...ns, borderColor: h ? '#ffcc44' : '#333' }}
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

      {/* Input A — top */}
      <div style={{ position: 'relative', marginBottom: 16 }}>
        <div style={{ ...hlabel, top: -14 }}>Input A</div>
        <Handle type="target" position={Position.Top} id="a" style={{ background: '#ff4444', width: 8, height: 8, border: 'none', left: '30%' }} />
      </div>

      {/* Input B — left */}
      <div style={{ position: 'absolute', left: 4, top: '50%', fontSize: '9px', color: '#4488ff' }}>B</div>
      <Handle type="target" position={Position.Left} id="b" style={{ background: '#4488ff', width: 8, height: 8, border: 'none', top: '50%' }} />

      <div style={{ fontWeight: 600, fontSize: '14px', color: '#ffcc44', textAlign: 'center', padding: '4px 0' }}>Add</div>
      <div style={{ fontSize: '10px', color: '#666', textAlign: 'center' }}>A + B</div>

      {/* Output — right */}
      <Handle type="source" position={Position.Right} style={{ background: '#44cc88', width: 8, height: 8, border: 'none', top: '50%' }} />
    </div>
  )
})
AddNode.displayName = 'AddNode'
export default AddNode
