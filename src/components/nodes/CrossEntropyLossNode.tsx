import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const n = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '150px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }

const CrossEntropyLossNode = memo((props: NodeProps<NodeData>) => {
  const { id } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)
  return (
    <div style={{ ...n, borderColor: h ? '#ff8888' : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }} title="Delete">x</button>}
      <Handle type="target" position={Position.Left} style={{ background: '#ff4444', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#ff8888', marginBottom: '4px', textTransform: 'uppercase' }}>CrossEntropyLoss</div>
      <Handle type="source" position={Position.Right} style={{ background: '#ff8844', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
CrossEntropyLossNode.displayName = 'CrossEntropyLossNode'
export default CrossEntropyLossNode
