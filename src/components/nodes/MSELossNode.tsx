import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'
import { BASE_NODE_STYLE, DELETE_BUTTON_STYLE } from './nodeStyles'

const MSELossNode = memo((props: NodeProps<NodeData>) => {
  const { id } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)
  return (
    <div style={{ ...BASE_NODE_STYLE, borderColor: h ? '#ff8888' : '#333', boxShadow: h ? '0 4px 16px rgba(0,0,0,0.5)' : BASE_NODE_STYLE.boxShadow }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={DELETE_BUTTON_STYLE} title="Delete">x</button>}
      <Handle type="target" position={Position.Left} style={{ background: '#ff4444', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#ff8888', marginBottom: '4px', textTransform: 'uppercase' }}>MSELoss</div>
      <Handle type="source" position={Position.Right} style={{ background: '#ff8844', width: 8, height: 8, border: 'none' }} />
    </div>
  )
})
MSELossNode.displayName = 'MSELossNode'
export default MSELossNode