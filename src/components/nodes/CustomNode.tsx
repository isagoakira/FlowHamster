import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'
import { BASE_NODE_STYLE, NODE_COLORS, HANDLE_TARGET_STYLE, HANDLE_SOURCE_STYLE, DELETE_BUTTON_STYLE } from './nodeStyles'

const CustomNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)
  return (
    <div style={{ ...BASE_NODE_STYLE, borderColor: h ? '#ff66ff' : '#333', boxShadow: h ? '0 4px 16px rgba(0,0,0,0.5)' : BASE_NODE_STYLE.boxShadow }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={DELETE_BUTTON_STYLE} title="Delete node">×</button>}
      <Handle type="target" position={Position.Left} style={{ background: '#ff66ff', width: 8, height: 8, border: 'none' }} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#ff66ff', marginBottom: '4px', textTransform: 'uppercase' }}>{data.label}</div>
      <div style={{ fontSize: '11px', color: '#666' }}>Custom</div>
      <Handle type="source" position={Position.Right} style={HANDLE_SOURCE_STYLE} />
    </div>
  )
})
CustomNode.displayName = 'CustomNode'
export default CustomNode