import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'
import { BASE_NODE_STYLE, HANDLE_SOURCE_STYLE, DELETE_BUTTON_STYLE } from './nodeStyles'

const ConstantNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)
  const value = data.params.value || '0'
  const shape = data.params.shape || '1'
  const trainable = data.params.trainable ? '✓' : '✗'
  return (
    <div style={{ ...BASE_NODE_STYLE, borderColor: h ? '#ffcc00' : '#333', boxShadow: h ? '0 4px 16px rgba(0,0,0,0.5)' : BASE_NODE_STYLE.boxShadow }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      {h && <button onClick={(e) => { e.stopPropagation(); removeNode(id) }} style={DELETE_BUTTON_STYLE} title="Delete node">×</button>}
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#ffcc00', marginBottom: '4px', textTransform: 'uppercase' }}>{data.label}</div>
      <div style={{ fontSize: '11px', color: '#888' }}>value: {value}</div>
      <div style={{ fontSize: '11px', color: '#888' }}>shape: [{shape}]</div>
      <div style={{ fontSize: '11px', color: '#888' }}>trainable: {trainable}</div>
      <Handle type="source" position={Position.Right} style={HANDLE_SOURCE_STYLE} />
    </div>
  )
})
ConstantNode.displayName = 'ConstantNode'
export default ConstantNode