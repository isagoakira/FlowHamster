import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'
import {
  BASE_NODE_STYLE,
  NODE_COLORS,
  HANDLE_TARGET_STYLE,
  HANDLE_SOURCE_STYLE,
  DELETE_BUTTON_STYLE,
} from './nodeStyles'

const Conv2dNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [hovered, setHovered] = useState(false)

  return (
    <div
      style={{
        ...BASE_NODE_STYLE,
        borderColor: hovered ? '#4488ff' : '#333',
        boxShadow: hovered ? '0 4px 16px rgba(0,0,0,0.5)' : BASE_NODE_STYLE.boxShadow,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <button
          onClick={(e) => { e.stopPropagation(); removeNode(id) }}
          style={DELETE_BUTTON_STYLE}
          title="Delete node"
        >
          x
        </button>
      )}
      <Handle type="target" position={Position.Left} style={HANDLE_TARGET_STYLE} />
      <div style={{ fontWeight: 600, fontSize: '12px', color: NODE_COLORS.layer, marginBottom: '4px', textTransform: 'uppercase' }}>
        {data.label}
      </div>
      <div style={{ fontSize: '11px', color: '#666' }}>Conv2d</div>
      <Handle type="source" position={Position.Right} style={HANDLE_SOURCE_STYLE} />
    </div>
  )
})
Conv2dNode.displayName = 'Conv2dNode'
export default Conv2dNode
