import { memo, useCallback, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'
import {
  BASE_NODE_STYLE,
  HANDLE_TARGET_STYLE,
  HANDLE_SOURCE_STYLE,
  PARAM_INPUT_STYLE,
  PARAM_ROW_STYLE,
  PARAM_LABEL_STYLE,
  DELETE_BUTTON_STYLE,
} from './nodeStyles'

const headerStyle: React.CSSProperties = {
  fontWeight: 600,
  fontSize: '12px',
  color: '#a0c0ff',
  marginBottom: '6px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
}

interface BaseNodeProps extends NodeProps<NodeData> {}

export const BaseNode = memo(({ id, data }: BaseNodeProps) => {
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const [hovered, setHovered] = useState(false)
  const removeNode = useGraphStore((s) => s.removeNode)

  const handleParamChange = useCallback(
    (key: string, value: string) => {
      const num = Number(value)
      updateNodeData(id, {
        params: { ...data.params, [key]: isNaN(num) ? value : num },
      })
    },
    [id, data.params, updateNodeData]
  )

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
      {/* Target handle (left) */}
      <Handle type="target" position={Position.Left} style={HANDLE_TARGET_STYLE} />

      <div style={headerStyle}>{data.label}</div>

      <div style={{ marginBottom: '4px' }}>
        {Object.entries(data.params).map(([key, val]) => (
          <div key={key} style={PARAM_ROW_STYLE}>
            <span style={PARAM_LABEL_STYLE}>{key}</span>
            <input
              style={PARAM_INPUT_STYLE}
              value={String(val)}
              onChange={(e) => handleParamChange(key, e.target.value)}
              type={typeof val === 'number' ? 'number' : 'text'}
            />
          </div>
        ))}
      </div>

      {/* Source handle (right) */}
      <Handle type="source" position={Position.Right} style={HANDLE_SOURCE_STYLE} />
    </div>
  )
})

BaseNode.displayName = 'BaseNode'
