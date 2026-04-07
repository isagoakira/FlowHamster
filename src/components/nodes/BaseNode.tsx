import { memo, useCallback, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

// Common styles shared by all nodes
const nodeStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '8px',
  padding: '10px 14px',
  minWidth: '160px',
  color: '#e0e0e0',
  fontSize: '13px',
}

const headerStyle: React.CSSProperties = {
  fontWeight: 600,
  fontSize: '12px',
  color: '#a0c0ff',
  marginBottom: '6px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
}

const paramRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: '4px',
  gap: '8px',
}

const labelStyle: React.CSSProperties = {
  color: '#888',
  fontSize: '11px',
  flex: '0 0 60px',
}

const inputStyle: React.CSSProperties = {
  background: '#111',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#e0e0e0',
  padding: '2px 6px',
  fontSize: '12px',
  width: '70px',
  textAlign: 'right',
}

interface BaseNodeProps extends NodeProps<NodeData> {}

export const BaseNode = memo(({ id, data }: BaseNodeProps) => {
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const [hovered, setHovered] = useState(false)

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
      style={{ ...nodeStyle, borderColor: hovered ? '#4488ff' : '#333' }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Target handle (left) */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#4488ff', width: 8, height: 8, border: 'none' }}
      />

      <div style={headerStyle}>{data.label}</div>

      <div style={{ marginBottom: '4px' }}>
        {Object.entries(data.params).map(([key, val]) => (
          <div key={key} style={paramRowStyle}>
            <span style={labelStyle}>{key}</span>
            <input
              style={inputStyle}
              value={String(val)}
              onChange={(e) => handleParamChange(key, e.target.value)}
              type={typeof val === 'number' ? 'number' : 'text'}
            />
          </div>
        ))}
      </div>

      {/* Source handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }}
      />
    </div>
  )
})

BaseNode.displayName = 'BaseNode'
