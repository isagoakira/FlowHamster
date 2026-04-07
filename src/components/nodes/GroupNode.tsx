import { memo } from 'react'
import { NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'

const GroupNode = memo(({ data, selected }: NodeProps<NodeData>) => {
  const childCount = Array.isArray(data.childNodeIds) ? data.childNodeIds.length : 0

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        borderRadius: '18px',
        background: selected ? 'rgba(41, 62, 104, 0.28)' : 'rgba(18, 29, 52, 0.16)',
        border: selected ? '1px solid #88aaff' : '1px solid rgba(96, 144, 255, 0.3)',
        boxSizing: 'border-box',
        position: 'relative',
      }}
    >
      <div
        className="group-drag-handle"
        style={{
          position: 'absolute',
          top: 12,
          left: 14,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          borderRadius: 10,
          background: 'rgba(10, 15, 28, 0.78)',
          border: '1px solid rgba(102, 136, 221, 0.35)',
          color: '#c7d7ff',
          fontSize: 12,
          fontWeight: 700,
          letterSpacing: '0.3px',
        }}
      >
        <span>🧩 {data.label || 'Group'}</span>
        <span style={{ fontSize: 10, fontWeight: 500, color: '#88aaff' }}>
          {childCount} modules
        </span>
      </div>
    </div>
  )
})

GroupNode.displayName = 'GroupNode'

export default GroupNode
