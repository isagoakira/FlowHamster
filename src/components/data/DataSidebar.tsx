import { useCallback, useState } from 'react'
import { DATA_NODE_CATEGORIES, DataNodeCategory, DataNodeDefinition } from '../../utils/dataNodeRegistry'

const sidebarStyle: React.CSSProperties = {
  width: '220px',
  background: '#111',
  borderRight: '1px solid #222',
  overflowY: 'auto',
  flexShrink: 0,
  display: 'flex',
  flexDirection: 'column',
}

const headerStyle: React.CSSProperties = {
  padding: '12px',
  borderBottom: '1px solid #222',
}

const contentStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '12px 8px',
}

const categoryStyle: React.CSSProperties = {
  fontSize: '10px',
  fontWeight: 700,
  color: '#555',
  textTransform: 'uppercase',
  letterSpacing: '1px',
  marginTop: '14px',
  marginBottom: '6px',
  padding: '0 4px',
}

const nodeItemStyle: React.CSSProperties = {
  padding: '6px 8px',
  borderRadius: '6px',
  cursor: 'grab',
  fontSize: '12px',
  color: '#ccc',
  marginBottom: '2px',
  transition: 'background 0.15s',
  border: '1px solid transparent',
}

export default function DataSidebar() {
  const [dragging, setDragging] = useState<string | null>(null)

  const onDragStart = useCallback((e: React.DragEvent, payload: { nodeType: string; label: string; defaultParams: Record<string, string | number | boolean>; fieldOrder?: string[] }) => {
    e.dataTransfer.setData('application/flowhamster-data', JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'move'
    setDragging(payload.nodeType)
  }, [])

  return (
    <div style={sidebarStyle}>
      <div style={headerStyle}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: '#a0c0ff' }}>
          Data Pipeline
        </div>
        <div style={{ fontSize: '10px', color: '#666', marginTop: '6px', lineHeight: 1.5 }}>
          数据图独立构建，不与模型图硬连线；只通过字段契约与 binding 关联。
        </div>
      </div>

      <div style={contentStyle}>
        {Object.values(DATA_NODE_CATEGORIES).map((category: DataNodeCategory) => (
          <div key={category.label}>
            <div style={categoryStyle}>{category.label}</div>
            {category.nodes.map((node: DataNodeDefinition) => (
              <div
                key={node.type}
                style={{
                  ...nodeItemStyle,
                  background: dragging === node.type ? '#1a2a3a' : 'transparent',
                  borderColor: dragging === node.type ? '#4488ff44' : 'transparent',
                }}
                draggable
                onDragStart={(e) => onDragStart(e, {
                  nodeType: node.type,
                  label: node.label,
                  defaultParams: node.defaultParams,
                  fieldOrder: node.fieldOrder,
                })}
                onDragEnd={() => setDragging(null)}
                title={node.description}
              >
                {node.label}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

