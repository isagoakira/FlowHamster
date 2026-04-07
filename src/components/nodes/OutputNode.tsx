import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const nodeStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #433',
  borderRadius: '8px',
  padding: '10px 14px',
  minWidth: '140px',
  color: '#e0e0e0',
  fontSize: '13px',
  position: 'relative',
}

const rs = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px', gap: '8px' }
const ls = { color: '#888', fontSize: '11px', flex: '0 0 72px' }
const is_ = { background: '#111', border: '1px solid #333', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '12px', width: '72px' }
const sel_ = { background: '#111', border: '1px solid #333', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '11px', width: '72px', cursor: 'pointer' }

const OutputNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const [hovered, setHovered] = useState(false)
  const removeNode = (nodeId: string) => { useGraphStore.getState().removeNode(nodeId) }

  const outputName = String(data.params?.output_name ?? '')
  const outputType = String(data.params?.output_type ?? 'main')

  const setOutputName = (v: string) => {
    updateNodeData(id, { params: { ...data.params, output_name: v } })
  }

  const setOutputType = (v: string) => {
    updateNodeData(id, { params: { ...data.params, output_type: v } })
  }

  // Color coding: main = orange, auxiliary = purple
  const borderColor = outputType === 'auxiliary' ? '#9966ff' : '#ff8844'

  return (
    <div
      style={{ ...nodeStyle, borderColor: hovered ? '#ffaa66' : borderColor }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <button
          onClick={(e) => { e.stopPropagation(); removeNode(id) }}
          style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }}
          title="Delete node"
        >
          ×
        </button>
      )}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: borderColor, width: 8, height: 8, border: 'none' }}
      />
      <div style={{ fontWeight: 600, fontSize: '12px', color: borderColor, marginBottom: '6px', textTransform: 'uppercase' }}>
        {data.label}
      </div>

      {/* Output Name */}
      <div style={rs}>
        <span style={ls}>Name</span>
        <input
          style={is_}
          value={outputName}
          onChange={(e) => setOutputName(e.target.value)}
          placeholder="e.g. logits"
        />
      </div>

      {/* Output Type */}
      <div style={rs}>
        <span style={ls}>Type</span>
        <select
          style={sel_}
          value={outputType}
          onChange={(e) => setOutputType(e.target.value)}
        >
          <option value="main">Main</option>
          <option value="auxiliary">Auxiliary</option>
        </select>
      </div>

      {/* Show resolved name when set */}
      {outputName && (
        <div style={{ fontSize: '10px', color: '#666', marginTop: '4px', textAlign: 'right' }}>
          → {outputName}
        </div>
      )}
    </div>
  )
})

OutputNode.displayName = 'OutputNode'
export default OutputNode
