import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const nodeStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #334',
  borderRadius: '8px',
  padding: '10px 14px',
  minWidth: '168px',
  color: '#e0e0e0',
  fontSize: '13px',
  position: 'relative',
}

const rs = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px', gap: '8px' }
const ls = { color: '#888', fontSize: '11px', flex: '0 0 54px' }
const is_ = { background: '#111', border: '1px solid #333', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '12px', width: '88px' }
const sel_ = { background: '#111', border: '1px solid #333', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '11px', width: '88px', cursor: 'pointer' }

const InputNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const removeNode = useGraphStore((s) => s.removeNode)
  const [hovered, setHovered] = useState(false)

  const setField = (key: string, value: string) => {
    updateNodeData(id, { params: { ...data.params, [key]: value } })
  }

  return (
    <div
      style={{ ...nodeStyle, borderColor: hovered ? '#6688cc' : '#334' }}
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
      <div style={{ fontWeight: 600, fontSize: '12px', color: '#a0c0ff', marginBottom: '4px', textTransform: 'uppercase' }}>
        {data.label}
      </div>
      <div style={rs}>
        <span style={ls}>Name</span>
        <input
          style={is_}
          value={String(data.params.name ?? '')}
          onChange={(e) => setField('name', e.target.value)}
          placeholder="image"
        />
      </div>
      <div style={rs}>
        <span style={ls}>Shape</span>
        <input
          style={is_}
          value={String(data.params.shape ?? '')}
          onChange={(e) => setField('shape', e.target.value)}
          placeholder="3,224,224"
        />
      </div>
      <div style={rs}>
        <span style={ls}>DType</span>
        <select
          style={sel_}
          value={String(data.params.dtype ?? 'float32')}
          onChange={(e) => setField('dtype', e.target.value)}
        >
          <option value="float32">float32</option>
          <option value="float16">float16</option>
          <option value="int64">int64</option>
        </select>
      </div>
      <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
        Contract: {String(data.params.name || 'input')} → {String(data.params.shape || '?')}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }}
      />
    </div>
  )
})

InputNode.displayName = 'InputNode'
export default InputNode
