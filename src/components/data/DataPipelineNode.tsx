import { memo, useState, useCallback, type SyntheticEvent } from 'react'
import { Handle, NodeProps, Position } from 'reactflow'
import { DataNodeData, FieldSpec, FieldDtype } from '../../types/dataGraph'
import { useDataGraphStore } from '../../hooks/useDataGraphStore'

// Category-based styling
const CATEGORY_STYLES: Record<string, { border: string; handleColor: string; labelColor: string }> = {
  sources: { border: '#3d6b3d', handleColor: '#44cc88', labelColor: '#7dd87d' },
  readers: { border: '#4d5d8d', handleColor: '#5d8dff', labelColor: '#8db5ff' },
  field_ops: { border: '#6d5d8d', handleColor: '#9d7dff', labelColor: '#c4a7ff' },
  transforms: { border: '#8d6d3d', handleColor: '#ffaa44', labelColor: '#ffc87d' },
  augmentation: { border: '#8d3d6d', handleColor: '#ff44aa', labelColor: '#ff7dc4' },
  compose: { border: '#5d6d8d', handleColor: '#88aacc', labelColor: '#aaccff' },
  organization: { border: '#5d8d6d', handleColor: '#44ddaa', labelColor: '#7dffc4' },
  batch: { border: '#8d5d4d', handleColor: '#dd6644', labelColor: '#ffaa88' },
  nlp: { border: '#5d4d8d', handleColor: '#9966ff', labelColor: '#c499ff' },
  output: { border: '#3d5d8d', handleColor: '#4488ff', labelColor: '#88bbff' },
}

const ns = { background: '#161a20', borderRadius: '8px', padding: '10px 14px', minWidth: '210px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }
const rs = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px', gap: '8px' }
const ls = { color: '#8892a0', fontSize: '11px', flex: '0 0 72px' }
const is_ = { background: '#0f1318', border: '1px solid #2f3b4d', borderRadius: '4px', color: '#e0e0e0', padding: '2px 6px', fontSize: '12px', width: '110px' }

const dtypeColors: Record<FieldDtype, string> = {
  tensor: '#88aaff',
  scalar: '#88ff88',
  string: '#ffaa88',
  image: '#aa88ff',
  label: '#ff88aa',
  mask: '#88ffff',
  text: '#ffff88',
  audio: '#ff8888',
  spectrogram: '#88ff88',
}

// Infer dtype from field name
function inferDtypeFromName(name: string): FieldDtype {
  const lowered = name.toLowerCase()
  if (lowered.includes('image') || lowered.includes('img') || lowered.includes('input')) return 'image'
  if (lowered.includes('label') || lowered.includes('target') || lowered.includes('class')) return 'label'
  if (lowered.includes('mask')) return 'mask'
  if (lowered.includes('text') || lowered.includes('token')) return 'text'
  if (lowered.includes('audio') || lowered.includes('waveform')) return 'audio'
  if (lowered.includes('spectrogram') || lowered.includes('mel') || lowered.includes('mfcc')) return 'spectrogram'
  return 'tensor'
}

// Parse fieldSpecs JSON
function parseFieldSpecs(fieldSpecsStr: string): FieldSpec[] {
  if (!fieldSpecsStr) return []
  try {
    const parsed = JSON.parse(fieldSpecsStr)
    if (Array.isArray(parsed)) return parsed
    return []
  } catch {
    return []
  }
}

// Serialize field specs to JSON
function serializeFieldSpecs(specs: FieldSpec[]): string {
  return JSON.stringify(specs)
}

function stopCanvasInteraction(event: SyntheticEvent) {
  event.stopPropagation()
}

const DataPipelineNode = memo((props: NodeProps<DataNodeData>) => {
  const { id, data } = props
  const updateNodeData = useDataGraphStore((s) => s.updateNodeData)
  const removeNode = useDataGraphStore((s) => s.removeNode)
  const [hovered, setHovered] = useState(false)
  const [showFieldSpecs, setShowFieldSpecs] = useState(false)

  // Get category style based on node type
  const getCategoryStyle = (nodeType: string) => {
    for (const [category, style] of Object.entries(CATEGORY_STYLES)) {
      if (nodeType.includes(category.replace('_', '')) || category.includes(nodeType.split('_')[0])) {
        return style
      }
    }
    return { border: '#2f3b4d', handleColor: '#5d8dff', labelColor: '#a9c7ff' }
  }

  const categoryStyle = getCategoryStyle(data.nodeType)

  const setField = (key: string, value: string) => {
    const maybeNumber = Number(value)
    updateNodeData(id, {
      params: {
        ...data.params,
        [key]: value === 'true' ? true : value === 'false' ? false : (Number.isNaN(maybeNumber) || value.trim() === '' ? value : maybeNumber),
      },
    })
  }

  // Get fields from the 'fields' param for dataset_output
  const getOutputFields = useCallback((): string[] => {
    if (data.nodeType !== 'dataset_output') return []
    const fieldsStr = String(data.params.fields || '')
    return fieldsStr.split(',').map(f => f.trim()).filter(Boolean)
  }, [data.nodeType, data.params.fields])

  // Get or infer field specs
  const getFieldSpecs = useCallback((): FieldSpec[] => {
    const specs = parseFieldSpecs(data.fieldSpecs || '')
    const fields = getOutputFields()
    // If specs exist and match fields, return them; otherwise infer from field names
    if (specs.length === fields.length && fields.every(f => specs.some(s => s.name === f))) {
      return specs
    }
    // Infer from field names
    return fields.map(name => ({
      name,
      dtype: inferDtypeFromName(name),
      shapeHint: name === 'image' ? 'N,C,H,W' : name === 'label' ? 'N' : undefined,
    }))
  }, [data.fieldSpecs, getOutputFields])

  const fieldSpecs = getFieldSpecs()

  const updateFieldSpec = (name: string, dtype: FieldDtype, shapeHint?: string) => {
    const newSpecs = fieldSpecs.map(spec =>
      spec.name === name ? { ...spec, dtype, shapeHint } : spec
    )
    updateNodeData(id, {
      params: { ...data.params },
      fieldSpecs: serializeFieldSpecs(newSpecs),
    })
  }

  const outputFields = getOutputFields()

  return (
    <div
      style={{
        ...ns,
        borderColor: hovered ? categoryStyle.handleColor : categoryStyle.border,
        borderWidth: '1px',
        borderStyle: 'solid',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <button
          onClick={(e) => { e.stopPropagation(); removeNode(id) }}
          style={{
            position: 'absolute',
            top: 4,
            right: 4,
            background: '#ff4444',
            border: 'none',
            borderRadius: '50%',
            width: 18,
            height: 18,
            cursor: 'pointer',
            color: '#fff',
            fontSize: 11,
            lineHeight: 1,
            padding: 0,
          }}
          title="Delete node"
        >
          ×
        </button>
      )}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: categoryStyle.handleColor, width: 8, height: 8, border: 'none' }}
      />
      <div style={{ fontWeight: 600, fontSize: '12px', color: categoryStyle.labelColor, marginBottom: '6px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
        {data.label}
        <span style={{ fontSize: 9, fontWeight: 400, color: '#555', background: '#222', padding: '1px 4px', borderRadius: 2, textTransform: 'none' }}>
          {data.nodeType}
        </span>
      </div>
      {(data.fieldOrder ?? Object.keys(data.params)).map((field) => (
        <div key={field} style={rs}>
          <span style={ls}>{field}</span>
          <input
            className="nodrag nowheel"
            style={is_}
            value={String(data.params[field] ?? '')}
            onPointerDown={stopCanvasInteraction}
            onMouseDown={stopCanvasInteraction}
            onWheel={stopCanvasInteraction}
            onChange={(e) => setField(field, e.target.value)}
          />
        </div>
      ))}

      {/* Field metadata badges for dataset_output */}
      {data.nodeType === 'dataset_output' && outputFields.length > 0 && (
        <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #2f3b4d' }}>
          <button
            onClick={() => setShowFieldSpecs(!showFieldSpecs)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#88bbff',
              fontSize: '10px',
              cursor: 'pointer',
              padding: '2px 0',
              textAlign: 'left' as const,
            }}
          >
            {showFieldSpecs ? '▼' : '▶'} Field Metadata ({fieldSpecs.length} fields)
          </button>

          {showFieldSpecs && (
            <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column' as const, gap: '4px' }}>
              {fieldSpecs.map((spec) => (
                <div key={spec.name} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                  <span style={{ color: '#ccc', minWidth: '50px' }}>{spec.name}</span>
                  <select
                    className="nodrag nowheel"
                    value={spec.dtype}
                    onPointerDown={stopCanvasInteraction}
                    onMouseDown={stopCanvasInteraction}
                    onWheel={stopCanvasInteraction}
                    onChange={(e) => updateFieldSpec(spec.name, e.target.value as FieldDtype, spec.shapeHint)}
                    style={{
                      background: '#0f1318',
                      border: '1px solid #2f3b4d',
                      borderRadius: '3px',
                      color: dtypeColors[spec.dtype] || '#ccc',
                      fontSize: '10px',
                      padding: '1px 4px',
                      cursor: 'pointer',
                    }}
                  >
                    <option value="tensor">tensor</option>
                    <option value="scalar">scalar</option>
                    <option value="string">string</option>
                    <option value="image">image</option>
                    <option value="label">label</option>
                    <option value="mask">mask</option>
                    <option value="text">text</option>
                    <option value="audio">audio</option>
                    <option value="spectrogram">spectrogram</option>
                  </select>
                  <input
                    className="nodrag nowheel"
                    value={spec.shapeHint || ''}
                    placeholder="N,C,H,W"
                    onPointerDown={stopCanvasInteraction}
                    onMouseDown={stopCanvasInteraction}
                    onWheel={stopCanvasInteraction}
                    onChange={(e) => updateFieldSpec(spec.name, spec.dtype, e.target.value || undefined)}
                    style={{
                      background: '#0f1318',
                      border: '1px solid #2f3b4d',
                      borderRadius: '3px',
                      color: '#aaa',
                      fontSize: '10px',
                      padding: '1px 4px',
                      width: '60px',
                    }}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Compact badges view */}
          {!showFieldSpecs && (
            <div style={{ display: 'flex', flexWrap: 'wrap' as const, gap: '4px', marginTop: '4px' }}>
              {fieldSpecs.map((spec) => (
                <span
                  key={spec.name}
                  style={{
                    background: '#0f1318',
                    border: `1px solid ${dtypeColors[spec.dtype] || '#444'}`,
                    borderRadius: '3px',
                    color: dtypeColors[spec.dtype] || '#ccc',
                    fontSize: '10px',
                    padding: '1px 5px',
                  }}
                  title={spec.shapeHint ? `Shape: ${spec.shapeHint}` : spec.dtype}
                >
                  {spec.name}:{spec.dtype}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: categoryStyle.handleColor, width: 8, height: 8, border: 'none' }}
      />
    </div>
  )
})

DataPipelineNode.displayName = 'DataPipelineNode'
export default DataPipelineNode
