/**
 * CompositeNodeViewer - Modal dialog for viewing and editing composite node internals
 * Allows users to view submodules, edit parameters, and save as a new custom class
 */

import { memo, useState, useCallback, useMemo } from 'react'
import { SubModule, InternalEdge } from '../../utils/nodeRegistry'
import { registerCustomClass } from '../../utils/customCompositeRegistry'
import { useGraphStore } from '../../hooks/useGraphStore'
import CompositeCanvas from './CompositeCanvas'

// Modal overlay style
const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  marginLeft: 220,
  background: 'rgba(0, 0, 0, 0.7)',
  zIndex: 1000,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

// Modal panel style
const panelStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: 12,
  width: 900,
  maxWidth: '95vw',
  maxHeight: '90vh',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 16px 64px rgba(0, 0, 0, 0.6)',
  overflow: 'hidden',
}

// Header style
const headerStyle: React.CSSProperties = {
  padding: '16px 20px',
  borderBottom: '1px solid #333',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  background: '#222',
}

// Content style
const contentStyle: React.CSSProperties = {
  flex: 1,
  overflow: 'auto',
  padding: 16,
  display: 'flex',
  flexDirection: 'column',
  gap: 16,
}

// Footer style
const footerStyle: React.CSSProperties = {
  padding: '12px 20px',
  borderTop: '1px solid #333',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  background: '#1a1a1a',
}

// Input style
const inputStyle: React.CSSProperties = {
  background: '#111',
  border: '1px solid #333',
  borderRadius: 4,
  color: '#e0e0e0',
  padding: '6px 10px',
  fontSize: 12,
  width: '100%',
}

// Button styles
const buttonBase: React.CSSProperties = {
  padding: '8px 16px',
  borderRadius: 6,
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
  border: 'none',
  transition: 'all 0.2s',
}

const primaryButtonStyle: React.CSSProperties = {
  ...buttonBase,
  background: '#4488ff',
  color: '#fff',
}

const secondaryButtonStyle: React.CSSProperties = {
  ...buttonBase,
  background: '#333',
  color: '#ccc',
  border: '1px solid #444',
}

// Parameter type definitions for submodule types
interface ParamDef {
  key: string
  label: string
  type: 'number' | 'boolean' | 'text'
  default?: number | boolean | string
}

const SUBMODULE_PARAM_DEFS: Record<string, ParamDef[]> = {
  linear: [
    { key: 'in_features', label: 'in_features', type: 'number', default: 512 },
    { key: 'out_features', label: 'out_features', type: 'number', default: 512 },
  ],
  conv1d: [
    { key: 'in_channels', label: 'in_channels', type: 'number', default: 64 },
    { key: 'out_channels', label: 'out_channels', type: 'number', default: 64 },
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 3 },
  ],
  conv2d: [
    { key: 'in_channels', label: 'in_channels', type: 'number', default: 3 },
    { key: 'out_channels', label: 'out_channels', type: 'number', default: 64 },
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 3 },
  ],
  dropout: [
    { key: 'p', label: 'p', type: 'number', default: 0.5 },
  ],
  layernorm: [
    { key: 'normalized_shape', label: 'normalized_shape', type: 'number', default: 512 },
  ],
  multiheadattention: [
    { key: 'embed_dim', label: 'embed_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
  ],
  softmax: [
    { key: 'dim', label: 'dim', type: 'number', default: -1 },
  ],
  add: [],  // Residual addition - no parameters
  relu: [],
  gelu: [],
  silu: [],
}

// Simple arithmetic expression evaluator
function evaluateArithmetic(expr: string): number {
  const tokens: (number | string)[] = []
  let currentNum = ''

  for (let i = 0; i < expr.length; i++) {
    const char = expr[i]
    if ('0123456789.'.includes(char)) {
      currentNum += char
    } else if (' \t'.includes(char)) {
      if (currentNum) {
        tokens.push(parseFloat(currentNum))
        currentNum = ''
      }
    } else if ('+-*/()'.includes(char)) {
      if (currentNum) {
        tokens.push(parseFloat(currentNum))
        currentNum = ''
      }
      tokens.push(char)
    }
  }
  if (currentNum) {
    tokens.push(parseFloat(currentNum))
  }

  function parseAddSub(tokens: (number | string)[]): number {
    let result = parseMulDiv(tokens)
    while (tokens.length > 0 && (tokens[0] === '+' || tokens[0] === '-')) {
      const op = tokens.shift() as string
      const right = parseMulDiv(tokens)
      result = op === '+' ? result + right : result - right
    }
    return result
  }

  function parseMulDiv(tokens: (number | string)[]): number {
    let result = parsePrimary(tokens)
    while (tokens.length > 0 && (tokens[0] === '*' || tokens[0] === '/')) {
      const op = tokens.shift() as string
      const right = parsePrimary(tokens)
      result = op === '*' ? result * right : result / right
    }
    return result
  }

  function parsePrimary(tokens: (number | string)[]): number {
    const token = tokens.shift()
    if (token === '(') {
      const result = parseAddSub(tokens)
      tokens.shift()
      return result
    }
    return token as number
  }

  return parseAddSub(tokens)
}

// Resolve template strings like ${param_name} against parentParams
function resolveValue(value: number | string | boolean, parentParams: Record<string, number | string | boolean>): number | string | boolean {
  if (typeof value !== 'string') return value

  const templateMatch = value.match(/^\$\{(\w+)\}$/)
  if (templateMatch) {
    const key = templateMatch[1]
    return parentParams[key] ?? value
  }

  const exprMatch = value.match(/^\$\{([^}]+)\}$/)
  if (exprMatch) {
    try {
      let expr = exprMatch[1]
      for (const [k, v] of Object.entries(parentParams)) {
        const re = new RegExp(`\\$\\{${k}\\}`, 'g')
        expr = expr.replace(re, String(v))
      }
      if (/^[\d\s+\-*/().]+$/.test(expr)) {
        const result = evaluateArithmetic(expr.trim())
        return typeof result === 'number' ? result : value
      }
    } catch {
      // If evaluation fails, return original
    }
  }

  return value
}

// Resolve all params in a submodule against parent params
function resolveSubModuleParams(
  subModule: SubModule,
  parentParams: Record<string, number | string | boolean>
): SubModule {
  const resolvedParams: Record<string, number | string | boolean> = {}
  for (const [key, value] of Object.entries(subModule.params)) {
    resolvedParams[key] = resolveValue(value, parentParams)
  }
  return { ...subModule, params: resolvedParams }
}

// Generate stack visualization for multi-layer nodes (each layer has multiple submodules)
function generateStackVisualization(
  baseLayerSubModules: SubModule[],
  numLayers: number
): SubModule[] {
  const modules: SubModule[] = []

  for (let layerIdx = 1; layerIdx <= numLayers; layerIdx++) {
    // Add each submodule from the base layer, prefixed with layer index
    for (const sm of baseLayerSubModules) {
      modules.push({
        id: `L${layerIdx}_${sm.id}`,
        type: sm.type,
        label: `L${layerIdx}_${sm.label}`,
        params: { ...sm.params },
      })
    }
  }

  return modules
}

// Generate edges for stack visualization - connects layers and preserves internal edges
function generateStackEdges(
  baseLayerSubModules: SubModule[],
  baseLayerEdges: InternalEdge[],
  numLayers: number
): InternalEdge[] {
  const edges: InternalEdge[] = []

  for (let layerIdx = 1; layerIdx <= numLayers; layerIdx++) {
    // Add internal edges for this layer
    for (const edge of baseLayerEdges) {
      edges.push({
        from: `L${layerIdx}_${edge.from}`,
        to: `L${layerIdx}_${edge.to}`,
      })
    }

    // Connect layer N to layer N+1
    if (layerIdx < numLayers) {
      // Find the "first" node in the layer (input node - has no incoming edges)
      const firstNode = baseLayerSubModules.find(sm => {
        return !baseLayerEdges.some(e => e.to === sm.id)
      })

      // Find the "last" node in the layer (output node - is not a source of any edge)
      const sourceIds = new Set(baseLayerEdges.map(e => e.from))
      const lastNode = baseLayerSubModules.find(sm => {
        return !sourceIds.has(sm.id)
      })

      if (firstNode && lastNode) {
        // Connect output of current layer to input of next layer
        edges.push({
          from: `L${layerIdx}_${lastNode.id}`,
          to: `L${layerIdx + 1}_${firstNode.id}`,
        })
      }
    }
  }

  return edges
}

interface CompositeNodeViewerProps {
  isOpen: boolean
  nodeId: string  // Use nodeId instead of passing all node data
  onClose: () => void
  nodeType: string
  nodeLabel: string
  subModules: SubModule[]
  internalEdges: InternalEdge[]
  outputVar: string
}

export default memo(function CompositeNodeViewer({
  isOpen,
  nodeId,
  onClose,
  nodeType,
  nodeLabel,
  subModules,
  internalEdges,
  outputVar,
}: CompositeNodeViewerProps) {
  const [selectedSubModuleId, setSelectedSubModuleId] = useState<string | null>(null)
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [customClassName, setCustomClassName] = useState('')
  const [customClassDescription, setCustomClassDescription] = useState('')
  const [customClassEmoji, setCustomClassEmoji] = useState('🔧')
  const [customClassCategory, setCustomClassCategory] = useState<'cv' | 'nlp' | 'gan' | 'other'>('other')
  const [editingMode, setEditingMode] = useState<'view' | 'edit'>('view')
  const [subModulesExpanded, setSubModulesExpanded] = useState(false)

  // Get all nodes from store - this will re-render when nodes array changes
  const allNodes = useGraphStore((s) => s.nodes)
  const updateNodeData = useGraphStore((s) => s.updateNodeData)

  // Find the specific node by id
  const nodeFromStore = useMemo(() => {
    return allNodes.find(n => n.id === nodeId)
  }, [allNodes, nodeId])

  // Get current params from store (real-time)
  const currentParams = useMemo(() => {
    return nodeFromStore?.data?.params || {}
  }, [nodeFromStore])

  // Get num_layers for stack visualization
  const numLayers = useMemo(() => {
    return Math.max(1, Number(currentParams['num_layers'] || 1))
  }, [currentParams['num_layers']])

  // Determine if this is a stack-based node
  const isStackNode = nodeType === 'transformerencoder' || nodeType === 'transformerdecoder'

  // Resolve submodule params - regenerate when params change
  const resolvedSubModules = useMemo(() => {
    if (isStackNode && numLayers > 1) {
      // Generate full stack visualization by repeating the entire layer structure
      // First resolve params for all submodules in one layer
      const resolvedBaseLayer = subModules.map(sm => resolveSubModuleParams(sm, currentParams))
      return generateStackVisualization(resolvedBaseLayer, numLayers)
    }
    return subModules.map(sm => resolveSubModuleParams(sm, currentParams))
  }, [subModules, currentParams, numLayers, isStackNode])

  // Generate edges for stack visualization
  const resolvedEdges = useMemo(() => {
    if (isStackNode && numLayers > 1) {
      // Generate edges for full stack - first resolve base layer edges, then duplicate
      const resolvedBaseLayer = subModules.map(sm => resolveSubModuleParams(sm, currentParams))
      return generateStackEdges(resolvedBaseLayer, internalEdges, numLayers)
    }
    return internalEdges
  }, [isStackNode, numLayers, internalEdges])

  // Get selected submodule
  const selectedSubModule = selectedSubModuleId
    ? resolvedSubModules.find(s => s.id === selectedSubModuleId)
    : null

  // Handle parameter change - update store directly
  const handleParamChange = useCallback((key: string, value: number | string | boolean) => {
    // Always allow parameter changes regardless of editing mode
    const newParams = { ...currentParams, [key]: value }
    updateNodeData(nodeId, { params: newParams })
  }, [currentParams, nodeId, updateNodeData])

  // Get param definitions for a submodule type
  const getParamDefs = (type: string): ParamDef[] => {
    return SUBMODULE_PARAM_DEFS[type] || []
  }

  // Handle save as new class
  const handleSaveAsNewClass = useCallback(() => {
    if (!customClassName.trim()) {
      alert('Please enter a name for the custom class')
      return
    }

    registerCustomClass(
      customClassName.trim(),
      nodeType,
      customClassCategory,
      customClassEmoji,
      customClassDescription.trim(),
      subModules,
      internalEdges,
      outputVar
    )

    setShowSaveDialog(false)
    setCustomClassName('')
    setCustomClassDescription('')
    setCustomClassEmoji('🔧')
    setCustomClassCategory('other')
    onClose()
  }, [customClassName, customClassDescription, customClassEmoji, customClassCategory, nodeType, subModules, internalEdges, outputVar, onClose])

  if (!isOpen) return null

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={panelStyle} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={headerStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 6,
                background: '#333',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 16,
              }}
            >
              🔍
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#e0e0e0' }}>
                {nodeLabel} - Internal Structure
              </div>
              <div style={{ fontSize: 11, color: '#666' }}>
                {numLayers > 1 ? `Stack of ${numLayers} layers` : 'Single layer'} | Click submodule to view params
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => setEditingMode(editingMode === 'view' ? 'edit' : 'view')}
              style={{
                ...secondaryButtonStyle,
                background: editingMode === 'edit' ? '#2a4a7a' : '#333',
              }}
            >
              {editingMode === 'view' ? '✏️ Edit' : '👁️ View'}
            </button>
            <button onClick={onClose} style={{ ...secondaryButtonStyle, padding: '6px 12px' }}>
              ✕ Close
            </button>
          </div>
        </div>

        {/* Content */}
        <div style={contentStyle}>
          {/* Parent parameters section */}
          <div
            style={{
              padding: 16,
              background: '#151515',
              borderRadius: 8,
              border: '1px solid #333',
            }}
          >
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: '#a0c0ff',
                marginBottom: 12,
                textTransform: 'uppercase',
              }}
            >
              {nodeType.toUpperCase()} Parameters
              {numLayers > 1 && (
                <span style={{ fontSize: 10, color: '#666', fontWeight: 400, marginLeft: 8 }}>
                  ({numLayers} layers)
                </span>
              )}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
              {Object.entries(currentParams).map(([key, value]) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <label style={{ fontSize: 12, color: '#888', minWidth: 120 }}>{key}:</label>
                  {typeof value === 'boolean' ? (
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={e => handleParamChange(key, e.target.checked)}
                      style={{ accentColor: '#4488ff' }}
                    />
                  ) : (
                    <input
                      type="number"
                      value={value}
                      onChange={e =>
                        handleParamChange(key, e.target.value === '' ? 0 : Number(e.target.value))
                      }
                      style={{
                        ...inputStyle,
                        width: 100,
                      }}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Mini canvas */}
          <CompositeCanvas
            subModules={resolvedSubModules}
            internalEdges={resolvedEdges}
            selectedSubModuleId={selectedSubModuleId}
            onSelectSubModule={setSelectedSubModuleId}
          />

          {/* Submodule parameter editor */}
          {selectedSubModule && (
            <div
              style={{
                padding: 16,
                background: '#111',
                borderRadius: 8,
                border: '1px solid #4488ff',
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: '#a0c0ff',
                  marginBottom: 12,
                  textTransform: 'uppercase',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: getSubModuleColor(selectedSubModule.type),
                  }}
                />
                {selectedSubModule.label}
                <span style={{ fontWeight: 400, color: '#666', textTransform: 'none' }}>
                  ({selectedSubModule.type})
                </span>
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                {getParamDefs(selectedSubModule.type).map(paramDef => {
                  const currentValue = selectedSubModule.params[paramDef.key] ?? paramDef.default
                  return (
                    <div key={paramDef.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <label style={{ fontSize: 11, color: '#888', minWidth: 100 }}>
                        {paramDef.label}:
                      </label>
                      <input
                        type="number"
                        value={currentValue as number}
                        style={{
                          ...inputStyle,
                          width: 100,
                        }}
                      />
                    </div>
                  )
                })}

                {getParamDefs(selectedSubModule.type).length === 0 && (
                  <div style={{ fontSize: 11, color: '#555', fontStyle: 'italic' }}>
                    No configurable parameters
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Submodule list - collapsible */}
          <div
            style={{
              background: '#151515',
              borderRadius: 8,
              border: '1px solid #333',
              overflow: 'hidden',
            }}
          >
            {/* Header - always visible */}
            <div
              onClick={() => setSubModulesExpanded(!subModulesExpanded)}
              style={{
                padding: '10px 12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: 12,
                fontWeight: 600,
                color: '#888',
                textTransform: 'uppercase',
                userSelect: 'none',
              }}
            >
              <span>All Sub-Modules ({resolvedSubModules.length})</span>
              <span style={{ fontSize: 10, color: '#666' }}>
                {subModulesExpanded ? '▲ 收起' : '▼ 展开'}
              </span>
            </div>

            {/* Content - only visible when expanded */}
            {subModulesExpanded && (
              <div style={{ padding: '0 12px 12px', display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {resolvedSubModules.map(sm => (
                  <div
                    key={sm.id}
                    onClick={() => setSelectedSubModuleId(sm.id === selectedSubModuleId ? null : sm.id)}
                    style={{
                      padding: '6px 12px',
                      background: selectedSubModuleId === sm.id ? '#2a3a5a' : '#222',
                      border: `1px solid ${selectedSubModuleId === sm.id ? '#4488ff' : '#333'}`,
                      borderRadius: 6,
                      cursor: 'pointer',
                      fontSize: 11,
                      color: '#ccc',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-block',
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: getSubModuleColor(sm.type),
                      }}
                    />
                    {sm.label}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Save dialog */}
          {showSaveDialog && (
            <div
              style={{
                padding: 16,
                background: '#111',
                borderRadius: 8,
                border: '1px solid #4488ff',
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 600, color: '#e0e0e0', marginBottom: 16 }}>
                Save as New Custom Class
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
                    Class Name *
                  </label>
                  <input
                    type="text"
                    value={customClassName}
                    onChange={e => setCustomClassName(e.target.value)}
                    placeholder="MyCustomMLP"
                    style={inputStyle}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
                    Category
                  </label>
                  <select
                    value={customClassCategory}
                    onChange={e => setCustomClassCategory(e.target.value as 'cv' | 'nlp' | 'gan' | 'other')}
                    style={{ ...inputStyle, width: 120 }}
                  >
                    <option value="cv">Computer Vision</option>
                    <option value="nlp">NLP</option>
                    <option value="gan">GAN</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
                    Emoji
                  </label>
                  <input
                    type="text"
                    value={customClassEmoji}
                    onChange={e => setCustomClassEmoji(e.target.value)}
                    style={{ ...inputStyle, width: 60, textAlign: 'center' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
                    Description
                  </label>
                  <input
                    type="text"
                    value={customClassDescription}
                    onChange={e => setCustomClassDescription(e.target.value)}
                    placeholder="My custom module"
                    style={inputStyle}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
                <button onClick={() => setShowSaveDialog(false)} style={secondaryButtonStyle}>
                  Cancel
                </button>
                <button onClick={handleSaveAsNewClass} style={primaryButtonStyle}>
                  💾 Save Class
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={footerStyle}>
          <div style={{ fontSize: 11, color: '#666' }}>
            {resolvedSubModules.length} sub-modules | {resolvedEdges.length} connections
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={onClose} style={secondaryButtonStyle}>
              Cancel
            </button>
            {!showSaveDialog && (
              <button onClick={() => setShowSaveDialog(true)} style={primaryButtonStyle}>
                💾 Save as New Class
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})

// Get color for submodule type
function getSubModuleColor(type: string): string {
  const colors: Record<string, string> = {
    linear: '#88cc44',
    conv1d: '#44aaff',
    conv2d: '#4488ff',
    conv3d: '#aa44ff',
    relu: '#ff6644',
    gelu: '#ff8844',
    silu: '#ffaa44',
    sigmoid: '#ffcc44',
    tanh: '#44ddff',
    leakyrelu: '#ff6644',
    dropout: '#ff8844',
    dropout1d: '#ff8844',
    droppath: '#ffaa66',
    layernorm: '#44ffaa',
    batchnorm2d: '#44ffaa',
    groupnorm: '#44ffcc',
    maxpool2d: '#aa44ff',
    avgpool2d: '#aa66ff',
    adaptiveavgpool2d: '#aa88ff',
    globalavgpool: '#aa99ff',
    softmax: '#ff44ff',
    flatten: '#888888',
    reshape: '#888888',
    multiheadattention: '#44ff88',
    embedding: '#cc44ff',
    add: '#ffdd44',       // Yellow for residual add
    q_proj: '#44ddff',    // Cyan for Q projection
    k_proj: '#44ffdd',    // Teal for K projection
    v_proj: '#dd44ff',    // Magenta for V projection
    out_proj: '#88ddaa',  // Light green for output projection
    attn: '#ffff44',       // Yellow for attention
  }
  return colors[type] || '#666666'
}
