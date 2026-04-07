/**
 * Binding Panel - Data binding configuration
 */

import { useMemo } from 'react'
import { useGraphStore } from '../../../hooks/useGraphStore'
import { useDataGraphStore } from '../../../hooks/useDataGraphStore'
import { WorkflowBinding } from '../../../schema/workflowDocument'
import { compileDataWorkflow, validateBindingCompatibility } from '../../../utils/dataWorkflowCompiler'
import { FieldDtype } from '../../../types/dataGraph'
import { trainingPanelStyle, fieldGroupStyle, labelStyle, inputStyle, btnStyle } from '../styles/toolbarSharedStyles'

interface BindingPanelProps {
  onClose: () => void
}

export function BindingPanel({ onClose }: BindingPanelProps) {
  const modelNodes = useGraphStore((s) => s.nodes)
  const bindings = useGraphStore((s) => s.bindings)
  const setBindings = useGraphStore((s) => s.setBindings)
  const dataNodes = useDataGraphStore((s) => s.nodes)
  const dataEdges = useDataGraphStore((s) => s.edges)
  const trainingConfig = useGraphStore((s) => s.trainingConfig)

  const modelInputs = useMemo(() => (
    modelNodes
      .filter((node) => node.data.nodeType === 'input')
      .map((node, index) => ({
        key: String(node.data.params.name || `input_${index + 1}`),
        label: node.data.label,
      }))
  ), [modelNodes])

  // Get data output fields with their dtype info
  const dataOutputs = useMemo(() => {
    const outputs: Array<{ field: string; dtype?: FieldDtype; shapeHint?: string }> = []
    const seen = new Set<string>()

    for (const node of dataNodes) {
      if (node.data.nodeType !== 'dataset_output') continue
      const fields = String(node.data.params.fields || '')
        .split(',')
        .map((f) => f.trim())
        .filter(Boolean)

      for (const field of fields) {
        if (seen.has(field)) continue
        seen.add(field)

        // Try to get dtype from fieldSpecs
        let dtype: FieldDtype | undefined
        let shapeHint: string | undefined

        if (node.data.fieldSpecs) {
          try {
            const specs = JSON.parse(node.data.fieldSpecs)
            const spec = specs.find((s: any) => s.name === field)
            if (spec) {
              dtype = spec.dtype
              shapeHint = spec.shapeHint
            }
          } catch {}
        }

        // Infer from field name if not found
        if (!dtype) {
          const lowered = field.toLowerCase()
          if (lowered.includes('image') || lowered.includes('img')) dtype = 'image'
          else if (lowered.includes('label') || lowered.includes('target') || lowered.includes('class')) dtype = 'label'
          else if (lowered.includes('mask')) dtype = 'mask'
          else if (lowered.includes('text') || lowered.includes('token')) dtype = 'text'
          else if (lowered.includes('audio') || lowered.includes('waveform')) dtype = 'audio'
          else if (lowered.includes('spectrogram') || lowered.includes('mel') || lowered.includes('mfcc')) dtype = 'spectrogram'
          else dtype = 'tensor'
        }

        outputs.push({ field, dtype, shapeHint })
      }
    }
    return outputs
  }, [dataNodes])

  const setBindingValue = (target: WorkflowBinding['target'], targetKey: string, sourceKey: string) => {
    const next = bindings.filter((binding) => !(binding.target === target && binding.targetKey === targetKey))
    if (sourceKey) {
      next.push({
        id: `${target}:${targetKey}`,
        sourceGraph: 'data',
        sourceKey,
        target,
        targetKey,
      })
    }
    setBindings(next)
  }

  const getBindingValue = (target: WorkflowBinding['target'], targetKey: string) =>
    bindings.find((binding) => binding.target === target && binding.targetKey === targetKey)?.sourceKey ?? ''

  const bindingStatus = useMemo(() => (
    compileDataWorkflow(modelNodes as any, dataNodes as any, dataEdges as any, bindings, trainingConfig)
  ), [modelNodes, dataNodes, dataEdges, bindings, trainingConfig])

  // Validate binding compatibility
  const bindingValidations = useMemo(() => (
    validateBindingCompatibility(bindings, dataNodes as any, modelNodes as any)
  ), [bindings, dataNodes, modelNodes])

  // Get dtype color
  const getDtypeColor = (dtype?: FieldDtype): string => {
    const colors: Record<FieldDtype, string> = {
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
    return dtype ? colors[dtype] || '#ccc' : '#666'
  }

  // Check if a specific binding has validation issues
  const getBindingValidation = (target: WorkflowBinding['target'], targetKey: string) => {
    const binding = bindings.find(b => b.target === target && b.targetKey === targetKey)
    if (!binding) return null
    return bindingValidations.find(v => v.sourceKey === binding.sourceKey && v.targetKey === binding.targetKey)
  }

  // Auto-bind: match data fields to model inputs by name similarity
  const autoBind = () => {
    const newBindings: WorkflowBinding[] = [...bindings]

    for (const input of modelInputs) {
      // Skip if already bound
      if (bindings.some(b => b.target === 'model_input' && b.targetKey === input.key)) continue

      // Try exact match first
      const exactMatch = dataOutputs.find(d => d.field.toLowerCase() === input.key.toLowerCase())
      if (exactMatch) {
        newBindings.push({
          id: `model_input:${input.key}`,
          sourceGraph: 'data',
          sourceKey: exactMatch.field,
          target: 'model_input',
          targetKey: input.key,
        })
        continue
      }

      // Try partial match (e.g., 'image' in data matches 'input_image')
      const partialMatch = dataOutputs.find(d =>
        d.field.toLowerCase().includes(input.key.toLowerCase()) ||
        input.key.toLowerCase().includes(d.field.toLowerCase())
      )
      if (partialMatch) {
        newBindings.push({
          id: `model_input:${input.key}`,
          sourceGraph: 'data',
          sourceKey: partialMatch.field,
          target: 'model_input',
          targetKey: input.key,
        })
      }
    }

    // Auto-bind training target (label field)
    const labelBinding = bindings.find(b => b.target === 'training_target')
    if (!labelBinding) {
      const labelField = dataOutputs.find(d =>
        d.field.toLowerCase().includes('label') ||
        d.field.toLowerCase().includes('target') ||
        d.field.toLowerCase().includes('class')
      )
      if (labelField) {
        newBindings.push({
          id: 'training_target:label',
          sourceGraph: 'data',
          sourceKey: labelField.field,
          target: 'training_target',
          targetKey: 'label',
        })
      }
    }

    setBindings(newBindings)
  }

  // Get suggested bindings (potential matches that aren't bound yet)
  const suggestedBindings = useMemo(() => {
    const suggestions: Array<{ target: 'model_input' | 'training_target'; targetKey: string; sourceKey: string; matchType: 'exact' | 'partial' }> = []

    for (const input of modelInputs) {
      if (bindings.some(b => b.target === 'model_input' && b.targetKey === input.key)) continue

      const exactMatch = dataOutputs.find(d => d.field.toLowerCase() === input.key.toLowerCase())
      if (exactMatch) {
        suggestions.push({ target: 'model_input', targetKey: input.key, sourceKey: exactMatch.field, matchType: 'exact' })
        continue
      }

      const partialMatch = dataOutputs.find(d =>
        d.field.toLowerCase().includes(input.key.toLowerCase()) ||
        input.key.toLowerCase().includes(d.field.toLowerCase())
      )
      if (partialMatch) {
        suggestions.push({ target: 'model_input', targetKey: input.key, sourceKey: partialMatch.field, matchType: 'partial' })
      }
    }

    // Suggest training target
    if (!bindings.some(b => b.target === 'training_target')) {
      const labelField = dataOutputs.find(d =>
        d.field.toLowerCase().includes('label') ||
        d.field.toLowerCase().includes('target') ||
        d.field.toLowerCase().includes('class')
      )
      if (labelField) {
        suggestions.push({ target: 'training_target', targetKey: 'label', sourceKey: labelField.field, matchType: 'exact' })
      }
    }

    return suggestions
  }, [modelInputs, dataOutputs, bindings])

  // Count errors and warnings
  const errorCount = bindingValidations.filter(v => !v.isCompatible && v.error).length
  const warningCount = bindingValidations.filter(v => !v.isCompatible && v.warning && !v.error).length

  return (
    <div style={trainingPanelStyle} onClick={(e) => e.stopPropagation()}>
      <div style={{ fontSize: '13px', fontWeight: 700, color: '#a0c0ff', marginBottom: '12px', letterSpacing: '0.5px' }}>
        🔗 Binding 配置
      </div>
      <div style={{ fontSize: '11px', color: '#666', marginBottom: '12px', lineHeight: 1.5 }}>
        将数据图输出字段映射到模型输入和训练目标；模型图与数据图保持解耦。
      </div>

      {/* Auto-bind section */}
      {dataOutputs.length > 0 && modelInputs.length > 0 && (
        <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '8px', background: '#1a1e24', border: '1px solid #333' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: '#a0c0ff' }}>
              ⚡ 智能绑定
            </div>
            <button
              onClick={autoBind}
              disabled={suggestedBindings.length === 0}
              style={{
                padding: '4px 12px',
                fontSize: '11px',
                background: suggestedBindings.length > 0 ? '#1e3a5f' : '#1a1a1a',
                border: `1px solid ${suggestedBindings.length > 0 ? '#4488cc' : '#333'}`,
                borderRadius: '4px',
                color: suggestedBindings.length > 0 ? '#88ccff' : '#555',
                cursor: suggestedBindings.length > 0 ? 'pointer' : 'not-allowed',
              }}
            >
              🔗 自动匹配 {suggestedBindings.length > 0 && `(${suggestedBindings.length})`}
            </button>
          </div>

          {/* Suggested bindings */}
          {suggestedBindings.length > 0 && (
            <div style={{ fontSize: '10px', color: '#666', marginBottom: '6px' }}>
              建议绑定:
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {suggestedBindings.map((s, idx) => (
              <button
                key={idx}
                onClick={() => {
                  if (s.target === 'model_input') {
                    setBindingValue('model_input', s.targetKey, s.sourceKey)
                  } else {
                    setBindingValue('training_target', s.targetKey, s.sourceKey)
                  }
                }}
                style={{
                  padding: '3px 8px',
                  fontSize: '10px',
                  background: '#111',
                  border: `1px solid ${s.matchType === 'exact' ? '#4488aa' : '#555'}`,
                  borderRadius: '4px',
                  color: s.matchType === 'exact' ? '#88ddff' : '#999',
                  cursor: 'pointer',
                }}
                title={s.matchType === 'exact' ? '精确匹配' : '模糊匹配'}
              >
                {s.sourceKey} → {s.targetKey} {s.matchType === 'exact' ? '✓' : '~'}
              </button>
            ))}
          </div>

          {/* Unbound data fields indicator */}
          {(() => {
            const boundFields = new Set(bindings.filter(b => b.sourceGraph === 'data').map(b => b.sourceKey))
            const unboundFields = dataOutputs.filter(d => !boundFields.has(d.field))
            if (unboundFields.length === 0) return null
            return (
              <div style={{ marginTop: '8px', fontSize: '10px', color: '#555' }}>
                未绑定字段: {unboundFields.map(f => f.field).join(', ')}
              </div>
            )
          })()}
        </div>
      )}

      {/* Validation errors - red panel */}
      {errorCount > 0 && (
        <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '8px', background: '#2a1515', border: '1px solid #8b4444' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: '#ff8888', marginBottom: '6px' }}>
            ❌ 类型不兼容 ({errorCount} 个错误)
          </div>
          {bindingValidations.filter(v => !v.isCompatible && v.error).map((v, i) => (
            <div key={i} style={{ fontSize: '11px', color: '#cc6666', lineHeight: 1.5 }}>
              - {v.sourceKey} → {v.targetKey}: {v.error}
            </div>
          ))}
        </div>
      )}

      {/* Warnings - yellow panel */}
      {(warningCount > 0 || bindingStatus.warnings.length > 0) && (
        <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '8px', background: '#231b12', border: '1px solid #5c4728' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: '#f0c27a', marginBottom: '6px' }}>
            ⚠️ 警告 ({warningCount + bindingStatus.warnings.length})
          </div>
          {bindingValidations.filter(v => !v.isCompatible && v.warning && !v.error).map((v, i) => (
            <div key={`w${i}`} style={{ fontSize: '11px', color: '#c9a16a', lineHeight: 1.5 }}>
              - {v.sourceKey}: {v.warning}
            </div>
          ))}
          {bindingStatus.warnings.map((warning, i) => (
            <div key={`sw${i}`} style={{ fontSize: '11px', color: '#c9a16a', lineHeight: 1.5 }}>
              - {warning}
            </div>
          ))}
        </div>
      )}

      {dataOutputs.length === 0 ? (
        <div style={{ fontSize: '11px', color: '#777', marginBottom: '14px' }}>
          还没有数据输出字段。请先切到 `Data` 工作区并添加 `Dataset Output` 节点。
        </div>
      ) : (
        <>
          {modelInputs.map((input) => {
            const validation = getBindingValidation('model_input', input.key)
            return (
              <div key={input.key} style={fieldGroupStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                  <label style={labelStyle}>Model Input · {input.key}</label>
                  {validation && !validation.isCompatible && validation.error && (
                    <span style={{ fontSize: '10px', color: '#ff8888' }}>⚠️ dtype不匹配</span>
                  )}
                  {validation && validation.isCompatible && validation.sourceDtype && (
                    <span style={{
                      fontSize: '10px',
                      color: getDtypeColor(validation.sourceDtype),
                      background: '#1a1a1a',
                      padding: '1px 5px',
                      borderRadius: '3px',
                      border: `1px solid ${getDtypeColor(validation.sourceDtype)}33`,
                    }}>
                      {validation.sourceDtype}
                    </span>
                  )}
                </div>
                <select
                  style={{
                    ...inputStyle,
                    borderColor: validation && !validation.isCompatible ? '#8b4444' : undefined,
                  }}
                  value={getBindingValue('model_input', input.key)}
                  onChange={(e) => setBindingValue('model_input', input.key, e.target.value)}
                >
                  <option value="">-- Select data field --</option>
                  {dataOutputs.map(({ field, dtype, shapeHint }) => (
                    <option key={field} value={field}>
                      {field} {dtype ? `[${dtype}${shapeHint ? `:${shapeHint}` : ''}]` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )
          })}

          <div style={fieldGroupStyle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <label style={labelStyle}>Training Target · label</label>
              {(() => {
                const validation = getBindingValidation('training_target', 'label')
                if (validation && !validation.isCompatible) {
                  return <span style={{ fontSize: '10px', color: '#f0c27a' }}>⚠️</span>
                }
                if (validation && validation.sourceDtype) {
                  return (
                    <span style={{
                      fontSize: '10px',
                      color: getDtypeColor(validation.sourceDtype),
                      background: '#1a1a1a',
                      padding: '1px 5px',
                      borderRadius: '3px',
                      border: `1px solid ${getDtypeColor(validation.sourceDtype)}33`,
                    }}>
                      {validation.sourceDtype}
                    </span>
                  )
                }
                return null
              })()}
            </div>
            <select
              style={inputStyle}
              value={getBindingValue('training_target', 'label')}
              onChange={(e) => setBindingValue('training_target', 'label', e.target.value)}
            >
              <option value="">-- Select target field --</option>
              {dataOutputs.map(({ field, dtype, shapeHint }) => (
                <option key={field} value={field}>
                  {field} {dtype ? `[${dtype}${shapeHint ? `:${shapeHint}` : ''}]` : ''}
                </option>
              ))}
            </select>
          </div>
        </>
      )}

      {/* Data output field legend with binding status */}
      {dataOutputs.length > 0 && (
        <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid #333' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <div style={{ fontSize: '10px', color: '#666' }}>可用数据字段:</div>
            <div style={{ fontSize: '10px', color: '#555' }}>
              {(() => {
                const boundCount = bindings.filter(b => b.sourceGraph === 'data').length
                return `${boundCount}/${dataOutputs.length} 已绑定`
              })()}
            </div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {dataOutputs.map(({ field, dtype }) => {
              const isBound = bindings.some(b => b.sourceGraph === 'data' && b.sourceKey === field)
              return (
                <span
                  key={field}
                  style={{
                    fontSize: '10px',
                    color: isBound ? '#88ff88' : getDtypeColor(dtype),
                    background: isBound ? '#0f1a0f' : '#0f1318',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    border: `1px solid ${isBound ? '#448844' : getDtypeColor(dtype)}44`,
                    opacity: isBound ? 0.7 : 1,
                  }}
                  title={isBound ? '已绑定' : dtype}
                >
                  {isBound && '✓ '}{field}:{dtype}
                </span>
              )
            })}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
        <button onClick={() => setBindings([])} style={{ ...btnStyle, flex: 1 }}>
          清空绑定
        </button>
        <button onClick={onClose} style={{ ...btnStyle, flex: 1, background: '#2a2a3a', borderColor: '#444' }}>
          关闭
        </button>
      </div>
    </div>
  )
}
