/**
 * WorkflowSettingsBar - Bottom settings bar for workflow configuration
 * Contains: Multi-output toggle, data preprocessing, loss function, optimizer, iterator settings
 */

import { memo, useMemo } from 'react'
import { useGraphStore } from '../hooks/useGraphStore'
import { useDataGraphStore } from '../hooks/useDataGraphStore'
import { WorkflowTrainingConfig } from '../schema/workflowDocument'
import { validateBindingCompatibility } from '../utils/dataWorkflowCompiler'

const barStyle: React.CSSProperties = {
  height: '48px',
  background: '#111',
  borderTop: '1px solid #222',
  display: 'flex',
  alignItems: 'center',
  padding: '0 16px',
  gap: '16px',
  flexShrink: 0,
  zIndex: 500,
}

const sectionStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
}

const labelStyle: React.CSSProperties = {
  fontSize: '11px',
  color: '#666',
  textTransform: 'uppercase' as const,
  letterSpacing: '0.5px',
}

const selectStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#ccc',
  padding: '4px 8px',
  fontSize: '11px',
  cursor: 'pointer',
  height: '28px',
}

const checkboxLabelStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  cursor: 'pointer',
  padding: '4px 8px',
  borderRadius: '4px',
  background: '#1a1a1a',
  border: '1px solid #333',
  fontSize: '11px',
  color: '#888',
  transition: 'all 0.15s',
}

const checkboxLabelActiveStyle: React.CSSProperties = {
  ...checkboxLabelStyle,
  color: '#88bbff',
  borderColor: '#4488ff',
  background: '#1a2a3a',
}

const dividerStyle: React.CSSProperties = {
  width: '1px',
  height: '24px',
  background: '#333',
}

interface WorkflowSettingsBarProps {
  onOpenDataWorkflow?: () => void
}

function WorkflowSettingsBar({ onOpenDataWorkflow }: WorkflowSettingsBarProps) {
  const features = useGraphStore((s) => s.features)
  const toggleFeature = useGraphStore((s) => s.toggleFeature)
  const trainingConfig = useGraphStore((s) => s.trainingConfig)
  const setTrainingConfig = useGraphStore((s) => s.setTrainingConfig)
  const bindings = useGraphStore((s) => s.bindings)
  const modelNodes = useGraphStore((s) => s.nodes)
  const dataNodes = useDataGraphStore((s) => s.nodes)

  // Validate bindings for errors/warnings
  const bindingValidation = useMemo(() => {
    if (bindings.length === 0) return { errors: 0, warnings: 0 }
    const validations = validateBindingCompatibility(bindings, dataNodes as any, modelNodes as any)
    const errors = validations.filter(v => !v.isCompatible && v.error).length
    const warnings = validations.filter(v => !v.isCompatible && v.warning && !v.error).length
    return { errors, warnings }
  }, [bindings, dataNodes, modelNodes])

  const updateConfig = (updater: (current: WorkflowTrainingConfig) => WorkflowTrainingConfig) => {
    setTrainingConfig(updater(trainingConfig))
  }

  return (
    <div style={barStyle}>
      {/* Multi-output toggle */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Features</span>
        <label style={features.multiOutput ? checkboxLabelActiveStyle : checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={features.multiOutput}
            onChange={() => toggleFeature('multiOutput')}
            style={{ display: 'none' }}
          />
          <span style={{ fontSize: '13px' }}>{features.multiOutput ? '☑' : '☐'}</span>
          <span>🌐 Multi-Output</span>
        </label>
      </div>

      <div style={dividerStyle} />

      {/* Data Preprocessing with binding validation indicator */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Data</span>
        <select style={selectStyle} onClick={onOpenDataWorkflow} title="Data workflow settings">
          <option value="">Select dataset...</option>
          {dataNodes.length > 0 && <option value="custom">Custom ({dataNodes.length} nodes)</option>}
        </select>
        {bindingValidation.errors > 0 && (
          <span style={{ fontSize: '12px', color: '#ff8888' }} title={`${bindingValidation.errors} binding errors`}>
            ❌
          </span>
        )}
        {bindingValidation.errors === 0 && bindingValidation.warnings > 0 && (
          <span style={{ fontSize: '12px', color: '#f0c27a' }} title={`${bindingValidation.warnings} binding warnings`}>
            ⚠️
          </span>
        )}
      </div>

      <div style={dividerStyle} />

      {/* Loss Function */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Loss</span>
        <select
          style={selectStyle}
          value={trainingConfig.loss.type}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            loss: { ...current.loss, type: e.target.value, enabled: true },
          }))}
        >
          <option value="cross_entropy">CrossEntropy</option>
          <option value="mse">MSE</option>
          <option value="bce">BCE</option>
          <option value="none">None</option>
        </select>
      </div>

      {/* Optimizer */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Optimizer</span>
        <select
          style={selectStyle}
          value={trainingConfig.optimizer.type}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            optimizer: { ...current.optimizer, type: e.target.value, enabled: true },
          }))}
        >
          <option value="adamw">AdamW</option>
          <option value="adam">Adam</option>
          <option value="sgd">SGD</option>
          <option value="rmsprop">RMSprop</option>
        </select>
      </div>

      {/* Learning Rate */}
      <div style={sectionStyle}>
        <span style={labelStyle}>LR</span>
        <input
          type="number"
          step="0.0001"
          min={0}
          value={Number(trainingConfig.optimizer.params.lr ?? 0.001)}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            optimizer: {
              ...current.optimizer,
              params: { ...current.optimizer.params, lr: Number(e.target.value) || 0.001 },
            },
          }))}
          style={{
            ...selectStyle,
            width: '80px',
            textAlign: 'right' as const,
          }}
        />
      </div>

      <div style={dividerStyle} />

      {/* Scheduler */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Scheduler</span>
        <select
          style={selectStyle}
          value={trainingConfig.scheduler.enabled ? trainingConfig.scheduler.type : 'none'}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            scheduler: {
              ...current.scheduler,
              enabled: e.target.value !== 'none',
              type: e.target.value === 'none' ? current.scheduler.type : e.target.value,
            },
          }))}
        >
          <option value="none">None</option>
          <option value="cosine_annealing">CosineAnnealing</option>
          <option value="step_lr">StepLR</option>
          <option value="reduce_on_plateau">ReduceLROnPlateau</option>
        </select>
      </div>

      <div style={dividerStyle} />

      {/* Epochs & Batch Size */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Epochs</span>
        <input
          type="number"
          min={1}
          value={trainingConfig.runtime.epochs}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            runtime: { ...current.runtime, epochs: Math.max(1, Number(e.target.value) || 1) },
          }))}
          style={{ ...selectStyle, width: '60px', textAlign: 'right' as const }}
        />
      </div>

      <div style={sectionStyle}>
        <span style={labelStyle}>Batch</span>
        <input
          type="number"
          min={1}
          value={trainingConfig.runtime.batchSize}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            runtime: { ...current.runtime, batchSize: Math.max(1, Number(e.target.value) || 1) },
          }))}
          style={{ ...selectStyle, width: '60px', textAlign: 'right' as const }}
        />
      </div>

      <div style={{ flex: 1 }} />

      {/* Device */}
      <div style={sectionStyle}>
        <span style={labelStyle}>Device</span>
        <select
          style={selectStyle}
          value={trainingConfig.runtime.device}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            runtime: { ...current.runtime, device: e.target.value },
          }))}
        >
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </select>
      </div>

      {/* AMP */}
      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '11px',
        color: trainingConfig.runtime.amp ? '#88bbff' : '#666',
        cursor: 'pointer',
      }}>
        <input
          type="checkbox"
          checked={trainingConfig.runtime.amp}
          onChange={(e) => updateConfig((current) => ({
            ...current,
            runtime: { ...current.runtime, amp: e.target.checked },
          }))}
          style={{ accentColor: '#4488ff' }}
        />
        AMP
      </label>
    </div>
  )
}

export default memo(WorkflowSettingsBar)
