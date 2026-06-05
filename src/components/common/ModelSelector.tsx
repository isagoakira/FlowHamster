/**
 * ModelSelector — Dropdown for selecting a model within the current provider.
 */
import { LlmModel } from '../../services/llmApi'

const selectStyle: React.CSSProperties = {
  width: '100%',
  background: '#111',
  border: '1px solid #333',
  borderRadius: '6px',
  color: '#ddd',
  padding: '6px 8px',
  fontSize: '12px',
  cursor: 'pointer',
}

interface ModelSelectorProps {
  models: LlmModel[]
  selectedModelId: string | null
  onSelect: (modelId: string) => void
  disabled?: boolean
}

export function ModelSelector({ models, selectedModelId, onSelect, disabled }: ModelSelectorProps) {
  return (
    <select
      style={selectStyle}
      value={selectedModelId ?? ''}
      onChange={(e) => onSelect(e.target.value)}
      disabled={disabled || models.length === 0}
    >
      {models.length === 0 && (
        <option value="">No models available</option>
      )}
      {models.map((model) => (
        <option key={model.id} value={model.id}>
          {model.name}
        </option>
      ))}
    </select>
  )
}
