/**
 * ProviderSelector — Dropdown for selecting an LLM vendor.
 */
import { LlmProvider } from '../../services/llmApi'

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

interface ProviderSelectorProps {
  providers: LlmProvider[]
  selectedProviderId: string | null
  onSelect: (providerId: string) => void
  disabled?: boolean
}

export function ProviderSelector({ providers, selectedProviderId, onSelect, disabled }: ProviderSelectorProps) {
  return (
    <select
      style={selectStyle}
      value={selectedProviderId ?? ''}
      onChange={(e) => onSelect(e.target.value)}
      disabled={disabled || providers.length === 0}
    >
      {providers.length === 0 && (
        <option value="">No providers available</option>
      )}
      {providers.map((provider) => (
        <option key={provider.id} value={provider.id}>
          {provider.name} {provider.api_key_configured ? '(configured)' : '(not configured)'}
        </option>
      ))}
    </select>
  )
}
