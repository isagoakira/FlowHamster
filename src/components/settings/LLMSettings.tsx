/**
 * LLMSettings — API Key management, provider/model selection.
 * SECURITY: API keys are submitted to backend and immediately discarded from local state.
 */
import { useState, useEffect, useCallback } from 'react'
import { useLlmStore } from '../../hooks/useLlmStore'
import { ProviderSelector } from '../common/ProviderSelector'
import { ModelSelector } from '../common/ModelSelector'
import { panelContainerStyle, labelStyle, inputStyle } from '../toolbar/styles/toolbarSharedStyles'

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '13px',
  fontWeight: 700,
  color: '#a0c0ff',
  marginBottom: '12px',
  letterSpacing: '0.5px',
}

const fieldGroupStyle: React.CSSProperties = {
  marginBottom: '12px',
}

const statusBadgeStyle = (configured: boolean): React.CSSProperties => ({
  display: 'inline-block',
  padding: '2px 8px',
  borderRadius: '4px',
  fontSize: '11px',
  fontWeight: 600,
  background: configured ? '#1a3a2a' : '#3a2a1a',
  color: configured ? '#88ffaa' : '#ffaa66',
  border: `1px solid ${configured ? '#2f7a5c' : '#8b6b3a'}`,
})

const btnPrimaryStyle: React.CSSProperties = {
  background: '#224488',
  border: '1px solid #3366aa',
  borderRadius: '6px',
  color: '#c0d8ff',
  padding: '6px 12px',
  fontSize: '12px',
  cursor: 'pointer',
}

const btnDangerStyle: React.CSSProperties = {
  background: '#5a2a2a',
  border: '1px solid #8b4444',
  borderRadius: '6px',
  color: '#ffaaaa',
  padding: '6px 12px',
  fontSize: '12px',
  cursor: 'pointer',
}

const btnSecondaryStyle: React.CSSProperties = {
  background: '#2a2a3a',
  border: '1px solid #444',
  borderRadius: '6px',
  color: '#ccc',
  padding: '6px 12px',
  fontSize: '12px',
  cursor: 'pointer',
}

const errorStyle: React.CSSProperties = {
  padding: '8px 10px',
  borderRadius: '6px',
  background: '#2a1515',
  border: '1px solid #8b4444',
  color: '#ffaaaa',
  fontSize: '11px',
  marginBottom: '10px',
}

interface LLMSettingsProps {
  onClose: () => void
}

export function LLMSettings({ onClose }: LLMSettingsProps) {
  const store = useLlmStore()

  const [keyInput, setKeyInput] = useState('')
  const [saving, setSaving] = useState(false)
  const [justSaved, setJustSaved] = useState(false)

  const selectedProvider = store.providers.find((p) => p.id === store.selectedProviderId)

  useEffect(() => {
    if (store.providers.length === 0) {
      store.loadProviders()
    }
  }, [])

  const handleSaveKey = useCallback(async () => {
    if (!store.selectedProviderId || !keyInput.trim()) return
    setSaving(true)
    try {
      await store.configureApiKey(store.selectedProviderId, keyInput.trim())
      setKeyInput('')
      setJustSaved(true)
      setTimeout(() => setJustSaved(false), 2000)
    } catch {
      // error is already captured in store.error
    } finally {
      setSaving(false)
    }
  }, [store, keyInput])

  const handleDeleteKey = useCallback(async () => {
    if (!store.selectedProviderId) return
    setSaving(true)
    try {
      await store.removeApiKey(store.selectedProviderId)
    } catch {
      // error is already captured in store.error
    } finally {
      setSaving(false)
    }
  }, [store])

  return (
    <div style={panelContainerStyle} onClick={(e) => e.stopPropagation()}>
      <div style={sectionTitleStyle}>LLM Provider Settings</div>

      {store.error && (
        <div style={errorStyle}>
          {store.error}
          <button
            onClick={() => store.clearError()}
            style={{ marginLeft: '8px', background: 'transparent', border: 'none', color: '#ff8888', cursor: 'pointer', fontSize: '11px' }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Provider selection */}
      <div style={fieldGroupStyle}>
        <span style={labelStyle}>Provider</span>
        <ProviderSelector
          providers={store.providers}
          selectedProviderId={store.selectedProviderId}
          onSelect={store.selectProvider}
          disabled={store.isLoading}
        />
      </div>

      {/* Model selection */}
      <div style={fieldGroupStyle}>
        <span style={labelStyle}>Model</span>
        <ModelSelector
          models={selectedProvider?.models ?? []}
          selectedModelId={store.selectedModelId}
          onSelect={store.selectModel}
          disabled={store.isLoading}
        />
      </div>

      <div style={{ height: '1px', background: '#333', margin: '12px 0' }} />

      {/* API Key management */}
      {selectedProvider && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#ccc' }}>
              {selectedProvider.name} API Key
            </span>
            <span style={statusBadgeStyle(selectedProvider.api_key_configured)}>
              {selectedProvider.api_key_configured ? 'Configured' : 'Not configured'}
            </span>
          </div>

          <div style={fieldGroupStyle}>
            <input
              type="password"
              style={inputStyle}
              placeholder="Enter API key..."
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              disabled={saving}
            />
            <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
              Key is sent to backend for encrypted storage and never persisted locally.
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
            <button
              style={{ ...btnPrimaryStyle, opacity: saving || !keyInput.trim() ? 0.6 : 1 }}
              onClick={handleSaveKey}
              disabled={saving || !keyInput.trim()}
            >
              {saving ? 'Saving...' : justSaved ? 'Saved!' : 'Save Key'}
            </button>
            {selectedProvider.api_key_configured && (
              <button
                style={{ ...btnDangerStyle, opacity: saving ? 0.6 : 1 }}
                onClick={handleDeleteKey}
                disabled={saving}
              >
                Delete Key
              </button>
            )}
          </div>
        </div>
      )}

      <button
        onClick={onClose}
        style={{ ...btnSecondaryStyle, marginTop: '12px', width: '100%' }}
      >
        Close
      </button>
    </div>
  )
}
