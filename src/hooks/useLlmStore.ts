/**
 * LLM State Management — Zustand store for provider/model selection and key status.
 * SECURITY: Never stores plaintext API keys.
 */
import { create } from 'zustand'
import { LlmProvider, fetchProviders, setApiKey, deleteApiKey } from '../services/llmApi'

interface LlmState {
  providers: LlmProvider[]
  selectedProviderId: string | null
  selectedModelId: string | null
  isLoading: boolean
  error: string | null

  // Actions
  loadProviders: () => Promise<void>
  selectProvider: (providerId: string) => void
  selectModel: (modelId: string) => void
  configureApiKey: (providerId: string, apiKey: string) => Promise<void>
  removeApiKey: (providerId: string) => Promise<void>
  clearError: () => void
}

export const useLlmStore = create<LlmState>((set, get) => ({
  providers: [],
  selectedProviderId: null,
  selectedModelId: null,
  isLoading: false,
  error: null,

  loadProviders: async () => {
    set({ isLoading: true, error: null })
    try {
      const providers = await fetchProviders()
      const { selectedProviderId, selectedModelId } = get()

      // Auto-select first provider if current selection is invalid
      let nextProviderId = selectedProviderId
      let nextModelId = selectedModelId
      const validProvider = providers.find((p) => p.id === selectedProviderId)

      if (!validProvider && providers.length > 0) {
        nextProviderId = providers[0].id
        nextModelId = providers[0].models[0]?.id ?? null
      } else if (validProvider) {
        const validModel = validProvider.models.find((m) => m.id === selectedModelId)
        if (!validModel) {
          nextModelId = validProvider.models[0]?.id ?? null
        }
      }

      set({ providers, isLoading: false, selectedProviderId: nextProviderId, selectedModelId: nextModelId })
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to load providers', isLoading: false })
    }
  },

  selectProvider: (providerId) => {
    const provider = get().providers.find((p) => p.id === providerId)
    set({
      selectedProviderId: providerId,
      selectedModelId: provider?.models[0]?.id ?? null,
    })
  },

  selectModel: (modelId) => {
    set({ selectedModelId: modelId })
  },

  configureApiKey: async (providerId, apiKey) => {
    set({ error: null })
    try {
      await setApiKey(providerId, apiKey)
      set((state) => ({
        providers: state.providers.map((p) =>
          p.id === providerId ? { ...p, api_key_configured: true } : p
        ),
      }))
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to set API key' })
      throw err
    }
  },

  removeApiKey: async (providerId) => {
    set({ error: null })
    try {
      await deleteApiKey(providerId)
      set((state) => ({
        providers: state.providers.map((p) =>
          p.id === providerId ? { ...p, api_key_configured: false } : p
        ),
      }))
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to delete API key' })
      throw err
    }
  },

  clearError: () => set({ error: null }),
}))
