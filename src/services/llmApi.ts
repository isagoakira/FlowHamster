/**
 * LLM API Service — Multi-vendor LLM provider management & chat completions
 */
import { API_BASE_URL } from '../utils/runtimeConfig'

export interface LlmModel {
  id: string
  name: string
}

export interface LlmProvider {
  id: string
  name: string
  models: LlmModel[]
  api_key_configured: boolean
}

export interface LlmProvidersResponse {
  providers: LlmProvider[]
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatCompletionRequest {
  provider: string
  model: string
  messages: ChatMessage[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
}

export interface ChatCompletionResponse {
  success: boolean
  content?: string
  error?: string
}

export interface StreamChunk {
  type: 'chunk'
  provider: string
  content: string
}

export interface StreamDone {
  type: 'done'
  provider: string
}

export type StreamEvent = StreamChunk | StreamDone

export async function fetchProviders(): Promise<LlmProvider[]> {
  const res = await fetch(`${API_BASE_URL}/llm/providers`)
  if (!res.ok) {
    throw new Error(`Failed to fetch providers: ${res.status}`)
  }
  const data: LlmProvidersResponse = await res.json()
  return data.providers
}

export async function setApiKey(providerId: string, apiKey: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/llm/providers/${providerId}/api-key`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey }),
  })
  if (!res.ok) {
    throw new Error(`Failed to set API key: ${res.status}`)
  }
}

export async function deleteApiKey(providerId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/llm/providers/${providerId}/api-key`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    throw new Error(`Failed to delete API key: ${res.status}`)
  }
}

export async function sendChatCompletion(
  request: ChatCompletionRequest
): Promise<ChatCompletionResponse> {
  const res = await fetch(`${API_BASE_URL}/llm/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    throw new Error(`Chat completion failed: ${res.status}`)
  }
  return res.json()
}

export async function streamChatCompletion(
  request: ChatCompletionRequest,
  onChunk: (event: StreamEvent) => void,
  onError?: (error: Error) => void
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/llm/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, stream: true }),
  })

  if (!res.ok) {
    throw new Error(`Stream chat completion failed: ${res.status}`)
  }

  const reader = res.body?.getReader()
  if (!reader) {
    throw new Error('Response body is not readable')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const payload = trimmed.slice(6)
        if (payload === '[DONE]') {
          onChunk({ type: 'done', provider: request.provider })
          continue
        }
        try {
          const event = JSON.parse(payload) as StreamEvent
          onChunk(event)
        } catch {
          // ignore malformed SSE lines
        }
      }
    }
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)))
  } finally {
    reader.releaseLock()
  }
}
