/**
 * LLM API Service — Multi-vendor LLM provider management & chat completions
 * Aligned with ISA-209 backend contract.
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

/* ---------- Tool / Function Calling Types ---------- */

export interface ToolFunction {
  name: string
  description?: string
  parameters: Record<string, unknown>
}

export interface Tool {
  type: 'function'
  function: ToolFunction
}

export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

/* ---------- Chat Message ---------- */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: ToolCall[]
  tool_call_id?: string
}

/* ---------- Chat Completion Request ---------- */

export interface ChatCompletionRequest {
  provider: string
  model: string
  messages: ChatMessage[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
  tools?: Tool[]
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } }
  api_key?: string
  extra?: Record<string, unknown>
}

/* ---------- Usage / Message in Response ---------- */

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ResponseMessage {
  role: string
  content: string
  tool_calls?: ToolCall[]
}

export interface ChatCompletionResponse {
  success: boolean
  provider?: string
  model?: string
  message?: ResponseMessage
  usage?: TokenUsage
  error?: string
  raw?: string
}

/* ---------- SSE Stream Types ---------- */

export interface StreamChunk {
  type: 'chunk'
  provider: string
  content?: string
  event?: string
  tool_calls?: ToolCall[]
}

export interface StreamDone {
  type: 'done'
  provider: string
}

export type StreamEvent = StreamChunk | StreamDone

/* ---------- Provider / Key APIs ---------- */

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

/* ---------- Chat Completion (Non-streaming) ---------- */

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

/* ---------- Chat Completion (Streaming SSE) ---------- */

function parseSsePayload(payload: string, defaultProvider: string): StreamEvent | null {
  if (payload === '[DONE]') {
    return { type: 'done', provider: defaultProvider }
  }
  try {
    const event = JSON.parse(payload) as StreamEvent
    if (event.type === 'chunk' || event.type === 'done') {
      return event
    }
  } catch {
    // ignore malformed JSON
  }
  return null
}

function flushBuffer(buffer: string, defaultProvider: string, onChunk: (event: StreamEvent) => void): string {
  const lines = buffer.split('\n')
  const remainder = lines.pop() ?? ''
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('data: ')) continue
    const event = parseSsePayload(trimmed.slice(6), defaultProvider)
    if (event) onChunk(event)
  }
  return remainder
}

export async function streamChatCompletion(
  request: ChatCompletionRequest,
  onChunk: (event: StreamEvent) => void
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
      buffer = flushBuffer(buffer, request.provider, onChunk)
    }

    // Flush any remaining buffer after stream ends
    if (buffer.trim()) {
      const event = parseSsePayload(buffer.trim(), request.provider)
      if (event) onChunk(event)
    }
  } finally {
    reader.releaseLock()
  }
}
