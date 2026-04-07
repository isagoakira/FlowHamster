const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env ?? {}
const API_ORIGIN = env.VITE_DEEPFORGE_API_ORIGIN ?? 'http://localhost:8000'
const WS_ORIGIN = env.VITE_DEEPFORGE_WS_ORIGIN ?? API_ORIGIN.replace(/^http/, 'ws')

export const API_BASE_URL = `${API_ORIGIN}/api`
export const HEALTH_URL = `${API_ORIGIN}/health`
export const WS_CODE_URL = `${WS_ORIGIN}/ws/code`
