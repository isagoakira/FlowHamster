import { useState } from 'react'
import { API_BASE_URL } from '../utils/runtimeConfig'

export interface TensorStats {
  shape: number[]
  dtype: string
  mean: number
  std: number
  min: number
  max: number
  numel: number
}

export interface TensorPreview {
  [nodeId: string]: TensorStats
}

export function useTensorExecutor() {
  const [preview, setPreview] = useState<TensorPreview>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runForward = async (nodes: any[], edges: any[], inputShape?: number[]) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE_URL}/execute/forward`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes,
          edges,
          input_shape: inputShape || [1, 3, 224, 224],
        }),
      })
      const data = await res.json()
      if (data.success) {
        setPreview(data.outputs)
      } else {
        setError(data.error || 'Unknown error')
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const clearPreview = () => {
    setPreview({})
    setError(null)
  }

  return { preview, loading, error, runForward, clearPreview }
}
