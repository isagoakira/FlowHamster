import { useCallback } from 'react'
import { useGraphStore } from './useGraphStore'

/**
 * Hook providing node action functions.
 * This replaces the anti-pattern of using window.__flowhamster_* directly.
 */
export function useNodeActions() {
  const removeNode = useCallback((id: string) => {
    useGraphStore.getState().removeNode(id)
  }, [])

  const updateNodeData = useCallback((id: string, data: Record<string, unknown>) => {
    useGraphStore.getState().updateNodeData(id, data)
  }, [])

  const updateNodeParams = useCallback((id: string, params: Record<string, unknown>) => {
    useGraphStore.getState().updateNodeParams(id, params)
  }, [])

  return {
    removeNode,
    updateNodeData,
    updateNodeParams,
  }
}
