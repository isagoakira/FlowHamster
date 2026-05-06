/**
 * Training Store — manages runs, metrics, and training lifecycle on the frontend.
 */

import { create } from 'zustand'

export interface Run {
  id: string
  workflow_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  config_json: string
  started_at: string | null
  ended_at: string | null
  best_metric: string | null
  best_metric_value: number | null
  output_dir: string
  error_message: string | null
}

export interface Metric {
  id: number
  run_id: string
  step: number
  phase: 'train' | 'val'
  loss: number | null
  accuracy: number | null
  learning_rate: number | null
  epoch: number | null
  created_at: string
}

export interface Checkpoint {
  id: string
  run_id: string
  step: number
  metric_value: number | null
  filepath: string
  created_at: string
}

export interface TrainingStore {
  runs: Run[]
  currentRunId: string | null
  metrics: Map<string, Metric[]>        // run_id → metrics array
  checkpoints: Map<string, Checkpoint[]> // run_id → checkpoints
  isLoading: boolean
  error: string | null

  // Actions
  fetchRuns: () => Promise<void>
  startTraining: (workflowDoc: object, trainingConfig: object) => Promise<string>
  cancelRun: (runId: string) => Promise<void>
  deleteRun: (runId: string) => Promise<void>
  fetchMetrics: (runId: string) => Promise<void>
  fetchCheckpoints: (runId: string) => Promise<void>
  pollRunStatus: (runId: string) => Promise<Run | null>
  setCurrentRun: (runId: string | null) => void
}

const API_BASE = 'http://localhost:8000'

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  runs: [],
  currentRunId: null,
  metrics: new Map(),
  checkpoints: new Map(),
  isLoading: false,
  error: null,

  fetchRuns: async () => {
    set({ isLoading: true, error: null })
    try {
      const res = await fetch(`${API_BASE}/api/runs`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: Run[] = await res.json()
      set({ runs: data, isLoading: false })
    } catch (err) {
      set({ error: (err as Error).message, isLoading: false })
    }
  },

  startTraining: async (workflowDoc: object, trainingConfig: object) => {
    set({ isLoading: true, error: null })
    try {
      const res = await fetch(`${API_BASE}/api/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflowDoc, trainingConfig }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      // Immediately refresh runs list
      get().fetchRuns()
      return data.run_id as string
    } catch (err) {
      set({ error: (err as Error).message, isLoading: false })
      throw err
    }
  },

  cancelRun: async (runId: string) => {
    await fetch(`${API_BASE}/api/runs/${runId}/cancel`, { method: 'POST' })
    get().fetchRuns()
  },

  deleteRun: async (runId: string) => {
    await fetch(`${API_BASE}/api/runs/${runId}`, { method: 'DELETE' })
    set((state) => ({
      runs: state.runs.filter(r => r.id !== runId),
      currentRunId: state.currentRunId === runId ? null : state.currentRunId,
    }))
  },

  fetchMetrics: async (runId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/runs/${runId}/metrics`)
      if (!res.ok) return
      const data: Metric[] = await res.json()
      set((state) => {
        const next = new Map(state.metrics)
        next.set(runId, data)
        return { metrics: next }
      })
    } catch { /* non-critical */ }
  },

  fetchCheckpoints: async (runId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/runs/${runId}/checkpoints`)
      if (!res.ok) return
      const data: Checkpoint[] = await res.json()
      set((state) => {
        const next = new Map(state.checkpoints)
        next.set(runId, data)
        return { checkpoints: next }
      })
    } catch { /* non-critical */ }
  },

  pollRunStatus: async (runId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/runs/${runId}`)
      if (!res.ok) return null
      const run: Run = await res.json()
      set((state) => ({
        runs: state.runs.map(r => r.id === runId ? run : r),
      }))
      return run
    } catch {
      return null
    }
  },

  setCurrentRun: (runId: string | null) => {
    set({ currentRunId: runId })
  },
}))