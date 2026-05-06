/**
 * Training Dashboard — real-time training runs, metrics, and checkpoint management.
 */

import { useEffect, useRef } from 'react'
import { useTrainingStore, Run } from '../../../hooks/useTrainingStore'
import { trainingPanelStyle, btnStyle } from '../styles/toolbarSharedStyles'

interface TrainingDashboardProps {
  onClose: () => void
}

const STATUS_CONFIG: Record<Run['status'], { color: string; emoji: string }> = {
  pending:   { color: '#888', emoji: '⏳' },
  running:   { color: '#4a9', emoji: '🔄' },
  completed: { color: '#4c4', emoji: '✅' },
  failed:    { color: '#c44', emoji: '❌' },
  cancelled: { color: '#888', emoji: '🚫' },
}

// Simple SVG line chart for loss/accuracy
function MiniChart({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null
  const w = 160, h = 50
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(' ')
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

function RunCard({ run, isSelected, onSelect }: { run: Run; isSelected: boolean; onSelect: () => void }) {
  const { cancelRun, deleteRun } = useTrainingStore()
  const statusCfg = STATUS_CONFIG[run.status] ?? STATUS_CONFIG.pending

  return (
    <div
      style={{
        padding: '8px 10px',
        background: isSelected ? '#1a2a3a' : '#141414',
        border: `1px solid ${isSelected ? '#4488ff' : '#2a2a2a'}`,
        borderRadius: '6px',
        cursor: 'pointer',
        marginBottom: '6px',
      }}
      onClick={onSelect}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
        <span style={{ fontSize: '11px', color: '#ccc', fontFamily: 'monospace' }}>{run.id}</span>
        <span style={{ fontSize: '11px', color: statusCfg.color }}>
          {statusCfg.emoji} {run.status}
        </span>
      </div>
      {run.best_metric && (
        <div style={{ fontSize: '10px', color: '#888' }}>
          Best {run.best_metric}: {run.best_metric_value?.toFixed(4)}
        </div>
      )}
      {run.started_at && (
        <div style={{ fontSize: '10px', color: '#555' }}>
          {new Date(run.started_at).toLocaleString()}
        </div>
      )}
      {run.status === 'running' && (
        <button
          style={{ ...btnStyle, fontSize: '10px', padding: '2px 8px', marginTop: '4px', background: '#2a1a1a', borderColor: '#844', color: '#c88' }}
          onClick={(e) => { e.stopPropagation(); cancelRun(run.id) }}
        >
          Cancel
        </button>
      )}
      {(run.status === 'completed' || run.status === 'failed' || run.status === 'cancelled') && (
        <button
          style={{ ...btnStyle, fontSize: '10px', padding: '2px 8px', marginTop: '4px', background: '#1a1a1a', borderColor: '#444', color: '#666' }}
          onClick={(e) => { e.stopPropagation(); deleteRun(run.id) }}
        >
          Delete
        </button>
      )}
    </div>
  )
}

function MetricsPanel({ runId }: { runId: string }) {
  const metrics = useTrainingStore((s) => s.metrics.get(runId) ?? [])
  const fetchMetrics = useTrainingStore((s) => s.fetchMetrics)

  useEffect(() => {
    fetchMetrics(runId)
  }, [runId, fetchMetrics])

  const trainLosses = metrics.filter(m => m.phase === 'train').map(m => m.loss).filter((v): v is number => v !== null)
  const valLosses = metrics.filter(m => m.phase === 'val').map(m => m.loss).filter((v): v is number => v !== null)
  const trainAccs = metrics.filter(m => m.phase === 'train').map(m => m.accuracy).filter((v): v is number => v !== null)
  const valAccs = metrics.filter(m => m.phase === 'val').map(m => m.accuracy).filter((v): v is number => v !== null)

  return (
    <div style={{ marginTop: '12px' }}>
      <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>Loss</div>
      <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
        <div>
          <div style={{ fontSize: '9px', color: '#4a4', marginBottom: '2px' }}>Train</div>
          <MiniChart data={trainLosses} color="#4a4" />
        </div>
        <div>
          <div style={{ fontSize: '9px', color: '#88aaff', marginBottom: '2px' }}>Val</div>
          <MiniChart data={valLosses} color="#88aaff" />
        </div>
      </div>
      {trainAccs.length > 0 && (
        <>
          <div style={{ fontSize: '11px', color: '#888', marginTop: '10px', marginBottom: '6px' }}>Accuracy</div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
            <div>
              <div style={{ fontSize: '9px', color: '#4a4', marginBottom: '2px' }}>Train</div>
              <MiniChart data={trainAccs} color="#4a4" />
            </div>
            <div>
              <div style={{ fontSize: '9px', color: '#88aaff', marginBottom: '2px' }}>Val</div>
              <MiniChart data={valAccs} color="#88aaff" />
            </div>
          </div>
        </>
      )}
      {metrics.length === 0 && (
        <div style={{ fontSize: '10px', color: '#444', marginTop: '8px' }}>No metrics yet</div>
      )}
    </div>
  )
}

export function TrainingDashboard({ onClose }: TrainingDashboardProps) {
  const { runs, currentRunId, fetchRuns, setCurrentRun, pollRunStatus } = useTrainingStore()
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  // Poll running runs every 5s
  useEffect(() => {
    const runningIds = runs.filter(r => r.status === 'running').map(r => r.id)
    if (runningIds.length === 0) {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      return
    }
    pollIntervalRef.current = setInterval(async () => {
      for (const id of runningIds) {
        const updated = await pollRunStatus(id)
        // Refresh metrics when run completes
        if (updated && updated.status !== 'running') {
          useTrainingStore.getState().fetchMetrics(id)
        }
      }
    }, 5000)
    return () => { if (pollIntervalRef.current) clearInterval(pollIntervalRef.current) }
  }, [runs])

  const selectedRun = runs.find(r => r.id === currentRunId)

  return (
    <div style={trainingPanelStyle} onClick={(e) => e.stopPropagation()}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: '#a0c0ff', letterSpacing: '0.5px' }}>
          🚀 Training Runs
        </div>
        <button
          style={{ ...btnStyle, background: '#1a1a1a', borderColor: '#333', color: '#666', fontSize: '11px', padding: '2px 8px' }}
          onClick={onClose}
        >
          ✕
        </button>
      </div>

      {/* Run list */}
      <div style={{ maxHeight: '220px', overflowY: 'auto', marginBottom: '10px' }}>
        {runs.length === 0 && (
          <div style={{ fontSize: '11px', color: '#555', textAlign: 'center', padding: '20px 0' }}>
            No runs yet. Click 🚀 Train to start.
          </div>
        )}
        {runs.map(run => (
          <RunCard
            key={run.id}
            run={run}
            isSelected={run.id === currentRunId}
            onSelect={() => setCurrentRun(run.id === currentRunId ? null : run.id)}
          />
        ))}
      </div>

      {/* Metrics panel for selected run */}
      {selectedRun && (
        <div style={{ borderTop: '1px solid #2a2a2a', paddingTop: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', color: '#88aaff' }}>Run: {selectedRun.id}</span>
            <span style={{ fontSize: '11px', color: STATUS_CONFIG[selectedRun.status].color }}>
              {STATUS_CONFIG[selectedRun.status].emoji} {selectedRun.status}
            </span>
          </div>
          <MetricsPanel runId={selectedRun.id} />
          {selectedRun.error_message && (
            <div style={{ fontSize: '10px', color: '#c44', marginTop: '8px', wordBreak: 'break-all', whiteSpace: 'pre-wrap', maxHeight: '80px', overflowY: 'auto' }}>
              Error: {selectedRun.error_message.slice(-500)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}