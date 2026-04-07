/**
 * All Outputs Panel - Tensor stats display
 */

import { TensorStats } from '../../../hooks/useTensorExecutor'
import { panelContainerStyle } from '../styles/toolbarSharedStyles'

interface AllOutputsPanelProps {
  preview: Record<string, TensorStats>
  onClose: () => void
}

export function AllOutputsPanel({ preview, onClose }: AllOutputsPanelProps) {
  return (
    <div style={{
      ...panelContainerStyle,
      minWidth: '300px',
      maxWidth: '420px',
      maxHeight: '70vh',
      overflowY: 'auto',
    }}>
      <div style={{ fontSize: '13px', fontWeight: 700, color: '#a0c0ff', marginBottom: '12px' }}>
        📊 All Node Outputs
      </div>
      {Object.keys(preview).length === 0 ? (
        <div style={{ fontSize: '11px', color: '#555' }}>No outputs yet. Run ▶ Preview first.</div>
      ) : (
        Object.entries(preview).map(([nodeId, stats]) => (
          <div key={nodeId} style={{ marginBottom: '12px', padding: '8px', background: '#222', borderRadius: '6px', border: '1px solid #333' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#88bbff', marginBottom: '4px' }}>{nodeId}</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 12px', fontSize: '11px' }}>
              <span style={{ color: '#555' }}>Shape</span><span style={{ color: '#ccc' }}>[{stats.shape.join(', ')}]</span>
              <span style={{ color: '#555' }}>Dtype</span><span style={{ color: '#ccc' }}>{stats.dtype}</span>
              <span style={{ color: '#555' }}>Mean</span><span style={{ color: '#ccc' }}>{stats.mean}</span>
              <span style={{ color: '#555' }}>Std</span><span style={{ color: '#ccc' }}>{stats.std}</span>
              <span style={{ color: '#555' }}>Range</span><span style={{ color: '#ccc' }}>[{stats.min}, {stats.max}]</span>
              <span style={{ color: '#555' }}>Numel</span><span style={{ color: '#ccc' }}>{stats.numel.toLocaleString()}</span>
            </div>
          </div>
        ))
      )}
      <button onClick={onClose} style={{ width: '100%', background: '#2a2a3a', border: '1px solid #444', borderRadius: '6px', color: '#ccc', padding: '6px', cursor: 'pointer', fontSize: '12px' }}>关闭</button>
    </div>
  )
}
