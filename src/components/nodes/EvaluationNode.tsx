import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeData } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  f1: 'F1 Score',
  precision: 'Precision',
  recall: 'Recall',
  confusion_matrix: 'Confusion Matrix',
  mean_iou: 'Mean IoU',
  roc_auc: 'ROC AUC',
}

const METRIC_COLORS: Record<string, string> = {
  accuracy: '#44cc88',
  f1: '#44aacc',
  precision: '#88aacc',
  recall: '#cc8844',
  confusion_matrix: '#cc88cc',
  mean_iou: '#88ccaa',
  roc_auc: '#aacc44',
}

const n = { background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '10px 14px', minWidth: '140px', color: '#e0e0e0', fontSize: '13px', position: 'relative' as const }

const EvaluationNode = memo((props: NodeProps<NodeData>) => {
  const { id, data } = props
  const removeNode = (id: string) => { useGraphStore.getState().removeNode(id) }
  const [h, setH] = useState(false)

  const metricType = String(data.params?.metric_type ?? 'accuracy')
  const label = METRIC_LABELS[metricType] || 'Metric'
  const color = METRIC_COLORS[metricType] || '#44cc88'

  return (
    <div style={{ ...n, borderColor: h ? color : '#333' }} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}>
      <Handle
        type="target"
        position={Position.Left}
        id="predictions"
        style={{ background: color, width: 8, height: 8, border: 'none', top: '30%' }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="targets"
        style={{ background: '#888', width: 8, height: 8, border: 'none', top: '70%' }}
      />
      <div style={{ fontWeight: 600, fontSize: '11px', color, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {label}
      </div>
      <div style={{ fontSize: '11px', color: '#666' }}>Evaluation</div>
      <Handle
        type="source"
        position={Position.Right}
        id="score"
        style={{ background: '#44cc88', width: 8, height: 8, border: 'none' }}
      />
      {h && (
        <button
          onClick={(e) => { e.stopPropagation(); removeNode(id) }}
          style={{ position: 'absolute', top: 4, right: 4, background: '#ff4444', border: 'none', borderRadius: '50%', width: 18, height: 18, cursor: 'pointer', color: '#fff', fontSize: 11, lineHeight: 1, padding: 0 }}
          title="Delete"
        >
          ×
        </button>
      )}
    </div>
  )
})
EvaluationNode.displayName = 'EvaluationNode'
export default EvaluationNode
