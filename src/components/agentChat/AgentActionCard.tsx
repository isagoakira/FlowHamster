import { AgentToolCall } from '../../types/agentChat'
import { buildActionDiffPreview } from '../../utils/agentApi'
import {
  actionCardStyle,
  actionDiffStyle,
  actionBtnRowStyle,
  confirmBtnStyle,
  rejectBtnStyle,
} from './agentChatStyles'

interface AgentActionCardProps {
  action: AgentToolCall
  onConfirm: () => void
  onReject: () => void
}

export function AgentActionCard({ action, onConfirm, onReject }: AgentActionCardProps) {
  const diff = action.diffPreview || buildActionDiffPreview(action)

  return (
    <div style={actionCardStyle}>
      <div style={{ fontSize: '12px', fontWeight: 700, color: '#ddd' }}>
        {action.description}
      </div>
      <div style={{ fontSize: '10px', color: '#888', marginTop: '2px' }}>
        Type: {action.type}
      </div>
      <pre style={actionDiffStyle}>{diff}</pre>
      <div style={actionBtnRowStyle}>
        <button style={rejectBtnStyle} onClick={onReject}>
          Reject
        </button>
        <button style={confirmBtnStyle} onClick={onConfirm}>
          Confirm
        </button>
      </div>
    </div>
  )
}
