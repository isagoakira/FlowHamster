import { useAgentChatStore } from '../../stores/useAgentChatStore'
import { toggleBtnStyle } from './agentChatStyles'

export function AgentChatToggle() {
  const { isOpen, toggleOpen } = useAgentChatStore()

  return (
    <button
      style={{
        ...toggleBtnStyle,
        transform: isOpen ? 'scale(0)' : 'scale(1)',
        opacity: isOpen ? 0 : 1,
        pointerEvents: isOpen ? 'none' : 'auto',
      }}
      onClick={toggleOpen}
      title="Open Agent Chat"
    >
      🤖
    </button>
  )
}

export default AgentChatToggle
