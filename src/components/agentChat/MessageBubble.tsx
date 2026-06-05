import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import tsx from 'react-syntax-highlighter/dist/esm/languages/prism/tsx'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { AgentChatMessage, AgentToolCall } from '../../types/agentChat'
import { AgentActionCard } from './AgentActionCard'
import { userBubbleStyle, agentBubbleStyle, systemBubbleStyle, codeBlockStyle } from './agentChatStyles'

SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('tsx', tsx)
SyntaxHighlighter.registerLanguage('typescript', tsx)
SyntaxHighlighter.registerLanguage('jsx', tsx)
SyntaxHighlighter.registerLanguage('javascript', tsx)

interface MessageBubbleProps {
  message: AgentChatMessage
  pendingActions?: AgentToolCall[]
  onConfirmAction?: (actionId: string) => void
  onRejectAction?: (actionId: string) => void
}

export function MessageBubble({ message, pendingActions, onConfirmAction, onRejectAction }: MessageBubbleProps) {
  const bubbleStyle = useMemo(() => {
    switch (message.role) {
      case 'user': return userBubbleStyle
      case 'system': return systemBubbleStyle
      default: return agentBubbleStyle
    }
  }, [message.role])

  const isAgent = message.role === 'agent'

  // Extract node references like [node:nodeId] or `nodeId` from content
  const nodeReferencePattern = /\[node:([^\]]+)\]/g

  const renderContent = () => {
    if (message.role === 'user') {
      return <div>{message.content}</div>
    }

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            const lang = match ? match[1] : ''
            const codeString = String(children).replace(/\n$/, '')

            if (lang) {
              return (
                <SyntaxHighlighter
                  style={vscDarkPlus as any}
                  language={lang}
                  PreTag="div"
                  customStyle={codeBlockStyle}
                  {...props}
                >
                  {codeString}
                </SyntaxHighlighter>
              )
            }
            return (
              <code
                style={{
                  background: '#2a2a2a',
                  padding: '1px 4px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  color: '#c9d7e8',
                }}
                {...props}
              >
                {children}
              </code>
            )
          },
          pre({ children }: any) {
            return <pre style={{ margin: 0 }}>{children}</pre>
          },
        }}
      >
        {message.content.replace(nodeReferencePattern, (_, nodeId) => `\`${nodeId}\``)}
      </ReactMarkdown>
    )
  }

  return (
    <div style={bubbleStyle}>
      {renderContent()}
      {isAgent && pendingActions && pendingActions.length > 0 && onConfirmAction && onRejectAction && (
        <div style={{ marginTop: '8px' }}>
          {pendingActions.map((action) => (
            <AgentActionCard
              key={action.id}
              action={action}
              onConfirm={() => onConfirmAction(action.id)}
              onReject={() => onRejectAction(action.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
