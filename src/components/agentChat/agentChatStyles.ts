export const chatPanelStyle: React.CSSProperties = {
  position: 'fixed',
  right: '16px',
  bottom: '16px',
  width: '380px',
  height: '560px',
  background: '#121212',
  border: '1px solid #2a2a2a',
  borderRadius: '12px',
  display: 'flex',
  flexDirection: 'column',
  zIndex: 1200,
  boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
  overflow: 'hidden',
}

export const chatHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '12px 14px',
  borderBottom: '1px solid #222',
  background: '#1a1a1a',
}

export const chatTitleStyle: React.CSSProperties = {
  fontSize: '13px',
  fontWeight: 700,
  color: '#a0c0ff',
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
}

export const chatMessagesStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '12px',
  display: 'flex',
  flexDirection: 'column',
  gap: '10px',
}

export const chatInputAreaStyle: React.CSSProperties = {
  padding: '10px 12px',
  borderTop: '1px solid #222',
  background: '#1a1a1a',
  display: 'flex',
  gap: '8px',
}

export const chatInputStyle: React.CSSProperties = {
  flex: 1,
  background: '#0e0e0e',
  border: '1px solid #333',
  borderRadius: '8px',
  padding: '8px 10px',
  color: '#ddd',
  fontSize: '12px',
  outline: 'none',
  resize: 'none',
  minHeight: '36px',
  maxHeight: '100px',
  fontFamily: 'inherit',
}

export const chatSendBtnStyle: React.CSSProperties = {
  background: '#2a4a8a',
  border: '1px solid #3a5aaa',
  borderRadius: '8px',
  color: '#fff',
  fontSize: '12px',
  fontWeight: 700,
  padding: '0 14px',
  cursor: 'pointer',
  transition: 'background 0.15s',
}

export const messageBubbleBaseStyle: React.CSSProperties = {
  maxWidth: '90%',
  padding: '10px 12px',
  borderRadius: '10px',
  fontSize: '12px',
  lineHeight: 1.5,
  wordBreak: 'break-word',
}

export const userBubbleStyle: React.CSSProperties = {
  ...messageBubbleBaseStyle,
  alignSelf: 'flex-end',
  background: '#1e3a5f',
  color: '#cce0ff',
  borderBottomRightRadius: '2px',
}

export const agentBubbleStyle: React.CSSProperties = {
  ...messageBubbleBaseStyle,
  alignSelf: 'flex-start',
  background: '#1e1e1e',
  color: '#ddd',
  border: '1px solid #2a2a2a',
  borderBottomLeftRadius: '2px',
}

export const systemBubbleStyle: React.CSSProperties = {
  ...messageBubbleBaseStyle,
  alignSelf: 'center',
  background: 'transparent',
  color: '#666',
  fontSize: '11px',
  fontStyle: 'italic',
  padding: '4px 8px',
}

export const actionCardStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #444',
  borderRadius: '8px',
  padding: '10px',
  marginTop: '8px',
}

export const actionDiffStyle: React.CSSProperties = {
  background: '#0e1117',
  border: '1px solid #273244',
  borderRadius: '6px',
  padding: '8px',
  fontSize: '11px',
  fontFamily: 'monospace',
  color: '#c9d7e8',
  lineHeight: 1.4,
  whiteSpace: 'pre-wrap',
  maxHeight: '160px',
  overflow: 'auto',
  marginTop: '6px',
  marginBottom: '8px',
}

export const actionBtnRowStyle: React.CSSProperties = {
  display: 'flex',
  gap: '8px',
  justifyContent: 'flex-end',
}

export const confirmBtnStyle: React.CSSProperties = {
  background: '#2f7a5c',
  border: '1px solid #3a9a6c',
  borderRadius: '6px',
  color: '#fff',
  fontSize: '11px',
  fontWeight: 700,
  padding: '5px 12px',
  cursor: 'pointer',
}

export const rejectBtnStyle: React.CSSProperties = {
  background: '#5a2a2a',
  border: '1px solid #7a3a3a',
  borderRadius: '6px',
  color: '#ffaaaa',
  fontSize: '11px',
  fontWeight: 700,
  padding: '5px 12px',
  cursor: 'pointer',
}

export const toggleBtnStyle: React.CSSProperties = {
  position: 'fixed',
  right: '20px',
  bottom: '20px',
  width: '48px',
  height: '48px',
  borderRadius: '50%',
  background: '#2a4a8a',
  border: '1px solid #3a5aaa',
  color: '#fff',
  fontSize: '20px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  zIndex: 1200,
  boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
  transition: 'transform 0.15s, background 0.15s',
}

export const codeBlockStyle: React.CSSProperties = {
  background: '#0e1117',
  border: '1px solid #273244',
  borderRadius: '6px',
  padding: '8px 10px',
  fontSize: '11px',
  fontFamily: 'monospace',
  color: '#c9d7e8',
  lineHeight: 1.4,
  overflowX: 'auto',
  marginTop: '6px',
  marginBottom: '6px',
}

export const nodeRefCardStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
  background: '#1e2a3a',
  border: '1px solid #3355aa44',
  borderRadius: '4px',
  padding: '2px 6px',
  fontSize: '11px',
  color: '#88aaff',
  cursor: 'pointer',
  margin: '0 2px',
}
