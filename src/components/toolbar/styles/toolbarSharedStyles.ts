/**
 * Shared style constants for Toolbar components
 */

export const toolbarStyle: React.CSSProperties = {
  height: '44px',
  background: '#111',
  borderBottom: '1px solid #222',
  display: 'flex',
  alignItems: 'center',
  padding: '0 16px',
  gap: '8px',
  flexShrink: 0,
}

export const logoStyle: React.CSSProperties = {
  fontSize: '15px',
  fontWeight: 700,
  color: '#a0c0ff',
  letterSpacing: '1px',
}

export const btnStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '6px',
  color: '#ccc',
  padding: '4px 12px',
  fontSize: '12px',
  cursor: 'pointer',
}

export const dangerBtn: React.CSSProperties = {
  ...btnStyle,
  background: '#2a1a1a',
  borderColor: '#663333',
  color: '#cc8888',
}

export const panelContainerStyle: React.CSSProperties = {
  position: 'absolute',
  top: '44px',
  right: '16px',
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '10px',
  padding: '16px',
  zIndex: 1000,
  minWidth: '320px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
}

export const trainingPanelStyle: React.CSSProperties = {
  position: 'absolute',
  top: '44px',
  right: '16px',
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '10px',
  padding: '16px',
  zIndex: 1000,
  minWidth: '360px',
  maxHeight: 'calc(100vh - 60px)',
  overflowY: 'auto',
  boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
}

export const fieldGroupStyle: React.CSSProperties = {
  marginBottom: '12px',
}

export const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '11px',
  color: '#888',
  marginBottom: '4px',
}

export const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#111',
  border: '1px solid #333',
  borderRadius: '6px',
  color: '#ddd',
  padding: '6px 8px',
  fontSize: '12px',
  boxSizing: 'border-box',
}

export const codeTextareaStyle: React.CSSProperties = {
  ...inputStyle,
  fontFamily: 'Monaco, Consolas, "Courier New", monospace',
  fontSize: '11px',
  minHeight: '80px',
  resize: 'vertical' as const,
  lineHeight: 1.4,
}
