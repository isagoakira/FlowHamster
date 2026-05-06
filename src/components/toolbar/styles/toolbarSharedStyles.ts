/**
 * Shared style constants for Toolbar components
 */

// Main toolbar row (upper) — editing operations
export const toolbarStyle: React.CSSProperties = {
  height: '48px',
  background: '#111',
  borderBottom: '1px solid #222',
  display: 'flex',
  alignItems: 'center',
  padding: '0 16px',
  gap: '4px',
  flexShrink: 0,
}

// Secondary toolbar row (lower) — panel/open and export actions
export const toolbarSecondaryStyle: React.CSSProperties = {
  height: '40px',
  background: '#0f0f0f',
  borderBottom: '1px solid #1e1e1e',
  display: 'flex',
  alignItems: 'center',
  padding: '0 16px',
  gap: '4px',
  flexShrink: 0,
}

// Vertical separator between functional groups
export const groupSeparatorStyle: React.CSSProperties = {
  width: '1px',
  height: '22px',
  background: '#2a2a2a',
  margin: '0 8px',
  flexShrink: 0,
}

export const logoStyle: React.CSSProperties = {
  fontSize: '15px',
  fontWeight: 700,
  color: '#a0c0ff',
  letterSpacing: '1px',
}

// Icon button — compact, for upper toolbar high-frequency actions
export const iconBtnStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #2e2e2e',
  borderRadius: '6px',
  color: '#aaa',
  padding: '5px 8px',
  fontSize: '13px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

// Text button — for secondary toolbar panel/export actions
export const textBtnStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #2e2e2e',
  borderRadius: '6px',
  color: '#ccc',
  padding: '4px 10px',
  fontSize: '12px',
  cursor: 'pointer',
}

// Active panel button state
export const activeBtnStyle: React.CSSProperties = {
  background: '#1a2a3a',
  border: '1px solid #3355aa',
  borderRadius: '6px',
  color: '#88aaff',
  padding: '4px 10px',
  fontSize: '12px',
  cursor: 'pointer',
}

// Settings gear — icon only
export const settingsBtnStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #2e2e2e',
  borderRadius: '6px',
  color: '#555',
  padding: '4px 7px',
  fontSize: '14px',
  cursor: 'pointer',
}

// Legacy shared button style
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
  borderColor: '#5a2a2a',
  color: '#b07070',
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