// Shared node styles for FlowHamster
// All model/data/training nodes should import from here for consistent styling

export const NODE_COLORS = {
  // Category-based colors for node header/border accents
  layer: '#a0c0ff',       // Conv2d, Linear, LSTM - blue
  activation: '#88dd88',  // ReLU, Sigmoid, Tanh - green
  normalization: '#dd88dd', // BatchNorm, LayerNorm - purple
  input: '#6688cc',       // Input node - blue
  output: '#ff8844',      // Output node - orange
  loss: '#ff6666',       // CrossEntropyLoss, MSELoss - red
  optimizer: '#44cc44',   // SGD, Adam, AdamW - green
  scheduler: '#ccaa44',   // CosineAnnealing, StepLR - yellow
  data: '#44ccff',       // Data nodes - cyan
  utility: '#aaaaaa',    // Reshape, Concat, etc - gray
  attention: '#dd88ff',  // Attention nodes - violet
  evaluation: '#ffaa44', // Evaluation - orange
  custom: '#44ffcc',     // Custom composite - teal
} as const

// Base node styles shared by all node types
export const BASE_NODE_STYLE: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '8px',
  padding: '10px 14px',
  minWidth: '140px',
  color: '#e0e0e0',
  fontSize: '13px',
  position: 'relative',
  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
  transition: 'border-color 0.15s, box-shadow 0.15s',
}

// Hover state style (to be merged with base style)
export const HOVER_STYLE: React.CSSProperties = {
  boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
}

// Delete button style (shared across all nodes)
export const DELETE_BUTTON_STYLE: React.CSSProperties = {
  position: 'absolute',
  top: 4,
  right: 4,
  background: '#ff4444',
  border: 'none',
  borderRadius: '50%',
  width: 18,
  height: 18,
  cursor: 'pointer',
  color: '#fff',
  fontSize: 11,
  lineHeight: 1,
  padding: 0,
}

// Handle styles
export const HANDLE_TARGET_STYLE: React.CSSProperties = {
  background: '#4488ff',
  width: 8,
  height: 8,
  border: 'none',
}

export const HANDLE_SOURCE_STYLE: React.CSSProperties = {
  background: '#44cc88',
  width: 8,
  height: 8,
  border: 'none',
}

// Common input styles for param inputs
export const PARAM_INPUT_STYLE: React.CSSProperties = {
  background: '#111',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#e0e0e0',
  padding: '2px 6px',
  fontSize: '12px',
  width: '72px',
  textAlign: 'right' as const,
}

// Common row styles for param layout
export const PARAM_ROW_STYLE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: '4px',
  gap: '8px',
}

// Common label styles
export const PARAM_LABEL_STYLE: React.CSSProperties = {
  color: '#888',
  fontSize: '11px',
  flex: '0 0 54px',
}
