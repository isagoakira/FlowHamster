import { memo, useState, useEffect } from 'react'
import { useGraphStore } from '../../hooks/useGraphStore'
import { NODE_REGISTRY, CompositeNodeDefinition } from '../../utils/nodeRegistry'
import { CustomCompositeNodeData, FlowHamsterNode } from '../../types/graph'
import { parsePythonCode, ParsedModule, generateModuleTemplate } from '../../utils/pythonCodeParser'
import CompositeNodeViewer from './CompositeNodeViewer'

// 模块参数配置
const NODE_PARAM_CONFIGS: Record<string, Array<{ key: string; label: string; type: 'number' | 'boolean' | 'text'; default: any }>> = {
  conv2d: [
    { key: 'in_channels', label: 'in_channels', type: 'number', default: 3 },
    { key: 'out_channels', label: 'out_channels', type: 'number', default: 64 },
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 3 },
    { key: 'stride', label: 'stride', type: 'number', default: 1 },
    { key: 'padding', label: 'padding', type: 'number', default: 1 },
    { key: 'bias', label: 'bias', type: 'boolean', default: false },
  ],
  conv1d: [
    { key: 'in_channels', label: 'in_channels', type: 'number', default: 64 },
    { key: 'out_channels', label: 'out_channels', type: 'number', default: 128 },
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 3 },
    { key: 'stride', label: 'stride', type: 'number', default: 1 },
    { key: 'padding', label: 'padding', type: 'number', default: 1 },
    { key: 'bias', label: 'bias', type: 'boolean', default: false },
  ],
  conv3d: [
    { key: 'in_channels', label: 'in_channels', type: 'number', default: 3 },
    { key: 'out_channels', label: 'out_channels', type: 'number', default: 64 },
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 3 },
    { key: 'stride', label: 'stride', type: 'number', default: 1 },
    { key: 'padding', label: 'padding', type: 'number', default: 1 },
    { key: 'bias', label: 'bias', type: 'boolean', default: false },
  ],
  linear: [
    { key: 'in_features', label: 'in_features', type: 'number', default: 512 },
    { key: 'out_features', label: 'out_features', type: 'number', default: 256 },
    { key: 'bias', label: 'bias', type: 'boolean', default: true },
  ],
  embedding: [
    { key: 'num_embeddings', label: 'num_embeddings', type: 'number', default: 50000 },
    { key: 'embedding_dim', label: 'embedding_dim', type: 'number', default: 512 },
  ],
  dropout: [
    { key: 'p', label: 'p', type: 'number', default: 0.5 },
  ],
  mamba: [
    { key: 'd_model', label: 'd_model', type: 'number', default: 512 },
    { key: 'd_state', label: 'd_state', type: 'number', default: 16 },
    { key: 'd_conv', label: 'd_conv', type: 'number', default: 4 },
    { key: 'expand', label: 'expand', type: 'number', default: 2 },
  ],
  transformerencoder: [
    { key: 'embed_dim', label: 'embed_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
    { key: 'num_layers', label: 'num_layers', type: 'number', default: 6 },
    { key: 'dim_feedforward', label: 'dim_feedforward', type: 'number', default: 2048 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0.1 },
  ],
  transformerdecoder: [
    { key: 'embed_dim', label: 'embed_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
    { key: 'num_layers', label: 'num_layers', type: 'number', default: 6 },
    { key: 'dim_feedforward', label: 'dim_feedforward', type: 'number', default: 2048 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0.1 },
  ],
  selfattention: [
    { key: 'embed_dim', label: 'embed_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0.1 },
  ],
  crossattention: [
    { key: 'query_dim', label: 'query_dim', type: 'number', default: 512 },
    { key: 'kv_dim', label: 'kv_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0.1 },
  ],
  multiheadattention: [
    { key: 'embed_dim', label: 'embed_dim', type: 'number', default: 512 },
    { key: 'num_heads', label: 'num_heads', type: 'number', default: 8 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0.1 },
    { key: 'bias', label: 'bias', type: 'boolean', default: true },
  ],
  lstm: [
    { key: 'input_size', label: 'input_size', type: 'number', default: 512 },
    { key: 'hidden_size', label: 'hidden_size', type: 'number', default: 512 },
    { key: 'num_layers', label: 'num_layers', type: 'number', default: 2 },
  ],
  batchnorm2d: [
    { key: 'num_features', label: 'num_features', type: 'number', default: 64 },
  ],
  layernorm: [
    { key: 'normalized_shape', label: 'normalized_shape', type: 'number', default: 512 },
  ],
  groupnorm: [
    { key: 'num_groups', label: 'num_groups', type: 'number', default: 32 },
    { key: 'num_channels', label: 'num_channels', type: 'number', default: 64 },
  ],
  maxpool2d: [
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 2 },
    { key: 'stride', label: 'stride', type: 'number', default: 2 },
    { key: 'padding', label: 'padding', type: 'number', default: 0 },
  ],
  avgpool2d: [
    { key: 'kernel_size', label: 'kernel_size', type: 'number', default: 2 },
    { key: 'stride', label: 'stride', type: 'number', default: 2 },
    { key: 'padding', label: 'padding', type: 'number', default: 0 },
  ],
  adaptiveavgpool2d: [
    { key: 'output_size', label: 'output_size', type: 'number', default: 1 },
  ],
  leakyrelu: [
    { key: 'negative_slope', label: 'negative_slope', type: 'number', default: 0.01 },
  ],
  ffn: [
    { key: 'dim', label: 'dim', type: 'number', default: 512 },
    { key: 'hidden_dim', label: 'hidden_dim', type: 'number', default: 2048 },
    { key: 'dropout', label: 'dropout', type: 'number', default: 0 },
  ],
  mlp: [
    { key: 'in_features', label: 'in_features', type: 'number', default: 784 },
    { key: 'hidden_features', label: 'hidden_features', type: 'number', default: 256 },
    { key: 'out_features', label: 'out_features', type: 'number', default: 10 },
  ],
  custom: [
    { key: 'custom_code', label: 'custom_code', type: 'text', default: '' },
  ],
}

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 60,
  right: 10,
  width: 360,
  maxHeight: 'calc(100vh - 120px)',
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: 12,
  zIndex: 100,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  overflow: 'hidden',
}

const headerStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderBottom: '1px solid #333',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  background: '#222',
}

const sectionStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderBottom: '1px solid #2a2a2a',
}

const paramRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '8px 0',
  borderBottom: '1px solid #2a2a2a',
}

const inputStyle: React.CSSProperties = {
  background: '#111',
  border: '1px solid #333',
  borderRadius: 4,
  color: '#e0e0e0',
  padding: '4px 8px',
  fontSize: 12,
  width: 120,
  textAlign: 'right',
}

const checkboxStyle: React.CSSProperties = {
  width: 18,
  height: 18,
  accentColor: '#4488ff',
}

const codeTextareaStyle: React.CSSProperties = {
  width: '100%',
  minHeight: 200,
  background: '#111',
  border: '1px solid #333',
  borderRadius: 6,
  color: '#e0e0e0',
  padding: 8,
  fontSize: 11,
  fontFamily: '"Fira Code", monospace',
  resize: 'vertical',
}

const structItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '4px 0',
  fontSize: 11,
  color: '#888',
}

const structTagStyle: React.CSSProperties = {
  background: '#333',
  borderRadius: 3,
  padding: '1px 6px',
  fontSize: 10,
  color: '#a0c0ff',
}

interface NodeEditPanelProps {
  node: FlowHamsterNode
  onClose: () => void
}

export default memo(function NodeEditPanel({ node, onClose }: NodeEditPanelProps) {
  const updateNodeData = useGraphStore((s) => s.updateNodeData)
  const updateNodeLabel = useGraphStore((s) => s.updateNodeLabel)
  const renamePackageClass = useGraphStore((s) => s.renamePackageClass)
  const [label, setLabel] = useState(node.data.label)
  const [className, setClassName] = useState((node.data as CustomCompositeNodeData).customClassId ?? '')
  const [customCode, setCustomCode] = useState('')
  const [showCode, setShowCode] = useState(false)
  const [showImport, setShowImport] = useState(false)
  const [parsedModule, setParsedModule] = useState<ParsedModule | null>(null)
  const [importError, setImportError] = useState<string | null>(null)
  const [showCompositeViewer, setShowCompositeViewer] = useState(false)

  const nodeType = node.data.nodeType
  const params = node.data.params || {}
  const paramConfig = NODE_PARAM_CONFIGS[nodeType] || []
  const isPackagedCustom = Boolean((node.data as CustomCompositeNodeData).isCustomComposite)
  const isCustomModule = nodeType === 'custom' && !isPackagedCustom

  // Find composite node definition if this is a composite node
  const compositeDef = NODE_REGISTRY.flatMap(cat => cat.nodes).find(
    n => n.type === nodeType && 'isComposite' in n && (n as CompositeNodeDefinition).isComposite
  ) as CompositeNodeDefinition | undefined

  const isCompositeNode = !!compositeDef

  useEffect(() => {
    setLabel(node.data.label)
  }, [node.data.label])

  useEffect(() => {
    setClassName((node.data as CustomCompositeNodeData).customClassId ?? '')
  }, [node.data])

  useEffect(() => {
    if (isCustomModule && params.custom_code) {
      setCustomCode(params.custom_code as string)
      handleParseCode(params.custom_code as string)
    }
  }, [isCustomModule])

  const handleParamChange = (key: string, value: any) => {
    updateNodeData(node.id, {
      params: { ...params, [key]: value }
    })
  }

  const handleLabelChange = () => {
    if (label.trim() && label !== node.data.label) {
      updateNodeLabel(node.id, label.trim())
    }
  }

  const handleClassNameChange = () => {
    const currentClassName = (node.data as CustomCompositeNodeData).customClassId
    if (className.trim() && currentClassName && className.trim() !== currentClassName) {
      renamePackageClass(node.id, className.trim())
    }
  }

  const handleParseCode = (code: string) => {
    setImportError(null)
    const parsed = parsePythonCode(code)
    if (parsed) {
      setParsedModule(parsed)
      // Update node params with parsed parameters
      const parsedParams: Record<string, any> = {}
      parsed.parameters.forEach(p => {
        if (p.default) {
          // Try to convert default value to appropriate type
          if (p.default === 'None') parsedParams[p.name] = null
          else if (p.default === 'True') parsedParams[p.name] = true
          else if (p.default === 'False') parsedParams[p.name] = false
          else if (/^\d+\.\d+$/.test(p.default)) parsedParams[p.name] = parseFloat(p.default)
          else if (/^\d+$/.test(p.default)) parsedParams[p.name] = parseInt(p.default)
          else parsedParams[p.name] = p.default
        } else {
          parsedParams[p.name] = null
        }
      })
      updateNodeData(node.id, {
        params: { ...params, custom_code: code, ...parsedParams }
      })
    } else {
      setImportError('Could not parse Python code. Make sure it contains a valid class definition.')
    }
  }

  const handleImportCode = () => {
    if (customCode.trim()) {
      handleParseCode(customCode)
    }
  }

  const handleGenerateTemplate = () => {
    const template = generateModuleTemplate('CustomModule', [
      { name: 'in_features', type: 'int', default: '512' },
      { name: 'out_features', type: 'int', default: '256' },
    ])
    setCustomCode(template)
    handleParseCode(template)
  }

  // 获取模块的类别颜色
  const getNodeColor = (type: string): string => {
    const colors: Record<string, string> = {
      conv2d: '#4488ff',
      conv1d: '#44aaff',
      conv3d: '#aa44ff',
      linear: '#88cc44',
      dropout: '#ff8844',
      mamba: '#ff44aa',
      transformerencoder: '#44ffaa',
      transformerdecoder: '#44ffaa',
      custom: '#ff66ff',
    }
    return colors[type] || '#a0c0ff'
  }

  const color = getNodeColor(nodeType)

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 12, height: 12, borderRadius: 3, background: color }} />
          <span style={{ fontWeight: 700, fontSize: 13, color: '#e0e0e0' }}>
            {nodeType.toUpperCase()}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#666',
            cursor: 'pointer',
            fontSize: 18,
            padding: '0 4px',
          }}
        >
          ×
        </button>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
	        {/* 名称编辑 */}
	        <div style={sectionStyle}>
	          <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 6, textTransform: 'uppercase' }}>
	            Instance Name
	          </label>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onBlur={handleLabelChange}
            onKeyDown={(e) => e.key === 'Enter' && handleLabelChange()}
            style={{
              ...inputStyle,
              width: '100%',
              textAlign: 'left',
            }}
	          />
	        </div>

	        {isPackagedCustom && (
	          <div style={sectionStyle}>
	            <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 6, textTransform: 'uppercase' }}>
	              Class Name
	            </label>
	            <input
	              value={className}
	              onChange={(e) => setClassName(e.target.value)}
	              onBlur={handleClassNameChange}
	              onKeyDown={(e) => e.key === 'Enter' && handleClassNameChange()}
	              style={{
	                ...inputStyle,
	                width: '100%',
	                textAlign: 'left',
	              }}
	            />
	          </div>
	        )}

        {/* 自定义模块导入区域 */}
        {isCustomModule && (
          <div style={sectionStyle}>
            <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 8, textTransform: 'uppercase' }}>
              <span
                style={{ cursor: 'pointer', color: '#4488ff' }}
                onClick={() => setShowImport(!showImport)}
              >
                {showImport ? '▼' : '▶'} Import Custom Module
              </span>
            </label>
            {showImport && (
              <div>
                <div style={{ marginBottom: 8 }}>
                  <button
                    onClick={handleGenerateTemplate}
                    style={{
                      background: '#333',
                      border: '1px solid #555',
                      borderRadius: 4,
                      color: '#ccc',
                      padding: '4px 12px',
                      fontSize: 11,
                      cursor: 'pointer',
                      marginRight: 8,
                    }}
                  >
                    Generate Template
                  </button>
                  <button
                    onClick={handleImportCode}
                    style={{
                      background: '#4488ff',
                      border: 'none',
                      borderRadius: 4,
                      color: '#fff',
                      padding: '4px 12px',
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    Parse Code
                  </button>
                </div>
                <textarea
                  value={customCode}
                  onChange={(e) => setCustomCode(e.target.value)}
                  placeholder={"class CustomModule(nn.Module):\n  def __init__(self):\n    ...\n  def forward(self, x):\n    ..."}
                  style={codeTextareaStyle}
                />
                {importError && (
                  <div style={{ color: '#ff6666', fontSize: 10, marginTop: 4 }}>
                    {importError}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* 解析后的模块结构 */}
        {isCustomModule && parsedModule && (
          <div style={sectionStyle}>
            <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 8, textTransform: 'uppercase' }}>
              Module Structure
            </label>
            <div style={{ marginBottom: 8 }}>
              <div style={{ ...structItemStyle, color: '#a0c0ff', fontWeight: 600 }}>
                <span style={structTagStyle}>CLASS</span>
                {parsedModule.className}
              </div>
            </div>

            {parsedModule.parameters.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>Parameters:</div>
                {parsedModule.parameters.map((p, i) => (
                  <div key={i} style={structItemStyle}>
                    <span style={{ color: '#ffcc44' }}>{p.name}</span>
                    <span style={{ color: '#666' }}>: {p.type}</span>
                    {p.default && <span style={{ color: '#666' }}> = {p.default}</span>}
                  </div>
                ))}
              </div>
            )}

            {parsedModule.forwardInputs.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>Forward Inputs:</div>
                {parsedModule.forwardInputs.map((input, i) => (
                  <div key={i} style={structItemStyle}>
                    <span style={{ color: '#44ff88' }}>{input}</span>
                  </div>
                ))}
              </div>
            )}

            {parsedModule.forwardOutputs.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>Forward Output:</div>
                {parsedModule.forwardOutputs.map((output, i) => (
                  <div key={i} style={structItemStyle}>
                    <span style={{ color: '#ff88ff' }}>{output}</span>
                  </div>
                ))}
              </div>
            )}

            {parsedModule.submodules.length > 0 && (
              <div>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>Sub-modules:</div>
                {parsedModule.submodules.map((sm, i) => (
                  <div key={i} style={structItemStyle}>
                    <span style={structTagStyle}>self.{sm.name}</span>
                    <span style={{ color: '#888' }}>= {sm.type}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* 参数编辑 */}
        {paramConfig.length > 0 && !isCustomModule && (
          <div style={sectionStyle}>
            <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 8, textTransform: 'uppercase' }}>
              Parameters
            </label>
            {paramConfig.map((config) => (
              <div key={config.key} style={paramRowStyle}>
                <span style={{ fontSize: 12, color: '#888' }}>{config.label}</span>
                {config.type === 'boolean' ? (
                  <input
                    type="checkbox"
                    checked={!!params[config.key]}
                    onChange={(e) => handleParamChange(config.key, e.target.checked)}
                    style={checkboxStyle}
                  />
                ) : config.type === 'text' ? (
                  <textarea
                    value={String(params[config.key] ?? '')}
                    onChange={(e) => handleParamChange(config.key, e.target.value)}
                    style={{
                      ...codeTextareaStyle,
                      minHeight: 60,
                      width: 100,
                    }}
                  />
                ) : (
                  <input
                    type="number"
                    value={params[config.key] as number ?? config.default}
                    onChange={(e) => {
                      const v = e.target.value
                      handleParamChange(config.key, v === '' ? config.default : Number(v))
                    }}
                    style={inputStyle}
                  />
                )}
              </div>
            ))}
          </div>
        )}

        {/* Mamba 自定义代码区域 */}
        {nodeType === 'mamba' && (
          <div style={sectionStyle}>
            <label style={{ display: 'block', fontSize: 11, color: '#666', marginBottom: 8, textTransform: 'uppercase' }}>
              <span
                style={{ cursor: 'pointer', color: '#4488ff' }}
                onClick={() => setShowCode(!showCode)}
              >
                {showCode ? '▼' : '▶'} Custom Code
              </span>
            </label>
            {showCode && (
              <textarea
                value={customCode}
                onChange={(e) => setCustomCode(e.target.value)}
                placeholder="# Enter custom module code..."
                style={codeTextareaStyle}
              />
            )}
          </div>
        )}

        {/* 复合节点展开按钮 */}
        {isCompositeNode && compositeDef && (
          <div style={sectionStyle}>
            <button
              onClick={() => setShowCompositeViewer(true)}
              style={{
                width: '100%',
                padding: '10px 16px',
                background: '#2a4a7a',
                border: '1px solid #4488ff',
                borderRadius: 6,
                color: '#a0c0ff',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
              }}
            >
              🔍 展开内部结构
            </button>
          </div>
        )}

        {/* 模块信息 */}
        <div style={{ ...sectionStyle, borderBottom: 'none' }}>
          <div style={{ fontSize: 10, color: '#555', lineHeight: 1.8 }}>
            <div><span style={{ color: '#666' }}>ID:</span> {node.id}</div>
            <div><span style={{ color: '#666' }}>Type:</span> {nodeType}</div>
            <div><span style={{ color: '#666' }}>Output:</span> {NODE_REGISTRY.flatMap(cat => cat.nodes).find(n => n.type === nodeType)?.outputType || 'Tensor'}</div>
          </div>
        </div>
      </div>

      {/* 复合节点查看器弹窗 */}
      {isCompositeNode && compositeDef && (
        <CompositeNodeViewer
          isOpen={showCompositeViewer}
          nodeId={node.id}
          onClose={() => setShowCompositeViewer(false)}
          nodeType={nodeType}
          nodeLabel={node.data.label}
          subModules={compositeDef.internalStructure}
          internalEdges={compositeDef.internalEdges}
          outputVar={compositeDef.outputVar}
        />
      )}
    </div>
  )
})
