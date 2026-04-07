import { useState } from 'react'
import { useDataGraphStore } from '../../hooks/useDataGraphStore'
import { useGraphStore } from '../../hooks/useGraphStore'
import { compileDataWorkflow } from '../../utils/dataWorkflowCompiler'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

const panelStyle: React.CSSProperties = {
  width: '380px',
  background: '#0d0d0d',
  borderLeft: '1px solid #222',
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
}

const headerStyle: React.CSSProperties = {
  padding: '8px 12px',
  borderBottom: '1px solid #222',
  fontSize: '11px',
  fontWeight: 700,
  color: '#a0c0ff',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  flexShrink: 0,
}

const tabStyle: React.CSSProperties = {
  display: 'flex',
  borderBottom: '1px solid #222',
  flexShrink: 0,
}

const tabItemStyle = (active: boolean): React.CSSProperties => ({
  flex: 1,
  padding: '8px 12px',
  fontSize: '11px',
  fontWeight: 600,
  color: active ? '#a0c0ff' : '#555',
  background: active ? '#0d0d0d' : 'transparent',
  borderBottom: active ? '2px solid #4488ff' : '2px solid transparent',
  cursor: 'pointer',
  textAlign: 'center',
  transition: 'all 0.15s',
})

const contentStyle: React.CSSProperties = {
  flex: 1,
  overflow: 'auto',
  display: 'flex',
  flexDirection: 'column',
}

const summaryStyle: React.CSSProperties = {
  padding: '12px',
  fontSize: '11px',
  color: '#888',
  lineHeight: 1.7,
}

export default function DataPreview() {
  const [activeTab, setActiveTab] = useState<'code' | 'summary' | 'bindings'>('code')
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)
  const modelNodes = useGraphStore((s) => s.nodes)
  const bindings = useGraphStore((s) => s.bindings)
  const trainingConfig = useGraphStore((s) => s.trainingConfig)
  const compiled = compileDataWorkflow(modelNodes as any, nodes as any, edges as any, bindings, trainingConfig)

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>Data Pipeline</div>

      {/* Tab Switcher */}
      <div style={tabStyle}>
        <div style={tabItemStyle(activeTab === 'code')} onClick={() => setActiveTab('code')}>
          Python Code
        </div>
        <div style={tabItemStyle(activeTab === 'summary')} onClick={() => setActiveTab('summary')}>
          Summary
        </div>
        <div style={tabItemStyle(activeTab === 'bindings')} onClick={() => setActiveTab('bindings')}>
          Bindings
        </div>
      </div>

      <div style={contentStyle}>
        {activeTab === 'code' && (
          <>
            {compiled.pythonScaffold ? (
              <>
                <div style={{ padding: '6px 12px', background: '#151515', borderBottom: '1px solid #222', fontSize: '10px', color: '#666' }}>
                  Generated data pipeline code • {compiled.hasWorkflowRuntime ? 'Runtime enabled' : 'No runtime needed'}
                </div>
                <SyntaxHighlighter
                  language="python"
                  style={vscDarkPlus}
                  customStyle={{
                    flex: 1,
                    margin: 0,
                    padding: '12px',
                    fontSize: '11px',
                    lineHeight: 1.5,
                    backgroundColor: '#0d0d0d',
                    overflow: 'auto',
                  }}
                  codeTagProps={{
                    style: {
                      fontFamily: '"Fira Code", "Cascadia Code", Consolas, monospace',
                    }
                  }}
                  showLineNumbers
                  lineNumberStyle={{
                    color: '#444',
                    fontSize: '10px',
                    minWidth: '2em',
                    paddingRight: '12px',
                    userSelect: 'none',
                  }}
                >
                  {compiled.pythonScaffold || '# No data pipeline defined'}
                </SyntaxHighlighter>
              </>
            ) : (
              <div style={{ padding: '12px', fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '40px' }}>
                <div style={{ fontSize: '24px', marginBottom: '8px' }}>📦</div>
                No data pipeline nodes yet.<br />
                Add nodes from the sidebar to generate code.
              </div>
            )}
          </>
        )}

        {activeTab === 'summary' && (
          <div style={summaryStyle}>
            <div style={{ marginBottom: '8px', display: 'flex', gap: '16px' }}>
              <div>Nodes: <span style={{ color: '#a0c0ff' }}>{nodes.length}</span></div>
              <div>Edges: <span style={{ color: '#a0c0ff' }}>{edges.length}</span></div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              Output Fields: <span style={{ color: '#88cc88' }}>{compiled.outputFields.length}</span>
            </div>

            {compiled.outputFields.length > 0 && (
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#888', marginBottom: '6px', fontWeight: 600 }}>Output Contracts</div>
                {compiled.outputFields.map((field) => (
                  <div key={field} style={{ padding: '6px 8px', background: '#151515', border: '1px solid #222', borderRadius: '4px', marginBottom: '4px', color: '#c0d8ff', fontSize: '10px' }}>
                    {field}
                  </div>
                ))}
              </div>
            )}

            {compiled.summaryLines.length > 0 && (
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#888', marginBottom: '6px', fontWeight: 600 }}>Compilation Order</div>
                {compiled.summaryLines.map((line, i) => (
                  <div key={i} style={{ color: '#666', fontSize: '10px', marginBottom: '2px', fontFamily: 'monospace' }}>
                    {line}
                  </div>
                ))}
              </div>
            )}

            {compiled.warnings.length > 0 && (
              <div style={{ padding: '8px', background: '#1b1610', border: '1px solid #4c3920', borderRadius: '6px' }}>
                <div style={{ color: '#f0c27a', fontWeight: 600, marginBottom: '6px', fontSize: '11px' }}>⚠ Warnings</div>
                {compiled.warnings.map((warning, i) => (
                  <div key={i} style={{ color: '#c9a16a', fontSize: '10px', marginBottom: '2px' }}>• {warning}</div>
                ))}
              </div>
            )}

            {!compiled.hasWorkflowRuntime && nodes.length === 0 && (
              <div style={{ marginTop: '20px', color: '#555', fontSize: '10px', textAlign: 'center' }}>
                Data graph is empty. Drag nodes from the sidebar to build a data pipeline.
              </div>
            )}
          </div>
        )}

        {activeTab === 'bindings' && (
          <div style={summaryStyle}>
            <div style={{ marginBottom: '12px' }}>
              <div style={{ color: '#888', marginBottom: '6px', fontWeight: 600, fontSize: '11px' }}>Model Input Bindings</div>
              {compiled.modelInputBindings.length === 0 ? (
                <div style={{ color: '#555', fontSize: '10px' }}>No model input bindings configured.</div>
              ) : (
                compiled.modelInputBindings.map((binding, i) => (
                  <div key={i} style={{ padding: '6px 8px', background: '#151515', border: '1px solid #222', borderRadius: '4px', marginBottom: '4px', fontSize: '10px' }}>
                    <span style={{ color: '#88cc88' }}>{binding.sourceKey}</span>
                    <span style={{ color: '#555' }}> → </span>
                    <span style={{ color: '#a0c0ff' }}>{binding.targetKey}</span>
                  </div>
                ))
              )}
            </div>

            {compiled.targetBindingSource && (
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#888', marginBottom: '6px', fontWeight: 600, fontSize: '11px' }}>Target Binding</div>
                <div style={{ padding: '6px 8px', background: '#151515', border: '1px solid #222', borderRadius: '4px', fontSize: '10px' }}>
                  <span style={{ color: '#88cc88' }}>{compiled.targetBindingSource}</span>
                  <span style={{ color: '#555' }}> → </span>
                  <span style={{ color: '#ffaa44' }}>training_target</span>
                </div>
              </div>
            )}

            {compiled.primaryModelInputKey && (
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#888', marginBottom: '6px', fontWeight: 600, fontSize: '11px' }}>Primary Input</div>
                <div style={{ padding: '6px 8px', background: '#151515', border: '1px solid #222', borderRadius: '4px', fontSize: '10px', color: '#a0c0ff' }}>
                  {compiled.primaryModelInputKey}
                </div>
              </div>
            )}

            {bindings.length === 0 && !compiled.primaryModelInputKey && (
              <div style={{ color: '#555', fontSize: '10px', textAlign: 'center', marginTop: '20px' }}>
                No bindings configured. Connect data nodes to model inputs using the Bindings panel.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
