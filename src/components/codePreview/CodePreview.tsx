import { useState, useEffect, useRef, useCallback } from 'react'
import { useGraphStore } from '../../hooks/useGraphStore'
import { useDataGraphStore } from '../../hooks/useDataGraphStore'
import { getCachedOrGenerateCode } from '../../utils/cachedCodeGenerator'
import { useWebSocketCode } from '../../hooks/useWebSocketCode'
import { API_BASE_URL, HEALTH_URL } from '../../utils/runtimeConfig'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodePreviewProps {
  width?: number
  onWidthChange?: (width: number) => void
}

const MIN_WIDTH = 200
const MAX_WIDTH = 800

export default function CodePreview({ width = 320, onWidthChange }: CodePreviewProps) {
  const [panelWidth, setPanelWidth] = useState(width)
  const [isResizing, setIsResizing] = useState(false)
  const [wsEnabled, setWsEnabled] = useState(false)
  const [executing, setExecuting] = useState(false)
  const [output, setOutput] = useState('')
  const [error, setError] = useState('')
  const [copySuccess, setCopySuccess] = useState('')
  const backendAvailable = useRef(false)
  const [localCode, setLocalCode] = useState('')

  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const features = useGraphStore((s) => s.features)
  const trainingConfig = useGraphStore((s) => s.trainingConfig)
  const bindings = useGraphStore((s) => s.bindings)
  const dataNodes = useDataGraphStore((s) => s.nodes)
  const dataEdges = useDataGraphStore((s) => s.edges)

  const { connected, code: wsCode, setCode: setWsCode, connect, disconnect } = useWebSocketCode()

  // 拖拽调整宽度
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = window.innerWidth - e.clientX
      const clampedWidth = Math.min(Math.max(newWidth, MIN_WIDTH), MAX_WIDTH)
      setPanelWidth(clampedWidth)
      onWidthChange?.(clampedWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, onWidthChange])

  const panelStyle: React.CSSProperties = {
    width: `${panelWidth}px`,
    background: '#0d0d0d',
    borderLeft: '1px solid #222',
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
    position: 'relative',
    cursor: isResizing ? 'col-resize' : 'auto',
  }

  const headerStyle: React.CSSProperties = {
    padding: '8px 12px',
    borderBottom: '1px solid #222',
    fontSize: '11px',
    fontWeight: 700,
    color: '#a0c0ff',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexShrink: 0,
  }

  const resizeHandleStyle: React.CSSProperties = {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 4,
    cursor: 'col-resize',
    background: isResizing ? '#4488ff' : 'transparent',
    transition: 'background 0.15s',
    zIndex: 10,
  }

  // 开启/关闭 WS
  useEffect(() => {
    if (wsEnabled) {
      connect()
    } else {
      disconnect()
      setWsCode('')
    }
  }, [wsEnabled])

  // Try backend on mount
  useEffect(() => {
    fetch(HEALTH_URL)
      .then((r) => r.ok)
      .then((ok) => { backendAvailable.current = ok })
      .catch(() => { backendAvailable.current = false })
  }, [])

  // Generate local code when nodes change
  useEffect(() => {
    try {
      if (nodes.length === 0) {
        setLocalCode('# Empty graph — drag nodes from the sidebar')
        return
      }
      const code = getCachedOrGenerateCode(nodes as any, edges as any, features as any, trainingConfig as any, {
        dataGraphNodes: dataNodes as any,
        dataGraphEdges: dataEdges as any,
        bindings,
      })
      setLocalCode(code)
    } catch (e: any) {
      console.error('Code generation failed:', e)
      setLocalCode(`# Code generation failed\n# ${e?.message || 'Unknown error'}`)
    }
  }, [nodes, edges, features, trainingConfig, dataNodes, dataEdges, bindings])

  const displayCode = wsEnabled && wsCode ? wsCode : localCode

  const handleRun = async () => {
    setExecuting(true)
    setOutput('')
    setError('')
    try {
      const res = await fetch(`${API_BASE_URL}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: displayCode, target_device: 'cpu' }),
      })
      if (!res.ok) {
        setError(`HTTP ${res.status}: ${res.statusText}`)
        setExecuting(false)
        return
      }
      let data
      try {
        data = await res.json()
      } catch {
        setError('Invalid JSON response from backend')
        setExecuting(false)
        return
      }
      if (data.success) setOutput(data.output ?? '')
      else setError(data.error || 'Unknown error')
    } catch (e: any) {
      setError('Backend not available. Start: uvicorn backend.main:app --host 0.0.0.0 --port 8000')
    }
    setExecuting(false)
  }

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(displayCode || '').then(() => {
      setCopySuccess('Copied!')
      setTimeout(() => setCopySuccess(''), 2000)
    })
  }, [displayCode])

  return (
    <div style={panelStyle}>
      {/* 拖拽调整宽度的把手 */}
      <div
        style={resizeHandleStyle}
        onMouseDown={handleMouseDown}
        title="拖拽调整宽度"
      />
      <div style={headerStyle}>
        <span>Generated Code</span>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          {connected && (
            <span style={{ fontSize: 9, color: '#44cc88', fontWeight: 700 }}>● WS</span>
          )}
          {wsEnabled && !connected && (
            <span style={{ fontSize: 9, color: '#ffaa44', fontWeight: 700 }}>● 连接中</span>
          )}
          {backendAvailable.current && (
            <button
              onClick={() => setWsEnabled(!wsEnabled)}
              style={{
                background: wsEnabled ? '#3a1a1a' : 'transparent',
                border: `1px solid ${wsEnabled ? '#663333' : '#333'}`,
                borderRadius: 4,
                color: wsEnabled ? '#ff8888' : '#666',
                fontSize: 10,
                cursor: 'pointer',
                padding: '2px 6px',
              }}
              title="Toggle WebSocket real-time preview"
            >
              ⚡ {wsEnabled ? 'WS' : '实时'}
            </button>
          )}
          <span style={{ color: '#444', fontWeight: 400, fontSize: '10px' }}>Python</span>
          {copySuccess && (
            <span style={{ fontSize: 10, color: '#44cc88', fontWeight: 700 }}>{copySuccess}</span>
          )}
          <button
            onClick={handleCopy}
            style={{
              background: 'transparent',
              border: '1px solid #333',
              borderRadius: 4,
              color: '#888',
              fontSize: 10,
              cursor: 'pointer',
              padding: '2px 6px',
            }}
            title="Copy code to clipboard"
          >
            Copy
          </button>
        </div>
      </div>
      <SyntaxHighlighter
        language="python"
        style={vscDarkPlus}
        customStyle={{
          flex: 1,
          margin: 0,
          padding: '12px',
          fontSize: '11.5px',
          lineHeight: 1.6,
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
        {displayCode || '# Generating...'}
      </SyntaxHighlighter>
      <div style={{ padding: '8px 12px', borderTop: '1px solid #1a1a1a', display: 'flex', gap: 6 }}>
        <button
          onClick={handleRun}
          disabled={executing || !displayCode}
          style={{
            flex: 1,
            background: executing ? '#1a2a1a' : '#1a2a1a',
            border: '1px solid #336633',
            borderRadius: 6,
            color: '#88cc88',
            padding: '5px 0',
            fontSize: 11,
            cursor: displayCode ? 'pointer' : 'not-allowed',
            opacity: displayCode ? 1 : 0.5,
          }}
        >
          {executing ? '⏳ Running...' : '▶ Run'}
        </button>
      </div>
      {(output || error) && (
        <div style={{ padding: '8px 12px', borderTop: '1px solid #1a1a1a', maxHeight: 120, overflow: 'auto' }}>
          {error && <div style={{ color: '#ff6666', fontSize: 10, fontFamily: 'monospace', whiteSpace: 'pre' }}>{error}</div>}
          {output && <div style={{ color: '#88cc88', fontSize: 10, fontFamily: 'monospace', whiteSpace: 'pre' }}>{output}</div>}
        </div>
      )}
    </div>
  )
}
