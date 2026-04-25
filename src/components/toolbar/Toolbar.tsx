import { useState, useEffect, useCallback } from 'react'
import { useGraphStore } from '../../hooks/useGraphStore'
import { useDataGraphStore } from '../../hooks/useDataGraphStore'
import { useAutoLayout } from '../../hooks/useAutoLayout'
import { generateLocalCode } from '../../utils/codeGenerator'
import { exportNotebook, downloadFile } from '../../utils/api'
import { useTensorExecutor } from '../../hooks/useTensorExecutor'
import { getExecutableGraph } from '../../utils/graphStructure'
import { API_BASE_URL } from '../../utils/runtimeConfig'
import { createWorkflowDocumentFromGraph } from '../../utils/workflowDocument'
import { WorkflowDocument } from '../../schema/workflowDocument'
import { useWorkflowStore } from '../../stores/useWorkflowStore'
import { WorkflowDialog } from '../dialogs/WorkflowDialog'
import { SettingsPanel } from './panels/SettingsPanel'
import { AllOutputsPanel } from './panels/AllOutputsPanel'
import { TrainingConfigPanel } from './panels/TrainingConfigPanel'
import { BindingPanel } from './panels/BindingPanel'
import { toolbarStyle, logoStyle, btnStyle, dangerBtn } from './styles/toolbarSharedStyles'

export function Toolbar() {
  const store = useGraphStore()
  const dataNodes = useDataGraphStore((s) => s.nodes)
  const dataEdges = useDataGraphStore((s) => s.edges)
  const loadDataGraph = useDataGraphStore((s) => s.loadGraph)
  const { layout } = useAutoLayout()
  const canUndo = useGraphStore((s) => s.canUndo())
  const canRedo = useGraphStore((s) => s.canRedo())
  const selectedNodeCount = useGraphStore((s) => s.selectedNodeIds.length)
  const selectedEdgeCount = useGraphStore((s) => s.selectedEdgeIds.length)
  const hasClipboard = useGraphStore((s) => Boolean(s.clipboard?.nodes.length))
  const nodes = useGraphStore((s) => s.nodes)
  const trainingConfig = useGraphStore((s) => s.trainingConfig)
  const bindings = useGraphStore((s) => s.bindings)
  const workspaceMode = useGraphStore((s) => s.workspaceMode)
  const { preview, loading: previewLoading, runForward } = useTensorExecutor()
  const executableGraph = getExecutableGraph(store.nodes as any, store.edges as any)
  const hasSelection = selectedNodeCount > 0 || selectedEdgeCount > 0
  const canCopySelection = selectedNodeCount > 0
  const selectedNodeIds = useGraphStore((s) => s.selectedNodeIds)

  // Package-related selectors (custom composites)
  // canPackage = selectedNodes are all non-composite, count >= 2
  const canPackage = selectedNodeIds.length >= 2 && selectedNodeIds.every(id => {
    const node = nodes.find(n => n.id === id)
    return !(node?.data as any)?.isCustomComposite
  })
  const hasExpandedPackage = selectedNodeIds.some(id => {
    const node = nodes.find(n => n.id === id)
    return (node?.data as any)?.isCustomComposite === true && node?.data?.isExpanded === true
  })
  const hasCollapsedPackage = selectedNodeIds.some(id => {
    const node = nodes.find(n => n.id === id)
    return (node?.data as any)?.isCustomComposite === true && node?.data?.isExpanded === false
  })

  const [showSettings, setShowSettings] = useState(false)
  const [showTrainingConfig, setShowTrainingConfig] = useState(false)
  const [showBindings, setShowBindings] = useState(false)
  const [showAllOutputs, setShowAllOutputs] = useState(false)
  const [showWorkflowDialog, setShowWorkflowDialog] = useState(false)
  const [gradientMode, setGradientMode] = useState(false)

  // Workflow state
  const workflowStore = useWorkflowStore()
  const { currentWorkflowName, isDirty, saveWorkflow, currentWorkflowId } = workflowStore

  // Build complete WorkflowDocument from current state
  const buildCurrentDocument = useCallback((): WorkflowDocument => {
    return createWorkflowDocumentFromGraph(store.nodes, store.edges, {
      name: currentWorkflowName || 'Untitled',
      description: '',
      trainingConfig,
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    })
  }, [store.nodes, store.edges, currentWorkflowName, trainingConfig, dataNodes, dataEdges, bindings])

  // Ctrl+S: Save workflow
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 's') {
          e.preventDefault()
          if (currentWorkflowId) {
            const doc = buildCurrentDocument()
            saveWorkflow(doc).then((ok) => {
              if (ok) console.log('Workflow saved')
            })
          }
        }
        if (e.key === 'a' && !e.shiftKey) {
          // Ctrl+A: Select all nodes
          e.preventDefault()
          const allNodeIds = store.nodes.map((n) => n.id)
          store.setSelection(allNodeIds, [])
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentWorkflowId, buildCurrentDocument, saveWorkflow, store.nodes, store])

  // Mark dirty when graph changes
  useEffect(() => {
    if (currentWorkflowId) {
      workflowStore.setDirty(true)
    }
  }, [store.nodes, store.edges, currentWorkflowId])

  const handleOpenWorkflow = (document: WorkflowDocument) => {
    store.setNodes(document.modelGraph.nodes)
    store.setEdges(document.modelGraph.edges)
    store.setTrainingConfig(document.trainingConfig)
    store.setBindings(document.bindings)
    loadDataGraph(document.dataGraph.nodes as any[], document.dataGraph.edges as any[])
    // Set as current workflow in store
    if (document.metadata?.name) {
      workflowStore.openWorkflow(document.metadata.name)
    }
  }

  const handleGradientAnalysis = async () => {
    if (gradientMode) {
      setGradientMode(false)
      ;(window as any).__flowhamster_exitGradient?.()
      return
    }
    try {
      const res = await fetch(`${API_BASE_URL}/execute/gradients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes: executableGraph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
          edges: executableGraph.edges.map((e) => ({
            source: e.source,
            target: e.target,
            sourceHandle: e.sourceHandle ?? null,
            targetHandle: e.targetHandle ?? null,
            data: e.data,
          })),
        }),
      })
      const data = await res.json()
      if (data.success) {
        setGradientMode(true)
        ;(window as any).__flowhamster_setGradient?.(data.data)
      } else {
        alert('Gradient analysis failed: ' + data.error)
      }
    } catch {
      alert('Backend not available. Start the backend server first.')
    }
  }

  const handleAutoLayout = () => {
    const { nodes: layouted } = layout(store.nodes, store.edges)
    store.setNodes(layouted as any)
  }

  const handleRunPreview = async () => {
    if (!store.features.tensorPreview) return
    await runForward(executableGraph.nodes as any[], executableGraph.edges as any[])
    setShowAllOutputs(true)
  }

  const handleExportPy = useCallback(() => {
    const result = generateLocalCode(store.nodes as any, store.edges as any, store.features as any, trainingConfig as any, {
      dataGraphNodes: dataNodes as any,
      dataGraphEdges: dataEdges as any,
      bindings,
    })
    const code = typeof result === 'string' ? result : (result as any).code ?? JSON.stringify(result)
    const blob = new Blob([code], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'flowhamster_model.py'
    a.click()
    URL.revokeObjectURL(url)
  }, [store.nodes, store.edges, store.features, trainingConfig, dataNodes, dataEdges, bindings])

  const handleClear = () => {
    if (confirm(workspaceMode === 'model' ? 'Clear all model nodes?' : 'Clear all data nodes?')) {
      if (workspaceMode === 'model') {
        store.setNodes([])
        store.setEdges([])
      } else {
        loadDataGraph([], [])
      }
    }
  }

  return (
    <>
      <div
        style={toolbarStyle}
        onClick={() => {
          if (showSettings) setShowSettings(false)
          if (showTrainingConfig) setShowTrainingConfig(false)
          if (showBindings) setShowBindings(false)
        }}
      >
        <span style={logoStyle}>FlowHamster</span>

        <button
          style={{ ...btnStyle, color: workspaceMode === 'model' ? '#88aaff' : '#ccc' }}
          onClick={() => store.setWorkspaceMode('model')}
          title="Model graph workspace"
        >
          🧠 Model
        </button>
        <button
          style={{ ...btnStyle, color: workspaceMode === 'data' ? '#88aaff' : '#ccc' }}
          onClick={() => store.setWorkspaceMode('data')}
          title="Data graph workspace"
        >
          🗂 Data
        </button>

        <button
          style={{ ...btnStyle, opacity: canUndo ? 1 : 0.4 }}
          onClick={() => store.undo()}
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
        >
          ↩ Undo
        </button>
        <button
          style={{ ...btnStyle, opacity: canRedo ? 1 : 0.4 }}
          onClick={() => store.redo()}
          disabled={!canRedo}
          title="Redo (Ctrl+Shift+Z / Ctrl+Y)"
        >
          ↪ Redo
        </button>
        <button
          style={{ ...btnStyle, opacity: canPackage ? 1 : 0.4 }}
          onClick={() => store.packageSelection()}
          disabled={!canPackage}
          title="Package selected modules into a single module"
        >
          📦 Package
        </button>
        <button
          style={{ ...btnStyle, opacity: hasExpandedPackage ? 1 : 0.4 }}
          onClick={() => {
            const groupId = selectedNodeIds.find(id => {
              const node = nodes.find(n => n.id === id)
              return (node?.data as any)?.isCustomComposite && node?.data?.isExpanded
            })
            if (groupId) store.collapseGroup(groupId)
          }}
          disabled={!hasExpandedPackage}
          title="Collapse expanded composite"
        >
          🔽 Collapse
        </button>
        <button
          style={{ ...btnStyle, opacity: hasCollapsedPackage ? 1 : 0.4 }}
          onClick={() => {
            const groupId = selectedNodeIds.find(id => {
              const node = nodes.find(n => n.id === id)
              return (node?.data as any)?.isCustomComposite && !node?.data?.isExpanded
            })
            if (groupId) store.expandGroup(groupId)
          }}
          disabled={!hasCollapsedPackage}
          title="Expand composite to see internal structure"
        >
          🔼 Expand
        </button>
        <button
          style={{ ...btnStyle, opacity: canCopySelection ? 1 : 0.4 }}
          onClick={() => store.copySelection()}
          disabled={!canCopySelection}
          title="Copy selected modules and connections"
        >
          📋 Copy
        </button>
        <button
          style={{ ...btnStyle, opacity: hasClipboard ? 1 : 0.4 }}
          onClick={() => store.pasteClipboard()}
          disabled={!hasClipboard}
          title="Paste copied modules"
        >
          📥 Paste
        </button>
        <button
          style={{ ...btnStyle, opacity: hasSelection ? 1 : 0.4 }}
          onClick={() => store.deleteSelection()}
          disabled={!hasSelection}
          title="Delete selected modules"
        >
          ⌫ Delete
        </button>

        <div style={{ flex: 1 }} />

        {/* Phase 5: Tensor Preview — only shown when feature is enabled */}
        {store.features.tensorPreview && (
          <button
            style={{ ...btnStyle, background: '#1a2a3a', borderColor: '#3355aa', color: '#88aaff' }}
            onClick={handleRunPreview}
            disabled={previewLoading}
            title="Run forward pass on entire graph, show all outputs"
          >
            {previewLoading ? '⏳ Running...' : '▶ Run Preview'}
          </button>
        )}

        <button style={btnStyle} onClick={handleAutoLayout} title="Dagre auto layout">
          ↗ Auto Layout
        </button>
        <button
          style={{ ...btnStyle, color: showTrainingConfig ? '#88aaff' : '#ccc' }}
          onClick={(e) => {
            e.stopPropagation()
            setShowSettings(false)
            setShowBindings(false)
            setShowTrainingConfig(!showTrainingConfig)
          }}
          title="Training configuration"
        >
          🏋️ Train
        </button>
        <button
          style={{ ...btnStyle, color: showBindings ? '#88aaff' : '#ccc' }}
          onClick={(e) => {
            e.stopPropagation()
            setShowSettings(false)
            setShowTrainingConfig(false)
            setShowBindings(!showBindings)
          }}
          title="Binding configuration"
        >
          🔗 Bind
        </button>
        {/* Workflow Manager */}
        <button
          style={{
            ...btnStyle,
            background: currentWorkflowName ? '#1a2a3a' : '#1a1a1a',
            borderColor: currentWorkflowName ? '#3355aa' : '#333',
            color: currentWorkflowName ? '#88aaff' : '#ccc',
          }}
          onClick={() => setShowWorkflowDialog(true)}
          title="Workflow Manager"
        >
          {currentWorkflowName ? `📂 ${currentWorkflowName}${isDirty ? ' *' : ''}` : '📂 Workflow'}
        </button>
        <button style={btnStyle} onClick={handleExportPy}>
          Export .py
        </button>
        <button
          style={btnStyle}
          onClick={async () => {
            const graph = {
              nodes: executableGraph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
              edges: executableGraph.edges.map((e) => ({
                source: e.source,
                target: e.target,
                sourceHandle: e.sourceHandle ?? null,
                targetHandle: e.targetHandle ?? null,
                data: e.data,
              })),
            }
            try {
              const res = await exportNotebook(
                graph,
                'flowhamster_model',
                trainingConfig,
                { nodes: dataNodes as any[], edges: dataEdges as any[] },
                bindings
              )
              if (res.success) {
                downloadFile(res.content, res.filename, res.mime_type)
              } else {
                alert('Export notebook failed: ' + res.filename)
              }
            } catch {
              alert('Export notebook failed: backend not available. Start backend first.')
            }
          }}
        >
          Export .ipynb
        </button>

        {/* Settings gear — always visible */}
        <button
          style={{ ...btnStyle, padding: '4px 8px', fontSize: '14px', color: showSettings ? '#88aaff' : '#666' }}
          onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings) }}
          title="Settings"
        >
          ⚙
        </button>

        {store.features.gradientViz && (
          <button
            style={{
              ...btnStyle,
              background: gradientMode ? '#2a4a2a' : '#1a2a1a',
              borderColor: gradientMode ? '#55aa55' : '#336633',
              color: gradientMode ? '#aaffaa' : '#88cc88',
            }}
            onClick={handleGradientAnalysis}
            title="Analyze gradient flow on canvas"
          >
            {gradientMode ? '✅ Exit Gradient' : '📉 Gradient Analysis'}
          </button>
        )}

        <button style={dangerBtn} onClick={handleClear} title="Clear all nodes">
          🗑 Clear
        </button>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowSettings(false)} />
          <SettingsPanel onClose={() => setShowSettings(false)} />
        </>
      )}

      {showTrainingConfig && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowTrainingConfig(false)} />
          <TrainingConfigPanel onClose={() => setShowTrainingConfig(false)} />
        </>
      )}

      {showBindings && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowBindings(false)} />
          <BindingPanel onClose={() => setShowBindings(false)} />
        </>
      )}

      {/* All outputs panel — shown after Run Preview */}
      {showAllOutputs && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowAllOutputs(false)} />
          <AllOutputsPanel preview={preview} onClose={() => setShowAllOutputs(false)} />
        </>
      )}

      {/* Workflow Dialog */}
      {showWorkflowDialog && (
        <WorkflowDialog
          onClose={() => setShowWorkflowDialog(false)}
          onOpenWorkflow={handleOpenWorkflow}
        />
      )}
    </>
  )
}

export default Toolbar
