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
import { useTrainingStore } from '../../hooks/useTrainingStore'
import { WorkflowDialog } from '../dialogs/WorkflowDialog'
import { SettingsPanel } from './panels/SettingsPanel'
import { AllOutputsPanel } from './panels/AllOutputsPanel'
import { TrainingConfigPanel } from './panels/TrainingConfigPanel'
import { TrainingDashboard } from './panels/TrainingDashboard'
import { BindingPanel } from './panels/BindingPanel'
import {
  toolbarStyle,
  toolbarSecondaryStyle,
  groupSeparatorStyle,
  logoStyle,
  iconBtnStyle,
  textBtnStyle,
  activeBtnStyle,
  settingsBtnStyle,
  dangerBtn,
} from './styles/toolbarSharedStyles'

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

  const canPackage = selectedNodeIds.length >= 2 && selectedNodeIds.every(id => {
    const node = nodes.find(n => n.id === id)
    const data = node?.data as any
    return Boolean(node) &&
      node?.type !== 'group' &&
      data?.nodeType !== 'group' &&
      !(data?.isCustomComposite && data?.isExpanded)
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
  const [showTrainingDashboard, setShowTrainingDashboard] = useState(false)
  const [gradientMode, setGradientMode] = useState(false)

  const workflowStore = useWorkflowStore()
  const { currentWorkflowName, isDirty, saveWorkflow, currentWorkflowId } = workflowStore

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

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 's') {
          e.preventDefault()
          if (currentWorkflowId) {
            const doc = buildCurrentDocument()
            saveWorkflow(doc).then((ok) => { if (ok) console.log('Workflow saved') })
          }
        }
        if (e.key === 'a' && !e.shiftKey) {
          e.preventDefault()
          const allNodeIds = store.nodes.map((n) => n.id)
          store.setSelection(allNodeIds, [])
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentWorkflowId, buildCurrentDocument, saveWorkflow, store.nodes, store])

  useEffect(() => {
    if (currentWorkflowId) workflowStore.setDirty(true)
  }, [store.nodes, store.edges, currentWorkflowId])

  const handleOpenWorkflow = (document: WorkflowDocument) => {
    store.setNodes(document.modelGraph.nodes)
    store.setEdges(document.modelGraph.edges)
    store.setTrainingConfig(document.trainingConfig)
    store.setBindings(document.bindings)
    loadDataGraph(document.dataGraph.nodes as any[], document.dataGraph.edges as any[])
    if (document.metadata?.name) workflowStore.openWorkflow(document.metadata.name)
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
            source: e.source, target: e.target,
            sourceHandle: e.sourceHandle ?? null, targetHandle: e.targetHandle ?? null,
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

  const handleExportNotebook = async () => {
    const graph = {
      nodes: executableGraph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
      edges: executableGraph.edges.map((e) => ({
        source: e.source, target: e.target,
        sourceHandle: e.sourceHandle ?? null, targetHandle: e.targetHandle ?? null,
        data: e.data,
      })),
    }
    try {
      const res = await exportNotebook(graph, 'flowhamster_model', trainingConfig,
        { nodes: dataNodes as any[], edges: dataEdges as any[] }, bindings)
      if (res.success) {
        downloadFile(res.content, res.filename, res.mime_type)
      } else {
        alert('Export notebook failed: ' + res.filename)
      }
    } catch {
      alert('Export notebook failed: backend not available. Start backend first.')
    }
  }

  const handleStartTraining = useCallback(async () => {
    const doc = buildCurrentDocument()
    const cfg = trainingConfig
    const config = {
      epochs: cfg.runtime?.epochs ?? 10,
      batchSize: cfg.runtime?.batchSize ?? 32,
      optimizer: cfg.optimizer?.type ?? 'adam',
      learningRate: (cfg.optimizer?.params?.learningRate as number) ?? 0.001,
      momentum: (cfg.optimizer?.params?.momentum as number) ?? 0.9,
      weightDecay: (cfg.optimizer?.params?.weightDecay as number) ?? 0.0,
      scheduler: cfg.scheduler?.type ?? 'cosine',
      stepSize: (cfg.scheduler?.params?.stepSize as number) ?? 10,
      gamma: (cfg.scheduler?.params?.gamma as number) ?? 0.1,
      warmupEpochs: (cfg.scheduler?.params?.warmupEpochs as number) ?? 0,
      checkpoint: {
        enabled: cfg.checkpoint?.enabled ?? true,
        saveTopK: cfg.checkpoint?.saveTopK ?? 3,
        monitor: cfg.checkpoint?.monitor ?? 'val_loss',
        mode: cfg.checkpoint?.mode ?? 'min',
      },
      dataConfig: {
        trainDir: '',
        valDir: '',
        numWorkers: cfg.runtime?.numWorkers ?? 4,
      },
    }
    try {
      await useTrainingStore.getState().startTraining(doc, config)
      setShowTrainingDashboard(true)
    } catch {
      alert('Failed to start training. Is the backend running?')
    }
  }, [buildCurrentDocument, trainingConfig])

  const closeAllPanels = () => {
    setShowSettings(false)
    setShowTrainingConfig(false)
    setShowBindings(false)
    setShowTrainingDashboard(false)
  }

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
      {/* ===== UPPER TOOLBAR: Editing operations ===== */}
      <div style={toolbarStyle}>
        <span style={logoStyle}>FlowHamster</span>

        <div style={groupSeparatorStyle} />

        {/* Workspace toggle */}
        <button
          style={{ ...iconBtnStyle, color: workspaceMode === 'model' ? '#88aaff' : '#aaa' }}
          onClick={() => store.setWorkspaceMode('model')}
          title="Model workspace"
        >
          🧠
        </button>
        <button
          style={{ ...iconBtnStyle, color: workspaceMode === 'data' ? '#88aaff' : '#aaa' }}
          onClick={() => store.setWorkspaceMode('data')}
          title="Data workspace"
        >
          🗂
        </button>

        <div style={groupSeparatorStyle} />

        {/* History */}
        <button
          style={{ ...iconBtnStyle, opacity: canUndo ? 1 : 0.35 }}
          onClick={() => store.undo()}
          disabled={!canUndo}
          title="Undo"
        >
          ↩
        </button>
        <button
          style={{ ...iconBtnStyle, opacity: canRedo ? 1 : 0.35 }}
          onClick={() => store.redo()}
          disabled={!canRedo}
          title="Redo"
        >
          ↪
        </button>

        <div style={groupSeparatorStyle} />

        {/* Clipboard */}
        <button
          style={{ ...iconBtnStyle, opacity: canCopySelection ? 1 : 0.35 }}
          onClick={() => store.copySelection()}
          disabled={!canCopySelection}
          title="Copy"
        >
          📋
        </button>
        <button
          style={{ ...iconBtnStyle, opacity: hasClipboard ? 1 : 0.35 }}
          onClick={() => store.pasteClipboard()}
          disabled={!hasClipboard}
          title="Paste"
        >
          📥
        </button>
        <button
          style={{ ...iconBtnStyle, opacity: hasSelection ? 1 : 0.35 }}
          onClick={() => store.deleteSelection()}
          disabled={!hasSelection}
          title="Delete"
        >
          ⌫
        </button>

        <div style={groupSeparatorStyle} />

        {/* Structure */}
        <button
          style={{ ...iconBtnStyle, opacity: canPackage ? 1 : 0.35 }}
          onClick={() => store.packageSelection()}
          disabled={!canPackage}
          title="Package"
        >
          📦
        </button>
        <button
          style={{ ...iconBtnStyle, opacity: hasExpandedPackage ? 1 : 0.35 }}
          onClick={() => {
            const groupId = selectedNodeIds.find(id => {
              const node = nodes.find(n => n.id === id)
              return (node?.data as any)?.isCustomComposite && node?.data?.isExpanded
            })
            if (groupId) store.collapseGroup(groupId)
          }}
          disabled={!hasExpandedPackage}
          title="Collapse"
        >
          🔽
        </button>
        <button
          style={{ ...iconBtnStyle, opacity: hasCollapsedPackage ? 1 : 0.35 }}
          onClick={() => {
            const groupId = selectedNodeIds.find(id => {
              const node = nodes.find(n => n.id === id)
              return (node?.data as any)?.isCustomComposite && !node?.data?.isExpanded
            })
            if (groupId) store.expandGroup(groupId)
          }}
          disabled={!hasCollapsedPackage}
          title="Expand"
        >
          🔼
        </button>

        <div style={{ flex: 1 }} />

        {/* Auto Layout — right pinned */}
        <button
          style={iconBtnStyle}
          onClick={handleAutoLayout}
          title="Auto Layout"
        >
          ↗
        </button>
      </div>

      {/* ===== LOWER TOOLBAR: Panel & Export actions ===== */}
      <div
        style={toolbarSecondaryStyle}
        onClick={closeAllPanels}
      >
        {/* Train */}
        <button
          style={showTrainingConfig ? activeBtnStyle : textBtnStyle}
          onClick={(e) => {
            e.stopPropagation()
            setShowBindings(false)
            setShowTrainingConfig(!showTrainingConfig)
          }}
        >
          🏋️ Train
        </button>

        {/* Bind */}
        <button
          style={showBindings ? activeBtnStyle : textBtnStyle}
          onClick={(e) => {
            e.stopPropagation()
            setShowTrainingConfig(false)
            setShowBindings(!showBindings)
          }}
        >
          🔗 Bind
        </button>

        {/* Workflow */}
        <button
          style={{
            ...textBtnStyle,
            background: currentWorkflowName ? '#1a2a3a' : '#1a1a1a',
            borderColor: currentWorkflowName ? '#3355aa' : '#2e2e2e',
            color: currentWorkflowName ? '#88aaff' : '#ccc',
          }}
          onClick={(e) => { e.stopPropagation(); setShowWorkflowDialog(true) }}
        >
          {currentWorkflowName ? `📂 ${currentWorkflowName}${isDirty ? ' *' : ''}` : '📂 Workflow'}
        </button>

        <div style={groupSeparatorStyle} />

        {/* Export */}
        <button style={textBtnStyle} onClick={(e) => { e.stopPropagation(); handleExportPy() }}>
          Export .py
        </button>
        <button style={textBtnStyle} onClick={(e) => { e.stopPropagation(); handleExportNotebook() }}>
          Export .ipynb
        </button>

        <div style={{ flex: 1 }} />

        {/* Feature-gated: Run Preview */}
        {store.features.tensorPreview && (
          <button
            style={activeBtnStyle}
            onClick={(e) => { e.stopPropagation(); handleRunPreview() }}
            disabled={previewLoading}
          >
            {previewLoading ? '⏳ Running...' : '▶ Run Preview'}
          </button>
        )}

        {/* Train button */}
        <button
          style={{ ...activeBtnStyle, background: '#c0392b', borderColor: '#922b21', color: '#fff' }}
          onClick={(e) => { e.stopPropagation(); handleStartTraining() }}
          title="Start training run"
        >
          🚀 Train
        </button>

        {/* Settings */}
        <button
          style={{ ...settingsBtnStyle, color: showSettings ? '#88aaff' : '#555' }}
          onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings) }}
        >
          ⚙
        </button>

        {/* Feature-gated: Gradient Analysis */}
        {store.features.gradientViz && (
          <button
            style={gradientMode ? activeBtnStyle : textBtnStyle}
            onClick={(e) => { e.stopPropagation(); handleGradientAnalysis() }}
          >
            {gradientMode ? '✅ Exit Gradient' : '📉 Gradient Analysis'}
          </button>
        )}

        {/* Clear */}
        <button
          style={dangerBtn}
          onClick={(e) => { e.stopPropagation(); handleClear() }}
        >
          🗑
        </button>
      </div>

      {/* Panels */}
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

      {showTrainingDashboard && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowTrainingDashboard(false)} />
          <TrainingDashboard onClose={() => setShowTrainingDashboard(false)} />
        </>
      )}

      {showAllOutputs && (
        <>
          <div style={{ position: 'absolute', inset: 0, zIndex: 999 }} onClick={() => setShowAllOutputs(false)} />
          <AllOutputsPanel preview={preview} onClose={() => setShowAllOutputs(false)} />
        </>
      )}

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