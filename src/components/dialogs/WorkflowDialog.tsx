/**
 * 工作流管理对话框
 *
 * 用于创建、打开、删除、导入、导出工作流
 */

import { useState, useEffect } from 'react'
import { useWorkflowStore, WorkflowInfo } from '../../stores/useWorkflowStore'
import { WorkflowDocument } from '../../schema/workflowDocument'

interface WorkflowDialogProps {
  onClose: () => void
  onOpenWorkflow: (document: WorkflowDocument) => void
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0,0,0,0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 2000,
}

const dialogStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '12px',
  padding: '24px',
  minWidth: '480px',
  maxWidth: '600px',
  maxHeight: '80vh',
  overflowY: 'auto',
  boxShadow: '0 16px 64px rgba(0,0,0,0.8)',
}

const headerStyle: React.CSSProperties = {
  fontSize: '16px',
  fontWeight: 700,
  color: '#a0c0ff',
  marginBottom: '20px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#111',
  border: '1px solid #333',
  borderRadius: '6px',
  color: '#ddd',
  padding: '8px 12px',
  fontSize: '13px',
  boxSizing: 'border-box',
  marginBottom: '12px',
}

const btnStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #333',
  borderRadius: '6px',
  color: '#ccc',
  padding: '6px 16px',
  fontSize: '12px',
  cursor: 'pointer',
}

const primaryBtn: React.CSSProperties = {
  ...btnStyle,
  background: '#1a2a3a',
  borderColor: '#3355aa',
  color: '#88aaff',
}

const dangerBtn: React.CSSProperties = {
  ...btnStyle,
  background: '#2a1a1a',
  borderColor: '#663333',
  color: '#cc8888',
}

const sectionStyle: React.CSSProperties = {
  marginBottom: '24px',
}

const sectionTitle: React.CSSProperties = {
  fontSize: '12px',
  fontWeight: 600,
  color: '#888',
  marginBottom: '10px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
}

const workflowItemStyle: React.CSSProperties = {
  background: '#222',
  border: '1px solid #333',
  borderRadius: '8px',
  padding: '12px 16px',
  marginBottom: '8px',
  cursor: 'pointer',
  transition: 'border-color 0.15s',
}

export function WorkflowDialog({ onClose, onOpenWorkflow }: WorkflowDialogProps) {
  const {
    workflows,
    isLoading,
    fetchWorkflows,
    createWorkflow,
    openWorkflow,
    deleteWorkflow,
  } = useWorkflowStore()

  const [tab, setTab] = useState<'open' | 'create' | 'import'>('open')
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    fetchWorkflows()
  }, [fetchWorkflows])

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    const workflow = await createWorkflow(newName.trim(), newDesc.trim())
    if (workflow) {
      // 创建后打开一个空的默认文档
      const defaultDoc: WorkflowDocument = {
        version: '2.0.0',
        metadata: {
          name: newName.trim(),
          description: newDesc.trim(),
          schemaVersion: '2.0.0',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          exportSource: 'flowhamster'
        },
        modelGraph: { kind: 'model', nodes: [], edges: [], contract: { inputs: [], outputs: [] } },
        dataGraph: { kind: 'data', nodes: [], edges: [], contract: { inputs: [], outputs: [] } },
        trainingConfig: {
          taskType: 'classification',
          loss: { type: 'cross_entropy', enabled: true, params: {} },
          optimizer: { type: 'adamw', enabled: true, params: { lr: 0.001, weight_decay: 0 } },
          scheduler: { type: 'step_lr', enabled: false, params: { step_size: 10, gamma: 0.1 } },
          metrics: [],
          runtime: { device: 'auto', epochs: 10, batchSize: 32, amp: false, gradClip: null, numWorkers: 0 },
          checkpoint: { enabled: false, saveTopK: 3, monitor: 'val_loss', mode: 'min', earlyStopPatience: null }
        },
        bindings: []
      }
      onOpenWorkflow(defaultDoc)
      onClose()
    }
    setCreating(false)
  }

  const handleOpen = async (wf: WorkflowInfo) => {
    const document = await openWorkflow(wf.id)
    if (document) {
      onOpenWorkflow(document)
      onClose()
    }
  }

  const handleDelete = async (e: React.MouseEvent, wf: WorkflowInfo) => {
    e.stopPropagation()
    if (confirm(`Delete workflow "${wf.name}"?`)) {
      await deleteWorkflow(wf.id)
    }
  }

  const handleImportJson = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = (e: any) => {
      const file = e.target.files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (ev: any) => {
        try {
          const document = JSON.parse(ev.target.result)
          // 验证是否是有效的 workflow document
          if (!document.version || !document.metadata || !document.modelGraph) {
            alert('Invalid workflow file format')
            return
          }
          onOpenWorkflow(document)
          onClose()
        } catch {
          alert('Failed to parse workflow file')
        }
      }
      reader.readAsText(file)
    }
    input.click()
  }

  const handleExportCurrentWorkflow = async () => {
    const { currentWorkflowId } = useWorkflowStore.getState()
    if (!currentWorkflowId) {
      alert('No workflow is currently open')
      return
    }
    const workflowDoc = await openWorkflow(currentWorkflowId)
    if (workflowDoc) {
      const blob = new Blob([JSON.stringify(workflowDoc, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement('a')
      a.href = url
      a.download = `${workflowDoc.metadata?.name || 'workflow'}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const formatDate = (ts: number) => {
    return new Date(ts * 1000).toLocaleString()
  }

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={dialogStyle} onClick={(e) => e.stopPropagation()}>
        <div style={headerStyle}>
          <span>Workflow Manager</span>
          <button style={{ ...btnStyle, padding: '4px 8px' }} onClick={onClose}>✕</button>
        </div>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
          <button
            style={{ ...btnStyle, ...(tab === 'open' ? primaryBtn : {}), flex: 1 }}
            onClick={() => setTab('open')}
          >
            Open
          </button>
          <button
            style={{ ...btnStyle, ...(tab === 'create' ? primaryBtn : {}), flex: 1 }}
            onClick={() => setTab('create')}
          >
            New
          </button>
          <button
            style={{ ...btnStyle, ...(tab === 'import' ? primaryBtn : {}), flex: 1 }}
            onClick={() => setTab('import')}
          >
            Import JSON
          </button>
        </div>

        {tab === 'open' && (
          <div style={sectionStyle}>
            {isLoading ? (
              <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>Loading...</div>
            ) : workflows.length === 0 ? (
              <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
                No workflows yet. Create one first!
              </div>
            ) : (
              workflows.map((wf) => (
                <div
                  key={wf.id}
                  style={workflowItemStyle}
                  onClick={() => handleOpen(wf)}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#4488ff')}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#333')}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ fontSize: '14px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>
                        {wf.name}
                      </div>
                      {wf.description && (
                        <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                          {wf.description}
                        </div>
                      )}
                      <div style={{ fontSize: '10px', color: '#555' }}>
                        Updated: {formatDate(wf.updated_at)}
                      </div>
                    </div>
                    <button
                      style={{ ...dangerBtn, padding: '4px 10px', fontSize: '11px' }}
                      onClick={(e) => handleDelete(e, wf)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {tab === 'create' && (
          <div style={sectionStyle}>
            <div style={{ marginBottom: '12px' }}>
              <label style={{ ...sectionTitle, marginBottom: '6px', display: 'block' }}>Workflow Name</label>
              <input
                style={inputStyle}
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g., My ResNet Project"
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ ...sectionTitle, marginBottom: '6px', display: 'block' }}>Description (optional)</label>
              <input
                style={inputStyle}
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Brief description of this workflow"
              />
            </div>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button style={btnStyle} onClick={onClose}>Cancel</button>
              <button
                style={primaryBtn}
                onClick={handleCreate}
                disabled={!newName.trim() || creating}
              >
                {creating ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        )}

        {tab === 'import' && (
          <div style={sectionStyle}>
            <div style={{ color: '#888', fontSize: '13px', marginBottom: '16px', lineHeight: 1.5 }}>
              <p style={{ marginBottom: '12px' }}>
                <strong style={{ color: '#a0c0ff' }}>Import JSON:</strong> Load a workflow file (.json) that was previously exported or shared.
              </p>
              <p>
                This will create a new workflow entry in the backend storage.
              </p>
            </div>
            <button style={{ ...primaryBtn, width: '100%', padding: '10px' }} onClick={handleImportJson}>
              Choose JSON File...
            </button>
          </div>
        )}

        {/* Footer with export current workflow */}
        {useWorkflowStore.getState().currentWorkflowName && (
          <div style={{ borderTop: '1px solid #333', paddingTop: '16px', marginTop: '8px' }}>
            <button
              style={{ ...btnStyle, width: '100%' }}
              onClick={handleExportCurrentWorkflow}
            >
              Export Current Workflow as JSON
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
