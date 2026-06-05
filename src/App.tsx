import Canvas from './components/canvas/Canvas'
import Sidebar from './components/sidebar/Sidebar'
import CodePreview from './components/codePreview/CodePreview'
import Toolbar from './components/toolbar/Toolbar'
import WorkflowSettingsBar from './components/WorkflowSettingsBar'
import ErrorBoundary from './components/common/ErrorBoundary'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useGraphStore } from './hooks/useGraphStore'
import DataSidebar from './components/data/DataSidebar'
import DataCanvas from './components/data/DataCanvas'
import DataPreview from './components/data/DataPreview'
import { AgentChatPanel } from './components/agentChat/AgentChatPanel'
import { AgentChatToggle } from './components/agentChat/AgentChatToggle'

const appStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  width: '100vw',
  overflow: 'hidden',
}

const bodyStyle: React.CSSProperties = {
  display: 'flex',
  flex: 1,
  overflow: 'hidden',
}

export default function App() {
  useKeyboardShortcuts()
  const workspaceMode = useGraphStore((s) => s.workspaceMode)
  return (
    <div style={appStyle}>
      <Toolbar />
      <div style={bodyStyle}>
        <ErrorBoundary level="section">
          {workspaceMode === 'model' ? <Sidebar /> : <DataSidebar />}
        </ErrorBoundary>
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <ErrorBoundary level="section">
            {workspaceMode === 'model' ? <Canvas /> : <DataCanvas />}
          </ErrorBoundary>
        </div>
        <ErrorBoundary level="section">
          {workspaceMode === 'model' ? <CodePreview /> : <DataPreview />}
        </ErrorBoundary>
      </div>
      {workspaceMode === 'model' && <WorkflowSettingsBar />}
      <AgentChatToggle />
      <AgentChatPanel />
    </div>
  )
}
