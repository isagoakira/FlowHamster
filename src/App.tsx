import Canvas from './components/canvas/Canvas'
import Sidebar from './components/sidebar/Sidebar'
import CodePreview from './components/codePreview/CodePreview'
import Toolbar from './components/toolbar/Toolbar'
import WorkflowSettingsBar from './components/WorkflowSettingsBar'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useGraphStore } from './hooks/useGraphStore'
import DataSidebar from './components/data/DataSidebar'
import DataCanvas from './components/data/DataCanvas'
import DataPreview from './components/data/DataPreview'

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
        {workspaceMode === 'model' ? <Sidebar /> : <DataSidebar />}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {workspaceMode === 'model' ? <Canvas /> : <DataCanvas />}
        </div>
        {workspaceMode === 'model' ? <CodePreview /> : <DataPreview />}
      </div>
      {workspaceMode === 'model' && <WorkflowSettingsBar />}
    </div>
  )
}
