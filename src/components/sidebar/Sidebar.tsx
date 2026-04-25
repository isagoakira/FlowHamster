import { useState, useCallback, useEffect } from 'react'
import { useGraphStore } from '../../hooks/useGraphStore'
import { Template, useTemplateStore } from '../../utils/templateRegistry'
import { NODE_REGISTRY } from '../../utils/nodeRegistry'
import { getCustomClassesAsNodeCategory } from '../../utils/customCompositeRegistry'
import { NodeType } from '../../types/graph'
import { normalizeNodeType } from '../../utils/nodeType'

const LEGACY_TRAINING_CATEGORIES = ['Loss Functions', 'Optimizers', 'Schedulers']
const EVALUATION_CATEGORIES = ['Evaluation']

const sidebarStyle: React.CSSProperties = {
  width: '220px',
  background: '#111',
  borderRight: '1px solid #222',
  overflowY: 'auto',
  flexShrink: 0,
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
  zIndex: 1100, // Higher than overlay (1000)
}

const headerStyle: React.CSSProperties = {
  padding: '12px 12px 0',
  borderBottom: '1px solid #222',
}

const tabsStyle: React.CSSProperties = {
  display: 'flex',
  gap: '2px',
  paddingBottom: '12px',
}

const tabStyle = (active: boolean): React.CSSProperties => ({
  flex: 1,
  padding: '6px 0',
  textAlign: 'center',
  fontSize: '12px',
  fontWeight: 700,
  borderRadius: '6px',
  cursor: 'pointer',
  transition: 'background 0.15s',
  background: active ? '#1e2a3a' : 'transparent',
  color: active ? '#a0c0ff' : '#555',
  border: 'none',
  outline: 'none',
})

const contentStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '12px 8px',
}

const categoryHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  fontSize: '10px',
  fontWeight: 700,
  color: '#555',
  textTransform: 'uppercase',
  letterSpacing: '1px',
  marginTop: '14px',
  marginBottom: '6px',
  padding: '4px 4px',
  cursor: 'pointer',
  userSelect: 'none' as const,
  borderRadius: '4px',
  transition: 'background 0.15s',
}

const categoryStyle: React.CSSProperties = {
  fontSize: '10px',
  fontWeight: 700,
  color: '#555',
  textTransform: 'uppercase',
  letterSpacing: '1px',
}

const nodeItemStyle: React.CSSProperties = {
  padding: '6px 8px',
  borderRadius: '6px',
  cursor: 'grab',
  fontSize: '12px',
  color: '#ccc',
  marginBottom: '2px',
  transition: 'background 0.15s',
  border: '1px solid transparent',
}

const templateCardStyle: React.CSSProperties = {
  background: '#1a1a1a',
  border: '1px solid #2a2a2a',
  borderRadius: '8px',
  padding: '10px 12px',
  marginBottom: '8px',
  cursor: 'pointer',
  transition: 'border-color 0.15s, background 0.15s',
}

export default function Sidebar() {
  const [activeTab, setActiveTab] = useState<'nodes' | 'templates'>('nodes')
  const [dragging, setDragging] = useState<string | null>(null)
  const [hoveredTemplate, setHoveredTemplate] = useState<string | null>(null)
  const [customClasses] = useState<ReturnType<typeof getCustomClassesAsNodeCategory> | null>(null)
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set())
  const [searchQuery, setSearchQuery] = useState('')
  const features = useGraphStore((s) => s.features)

  // Load templates from backend API on mount
  const templates = useTemplateStore((s) => s.templates)
  const templatesLoading = useTemplateStore((s) => s.isLoading)
  useEffect(() => { useTemplateStore.getState().loadTemplates() }, [])

  // Get visible categories
  const visibleCategories = NODE_REGISTRY.filter(
    (cat) => !LEGACY_TRAINING_CATEGORIES.includes(cat.label) &&
             (features.evaluationNodes || !EVALUATION_CATEGORIES.includes(cat.label))
  ).map((cat) => {
    if (!searchQuery) return cat
    const q = searchQuery.toLowerCase()
    const filteredNodes = cat.nodes.filter(
      (n) => n.label.toLowerCase().includes(q) || n.type.toLowerCase().includes(q)
    )
    return filteredNodes.length > 0 ? { ...cat, nodes: filteredNodes } : null
  }).filter(Boolean) as typeof NODE_REGISTRY

  // Initialize collapsed state: Input/Output and Attention are expanded by default
  useEffect(() => {
    const initialCollapsed = new Set<string>()
    visibleCategories.forEach((cat) => {
      if (cat.label !== 'Input / Output' && cat.label !== 'Attention / Transformer') {
        initialCollapsed.add(cat.label)
      }
    })
    setCollapsedCategories(initialCollapsed)
  }, [])

  const toggleCategory = useCallback((label: string) => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(label)) {
        next.delete(label)
      } else {
        next.add(label)
      }
      return next
    })
  }, [])

  const expandAll = useCallback(() => {
    setCollapsedCategories(new Set())
  }, [])

  const collapseAll = useCallback(() => {
    setCollapsedCategories(new Set(visibleCategories.map((c) => c.label)))
  }, [visibleCategories])

  const onDragStart = useCallback((e: React.DragEvent, nodeType: NodeType, label: string, defaultParams: Record<string, number | string | boolean>) => {
    e.dataTransfer.setData('application/flowhamster', JSON.stringify({ nodeType: normalizeNodeType(nodeType), label, defaultParams }))
    e.dataTransfer.effectAllowed = 'move'
    setDragging(normalizeNodeType(nodeType))
  }, [])

  const onDragEnd = useCallback(() => setDragging(null), [])

  const handleTemplateClick = useCallback((tpl: Template) => {
    useGraphStore.getState().loadTemplate(tpl)
  }, [])

  return (
    <div style={sidebarStyle}>
      <div style={headerStyle}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: '#a0c0ff', padding: '0 0 10px' }}>
          FlowHamster
        </div>
        <div style={tabsStyle}>
          <button
            style={tabStyle(activeTab === 'nodes')}
            onClick={() => setActiveTab('nodes')}
          >
            Nodes
          </button>
          <button
            style={tabStyle(activeTab === 'templates')}
            onClick={() => setActiveTab('templates')}
          >
            Templates
          </button>
        </div>
      </div>

      <div style={contentStyle}>
        {activeTab === 'nodes' ? (
          <>
            {/* Node search */}
            <div style={{ marginBottom: '8px', padding: '0 4px' }}>
              <input
                type="text"
                placeholder="搜索节点..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  fontSize: '11px',
                  background: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '6px',
                  color: '#ccc',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                onFocus={(e) => { e.target.style.borderColor = '#4488ff' }}
                onBlur={(e) => { e.target.style.borderColor = '#333' }}
              />
            </div>

            {/* Expand/Collapse controls */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '8px', padding: '0 4px' }}>
              <button
                onClick={expandAll}
                style={{
                  flex: 1,
                  padding: '4px 8px',
                  fontSize: '10px',
                  background: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '4px',
                  color: '#888',
                  cursor: 'pointer',
                }}
              >
                全部展开
              </button>
              <button
                onClick={collapseAll}
                style={{
                  flex: 1,
                  padding: '4px 8px',
                  fontSize: '10px',
                  background: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '4px',
                  color: '#888',
                  cursor: 'pointer',
                }}
              >
                全部折叠
              </button>
            </div>

            {visibleCategories.map((cat) => {
              const isCollapsed = collapsedCategories.has(cat.label)
              return (
                <div key={cat.label}>
                  <div
                    style={categoryHeaderStyle}
                    onClick={() => toggleCategory(cat.label)}
                  >
                    <span style={categoryStyle}>
                      {isCollapsed ? '▶' : '▼'} {cat.label}
                    </span>
                    <span style={{ fontSize: '9px', color: '#444' }}>
                      {cat.nodes.length}
                    </span>
                  </div>
                  {!isCollapsed && cat.nodes.map((n) => (
                    <div
                      key={n.type}
                      style={{
                        ...nodeItemStyle,
                        background: dragging === n.type ? '#1a2a3a' : 'transparent',
                        borderColor: dragging === n.type ? '#4488ff44' : 'transparent',
                        paddingLeft: '16px',
                      }}
                      draggable
                      onDragStart={(e) => onDragStart(e, n.type, n.label, n.defaultParams)}
                      onDragEnd={onDragEnd}
                      title={n.description}
                    >
                      {n.label}
                    </div>
                  ))}
                </div>
              )
            })}

            {/* Custom classes section */}
            {customClasses && customClasses.nodes.length > 0 && (
              <div key="custom">
                <div style={{ ...categoryHeaderStyle, marginTop: '14px' }} onClick={() => toggleCategory('Custom')}>
                  <span style={categoryStyle}>
                    {collapsedCategories.has('Custom') ? '▶' : '▼'} Custom
                  </span>
                  <span style={{ fontSize: '9px', color: '#444' }}>
                    {customClasses.nodes.length}
                  </span>
                </div>
                {!collapsedCategories.has('Custom') && customClasses.nodes.map((n: any) => (
                  <div
                    key={n.type}
                    style={{
                      ...nodeItemStyle,
                      background: dragging === n.type ? '#2a3a4a' : 'transparent',
                      borderColor: dragging === n.type ? '#44aaff44' : 'transparent',
                      paddingLeft: '16px',
                    }}
                    draggable
                    onDragStart={(e) => onDragStart(e, n.type, n.label, n.defaultParams)}
                    onDragEnd={onDragEnd}
                    title={n.description}
                  >
                    {n.label}
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            {templatesLoading ? (
              <div style={{ color: '#666', fontSize: '12px', padding: '8px' }}>Loading templates...</div>
            ) : (
              templates.map((tpl) => (
              <div
                key={tpl.id}
                style={{
                  ...templateCardStyle,
                  borderColor: hoveredTemplate === tpl.id ? '#4488ff66' : '#2a2a2a',
                  background: hoveredTemplate === tpl.id ? '#1e2a3a' : '#1a1a1a',
                }}
                onClick={() => handleTemplateClick(tpl)}
                onMouseEnter={() => setHoveredTemplate(tpl.id)}
                onMouseLeave={() => setHoveredTemplate(null)}
              >
                <div style={{ fontSize: '14px', fontWeight: 700, color: '#e0e0e0', marginBottom: '4px' }}>
                  {tpl.emoji} {tpl.name}
                </div>
                <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.4' }}>
                  {tpl.description}
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  )
}
