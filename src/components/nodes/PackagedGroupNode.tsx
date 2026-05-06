/**
 * 打包组节点组件（自定义复合模块）v3
 *
 * 改进:
 * 1. 双击打开弹窗查看内部结构（不在内联展开）
 * 2. 右键菜单支持解包操作
 * 3. 更好的视觉反馈
 * 4. 对话框状态提升到 Canvas 层级，避免 ReactFlow 渲染冲突
 */

import { memo, useState, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { CustomCompositeNodeData, GroupPort } from '../../types/graph'
import { useGraphStore } from '../../hooks/useGraphStore'

interface PackagedGroupNodeProps extends NodeProps<CustomCompositeNodeData> {}

const PackagedGroupNode = memo(({ data, id }: PackagedGroupNodeProps) => {
  // 防御性代码：确保数据定义完整
  const inputs = Array.isArray(data.inputs) ? data.inputs : []
  const outputs = Array.isArray(data.outputs) ? data.outputs : []
  const childNodeIds = Array.isArray(data.childNodeIds) ? data.childNodeIds : []
  const childCount = childNodeIds.length

  const unpackageGroup = useGraphStore((s) => s.unpackageGroup)
  const openPackageViewer = useGraphStore((s) => s.openPackageViewer)
  const closePackageViewer = useGraphStore((s) => s.closePackageViewer)
  const renamePackage = useGraphStore((s) => s.renamePackage)
  const renamePackageClass = useGraphStore((s) => s.renamePackageClass)

  const [showMenu, setShowMenu] = useState(false)

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    openPackageViewer(id)
    setShowMenu(false)
  }, [id, openPackageViewer])

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setShowMenu(true)
  }, [])

  const handleUnpackage = useCallback(() => {
    unpackageGroup(id)
    setShowMenu(false)
    closePackageViewer()
  }, [id, unpackageGroup, closePackageViewer])

  const handleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    openPackageViewer(id)
    setShowMenu(false)
  }, [id, openPackageViewer])

  const handleRename = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    const currentName = data.label || 'Module'
    const newName = window.prompt('请输入新的实例名称:', currentName)
    if (newName && newName.trim() && newName !== currentName) {
      renamePackage(id, newName.trim())
    }
    setShowMenu(false)
  }, [id, data.label, renamePackage])

  const handleRenameClass = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    const currentClassName = data.customClassId || 'Module_1'
    const newClassName = window.prompt('请输入新的类名称:', currentClassName)
    if (newClassName && newClassName.trim() && newClassName !== currentClassName) {
      renamePackageClass(id, newClassName.trim())
    }
    setShowMenu(false)
  }, [id, data.customClassId, renamePackageClass])

  const handleEditClass = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    openPackageViewer(id, 'edit')
    setShowMenu(false)
  }, [id, openPackageViewer])

  // 主容器样式
  const containerStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    minWidth: 160,
    minHeight: 80,
    borderRadius: 10,
    background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95))',
    border: '1px solid rgba(168, 85, 247, 0.4)',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    position: 'relative',
    overflow: 'visible',
  }

  return (
    <>
      <div
        style={{ width: '100%', height: '100%', position: 'relative' }}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
      >
        {/* 输入端口 - 左侧 */}
        {inputs.map((input: GroupPort, idx: number) => (
          <Handle
            key={`input-${input.id}`}
            type="target"
            position={Position.Left}
            id={input.handleId}
            style={{
              top: `${((idx + 1) / (inputs.length + 1)) * 100}%`,
              background: '#22d3ee',
              border: '2px solid #0e7490',
              width: 10,
              height: 10,
            }}
          />
        ))}

        {/* 主内容区 */}
        <div style={containerStyle}>
          {/* 状态指示器 */}
          <div
            style={{
              position: 'absolute',
              top: 4,
              left: 8,
              fontSize: 9,
              color: 'rgba(148, 163, 184, 0.5)',
              fontWeight: 500,
            }}
          >
            📦
          </div>

          {/* 主内容 */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 4,
              padding: '8px 16px',
            }}
          >
            <div style={{ fontSize: 18, opacity: 0.9 }}>📦</div>
            <div
              style={{
                color: '#e2e8f0',
                fontSize: 11,
                fontWeight: 600,
                textAlign: 'center',
                maxWidth: 140,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {data.label}
            </div>
            <div
              style={{
                color: '#a78bfa',
                fontSize: 10,
                display: 'flex',
                gap: 6,
              }}
            >
              <span>{childCount} 个模块</span>
            </div>
          </div>

          {/* 端口信息行 */}
          <div
            style={{
              position: 'absolute',
              bottom: 4,
              display: 'flex',
              gap: 12,
              fontSize: 9,
              color: 'rgba(148, 163, 184, 0.6)',
            }}
          >
            {inputs.length > 0 && (
              <span style={{ color: '#22d3ee' }}>◂{inputs.length}</span>
            )}
            {outputs.length > 0 && (
              <span style={{ color: '#4ade80' }}>{outputs.length}▸</span>
            )}
          </div>

          {/* 双击提示 */}
          <div
            style={{
              position: 'absolute',
              top: 4,
              right: 4,
              fontSize: 8,
              color: 'rgba(148, 163, 184, 0.4)',
            }}
          >
            双击查看
          </div>
        </div>

        {/* 输出端口 - 右侧 */}
        {outputs.map((output: GroupPort, idx: number) => (
          <Handle
            key={`output-${output.id}`}
            type="source"
            position={Position.Right}
            id={output.handleId}
            style={{
              top: `${((idx + 1) / (outputs.length + 1)) * 100}%`,
              background: '#4ade80',
              border: '2px solid #15803d',
              width: 10,
              height: 10,
            }}
          />
        ))}

        {/* 右键菜单 */}
        {showMenu && (
          <>
            <div
              style={{
                position: 'fixed',
                inset: 0,
                zIndex: 999,
              }}
              onClick={(e) => { e.stopPropagation(); setShowMenu(false) }}
            />
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '100%',
                marginLeft: 8,
                transform: 'translateY(-50%)',
                background: '#1a1a2e',
                border: '1px solid #333',
                borderRadius: 8,
                padding: 6,
                zIndex: 1000,
                minWidth: 140,
                boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={handleExpand}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: '#1a2a2e',
                  border: '1px solid #2d4a4d',
                  borderRadius: 6,
                  color: '#a3e635',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                🔍 查看内部结构
              </button>
              <div style={{ height: 4 }} />
              <button
                onClick={handleRename}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: '#1a2a2e',
                  border: '1px solid #2d4a4d',
                  borderRadius: 6,
                  color: '#22d3ee',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                ✏️ 重命名实例
              </button>
              <div style={{ height: 4 }} />
              <button
                onClick={handleRenameClass}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: '#1a2a2e',
                  border: '1px solid #2d4a4d',
                  borderRadius: 6,
                  color: '#a78bfa',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                🏷️ 重命名类
              </button>
              <div style={{ height: 4 }} />
              <button
                onClick={handleEditClass}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: '#1a2a2e',
                  border: '1px solid #2d4a4d',
                  borderRadius: 6,
                  color: '#f97316',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                ✏️ 编辑类定义
              </button>
              <div style={{ height: 4 }} />
              <button
                onClick={handleUnpackage}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: '#2a1a2e',
                  border: '1px solid #5c3a6e',
                  borderRadius: 6,
                  color: '#d8b4fe',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                📤 解包（还原）
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
})

PackagedGroupNode.displayName = 'PackagedGroupNode'

export default PackagedGroupNode
