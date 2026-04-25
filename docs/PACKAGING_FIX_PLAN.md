# 打包功能修复与完善计划

## 实现状态：✅ 全部完成

## 修复内容

### 1. 移除 Group 功能 ✅
- Toolbar.tsx - 删除 Group 按钮
- useGraphStore.ts - 删除 `groupSelection` 函数及相关导入

### 2. 修复展开/收起 ✅
- 正确转换 `nodeType` → React Flow `type`（如 `'conv2d'` → `'conv2dNode'`）
- 完整保留 `params` 参数
- 正确处理边的 `sourceHandle` 和 `targetHandle`

### 3. 端口显示 ✅
- 显示输入/输出端口数量
- 收起时显示 `◂N`（输入）和 `N▸`（输出）
- 端口排序固定

### 4. 多选高亮 ✅
- Ctrl 多选时节点添加发光边框
- 高亮颜色：`#6366f1`（紫色）

### 5. 内部预览 ✅
- 显示模块数量
- 显示展开/收起提示

### 6. 边连接修复 ✅
- 打包时正确保存跨边界边（外部→内部、内部→外部）
- 展开时恢复外部边
- 收起时移除外部边
- 解包时完整恢复所有边

---

## 核心修复：边管理

### 问题根因

**原问题**：打包时只保存了完全在组内的边（`A→B`, `B→C`），丢弃了跨边界边（`X→A`, `C→Y`）

### 修复方案

#### GroupPort 保存原始边 ID

```typescript
interface GroupPort {
  id: string
  handleId: string
  label: string
  nodeId: string
  edgeId?: string  // 新增：原始边 ID
}
```

#### 打包时推断外部边

```typescript
function inferInputPorts(nodeIds, edges) {
  // 边从组外进入组内
  if (!nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
    inputPorts.push({
      ...
      nodeId: edge.target,  // 连接到内部节点
      edgeId: edge.id,      // 保存原始边 ID
    })
  }
}
```

#### 展开时恢复外部边

```typescript
function expandPackage(...) {
  // 恢复外部边（从外部进入内部）
  for (const input of inputs) {
    if (input.edgeId) {
      const originalEdge = allEdges.find((e) => e.id === input.edgeId)
      if (originalEdge) {
        restoredExternalEdges.push({
          ...originalEdge,
          target: input.nodeId,
          targetHandle: input.handleId,
        })
      }
    }
  }
}
```

#### 收起时移除外部边

```typescript
function collapsePackage(...) {
  // 移除内部边 AND 连接到内部节点的外部边
  const childNodeAndBoundaryEdgeIds = new Set([
    ...childNodeIds,
    ...inputs.map((i) => i.edgeId).filter(Boolean),
    ...outputs.map((o) => o.edgeId).filter(Boolean),
  ])

  const filteredEdges = allEdges.filter((e) => {
    if (internalEdgeIdSet.has(e.id)) return false
    if (childNodeAndBoundaryEdgeIds.has(e.id)) return false
    return true
  })
}
```

---

## 用户交互

| 操作 | 方式 | 效果 |
|------|------|------|
| 打包 | 选中节点 → 📦 Package | 创建黑盒模块，保留外部连接 |
| 展开 | 双击 或 选中 → 🔼 Expand | 显示内部结构，恢复外部连接 |
| 收起 | 双击 或 选中 → 🔽 Collapse | 隐藏内部结构，移除外部连接 |
| 解包 | 右键 → 📦 解包为普通节点 | 完整还原为普通节点 |

---

## 待完善项

- [x] 移除 Group 功能 ✅
- [x] 展开/收起修复 ✅
- [x] 端口显示 ✅
- [x] 多选高亮 ✅
- [x] 内部预览 ✅
- [x] 边连接修复 ✅
