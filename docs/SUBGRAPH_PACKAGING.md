# 子图打包 / 自定义复合模块

## 实现状态：✅ 全部完成（2026-04-20 重构）

## 概述

**打包组**和**复合模块**是同一概念的不同名称：
- 前端：显示为可展开的复合模块
- 后端：生成对应的 Python 类

**关键设计原则**：类的 identity（`customClassId`）和实例的 identity（`node.id`）严格分离。
- `customClassId` = 类名，跨实例共享，表示"这个节点是哪个类"  
- `node.id` = 实例唯一 ID，每个实例独立，用于边引用

## 架构统一

### 节点类型体系

```
NodeData
├── 基础节点 (conv2d, linear, etc.)
├── 预定义复合模块 (transformerencoder, selfattention, etc.)
└── 自定义复合模块 (user-created via packaging)
    - customClassId: 类名（不变）
    - node.id: 实例唯一ID（展开时分配fresh ID，solve嵌套冲突）
```

### 数据结构

```typescript
// 复合模块基类
interface CompositeNodeBase extends NodeData {
  isComposite: true
  customClassId: string        // 类身份，恒定不变
  isExpanded: boolean         // true = 展开显示内部，false = 折叠显示图标
  internalStructure: SubModuleData[]
  internalEdges: InternalEdgeData[]
  outputVar: string
}

// 预定义复合模块（编辑器内置）
interface PredefinedCompositeNodeData extends CompositeNodeBase {
  isPredefined: true
  compositeType: string  // 'transformerencoder' | 'selfattention' | etc.
}

// 用户创建的复合模块（通过打包）
interface CustomCompositeNodeData extends CompositeNodeBase {
  isCustomComposite: true
  customClassId: string
  inputs: GroupPort[]      // 自动推断的输入端口
  outputs: GroupPort[]     // 自动推断的输出端口
  isExpanded: boolean       // UI 状态
  childNodeIds: string[]
  internalEdgeIds: string[]
}
```

### 端口自动推断

打包时自动分析边的连接：
- **输入端口**：从组外连接到组内的边
- **输出端口**：从组内连接到组外的边

---

## 核心操作

| 操作 | 函数 | 说明 |
|------|------|------|
| 打包 | `packageSelection()` | 选中节点 → 创建复合模块 |
| 展开 | `expandGroup(id)` | 显示内部结构 |
| 收起 | `collapseGroup(id)` | 隐藏内部结构 |
| 解包 | `unpackageGroup(id)` | 还原为普通节点 |

---

## 文件变更

### 新增/修改

| 文件 | 变更 |
|------|------|
| `src/types/graph.ts` | 统一复合模块类型体系 |
| `src/utils/subgraphPackager.ts` | 打包/展开核心逻辑 |
| `src/components/nodes/PackagedGroupNode.tsx` | 复合模块 UI 组件 + 双击展开/收起 |
| `src/hooks/useGraphStore.ts` | 添加 packageSelection + registerCustomClass |
| `src/utils/nodeComponentRegistry.ts` | 注册 `custom` 类型 → PackagedGroupNode |
| `src/components/toolbar/Toolbar.tsx` | 添加 Package/Expand/Collapse 按钮 |
| `src/utils/codeGenerator.ts` | 支持 `custom` 类型代码生成 |
| `src/components/canvas/Canvas.tsx` | 添加右键菜单解包选项 |

---

## 用户交互

1. **打包**：选中节点 → 点击"📦 Package" → 自动推断 I/O → 创建黑盒模块
2. **展开**：选中复合模块 → 点击"🔼 Expand" 或 **双击** → 显示内部结构
3. **收起**：选中已展开模块 → 点击"🔽 Collapse" 或 **双击** → 恢复黑盒状态
4. **解包**：右键点击复合模块 → "📦 解包为普通节点" → 还原为普通节点

---

## 代码生成流程

1. 打包时调用 `registerCustomClass()` 注册到 `customCompositeRegistry`
2. 代码生成时检测 `block.opType === 'custom'`
3. 使用 `block.nodeId`（等于 `customClassId`）从注册表查找
4. 生成对应的 Python 类代码

---

## 嵌套打包

支持嵌套打包：
1. 展开一个复合模块
2. 选中其内部的部分节点
3. 再次打包 → 创建嵌套的复合模块

---

## 待完成

- [x] 代码生成：自定义复合模块生成 Python 类 ✅
- [x] 双击交互展开/收起 ✅
- [x] 右键菜单解包选项 ✅
- [x] 嵌套打包支持 ✅
