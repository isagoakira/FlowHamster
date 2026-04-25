# FlowHamster 架构优化计划

## 当前问题
1. 节点类型定义分散在多个文件，无统一源头
2. Handle ID 使用字符串字面量，无枚举约束
3. 节点组件注册、模板定义、代码生成三者信息不同步
4. 复合节点的 internalStructure 与代码生成逻辑重复定义

---

## 优化步骤

### Step 1: 统一节点类型定义
**目标**：一个文件定义所有节点类型、参数、Handle IDs

**文件**：`src/utils/nodeDefinition.ts`

**内容**：
```typescript
export enum NodeCategory {
  INPUT_OUTPUT = 'io',
  CONVOLUTION = 'conv',
  ACTIVATION = 'activation',
  NORMALIZATION = 'norm',
  POOLING = 'pool',
  ATTENTION = 'attention',
  TENSOR_OP = 'tensor',
  REGULARIZATION = 'reg',
  LOSS = 'loss',
  OPTIMIZER = 'optimizer',
}

export interface HandleDefinition {
  id: string
  position: 'left' | 'right' | 'top' | 'bottom'
  label?: string
}

export interface NodeDefinition {
  type: string                    // 节点类型，如 'transformerencoder'
  label: string                  // 显示名称
  category: NodeCategory
  description: string
  params: Record<string, ParamDefinition>
  handles: {
    inputs: HandleDefinition[]
    outputs: HandleDefinition[]
  }
  isComposite: boolean
  internalStructure?: SubModuleDefinition[]  // 复合节点内部定义
}

export interface ParamDefinition {
  type: 'number' | 'string' | 'boolean' | 'select'
  default: any
  description?: string
}
```

**状态**：TODO

---

### Step 2: 组件注册哈希化
**目标**：用 Map/Object 替代 switch-case，用枚举替代字符串字面量

**文件**：`src/components/canvas/Canvas.tsx`

**改动**：
1. 创建 `NODE_COMPONENT_MAP` 常量
2. 使用 `Object.freeze()` 冻结防止修改
3. 移除硬编码的 switch-case

**状态**：TODO

---

### Step 3: Handle ID 枚举化
**目标**：消除字符串字面量，用枚举替代

**文件**：`src/constants/handleIds.ts`

```typescript
export enum HandleId {
  // 基础
  INPUT = 'a',
  OUTPUT = 'result',

  // 复合节点
  ENCODER_X = 'x',
  ENCODER_TGT = 'tgt',

  // Concat
  CONCAT_IN_0 = 'in_0',
  CONCAT_IN_1 = 'in_1',
  CONCAT_IN_2 = 'in_2',

  // Add/Mul
  ADD_A = 'a',
  ADD_B = 'b',

  // Attention
  ATTN_Q = 'q',
  ATTN_K = 'k',
  ATTN_V = 'v',
  ATTN_WEIGHT = 'weight',
}
```

**状态**：TODO

---

### Step 4: 功能层面剥离去耦合
**目标**：各模块职责清晰，减少耦合

**拆分方案**：

1. **`src/utils/templateRegistry.ts`** → 纯数据定义，移除业务逻辑
2. **`src/utils/codeGenerator.ts`** → 纯代码生成，与 UI 解耦
3. **`src/utils/nodeRegistry.ts`** → 节点元数据，与代码生成解耦
4. **新建 `src/utils/handleRegistry.ts`** → Handle ID 统一管理

**状态**：TODO

---

### Step 5: 模板验证机制
**目标**：加载模板时验证节点类型和 Handle ID 合法性

**文件**：`src/utils/templateValidator.ts`

```typescript
export function validateTemplate(template: Template): ValidationResult {
  const errors: string[] = []

  for (const node of template.graph.nodes) {
    if (!NODE_DEFINITIONS[node.data.nodeType]) {
      errors.push(`Unknown node type: ${node.data.nodeType}`)
    }
  }

  for (const edge of template.graph.edges) {
    const targetNode = template.graph.nodes.find(n => n.id === edge.target)
    if (targetNode) {
      const nodeDef = NODE_DEFINITIONS[targetNode.data.nodeType]
      const validHandles = nodeDef.handles.inputs.map(h => h.id)
      if (!validHandles.includes(edge.targetHandle)) {
        errors.push(`Invalid handle ${edge.targetHandle} for node ${targetNode.data.nodeType}`)
      }
    }
  }

  return { valid: errors.length === 0, errors }
}
```

**状态**：TODO

---

## 实施顺序

1. ✅ **Step 0**: 备份当前版本
2. ⬜ **Step 1**: 统一节点类型定义 (`nodeDefinition.ts`)
3. ⬜ **Step 2**: 创建 Handle ID 枚举 (`handleIds.ts`)
4. ⬜ **Step 3**: 重构 nodeRegistry.ts 使用新枚举
5. ⬜ **Step 4**: 组件注册哈希化
6. ⬜ **Step 5**: 剥离去耦合
7. ⬜ **Step 6**: 模板验证机制

---

## 当前进度

| Step | 状态 | 完成日期 |
|------|------|----------|
| Step 0 (备份) | ✅ 完成 | 2026-04-02 |
| Step 1 (统一节点类型定义) | ✅ 完成 | 2026-04-02 |
| Step 2 (组件注册哈希化) | ✅ 完成 | 2026-04-02 |
| Step 3 (节点类型枚举化) | ✅ 完成 | 2026-04-02 |
| Step 4 (模板验证) | ✅ 完成 | 2026-04-02 |
| Step 5 (剥离去耦合) | ✅ 完成 | 2026-04-02 |

## Step 1 完成内容

- 创建 `src/constants/handleIds.ts` - Handle ID 枚举定义
- 创建 `src/utils/nodeDefinition.ts` - 统一节点定义（Single Source of Truth）
  - 定义了 50+ 种节点类型
  - 每个节点包含: type, label, category, description, params, handles, isComposite
  - 复合节点包含: internalStructure, internalEdges, outputVar
  - 辅助函数: getNodeDef, nodeTypeExists, isValidHandleId, getAllNodeTypes, getNodeTypesByCategory

## Step 2 完成内容

- 创建 `src/utils/nodeComponentRegistry.ts` - 节点组件注册表
  - 统一的组件映射管理
  - 使用 Object.freeze() 防止运行时修改
  - getReactFlowNodeTypes() 生成 React Flow nodeTypes
  - hasNodeComponent() 检查组件是否存在
- 更新 `Canvas.tsx` 使用新的注册表替代硬编码

## Step 3 完成内容

- 更新 `src/utils/nodeType.ts` 使用统一的节点定义
  - normalizeNodeType() 现在使用 nodeDefinition.ts 的 nodeTypeExists()
  - getNodeComponentType() 使用 nodeComponentRegistry.ts 的 getComponentName()
  - 新增 isValidNodeType() 和 getAllValidNodeTypes() 辅助函数

## Step 4 完成内容 (模板验证)

- 创建 `src/utils/templateValidator.ts` - 模板验证器
  - validateNodeTypes() - 验证节点类型是否有效
  - validateEdges() - 验证边的连接是否有效
  - validateTemplate() - 完整验证模板
  - validateTemplateWithMessages() - 返回人类可读的错误信息
- 更新 `useGraphStore.ts` 的 loadTemplate() 使用验证器
  - 加载模板时自动验证，警告信息输出到 console

## Step 5 完成内容 (剥离去耦)

- 更新 `codeGenerator.ts` 使用 `nodeDefinition.ts`
  - getCompositeNodeDef() 现在优先从 nodeDefinition.ts 获取定义
  - 向后兼容 NODE_REGISTRY（nodeRegistry.ts）
  - 新定义优先，旧的 NODE_REGISTRY 作为回退
