# FlowHamster 代码优化计划

## 问题汇总

### 1. 安全问题 (Critical) - ✅ 已修复
- `new Function()` 代码注入风险 → 使用安全求值器 `safeEval.ts`

### 2. 代码重复 (High) - ✅ 已修复
- BFS 遍历逻辑重复 → 提取到 `graphUtils.ts`
- 拓扑排序重复 → 提取到 `graphUtils.ts`
- reverseAdj 构建重复 → 提取到 `graphUtils.ts`
- `makeGraphId()` 重复 → 提取到 `graphUtils.ts`，统一使用
- `inferInputPorts`/`inferOutputPorts` 重复 → 合并为 `inferBoundaryPorts`

### 3. 类型安全 (Medium) - ✅ 已改进
- `as any` 用法已审查（多为 React Flow 类型系统问题）
- `allEdges.find()` 已添加 null 检查

### 4. 可维护性 (Medium) - 进行中
- codeGenerator.ts 1736 行过大 → 待拆分
- useGraphStore.ts 815 行 → 待拆分
- subgraphPackager.ts 逻辑重复 → ✅ 已优化

### 5. 魔法数字 (Low) - 待处理
- ID 生成参数 `slice(2, 9)` 统一

---

## 已完成

### Step 1: 提取公共工具函数 ✅
- [x] 创建 `src/utils/graphUtils.ts`
- [x] 提取 `buildReverseAdj()`
- [x] 提取 `topologicalSort()`
- [x] 提取 `makeGraphId()` - 统一使用
- [x] 提取 `bfsFromNodes()`
- [x] 提取 `safeArrayGet()` 和 `isDefined()`

### Step 2: 修复安全问题 ✅
- [x] 创建 `src/utils/safeEval.ts`
- [x] 替换 `new Function()` 为 `safeEvaluate()`
- [x] `codeGenerator.ts` 使用安全求值器

### Step 3: 添加强类型 ✅
- [x] 代码已有基本 null 检查

### Step 4: 简化 subgraphPackager ✅
- [x] 合并 `inferInputPorts`/`inferOutputPorts` 为 `inferBoundaryPorts`
- [x] 使用 `graphUtils.ts` 的 `makeGraphId`

### Step 5: 工作流持久化 ✅
- [x] 创建 `backend/routers/workflows.py` - REST API 后端
- [x] 注册 workflows 路由到 `backend/main.py`
- [x] 创建 `src/stores/useWorkflowStore.ts` - 前端状态管理
- [x] 创建 `src/components/dialogs/WorkflowDialog.tsx` - 工作流管理对话框
- [x] Toolbar 添加工作流按钮和 Ctrl+S/Ctrl+A 快捷键
- [x] 工作流存储于 `backend/workflows/{workflow_id}/` 目录

---

## 待处理

### Step 6: 重构大文件（风险高）
- codeGenerator.ts 拆分（风险高，可能破坏功能）
- useGraphStore.ts 拆分（风险高）

### Step 7: 清理代码
- [ ] 统一命名规范
- [ ] 添加 JSDoc 注释
- [ ] 移除未使用导入
- [ ] 提取魔法数字为常量
