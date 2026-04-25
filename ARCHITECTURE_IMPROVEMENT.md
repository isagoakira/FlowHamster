# FlowHamster 架构改进方案

## 当前问题

### 1. 前后端模板分离
```
前端: src/utils/templateRegistry.ts (硬编码 UI 模板)
       ↓ 不同步
后端: backend/templates/*.py (真实 PyTorch 实现)
```

用户修改后端 `.py` 文件，前端 UI 不会更新。

### 2. 节点定义分散
```
nodeDefinition.ts (节点元数据)
       ↓ 不同步
codeGenerator.ts (代码生成逻辑)
       ↓ 不同步
nodeRegistry.ts (旧定义)
       ↓ 不同步
Canvas.tsx (UI 组件)
```

### 3. 没有实时同步机制
前端改代码需要重新构建，后端改代码需要重启服务。

---

## 改进目标

**后端作为单一数据源，前端动态加载**

```
backend/templates/ (真实实现)
        ↓ 模板定义
   /api/templates (REST API)
        ↓ JSON
frontend (动态加载)
        ↓
验证 + 渲染 + 代码生成
```

---

## 实施步骤

### Step A: 定义统一模板 Schema
**目标**：前后端共享同一个模板结构定义

**文件**：`src/schema/template.ts`

**状态**：✅ 完成

---

### Step B: 后端模板 API
**目标**：后端提供 `/api/templates` 返回所有模板

**文件**：
- `backend/templates/registry.py` - 模板注册表
- `backend/routers/templates.py` - API 路由

**状态**：✅ 完成

---

### Step C: 前端动态加载模板
**目标**：前端从后端 API 加载模板，替代硬编码

**文件**：`src/hooks/useTemplateStore.ts`

```typescript
interface TemplateStore {
  templates: Template[]
  loadTemplates: () => Promise<void>
  getTemplate: (id: string) => Template | undefined
}

// 使用 zustand
export const useTemplateStore = create<TemplateStore>((set, get) => ({
  templates: [],

  loadTemplates: async () => {
    const response = await fetch('/api/templates')
    const templates = await response.json()
    set({ templates })
  },

  getTemplate: (id) => get().templates.find(t => t.id === id)
}))
```

**状态**：✅ 完成

---

### Step D: 模板 Schema 验证
**目标**：验证从后端加载的模板是否符合 Schema

**文件**：`src/utils/validateTemplateSchema.ts`

**状态**：✅ 完成

---

### Step E: 代码生成结果缓存
**目标**：避免重复生成代码，缓存后端返回的代码

**文件**：`src/stores/codeCache.ts`

```typescript
interface CodeCache {
  get: (templateId: string) => string | undefined
  set: (templateId: string, code: string) => void
  invalidate: (templateId: string) => void
}
```

**状态**：✅ 完成

---

## 实施顺序

| Step | 内容 | 依赖 |
|------|------|------|
| Step A | 定义统一模板 Schema | - |
| Step B | 后端模板 API | Step A |
| Step C | 前端动态加载模板 | Step B |
| Step D | Schema 验证 | Step A, C |
| Step E | 代码生成缓存 | Step C |

---

## 文件变更清单

### 新增文件
- `src/schema/template.ts` - 模板 Schema
- `src/hooks/useTemplateStore.ts` - 模板状态管理
- `src/utils/validateTemplateSchema.ts` - Schema 验证
- `backend/templates/registry.py` - 模板注册表

### 修改文件
- `backend/main.py` - 添加 `/api/templates` 路由
- `src/utils/templateRegistry.ts` - 改为从 store 加载
- `src/components/*` - 使用新的模板加载机制

### 删除文件
- `src/utils/templateRegistry.ts` (将被替代)
- 硬编码的模板定义

---

## 预期效果

1. **同步**：修改后端模板文件后，前端自动加载最新版本
2. **验证**：加载的模板经过 Schema 验证
3. **解耦**：前后端通过 API 通信，不再硬编码
4. **可测试**：可以单独测试后端模板 API
