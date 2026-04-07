/**
 * FlowHamster 模板 Schema 定义
 *
 * 前后端共享的模板结构定义
 * 后端模板 API 返回的数据结构必须符合此 Schema
 */

export interface TemplateNodePosition {
  x: number
  y: number
}

export interface TemplateNode {
  id: string
  type: string           // 节点类型，如 'conv2d', 'transformerencoder'
  label: string          // 显示名称
  params: Record<string, number | string | boolean>  // 节点参数
  position?: TemplateNodePosition  // 可选，布局位置
}

export interface TemplateEdge {
  id?: string
  source: string         // 源节点 ID
  target: string         // 目标节点 ID
  sourceHandle?: string  // 源端口，如 'result'
  targetHandle?: string  // 目标端口，如 'a', 'x', 'in_0'
}

export interface TemplateGraph {
  nodes: TemplateNode[]
  edges: TemplateEdge[]
}

export interface TemplateSchema {
  id: string
  name: string
  description: string
  category: 'cv' | 'nlp' | 'audio' | 'generic'
  emoji: string
  version: string
  graph: TemplateGraph
  // 后端可选：直接提供生成的代码
  backend_code?: {
    python?: string      // 生成的 Python 代码
    notebook?: string     // Jupyter Notebook 格式
  }
  // 元数据
  metadata?: {
    author?: string
    created_at?: string
    updated_at?: string
    tags?: string[]
  }
}

/**
 * 验证模板 Schema 是否合法
 */
export interface ValidationError {
  field: string
  message: string
}

export interface TemplateValidationResult {
  valid: boolean
  errors: ValidationError[]
}

/**
 * 默认模板分类
 */
export const TEMPLATE_CATEGORIES = {
  cv: 'Computer Vision',
  nlp: 'Natural Language Processing',
  audio: 'Audio',
  generic: 'Generic',
} as const

export type TemplateCategory = keyof typeof TEMPLATE_CATEGORIES
