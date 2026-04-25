/**
 * Custom Composite Class Registry
 * Manages user-defined custom composite modules saved from composite node viewers
 * Persists to localStorage
 */

import { SubModule, InternalEdge } from './nodeRegistry'
import { SubModuleData, InternalEdgeData } from '../types/graph'

export interface CustomCompositeClass {
  id: string
  name: string
  baseType: string  // 'mlp' | 'ffn' | 'transformerencoder' | etc.
  category: 'cv' | 'nlp' | 'gan' | 'other'
  emoji: string
  description: string
  internalStructure: SubModuleData[]
  internalEdges: InternalEdgeData[]
  outputVar: string
  codeTemplate: string
  createdAt: number
}

const STORAGE_KEY = 'flowhamster_custom_composites'

// Convert SubModule[] to SubModuleData[]
function toSubModuleData(structure: SubModule[]): SubModuleData[] {
  return structure.map(s => ({
    id: s.id,
    type: s.type,
    label: s.label,
    params: { ...s.params },
  }))
}

// Convert InternalEdge[] to InternalEdgeData[]
function toInternalEdgeData(edges: InternalEdge[]): InternalEdgeData[] {
  return edges.map(e => ({ from: e.from, to: e.to }))
}

/**
 * Load all custom classes from localStorage
 */
export function loadCustomClasses(): CustomCompositeClass[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return []
    return JSON.parse(stored)
  } catch {
    console.error('Failed to load custom classes from localStorage')
    return []
  }
}

/**
 * Save all custom classes to localStorage
 */
function saveCustomClasses(classes: CustomCompositeClass[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(classes))
  } catch {
    console.error('Failed to save custom classes to localStorage')
  }
}

/**
 * Generate a unique ID for a new custom class
 */
function generateId(): string {
  return `custom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Topological sort using Kahn's algorithm — handles branching (DAG) correctly.
 * Returns nodes in execution order.
 */
function topologicalSort(nodes: SubModuleData[], edges: InternalEdgeData[]): string[] {
  const nodeIds = new Set(nodes.map(n => n.id))
  const inDegree = new Map<string, number>()
  const adj = new Map<string, string[]>()

  nodes.forEach(n => { inDegree.set(n.id, 0); adj.set(n.id, []) })
  edges.forEach(e => {
    if (nodeIds.has(e.from) && nodeIds.has(e.to)) {
      inDegree.set(e.to, (inDegree.get(e.to) ?? 0) + 1)
      adj.get(e.from)!.push(e.to)
    }
  })

  const queue: string[] = []
  inDegree.forEach((deg, id) => { if (deg === 0) queue.push(id) })

  const order: string[] = []
  while (queue.length > 0) {
    const current = queue.shift()!
    order.push(current)
    for (const next of adj.get(current) ?? []) {
      const newDeg = (inDegree.get(next) ?? 1) - 1
      inDegree.set(next, newDeg)
      if (newDeg === 0) queue.push(next)
    }
  }

  // Any remaining nodes (cycles) — append them
  nodes.forEach(n => { if (!order.includes(n.id)) order.push(n.id) })

  return order
}

/**
 * Build a map: nodeId → [successorIds]
 */
function buildSuccessorMap(edges: InternalEdgeData[]): Map<string, string[]> {
  const succ = new Map<string, string[]>()
  edges.forEach(e => {
    if (!succ.has(e.from)) succ.set(e.from, [])
    succ.get(e.from)!.push(e.to)
  })
  return succ
}

/**
 * Convert node type to a readable PyTorch module name.
 */
function toPyTorchModule(type: string): string {
  const MAP: Record<string, string> = {
    conv1d: 'Conv1d', conv2d: 'Conv2d', conv3d: 'Conv3d',
    linear: 'Linear', relu: 'ReLU', gelu: 'GELU', silu: 'SiLU',
    sigmoid: 'Sigmoid', tanh: 'Tanh', leakyrelu: 'LeakyReLU',
    maxpool2d: 'MaxPool2d', avgpool2d: 'AvgPool2d',
    adaptiveavgpool2d: 'AdaptiveAvgPool2d', globalavgpool: 'AdaptiveAvgPool2d',
    batchnorm2d: 'BatchNorm2d', layernorm: 'LayerNorm', groupnorm: 'GroupNorm',
    dropout: 'Dropout', softmax: 'Softmax', flatten: 'Flatten',
    reshape: 'Reshape',
  }
  return MAP[type] ?? type.charAt(0).toUpperCase() + type.slice(1)
}

/**
 * Format params as Python keyword arguments.
 */
function formatParams(params: Record<string, unknown>): string {
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(', ')
}

/**
 * Generate a clean, readable Python class for a custom composite module.
 * Handles branching correctly via Kahn topological sort.
 */
export function generateCustomClassCode(cls: CustomCompositeClass): string {
  const { name, internalStructure, internalEdges } = cls

  if (internalStructure.length === 0) {
    return `class ${name}(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x`
  }

  const nodeMap = new Map<string, SubModuleData>(
    internalStructure.map(n => [n.id, n])
  )

  const order = topologicalSort(internalStructure, internalEdges)
  const succMap = buildSuccessorMap(internalEdges)

  // ── __init__: one submodule per node ──────────────────────────────────────
  const initLines: string[] = []
  for (const node of internalStructure) {
    const mod = toPyTorchModule(node.type)
    const params = formatParams(node.params)
    initLines.push(`        self.${node.id} = nn.${mod}(${params})`)
  }

  // ── Forward: topological order with intermediate variables ──────────────────
  // Determine which nodes need intermediate variables (nodes used as inputs
  // by multiple successors or receiving the initial input).
  const successors = buildSuccessorMap(internalEdges)
  const nodeUsedBy = new Map<string, number>()
  internalEdges.forEach(e => nodeUsedBy.set(e.to, (nodeUsedBy.get(e.to) ?? 0) + 1))

  // Identify the input node(s) — nodes with no incoming edges (or the first in order)
  const targets = new Set(internalEdges.map(e => e.to))
  const inputNodeId = order[0] // guaranteed by topological sort to have in-degree 0 or be first

  // Determine if each node produces a named intermediate (not just pass-through)
  const needsVar = new Set<string>()
  order.forEach(id => {
    const sucs = successors.get(id) ?? []
    const used = nodeUsedBy.get(id) ?? 0
    if (used !== 1 || sucs.length > 1) needsVar.add(id)
  })

  const fwdLines: string[] = []
  for (const id of order) {
    const node = nodeMap.get(id)!
    const sucs = successors.get(id) ?? []

    if (sucs.length === 0) {
      // Output node — last in the chain
      fwdLines.push(`        x = self.${id}(x)`)
    } else if (id === inputNodeId && needsVar.has(id)) {
      // First node that also branches
      fwdLines.push(`        x = self.${id}(x)`)
    } else if (needsVar.has(id)) {
      // Intermediate node with multiple consumers
      fwdLines.push(`        x = self.${id}(x)`)
    } else {
      // Pass-through — inlining
      fwdLines.push(`        x = self.${id}(x)`)
    }
  }

  fwdLines.push('        return x')

  return `class ${name}(nn.Module):
    def __init__(self):
        super().__init__()
${initLines.join('\n')}

    def forward(self, x):
${fwdLines.join('\n')}`
}

/**
 * Register a new custom composite class
 * 如果同名类已存在，则不注册，直接返回已存在的类
 */
export function registerCustomClass(
  name: string,
  baseType: string,
  category: 'cv' | 'nlp' | 'gan' | 'other',
  emoji: string,
  description: string,
  internalStructure: SubModule[],
  internalEdges: InternalEdge[],
  outputVar: string
): CustomCompositeClass {
  const classes = loadCustomClasses()

  // 如果同名类已存在，直接返回
  const existing = classes.find(c => c.name === name)
  if (existing) {
    return existing
  }

  const newClass: CustomCompositeClass = {
    id: generateId(),
    name,
    baseType,
    category,
    emoji,
    description,
    internalStructure: toSubModuleData(internalStructure),
    internalEdges: toInternalEdgeData(internalEdges),
    outputVar,
    codeTemplate: '',
    createdAt: Date.now(),
  }

  // Generate code template
  newClass.codeTemplate = generateCustomClassCode(newClass)

  classes.push(newClass)
  saveCustomClasses(classes)

  return newClass
}

/**
 * Update an existing custom class
 */
export function updateCustomClass(
  id: string,
  updates: Partial<Omit<CustomCompositeClass, 'id' | 'createdAt'>>
): CustomCompositeClass | null {
  const classes = loadCustomClasses()
  const index = classes.findIndex(c => c.id === id)

  if (index === -1) return null

  const updated = { ...classes[index], ...updates }

  // Regenerate code template if structure changed
  if (updates.internalStructure || updates.internalEdges || updates.name) {
    updated.codeTemplate = generateCustomClassCode(updated)
  }

  classes[index] = updated
  saveCustomClasses(classes)

  return updated
}

/**
 * Unregister a custom class by ID
 */
export function unregisterCustomClass(id: string): boolean {
  const classes = loadCustomClasses()
  const filtered = classes.filter(c => c.id !== id)

  if (filtered.length === classes.length) return false

  saveCustomClasses(filtered)
  return true
}

/**
 * Get a custom class by ID
 */
export function getCustomClass(id: string): CustomCompositeClass | undefined {
  const classes = loadCustomClasses()
  return classes.find(c => c.id === id)
}

/**
 * Get all custom classes
 */
export function getAllCustomClasses(): CustomCompositeClass[] {
  return loadCustomClasses()
}

/**
 * Get custom classes by category
 */
export function getCustomClassesByCategory(category: 'cv' | 'nlp' | 'gan' | 'other'): CustomCompositeClass[] {
  const classes = loadCustomClasses()
  return classes.filter(c => c.category === category)
}

/**
 * Convert custom classes to NodeCategory format for sidebar display
 */
export function getCustomClassesAsNodeCategory(): { label: string; nodes: any[] } {
  const classes = loadCustomClasses()

  const categoryMap: Record<string, CustomCompositeClass[]> = {
    cv: [],
    nlp: [],
    gan: [],
    other: [],
  }

  classes.forEach(cls => {
    if (categoryMap[cls.category]) {
      categoryMap[cls.category].push(cls)
    }
  })

  const categories: { label: string; nodes: any[] }[] = []

  if (categoryMap.cv.length > 0) {
    categories.push({
      label: 'Custom / Computer Vision',
      nodes: categoryMap.cv.map(cls => ({
        type: cls.id,
        label: `${cls.emoji} ${cls.name}`,
        description: cls.description || `Custom ${cls.baseType} module`,
        defaultParams: {},
        outputType: 'Tensor',
      })),
    })
  }

  if (categoryMap.nlp.length > 0) {
    categories.push({
      label: 'Custom / NLP',
      nodes: categoryMap.nlp.map(cls => ({
        type: cls.id,
        label: `${cls.emoji} ${cls.name}`,
        description: cls.description || `Custom ${cls.baseType} module`,
        defaultParams: {},
        outputType: 'Tensor',
      })),
    })
  }

  if (categoryMap.gan.length > 0) {
    categories.push({
      label: 'Custom / GAN',
      nodes: categoryMap.gan.map(cls => ({
        type: cls.id,
        label: `${cls.emoji} ${cls.name}`,
        description: cls.description || `Custom ${cls.baseType} module`,
        defaultParams: {},
        outputType: 'Tensor',
      })),
    })
  }

  if (categoryMap.other.length > 0) {
    categories.push({
      label: 'Custom / Other',
      nodes: categoryMap.other.map(cls => ({
        type: cls.id,
        label: `${cls.emoji} ${cls.name}`,
        description: cls.description || `Custom ${cls.baseType} module`,
        defaultParams: {},
        outputType: 'Tensor',
      })),
    })
  }

  return { label: 'Custom', nodes: categories.flatMap(c => c.nodes) }
}
