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
 * Generate Python code template for a custom composite class
 */
export function generateCustomClassCode(cls: CustomCompositeClass): string {
  const { name, internalStructure, internalEdges } = cls

  // Build the module dictionary for __init__
  const module_inits: string[] = []
  const forward_lines: string[] = []

  // Input assignment - pass through the input tensor, processing will be done in topological order
  // The first node in topological order will receive x as input

  // Process edges to determine order
  const processed = new Set<string>()
  const edgeMap = new Map<string, string>()
  internalEdges.forEach(e => {
    edgeMap.set(e.from, e.to)
  })

  // Find starting node(s) - nodes that aren't targets of any edge
  const targets = new Set(internalEdges.map(e => e.to))
  const starts = internalStructure.filter(s => !targets.has(s.id))
  if (starts.length === 0 && internalStructure.length > 0) {
    starts.push(internalStructure[0])
  }

  // Topological sort
  const order: string[] = []
  const queue = starts.map(s => s.id)
  while (queue.length > 0) {
    const current = queue.shift()!
    if (processed.has(current)) continue
    processed.add(current)
    order.push(current)

    const next = edgeMap.get(current)
    if (next && !processed.has(next)) {
      queue.push(next)
    }
  }

  // Add any unprocessed nodes
  internalStructure.forEach(s => {
    if (!processed.has(s.id)) {
      order.push(s.id)
    }
  })

  // Build init lines and forward lines
  internalStructure.forEach(s => {
    const paramsStr = Object.entries(s.params)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(', ')
    module_inits.push(`        self.${s.id} = nn.${s.type.charAt(0).toUpperCase() + s.type.slice(1)}(${paramsStr})`)
  })

  // Build forward pass
  order.forEach(id => {
    const node = internalStructure.find(s => s.id === id)
    if (!node) return

    const next = edgeMap.get(id)
    if (node.type === 'relu' || node.type === 'gelu' || node.type === 'silu' ||
        node.type === 'sigmoid' || node.type === 'tanh' || node.type === 'leakyrelu') {
      forward_lines.push(`        x = self.${node.id}(x)`)
    } else if (node.type === 'dropout' || node.type === 'droppath') {
      forward_lines.push(`        x = self.${node.id}(x)`)
    } else if (node.type === 'layernorm' || node.type === 'batchnorm2d' || node.type === 'groupnorm') {
      forward_lines.push(`        x = self.${node.id}(x)`)
    } else if (node.type === 'softmax') {
      forward_lines.push(`        x = self.${node.id}(x)`)
    } else if (next) {
      forward_lines.push(`        x = self.${node.id}(x)`)
    }
  })

  forward_lines.push(`        return x`)

  return `class ${name}(nn.Module):
    def __init__(self):
        super().__init__()
${module_inits.join('\n')}

    def forward(self, x):
${forward_lines.join('\n')}`
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
