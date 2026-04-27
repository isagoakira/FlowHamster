/**
 * Custom Composite Class Registry
 * Manages user-defined custom composite modules saved from composite node viewers
 * Persists to localStorage
 */

import { SubModule, InternalEdge } from './nodeRegistry'
import { SubModuleData, InternalEdgeData, GroupPort, BoundaryEdge } from '../types/graph'
import { genPythonNodeForward, genPythonNodeInit, safePythonName } from './pythonNodeRegistry'

export interface CustomCompositeClass {
  id: string
  name: string
  originClassId?: string
  parentClassId?: string
  baseType: string  // 'mlp' | 'ffn' | 'transformerencoder' | etc.
  category: 'cv' | 'nlp' | 'gan' | 'other'
  emoji: string
  description: string
  internalStructure: SubModuleData[]
  internalEdges: InternalEdgeData[]
  outputVar: string
  inputs?: GroupPort[]
  outputs?: GroupPort[]
  boundaryEdges?: BoundaryEdge[]
  codeTemplate: string
  createdAt: number
}

const STORAGE_KEY = 'flowhamster_custom_composites'
export const CUSTOM_CLASSES_CHANGED_EVENT = 'flowhamster:custom-classes-changed'

type RegistrableSubModule = SubModule | SubModuleData
type RegistrableInternalEdge = InternalEdge | InternalEdgeData

function readFromHandle(edge: RegistrableInternalEdge): string | undefined {
  return 'fromHandle' in edge ? edge.fromHandle : (edge as InternalEdge).fromPort
}

function readToHandle(edge: RegistrableInternalEdge): string | undefined {
  return 'toHandle' in edge ? edge.toHandle : (edge as InternalEdge).toPort
}

// Convert SubModule[] to SubModuleData[]
function toSubModuleData(structure: RegistrableSubModule[]): SubModuleData[] {
  return structure.map(s => ({
    id: s.id,
    type: s.type,
    label: s.label,
    params: { ...s.params },
    position: 'position' in s ? s.position : undefined,
    customClassId: 'customClassId' in s ? s.customClassId : undefined,
    data: 'data' in s ? s.data : undefined,
  }))
}

// Convert InternalEdge[] to InternalEdgeData[]
function toInternalEdgeData(edges: RegistrableInternalEdge[]): InternalEdgeData[] {
  return edges.map(e => ({
    id: 'id' in e ? e.id : undefined,
    from: e.from,
    to: e.to,
    fromHandle: readFromHandle(e),
    toHandle: readToHandle(e),
  }))
}

/**
 * Load all custom classes from localStorage
 */
export function loadCustomClasses(): CustomCompositeClass[] {
  try {
    if (typeof localStorage === 'undefined') return []
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
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent(CUSTOM_CLASSES_CHANGED_EVENT))
    }
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
 * Generate a clean, readable Python class for a custom composite module.
 * Handles branching correctly via Kahn topological sort.
 */
export function generateCustomClassCode(cls: CustomCompositeClass): string {
  const { name, internalStructure, internalEdges, outputVar } = cls
  const className = safePythonName(name)

  if (internalStructure.length === 0) {
    return `class ${className}(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x`
  }

  const order = topologicalSort(internalStructure, internalEdges)
  const nodeMap = new Map(internalStructure.map((node) => [node.id, node]))
  const incoming = new Map<string, InternalEdgeData[]>()
  for (const edge of internalEdges) {
    if (!incoming.has(edge.to)) incoming.set(edge.to, [])
    incoming.get(edge.to)!.push(edge)
  }

  const initLines: string[] = []
  for (const node of internalStructure) {
    const instanceName = safePythonName(node.id)
    const initLine = genPythonNodeInit({
      instanceName,
      opType: node.customClassId ? 'custom' : node.type,
      fields: node.params,
      customClassName: node.customClassId,
    })
    if (initLine) initLines.push(`        ${initLine}`)
  }

  const fwdLines: string[] = []
  const varMap = new Map<string, string>()
  for (const id of order) {
    const node = nodeMap.get(id)
    if (!node) continue

    const instanceName = safePythonName(node.id)
    const outVar = `x_${instanceName}`
    const inputVars = (incoming.get(id) ?? [])
      .map((edge) => varMap.get(edge.from))
      .filter((value): value is string => Boolean(value))
    const opType = node.customClassId ? 'custom' : node.type
    const line = genPythonNodeForward(opType, outVar, instanceName, inputVars.length > 0 ? inputVars : ['x'], node.params)
    if (line) fwdLines.push(`        ${line}`)
    varMap.set(id, outVar)
  }

  const finalVar = varMap.get(outputVar) ?? (order.length > 0 ? varMap.get(order[order.length - 1]) : null) ?? 'x'
  fwdLines.push(`        return ${finalVar}`)

  return `class ${className}(nn.Module):
    def __init__(self):
        super().__init__()
${initLines.length > 0 ? initLines.join('\n') : '        pass'}

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
  internalStructure: RegistrableSubModule[],
  internalEdges: RegistrableInternalEdge[],
  outputVar: string,
  options: {
    inputs?: GroupPort[]
    outputs?: GroupPort[]
    boundaryEdges?: BoundaryEdge[]
    originClassId?: string
    parentClassId?: string
  } = {}
): CustomCompositeClass {
  const classes = loadCustomClasses()

  // 如果同名类已存在，直接返回
  const existing = classes.find(c => c.name === name)
  if (existing) {
    return existing
  }

  const id = generateId()
  const newClass: CustomCompositeClass = {
    id,
    name,
    originClassId: options.originClassId ?? id,
    parentClassId: options.parentClassId,
    baseType,
    category,
    emoji,
    description,
    internalStructure: toSubModuleData(internalStructure),
    internalEdges: toInternalEdgeData(internalEdges),
    outputVar,
    inputs: options.inputs ?? [],
    outputs: options.outputs ?? [],
    boundaryEdges: options.boundaryEdges ?? [],
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
        id: cls.id,
        type: 'custom',
        classId: cls.id,
        customClassId: cls.name,
        originClassId: cls.originClassId ?? cls.id,
        label: cls.name,
        displayLabel: `${cls.emoji} ${cls.name}`,
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
        id: cls.id,
        type: 'custom',
        classId: cls.id,
        customClassId: cls.name,
        originClassId: cls.originClassId ?? cls.id,
        label: cls.name,
        displayLabel: `${cls.emoji} ${cls.name}`,
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
        id: cls.id,
        type: 'custom',
        classId: cls.id,
        customClassId: cls.name,
        originClassId: cls.originClassId ?? cls.id,
        label: cls.name,
        displayLabel: `${cls.emoji} ${cls.name}`,
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
        id: cls.id,
        type: 'custom',
        classId: cls.id,
        customClassId: cls.name,
        originClassId: cls.originClassId ?? cls.id,
        label: cls.name,
        displayLabel: `${cls.emoji} ${cls.name}`,
        description: cls.description || `Custom ${cls.baseType} module`,
        defaultParams: {},
        outputType: 'Tensor',
      })),
    })
  }

  return { label: 'Custom', nodes: categories.flatMap(c => c.nodes) }
}
