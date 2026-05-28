/**
 * 子图打包器 v2 - 重构版
 *
 * 改进:
 * 1. Edge ID 保留 - 解包时恢复原始 edge ID
 * 2. 拓扑布局 - 展开时使用拓扑排序计算位置
 * 3. 自动尺寸 - Package 节点大小自适应
 * 4. 完整解包 - 支持完整解包操作
 * 5. 清晰视觉状态 - 明确区分收起/展开状态
 */

import { Node } from 'reactflow'
import {
  FlowHamsterNode,
  FlowHamsterEdge,
  CustomCompositeNodeData,
  GroupPort,
  SubModuleData,
  InternalEdgeData,
  NodeData,
  NodeType,
} from '../types/graph'
import { getNodeComponentType } from './nodeType'

// 自定义复合模块的节点类型 (必须是 'customNode' 以匹配 nodeTypes 注册表)
const CUSTOM_COMPOSITE_TYPE = 'customNode'

function cloneData<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function uniqueId(baseId: string, usedIds: Set<string>): string {
  if (!usedIds.has(baseId)) {
    usedIds.add(baseId)
    return baseId
  }

  let index = 1
  let candidate = `${baseId}_${index}`
  while (usedIds.has(candidate)) {
    index += 1
    candidate = `${baseId}_${index}`
  }
  usedIds.add(candidate)
  return candidate
}

function createIdMap(
  internalStructure: SubModuleData[],
  allNodes: FlowHamsterNode[],
  groupId: string
): Map<string, string> {
  const usedIds = new Set(allNodes.filter((node) => node.id !== groupId).map((node) => node.id))
  const idMap = new Map<string, string>()

  for (const sub of internalStructure) {
    idMap.set(sub.id, uniqueId(sub.id, usedIds))
  }

  return idMap
}

function restoreNodeFromSubModule(
  sub: SubModuleData,
  idMap: Map<string, string>,
  fallbackPosition: { x: number; y: number },
  parentNode?: string
): FlowHamsterNode {
  const nodeData = sub.data
    ? cloneData(sub.data)
    : ({
        nodeType: sub.type as NodeType,
        label: sub.label,
        params: { ...sub.params },
      } as NodeData)

  const restored: FlowHamsterNode = {
    id: idMap.get(sub.id) ?? sub.id,
    type: getNodeComponentType(String(nodeData.nodeType ?? sub.type)),
    position: sub.position || fallbackPosition,
    data: nodeData,
    ...(parentNode ? { parentNode, extent: 'parent' as const } : {}),
  } as FlowHamsterNode

  if (sub.customClassId) {
    ;(restored.data as CustomCompositeNodeData).isComposite = true
    ;(restored.data as CustomCompositeNodeData).isCustomComposite = true
    ;(restored.data as CustomCompositeNodeData).customClassId = sub.customClassId
    ;(restored.data as CustomCompositeNodeData).isExpanded = false
  }

  return restored
}

function uniqueEdgeId(baseId: string, usedIds: Set<string>): string {
  return uniqueId(baseId, usedIds)
}

function getVisibleGroupChildIds(
  groupId: string,
  allNodes: FlowHamsterNode[],
  data: CustomCompositeNodeData
): Set<string> {
  const childNodeIds = new Set(data.childNodeIds ?? [])
  return new Set(
    allNodes
      .filter((node) => {
        const parentId = node.parentNode ?? node.parentId
        return parentId === groupId || (data.isExpanded && childNodeIds.has(node.id))
      })
      .map((node) => node.id)
  )
}

// === Boundary Edge Mapping (Deterministic, no fallback) ===

/**
 * Lookup by original edge ID (for restoring exact edges during unpackage).
 */
export function findBoundaryMappingByOriginalEdgeId(
  boundaryEdges: BoundaryEdgeData[],
  originalEdgeId: string
): BoundaryEdgeData | undefined {
  return boundaryEdges.find((be) => be.originalEdgeId === originalEdgeId)
}

function restoreCurrentBoundaryEdges(
  groupId: string,
  allEdges: FlowHamsterEdge[],
  boundaryEdges: BoundaryEdgeData[] | undefined,
  idMap: Map<string, string>
): FlowHamsterEdge[] {
  if (!boundaryEdges || boundaryEdges.length === 0) return []

  const restored: FlowHamsterEdge[] = []

  const findMapping = (
    direction: 'input' | 'output',
    currentEdgeId: string,
    currentGroupHandle?: string | null
  ): BoundaryEdgeData | undefined => {
    const candidates = boundaryEdges.filter((be) => be.direction === direction)

    const exact = candidates.find((be) => be.originalEdgeId === currentEdgeId)
    if (exact) return exact

    if (currentGroupHandle) {
      const byGroupHandle = candidates.find((be) => be.groupHandleId === currentGroupHandle)
      if (byGroupHandle) return byGroupHandle

      // Backward compatibility for packages created before groupHandleId was
      // guaranteed to be a stable package-port ID.
      const byInternalHandle = candidates.find((be) => be.internalHandle === currentGroupHandle)
      if (byInternalHandle) return byInternalHandle
    }

    if (candidates.length === 1) return candidates[0]
    return undefined
  }

  for (const edge of allEdges) {
    if (edge.target === groupId) {
      // INPUT edge: external source -> group
      // 1. Exact edge ID match (stable, preferred)
      const mapping = findMapping('input', edge.id, edge.targetHandle)
      if (!mapping) continue

      const internalNodeId = idMap.get(mapping.internalNodeId) ?? mapping.internalNodeId
      const targetHandle = mapping.internalHandle !== 'default'
        ? mapping.internalHandle
        : null

      restored.push({
        id: edge.id,
        source: edge.source,
        target: internalNodeId,
        sourceHandle: edge.sourceHandle,
        targetHandle,
      })
    } else if (edge.source === groupId) {
      // OUTPUT edge: group -> external target
      // 1. Exact edge ID match (stable, preferred)
      const mapping = findMapping('output', edge.id, edge.sourceHandle)
      if (!mapping) continue

      const internalNodeId = idMap.get(mapping.internalNodeId) ?? mapping.internalNodeId
      const sourceHandle = mapping.internalHandle !== 'default'
        ? mapping.internalHandle
        : null

      restored.push({
        id: edge.id,
        source: internalNodeId,
        target: edge.target,
        sourceHandle,
        targetHandle: edge.targetHandle,
      })
    }
  }

  return restored
}

// === Deep ID Remapping (for paste with nested packages) ===

/**
 * Recursively remaps all IDs in a SubModuleData array.
 * Handles nested custom composites inside internalStructure.
 */
export function deepRemapNodeIds(
  structure: SubModuleData[],
  nodeIdMap: Map<string, string>
): SubModuleData[] {
  return structure.map((sub) => {
    const remappedId = nodeIdMap.get(sub.id) ?? sub.id

    const remappedInternalStructure = sub.data?.internalStructure
      ? deepRemapNodeIds(sub.data.internalStructure as SubModuleData[], nodeIdMap)
      : undefined

    return {
      ...sub,
      id: remappedId,
      data: sub.data ? { ...sub.data, internalStructure: remappedInternalStructure } : undefined,
    }
  })
}

/**
 * Recursively remaps all from/to IDs in an InternalEdgeData array.
 */
export function deepRemapInternalEdges(
  edges: InternalEdgeData[],
  nodeIdMap: Map<string, string>
): InternalEdgeData[] {
  return edges.map((edge) => ({
    ...edge,
    from: nodeIdMap.get(edge.from) ?? edge.from,
    to: nodeIdMap.get(edge.to) ?? edge.to,
  }))
}

// Kept only for backwards-compatible tests; names are now derived from the
// active graph rather than session/global localStorage state.
let moduleCounter = 0

/**
 * 获取下一个模块名称
 * 确保持续生成唯一的模块名称，避免与已存在的类名冲突
 */
export function getNextModuleName(): string {
  return getNextModuleNameForNodes([])
}

export function getNextModuleNameForNodes(nodes: FlowHamsterNode[]): string {
  const existingNames = new Set(
    nodes.flatMap((node) => [
      node.id,
      (node.data as CustomCompositeNodeData)?.customClassId,
    ]).filter((name): name is string => Boolean(name))
  )

  for (let index = 1; index <= 1000; index++) {
    const name = `Module_${index}`
    if (!existingNames.has(name)) {
      return name
    }
  }
  moduleCounter += 1
  return `Module_${Date.now()}_${moduleCounter}`
}

/**
 * 重置模块计数器（用于测试）
 */
export function resetModuleCounter(): void {
  moduleCounter = 0
}

/**
 * 边界边数据 - 保留原始 edge ID
 */
export interface BoundaryEdgeData {
  originalEdgeId: string
  direction: 'input' | 'output'
  internalNodeId: string
  internalHandle: string
  groupHandleId: string
  // 原始边的完整信息（用于恢复）
  source: string
  target: string
  sourceHandle?: string | null
  targetHandle?: string | null
}

/**
 * 推断边界端口和边界边
 */
function inferBoundaryInfo(
  nodeIds: Set<string>,
  edges: FlowHamsterEdge[]
): { inputPorts: GroupPort[]; outputPorts: GroupPort[]; boundaryEdges: BoundaryEdgeData[] } {
  const inputPorts: GroupPort[] = []
  const outputPorts: GroupPort[] = []
  const boundaryEdges: BoundaryEdgeData[] = []

  for (const edge of edges) {
    const isInput = !nodeIds.has(edge.source) && nodeIds.has(edge.target)
    const isOutput = nodeIds.has(edge.source) && !nodeIds.has(edge.target)

    if (isInput) {
      const actualHandle = edge.targetHandle || 'input'
      let existingPort = inputPorts.find((port) => port.nodeId === edge.target && port.label === actualHandle)
      if (!existingPort) {
        const groupHandleId = `input_${inputPorts.length}`
        inputPorts.push({
          id: groupHandleId,
          handleId: groupHandleId,
          label: actualHandle,
          nodeId: edge.target,
          edgeId: edge.id,
        })
        existingPort = inputPorts[inputPorts.length - 1]
      }
      boundaryEdges.push({
        originalEdgeId: edge.id,
        direction: 'input',
        internalNodeId: edge.target,
        internalHandle: edge.targetHandle || 'default',
        groupHandleId: existingPort.handleId,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      })
    } else if (isOutput) {
      const actualHandle = edge.sourceHandle || 'output'
      let existingPort = outputPorts.find((port) => port.nodeId === edge.source && port.label === actualHandle)
      if (!existingPort) {
        const groupHandleId = `output_${outputPorts.length}`
        outputPorts.push({
          id: groupHandleId,
          handleId: groupHandleId,
          label: actualHandle,
          nodeId: edge.source,
          edgeId: edge.id,
        })
        existingPort = outputPorts[outputPorts.length - 1]
      }
      boundaryEdges.push({
        originalEdgeId: edge.id,
        direction: 'output',
        internalNodeId: edge.source,
        internalHandle: edge.sourceHandle || 'default',
        groupHandleId: existingPort.handleId,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      })
    }
  }

  return { inputPorts, outputPorts, boundaryEdges }
}

/**
 * 计算拓扑排序和节点层级
 */
export function computeTopologicalLayout(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[]
): Map<string, { layer: number; order: number }> {
  const nodeIds = new Set(nodes.map((n) => n.id))
  const inDegree = new Map<string, number>()
  const adjList = new Map<string, string[]>()

  // 初始化
  for (const node of nodes) {
    inDegree.set(node.id, 0)
    adjList.set(node.id, [])
  }

  // 构建邻接表和入度
  for (const edge of edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      adjList.get(edge.source)!.push(edge.target)
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1)
    }
  }

  // BFS 层序遍历
  const layout = new Map<string, { layer: number; order: number }>()
  const queue: string[] = []
  const layerGroups = new Map<number, string[]>()

  // 找到所有入度为 0 的节点
  for (const [nodeId, degree] of inDegree) {
    if (degree === 0) {
      queue.push(nodeId)
    }
  }

  let layer = 0
  while (queue.length > 0) {
    const size = queue.length
    const currentLayer: string[] = []
    for (let i = 0; i < size; i++) {
      const nodeId = queue.shift()!
      currentLayer.push(nodeId)
      layout.set(nodeId, { layer, order: i })

      for (const neighbor of adjList.get(nodeId) || []) {
        const newDegree = (inDegree.get(neighbor) || 0) - 1
        inDegree.set(neighbor, newDegree)
        if (newDegree === 0) {
          queue.push(neighbor)
        }
      }
    }
    layerGroups.set(layer, currentLayer)
    layer++
  }

  return layout
}

/**
 * 打包节点为自定义复合模块
 * @param moduleName 可选的模块名称，如果不提供则自动生成 Module_X 格式的名称
 */
export function packageNodes(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  nodeIds: string[],
  position: { x: number; y: number },
  moduleName?: string
): Node<CustomCompositeNodeData> {
  const nodeIdSet = new Set(nodeIds)

  // 内部边：两端都在组内
  const internalEdgesRaw = edges.filter(
    (e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target)
  )

  // 推断边界信息
  const { inputPorts, outputPorts, boundaryEdges } = inferBoundaryInfo(nodeIdSet, edges)

  // 选中的节点
  const selectedNodes = nodes.filter((n) => nodeIdSet.has(n.id))

  // 转换子模块数据（保存原始位置）
  // 对于自定义复合节点，也保存 customClassId
  const internalStructure: SubModuleData[] = selectedNodes.map((node) => {
    const base = {
      id: node.id,
      type: node.data.nodeType as string,
      label: node.data.label,
      params: { ...node.data.params },
      position: { x: node.position.x, y: node.position.y },
      data: cloneData(node.data),
    }
    // If it's a custom composite node, preserve the customClassId
    if (node.data.nodeType === 'custom' && (node.data as CustomCompositeNodeData).customClassId) {
      return {
        ...base,
        customClassId: (node.data as CustomCompositeNodeData).customClassId,
      }
    }
    return base
  })
  const internalEdges: InternalEdgeData[] = internalEdgesRaw.map((edge) => ({
    id: edge.id,
    from: edge.source,
    to: edge.target,
    fromHandle: edge.sourceHandle ?? undefined,
    toHandle: edge.targetHandle ?? undefined,
  }))

  // 生成唯一类 ID（使用提供的名称或自动生成）
  const customClassId = moduleName || getNextModuleNameForNodes(nodes)

  // 计算包节点大小（基于端口数量）
  const portCount = Math.max(inputPorts.length, outputPorts.length, 1)
  const width = 180
  const height = Math.max(100, 60 + portCount * 24)

  const groupNode: Node<CustomCompositeNodeData> = {
    id: customClassId,
    type: CUSTOM_COMPOSITE_TYPE,
    position,
    data: {
      nodeType: 'custom',
      label: customClassId,
      params: {},
      isComposite: true,
      isCustomComposite: true,
      customClassId,
      isExpanded: false,
      internalStructure,
      internalEdges,
      outputVar: outputPorts[0]?.nodeId || 'x',
      inputs: inputPorts,
      outputs: outputPorts,
      childNodeIds: nodeIds,
      internalEdgeIds: internalEdgesRaw.map((e) => e.id),
      boundaryEdges: boundaryEdges as any,
    },
    style: {
      width,
      height,
    },
  }

  return groupNode
}

/**
 * 展开复合模块，显示内部结构
 */
export function expandPackage(
  groupNode: Node<CustomCompositeNodeData>,
  allNodes: FlowHamsterNode[],
  allEdges: FlowHamsterEdge[]
): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  if (!groupNode.data.isCustomComposite) {
    return { nodes: allNodes, edges: allEdges }
  }

  const liveGroup = allNodes.find((node) => node.id === groupNode.id) as Node<CustomCompositeNodeData> | undefined
  if ((liveGroup?.data as CustomCompositeNodeData | undefined)?.isExpanded) {
    return { nodes: allNodes, edges: allEdges }
  }

  // F4: Validate before expand
  const validation = preExpandValidation(groupNode, allNodes, allEdges)
  if (!validation.isValid) {
    console.error('[subgraphPackager] Pre-expand validation failed:', validation.errors)
  }
  for (const w of validation.warnings) {
    console.warn('[subgraphPackager] Pre-expand validation warning:', w.message)
  }

  const data = groupNode.data as CustomCompositeNodeData
  const { internalStructure, internalEdges: storedEdges, boundaryEdges } = data

  // 计算组内节点的位置（使用原始位置或包节点位置作为基础）
  const baseX = groupNode.position.x
  const baseY = groupNode.position.y

  const visibleChildIds = getVisibleGroupChildIds(groupNode.id, allNodes, data)
  const collisionNodes = allNodes.filter((node) => !visibleChildIds.has(node.id))
  const idMap = createIdMap(internalStructure, collisionNodes, groupNode.id)

  // 恢复内部节点（使用原始位置，composite 子节点使用新 ID）
  const internalNodes: FlowHamsterNode[] = internalStructure.map((sub) => (
    restoreNodeFromSubModule(sub, idMap, { x: baseX + 220, y: baseY + 60 }, groupNode.id)
  ))

  // 恢复当前实例真实存在的外部边；boundaryEdges 只作为 handle -> 内部节点的映射模板。
  const restoredExternalEdges = restoreCurrentBoundaryEdges(
    groupNode.id,
    allEdges,
    boundaryEdges as BoundaryEdgeData[] | undefined,
    idMap
  )

  // 标记组为展开状态
  const updatedGroup: FlowHamsterNode = {
    ...groupNode,
    data: {
      ...data,
      isExpanded: true,
    },
  } as FlowHamsterNode

  // 过滤掉被包装的子节点，但排除组节点本身（由 updatedGroup 替代）
  const filteredNodes = allNodes.filter(
    (n) => !visibleChildIds.has(n.id) && n.id !== groupNode.id
  )

  // 移除当前实例的 package 边和已存在的可见内部边，避免误删其他实例/其他节点的同名边。
  const filteredEdges = allEdges.filter((edge) =>
    edge.source !== groupNode.id &&
    edge.target !== groupNode.id &&
    !visibleChildIds.has(edge.source) &&
    !visibleChildIds.has(edge.target)
  )

  // 恢复内部边（使用新 ID 映射）
  // 检查是否已存在同名边，避免重复添加（处理重新展开的情况）
  const existingEdgeIds = new Set(filteredEdges.map(e => e.id))
  const internalEdgeList: FlowHamsterEdge[] = storedEdges
    .map((edge) => ({
      id: uniqueEdgeId(edge.id || `e_${edge.from}_${edge.to}`, existingEdgeIds),
      source: idMap.get(edge.from) ?? edge.from,
      target: idMap.get(edge.to) ?? edge.to,
      sourceHandle: edge.fromHandle,
      targetHandle: edge.toHandle,
    }))

  return {
    nodes: [...filteredNodes, ...internalNodes, updatedGroup],
    edges: [...filteredEdges, ...internalEdgeList, ...restoredExternalEdges],
  }
}

/**
 * 收起展开的组
 */
export function collapsePackage(
  groupId: string,
  allNodes: FlowHamsterNode[],
  allEdges: FlowHamsterEdge[]
): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  const groupNode = allNodes.find(
    (n) => n.id === groupId && (n.data as CustomCompositeNodeData)?.isCustomComposite
  ) as Node<CustomCompositeNodeData> | undefined

  if (!groupNode || !(groupNode.data as CustomCompositeNodeData).isExpanded) {
    return { nodes: allNodes, edges: allEdges }
  }

  const data = groupNode.data as CustomCompositeNodeData
  const visibleChildIds = getVisibleGroupChildIds(groupId, allNodes, data)

  // 标记组为收起状态
  const updatedGroup: FlowHamsterNode = {
    ...groupNode,
    data: {
      ...data,
      isExpanded: false,
    },
  } as FlowHamsterNode

  // 移除内部节点
  const filteredNodes = allNodes.filter(
    (n) => !visibleChildIds.has(n.id) && n.id !== groupId
  )

  const filteredEdges = allEdges.filter((edge) =>
    edge.source !== groupId &&
    edge.target !== groupId &&
    !visibleChildIds.has(edge.source) &&
    !visibleChildIds.has(edge.target)
  )

  return {
    nodes: [...filteredNodes, updatedGroup],
    edges: filteredEdges,
  }
}

/**
 * 解包组，完全还原为普通节点
 * 恢复所有原始边 ID 和节点位置
 */
export function fullyUnpackageGroup(
  groupId: string,
  allNodes: FlowHamsterNode[],
  allEdges: FlowHamsterEdge[]
): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  const groupNode = allNodes.find(
    (n) => n.id === groupId && (n.data as CustomCompositeNodeData)?.isCustomComposite
  ) as Node<CustomCompositeNodeData> | undefined

  if (!groupNode) {
    return { nodes: allNodes, edges: allEdges }
  }

  // F4: Validate before unpackage
  const validation = preExpandValidation(groupNode, allNodes, allEdges)
  if (!validation.isValid) {
    console.error('[subgraphPackager] Pre-unpackage validation failed:', validation.errors)
    return { nodes: allNodes, edges: allEdges }
  }

  const data = groupNode.data as CustomCompositeNodeData
  const { internalStructure, internalEdges: storedEdges, boundaryEdges } = data

  const visibleChildIds = getVisibleGroupChildIds(groupId, allNodes, data)
  const collisionNodes = allNodes.filter((node) => !visibleChildIds.has(node.id))
  const idMap = createIdMap(internalStructure, collisionNodes, groupId)

  // 恢复内部节点（使用原始位置，composite 子节点使用新 ID）
  const restoredNodes: FlowHamsterNode[] = internalStructure.map((sub) => (
    restoreNodeFromSubModule(sub, idMap, { x: groupNode.position.x + 220, y: groupNode.position.y + 60 })
  ))

  // 恢复内部边（使用新 ID 映射）
  const usedEdgeIds = new Set(allEdges.filter((edge) => edge.id && edge.id !== groupId).map((edge) => edge.id!))
  const restoredEdges: FlowHamsterEdge[] = storedEdges.map((edge) => ({
    id: uniqueEdgeId(edge.id || `e_${edge.from}_${edge.to}`, usedEdgeIds),
    source: idMap.get(edge.from) ?? edge.from,
    target: idMap.get(edge.to) ?? edge.to,
    sourceHandle: edge.fromHandle,
    targetHandle: edge.toHandle,
  }))

  restoredEdges.push(...restoreCurrentBoundaryEdges(
    groupId,
    allEdges,
    boundaryEdges as BoundaryEdgeData[] | undefined,
    idMap
  ))

  // 只移除这个实例当前可见的壳、子节点和连接，不能按历史 hidden edge id 误删其他实例的边。
  const filteredNodes = allNodes.filter((n) => n.id !== groupId && !visibleChildIds.has(n.id))
  const filteredEdges = allEdges.filter((edge) =>
    edge.source !== groupId &&
    edge.target !== groupId &&
    !visibleChildIds.has(edge.source) &&
    !visibleChildIds.has(edge.target)
  )

  return {
    nodes: [...filteredNodes, ...restoredNodes],
    edges: [...filteredEdges, ...restoredEdges],
  }
}

export const unpackageGroup = fullyUnpackageGroup

// === Validation (F4) ===

export interface ValidationError {
  type: 'MISSING_NODE' | 'ORPHANED_EDGE' | 'INVALID_REFERENCE' | 'DUPLICATE_ID' | 'BROKEN_BOUNDARY'
  message: string
  nodeId?: string
  edgeId?: string
}

export interface ValidationWarning {
  type: 'DEGENERATE' | 'UNUSED' | 'CIRCULAR'
  message: string
  nodeId?: string
}

export interface ValidationResult {
  isValid: boolean
  errors: ValidationError[]
  warnings: ValidationWarning[]
}

/**
 * Validates internal structure integrity before expand/unpackage.
 */
export function validateInternalStructure(
  internalStructure: SubModuleData[],
  internalEdges: InternalEdgeData[],
  boundaryEdges?: BoundaryEdgeData[],
  options: { strictBoundaryCheck?: boolean } = {}
): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // 1. Check for duplicate IDs
  const nodeIdCounts = new Map<string, number>()
  for (const sub of internalStructure) {
    const count = nodeIdCounts.get(sub.id) ?? 0
    nodeIdCounts.set(sub.id, count + 1)
  }
  for (const [id, count] of nodeIdCounts) {
    if (count > 1) {
      errors.push({ type: 'DUPLICATE_ID', message: `Duplicate node ID "${id}"`, nodeId: id })
    }
  }

  // 2. Check all internalEdges reference valid nodes
  const validNodeIds = new Set(internalStructure.map(s => s.id))
  for (const edge of internalEdges) {
    if (!validNodeIds.has(edge.from)) {
      errors.push({ type: 'ORPHANED_EDGE', message: `Edge references non-existent node "${edge.from}"`, edgeId: edge.id, nodeId: edge.from })
    }
    if (!validNodeIds.has(edge.to)) {
      errors.push({ type: 'ORPHANED_EDGE', message: `Edge references non-existent node "${edge.to}"`, edgeId: edge.id, nodeId: edge.to })
    }
  }

  // 3. Check boundary edge integrity
  if (boundaryEdges && options.strictBoundaryCheck) {
    for (const be of boundaryEdges) {
      if (!validNodeIds.has(be.internalNodeId)) {
        errors.push({ type: 'BROKEN_BOUNDARY', message: `Boundary edge references internal node "${be.internalNodeId}" that does not exist`, nodeId: be.internalNodeId })
      }
    }
  }

  // 4. Degenerate structure warnings
  if (internalStructure.length === 0) {
    warnings.push({ type: 'DEGENERATE', message: 'Internal structure is empty' })
  }
  if (internalStructure.length === 1 && internalEdges.length > 0) {
    warnings.push({ type: 'DEGENERATE', message: 'Single-node internal structure with edges', nodeId: internalStructure[0].id })
  }

  return { isValid: errors.length === 0, errors, warnings }
}

/**
 * Validates boundary edge mapping is unambiguous (no handle maps to multiple internal nodes).
 */
export function validateBoundaryMappingConsistency(
  boundaryEdges: BoundaryEdgeData[]
): ValidationResult {
  const errors: ValidationError[] = []

  const handleToNodes = new Map<string, Set<string>>()
  for (const be of boundaryEdges) {
    const key = `${be.direction}:${be.groupHandleId}`
    if (!handleToNodes.has(key)) handleToNodes.set(key, new Set())
    handleToNodes.get(key)!.add(be.internalNodeId)
  }

  for (const [key, nodeSet] of handleToNodes) {
    if (nodeSet.size > 1) {
      const [direction, handleId] = key.split(':')
      errors.push({ type: 'INVALID_REFERENCE', message: `Ambiguous boundary mapping: handle "${handleId}" (${direction}) maps to multiple internal nodes: ${[...nodeSet].join(', ')}` })
    }
  }

  return { isValid: errors.length === 0, errors, warnings: [] }
}

/**
 * Pre-expand validation for a package node.
 */
export function preExpandValidation(
  groupNode: Node<CustomCompositeNodeData>,
  _allNodes: FlowHamsterNode[],
  _allEdges: FlowHamsterEdge[]
): ValidationResult {
  const data = groupNode.data as CustomCompositeNodeData

  if (!data.internalStructure || data.internalStructure.length === 0) {
    return {
      isValid: false,
      errors: [{ type: 'INVALID_REFERENCE', message: 'Package has no internalStructure', nodeId: groupNode.id }],
      warnings: [],
    }
  }

  const result = validateInternalStructure(
    data.internalStructure,
    data.internalEdges ?? [],
    data.boundaryEdges as BoundaryEdgeData[],
    { strictBoundaryCheck: true }
  )

  // Prefix errors with group node context
  result.errors = result.errors.map(e => ({ ...e, message: `[${groupNode.id}] ${e.message}` }))
  result.warnings = result.warnings.map(w => ({ ...w, message: `[${groupNode.id}] ${w.message}` }))

  return result
}

/**
 * 获取组的大小（基于内容和端口）
 */
export function getGroupDimensions(
  _nodes: FlowHamsterNode[],
  inputCount: number,
  outputCount: number
): { width: number; height: number } {
  const portCount = Math.max(inputCount, outputCount, 1)
  const height = Math.max(100, 60 + portCount * 24)
  return {
    width: 180,
    height,
  }
}
