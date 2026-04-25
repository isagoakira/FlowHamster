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
  NodeType,
} from '../types/graph'
import { getNodeComponentType } from './nodeType'
import { getAllCustomClasses } from './customCompositeRegistry'

// 自定义复合模块的节点类型 (必须是 'customNode' 以匹配 nodeTypes 注册表)
const CUSTOM_COMPOSITE_TYPE = 'customNode'

// 模块计数器，用于生成可读的模块名称
// 注意：此计数器仅在当前会话内有效，不持久化到 localStorage
// 这样可以避免刷新页面后编号继续递增的问题
let moduleCounter = 0

/**
 * 获取下一个模块名称
 * 确保持续生成唯一的模块名称，避免与已存在的类名冲突
 */
export function getNextModuleName(): string {
  const existingNames = new Set(getAllCustomClasses().map(c => c.name))

  // 如果当前计数器对应的名称已被使用，则递增直到找到可用名称
  while (true) {
    moduleCounter++
    const name = `Module_${moduleCounter}`
    if (!existingNames.has(name)) {
      return name
    }
    // 如果计数器过大（超过1000），重置以避免无限循环
    if (moduleCounter > 1000) {
      moduleCounter = 0
    }
  }
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

  // 用于去重 - 同一个 handle 只取第一个 edge
  const seenInputs = new Set<string>()
  const seenOutputs = new Set<string>()

  for (const edge of edges) {
    const isInput = !nodeIds.has(edge.source) && nodeIds.has(edge.target)
    const isOutput = nodeIds.has(edge.source) && !nodeIds.has(edge.target)

    if (isInput) {
      const key = `${edge.target}-${edge.targetHandle || 'default'}`
      if (!seenInputs.has(key)) {
        seenInputs.add(key)
        inputPorts.push({
          id: `input_${inputPorts.length}`,
          handleId: edge.targetHandle || 'default',
          label: edge.targetHandle || 'input',
          nodeId: edge.target,
          edgeId: edge.id,
        })
        boundaryEdges.push({
          originalEdgeId: edge.id,
          direction: 'input',
          internalNodeId: edge.target,
          internalHandle: edge.targetHandle || 'default',
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
        })
      }
    } else if (isOutput) {
      const key = `${edge.source}-${edge.sourceHandle || 'default'}`
      if (!seenOutputs.has(key)) {
        seenOutputs.add(key)
        outputPorts.push({
          id: `output_${outputPorts.length}`,
          handleId: edge.sourceHandle || 'default',
          label: edge.sourceHandle || 'output',
          nodeId: edge.source,
          edgeId: edge.id,
        })
        boundaryEdges.push({
          originalEdgeId: edge.id,
          direction: 'output',
          internalNodeId: edge.source,
          internalHandle: edge.sourceHandle || 'default',
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
        })
      }
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
    from: edge.source,
    to: edge.target,
    fromHandle: edge.sourceHandle ?? undefined,
    toHandle: edge.targetHandle ?? undefined,
  }))

  // 生成唯一类 ID（使用提供的名称或自动生成）
  const customClassId = moduleName || getNextModuleName()

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

  const data = groupNode.data as CustomCompositeNodeData
  const { internalStructure, internalEdges: storedEdges, childNodeIds, boundaryEdges } = data

  // 计算组内节点的位置（使用原始位置或包节点位置作为基础）
  const baseX = groupNode.position.x
  const baseY = groupNode.position.y

  // Build a mapping: composite sub-nodes get fresh instance IDs to avoid
  // collisions when the same composite class has multiple instances.
  // Regular nodes keep their original IDs.
  const idMap = new Map<string, string>()
  for (const sub of internalStructure) {
    if ((sub as any).customClassId) {
      // Composite sub-node: assign a fresh instance ID
      idMap.set(sub.id, `${(sub as any).customClassId}_instance_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`)
    }
    // else: keep original node ID
  }

  // 恢复内部节点（使用原始位置，composite 子节点使用新 ID）
  const internalNodes: FlowHamsterNode[] = internalStructure.map((sub) => {
    const newId = idMap.get(sub.id) ?? sub.id
    const position = sub.position || {
      x: baseX + 220,
      y: baseY + 60,
    }
    const baseNode: FlowHamsterNode = {
      id: newId,
      type: getNodeComponentType(sub.type),
      position,
      data: {
        nodeType: sub.type as NodeType,
        label: sub.label,
        params: { ...sub.params },
      },
      parentNode: groupNode.id,
      extent: 'parent',
    } as FlowHamsterNode

    // If it's a nested composite, preserve its class identity and mark non-expanded
    if ((sub as any).customClassId) {
      ;(baseNode.data as any).isCustomComposite = true
      ;(baseNode.data as any).customClassId = (sub as any).customClassId
      ;(baseNode.data as any).isExpanded = false
    }

    return baseNode
  })

  // 恢复外部边（使用边界边信息，重定向到内部节点）
  const restoredExternalEdges: FlowHamsterEdge[] = []

  if (boundaryEdges) {
    for (const be of boundaryEdges as BoundaryEdgeData[]) {
      if (be.direction === 'input') {
        // 输入边：外部 -> 内部，重定向到外部 -> 内部节点
        restoredExternalEdges.push({
          id: be.originalEdgeId,
          source: be.source,
          target: idMap.get(be.internalNodeId) ?? be.internalNodeId, // 重定向到内部节点
          sourceHandle: be.sourceHandle,
          targetHandle: be.internalHandle,
        })
      } else {
        // 输出边：内部 -> 外部，重定向到内部节点 -> 外部
        restoredExternalEdges.push({
          id: be.originalEdgeId,
          source: idMap.get(be.internalNodeId) ?? be.internalNodeId, // 重定向到内部节点
          target: be.target,
          sourceHandle: be.internalHandle,
          targetHandle: be.targetHandle,
        })
      }
    }
  }

  // 标记组为展开状态
  const updatedGroup: FlowHamsterNode = {
    ...groupNode,
    data: {
      ...data,
      isExpanded: true,
    },
  } as FlowHamsterNode

  // 过滤掉被包装的子节点，但排除组节点本身（由 updatedGroup 替代）
  const childNodeIdSet = new Set(childNodeIds)
  const filteredNodes = allNodes.filter(
    (n) => !childNodeIdSet.has(n.id) && n.id !== groupNode.id
  )

  // 过滤掉内部边和边界边（它们会在下面重新添加）
  const internalEdgeIdSet = new Set(data.internalEdgeIds)
  const boundaryEdgeIds = new Set((boundaryEdges as BoundaryEdgeData[])?.map((be) => be.originalEdgeId) || [])
  const allRemovedEdgeIds = new Set([...internalEdgeIdSet, ...boundaryEdgeIds])
  const filteredEdges = allEdges.filter((e) => !allRemovedEdgeIds.has(e.id))

  // 恢复内部边（使用新 ID 映射）
  // 检查是否已存在同名边，避免重复添加（处理重新展开的情况）
  const existingEdgeIds = new Set(filteredEdges.map(e => e.id))
  const internalEdgeList: FlowHamsterEdge[] = storedEdges
    .map((edge) => ({
      id: `e_${edge.from}_${edge.to}`,
      source: idMap.get(edge.from) ?? edge.from,
      target: idMap.get(edge.to) ?? edge.to,
      sourceHandle: edge.fromHandle,
      targetHandle: edge.toHandle,
    }))
    .filter(edge => !existingEdgeIds.has(edge.id))

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
  const { childNodeIds, internalEdgeIds, internalEdges, boundaryEdges } = data
  const childNodeIdSet = new Set(childNodeIds)

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
    (n) => !childNodeIdSet.has(n.id) && n.id !== groupId
  )

  // 移除内部边和边界边
  // 也需要移除 expandPackage 生成的新边 ID（格式：e_${from}_${to}）
  const generatedEdgeIds = new Set(
    (internalEdges || []).map((ie: InternalEdgeData) => `e_${ie.from}_${ie.to}`)
  )
  const internalEdgeIdSet = new Set(internalEdgeIds)
  const boundaryEdgeIds = new Set((boundaryEdges as BoundaryEdgeData[])?.map((be) => be.originalEdgeId) || [])
  const allRemovedIds = new Set([...internalEdgeIdSet, ...generatedEdgeIds, ...boundaryEdgeIds])

  const filteredEdges = allEdges.filter((e) => !allRemovedIds.has(e.id))

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

  const data = groupNode.data as CustomCompositeNodeData
  const { internalStructure, internalEdges: storedEdges, boundaryEdges } = data

  // Build ID map: give composite sub-nodes fresh instance IDs to avoid
  // collisions with other instances of the same class elsewhere in the graph.
  const idMap = new Map<string, string>()
  for (const sub of internalStructure) {
    if ((sub as any).customClassId) {
      idMap.set(sub.id, `${(sub as any).customClassId}_instance_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`)
    }
  }

  // 恢复内部节点（使用原始位置，composite 子节点使用新 ID）
  const restoredNodes: FlowHamsterNode[] = internalStructure.map((sub) => {
    const newId = idMap.get(sub.id) ?? sub.id
    const position = sub.position || {
      x: groupNode.position.x + 220,
      y: groupNode.position.y + 60,
    }
    const base: FlowHamsterNode = {
      id: newId,
      type: getNodeComponentType(sub.type),
      position,
      data: {
        nodeType: sub.type as NodeType,
        label: sub.label,
        params: { ...sub.params },
      },
    } as FlowHamsterNode

    // If it's a nested composite, preserve class identity as collapsed instance
    if ((sub as any).customClassId) {
      ;(base.data as any).isCustomComposite = true
      ;(base.data as any).customClassId = (sub as any).customClassId
      ;(base.data as any).isExpanded = false
    }

    return base
  })

  // 恢复内部边（使用新 ID 映射）
  const restoredEdges: FlowHamsterEdge[] = storedEdges.map((edge) => ({
    id: `e_${edge.from}_${edge.to}`,
    source: idMap.get(edge.from) ?? edge.from,
    target: idMap.get(edge.to) ?? edge.to,
    sourceHandle: edge.fromHandle,
    targetHandle: edge.toHandle,
  }))

  // 恢复外部边（使用边界边信息，保留原始 ID，内部引用使用新 ID）
  if (boundaryEdges) {
    for (const be of boundaryEdges as BoundaryEdgeData[]) {
      restoredEdges.push({
        id: be.originalEdgeId,
        source: be.source,
        target: idMap.get(be.internalNodeId) ?? be.internalNodeId,
        sourceHandle: be.sourceHandle,
        targetHandle: be.internalHandle,
      })
    }
  }

  // 过滤掉组节点和内部边
  const filteredNodes = allNodes.filter((n) => n.id !== groupId)
  const internalEdgeIdSet = new Set(data.internalEdgeIds)
  const boundaryEdgeIds = new Set((boundaryEdges as BoundaryEdgeData[])?.map((be) => be.originalEdgeId) || [])
  const allRemovedIds = new Set([...internalEdgeIdSet, ...boundaryEdgeIds])

  const filteredEdges = allEdges.filter((e) => !allRemovedIds.has(e.id))

  return {
    nodes: [...filteredNodes, ...restoredNodes],
    edges: [...filteredEdges, ...restoredEdges],
  }
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
