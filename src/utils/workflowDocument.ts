import { FlowHamsterEdge, FlowHamsterNode, NodeData } from '../types/graph'
import { DataFlowEdge, DataFlowNode } from '../types/dataGraph'
import {
  LegacyWorkflowProject,
  WorkflowBinding,
  WorkflowDocument,
  WorkflowPortContract,
  WorkflowTrainingConfig,
  WORKFLOW_DOCUMENT_VERSION,
} from '../schema/workflowDocument'
import { getNodeComponentType, normalizeNodeData, normalizeNodeType } from './nodeType'

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function toNodeData(value: unknown): NodeData | null {
  if (!isRecord(value) || typeof value.nodeType !== 'string') return null

  return normalizeNodeData({
    ...value,
    label: typeof value.label === 'string' ? value.label : value.nodeType,
    params: isRecord(value.params) ? value.params as Record<string, number | string | boolean> : {},
  } as NodeData)
}

function normalizeImportedNodes(nodes: unknown[]): FlowHamsterNode[] {
  return nodes.flatMap((node) => {
    if (!isRecord(node) || typeof node.id !== 'string') return []

    const data = toNodeData(node.data)
    if (!data) return []

    const position = isRecord(node.position)
      ? {
          x: typeof node.position.x === 'number' ? node.position.x : 0,
          y: typeof node.position.y === 'number' ? node.position.y : 0,
        }
      : { x: 0, y: 0 }

    return [{
      ...node,
      id: node.id,
      type: typeof node.type === 'string' ? node.type : getNodeComponentType(data.nodeType),
      position,
      data,
    } as FlowHamsterNode]
  })
}

function normalizeImportedEdges(edges: unknown[]): FlowHamsterEdge[] {
  return edges.flatMap((edge) => {
    if (!isRecord(edge) || typeof edge.source !== 'string' || typeof edge.target !== 'string') return []

    return [{
      ...edge,
      id:
        typeof edge.id === 'string'
          ? edge.id
          : `${edge.source}:${String(edge.sourceHandle ?? 'result')}->${edge.target}:${String(edge.targetHandle ?? 'a')}`,
      source: edge.source,
      target: edge.target,
    } as FlowHamsterEdge]
  })
}

function createDefaultTrainingConfig(): WorkflowTrainingConfig {
  return {
    taskType: 'classification',
    loss: {
      type: 'cross_entropy',
      enabled: true,
      params: {},
    },
    optimizer: {
      type: 'adamw',
      enabled: true,
      params: {
        lr: 1e-3,
        weight_decay: 1e-2,
      },
    },
    scheduler: {
      type: 'cosine_annealing',
      enabled: false,
      params: {},
    },
    metrics: [
      {
        type: 'accuracy',
        enabled: true,
        params: {},
      },
    ],
    runtime: {
      device: 'auto',
      epochs: 10,
      batchSize: 32,
      amp: false,
      gradClip: null,
      numWorkers: 4,
    },
    checkpoint: {
      enabled: true,
      saveTopK: 1,
      monitor: 'val_loss',
      mode: 'min',
      earlyStopPatience: null,
    },
  }
}

export { createDefaultTrainingConfig }

function collectModelContract(nodes: FlowHamsterNode[]): { inputs: WorkflowPortContract[]; outputs: WorkflowPortContract[] } {
  const inputs = nodes
    .filter((node) => normalizeNodeType(node.data.nodeType) === 'input')
    .map((node, index) => ({
      name:
        typeof node.data.params?.name === 'string'
          ? String(node.data.params.name)
          : `input_${index + 1}`,
      dtype:
        typeof node.data.params?.dtype === 'string'
          ? String(node.data.params.dtype)
          : undefined,
      shapeHint:
        typeof node.data.params?.shape === 'string'
          ? String(node.data.params.shape)
          : undefined,
      semanticRole: 'model_input',
      description: typeof node.data.label === 'string' ? node.data.label : undefined,
    }))

  const outputs = nodes
    .filter((node) => normalizeNodeType(node.data.nodeType) === 'output')
    .map((node, index) => ({
      name:
        typeof node.data.params?.output_name === 'string'
          ? String(node.data.params.output_name)
          : `output_${index + 1}`,
      semanticRole: 'model_output',
      description: typeof node.data.label === 'string' ? node.data.label : undefined,
    }))

  return { inputs, outputs }
}

function collectDataContract(nodes: DataFlowNode[]): { inputs: WorkflowPortContract[]; outputs: WorkflowPortContract[] } {
  const outputNodes = nodes
    .filter((node) => node.data.nodeType === 'dataset_output')
    .flatMap((node) => String(node.data.params.fields || '')
      .split(',')
      .map((field) => field.trim())
      .filter(Boolean)
      .map((field) => ({
        name: field,
        semanticRole: 'data_output',
        description: node.data.label,
      } satisfies WorkflowPortContract)))

  return {
    inputs: [],
    outputs: outputNodes,
  }
}

export function createWorkflowDocumentFromGraph(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  options?: {
    name?: string
    description?: string
    trainingConfig?: WorkflowTrainingConfig
    dataGraphNodes?: DataFlowNode[]
    dataGraphEdges?: DataFlowEdge[]
    bindings?: WorkflowBinding[]
  }
): WorkflowDocument {
  const now = new Date().toISOString()
  const dataGraphNodes = options?.dataGraphNodes ?? []
  const dataGraphEdges = options?.dataGraphEdges ?? []

  return {
    version: WORKFLOW_DOCUMENT_VERSION,
    metadata: {
      name: options?.name ?? 'FlowHamster Workflow',
      description: options?.description ?? 'Model graph driven workflow document',
      schemaVersion: WORKFLOW_DOCUMENT_VERSION,
      createdAt: now,
      updatedAt: now,
      exportSource: 'flowhamster',
    },
    modelGraph: {
      kind: 'model',
      nodes,
      edges,
      contract: collectModelContract(nodes),
    },
    dataGraph: {
      kind: 'data',
      nodes: dataGraphNodes,
      edges: dataGraphEdges,
      contract: collectDataContract(dataGraphNodes),
    },
    trainingConfig: options?.trainingConfig ?? createDefaultTrainingConfig(),
    bindings: options?.bindings ?? [],
  }
}

export function parseWorkflowDocument(payload: unknown): WorkflowDocument {
  if (isRecord(payload) && isRecord(payload.modelGraph)) {
    const nodes = normalizeImportedNodes(Array.isArray(payload.modelGraph.nodes) ? payload.modelGraph.nodes : [])
    const edges = normalizeImportedEdges(Array.isArray(payload.modelGraph.edges) ? payload.modelGraph.edges : [])
    const migrated = createWorkflowDocumentFromGraph(nodes, edges, {
      name:
        isRecord(payload.metadata) && typeof payload.metadata.name === 'string'
          ? payload.metadata.name
          : 'FlowHamster Workflow',
      description:
        isRecord(payload.metadata) && typeof payload.metadata.description === 'string'
          ? payload.metadata.description
          : 'Migrated workflow document',
    })

    return {
      ...migrated,
      version: typeof payload.version === 'string' ? payload.version : migrated.version,
      metadata: {
        ...migrated.metadata,
        ...(isRecord(payload.metadata) ? payload.metadata : {}),
        name:
          isRecord(payload.metadata) && typeof payload.metadata.name === 'string'
            ? payload.metadata.name
            : migrated.metadata.name,
        description:
          isRecord(payload.metadata) && typeof payload.metadata.description === 'string'
            ? payload.metadata.description
            : migrated.metadata.description,
      },
      dataGraph:
        isRecord(payload.dataGraph) && Array.isArray(payload.dataGraph.nodes) && Array.isArray(payload.dataGraph.edges)
          ? {
              kind: 'data',
              nodes: payload.dataGraph.nodes as Record<string, unknown>[],
              edges: payload.dataGraph.edges as Record<string, unknown>[],
              contract:
                isRecord(payload.dataGraph.contract)
                  ? {
                      inputs: Array.isArray(payload.dataGraph.contract.inputs)
                        ? payload.dataGraph.contract.inputs as WorkflowPortContract[]
                        : [],
                      outputs: Array.isArray(payload.dataGraph.contract.outputs)
                        ? payload.dataGraph.contract.outputs as WorkflowPortContract[]
                        : [],
                    }
                  : { inputs: [], outputs: [] },
            }
          : migrated.dataGraph,
      trainingConfig:
        isRecord(payload.trainingConfig)
          ? { ...migrated.trainingConfig, ...payload.trainingConfig }
          : migrated.trainingConfig,
      bindings: Array.isArray(payload.bindings) ? payload.bindings as WorkflowDocument['bindings'] : migrated.bindings,
    }
  }

  const legacy = payload as LegacyWorkflowProject
  const nodes = normalizeImportedNodes(Array.isArray(legacy?.nodes) ? legacy.nodes : [])
  const edges = normalizeImportedEdges(Array.isArray(legacy?.edges) ? legacy.edges : [])

  return createWorkflowDocumentFromGraph(nodes, edges)
}

export function getModelGraphFromWorkflowDocument(document: WorkflowDocument): {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
} {
  return {
    nodes: document.modelGraph.nodes,
    edges: document.modelGraph.edges,
  }
}
