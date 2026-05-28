import { afterEach, describe, expect, it, vi } from 'vitest'
import { executeWorkflow, generateCode } from './api'
import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'
import { DataFlowEdge, DataFlowNode } from '../types/dataGraph'
import { WorkflowBinding, WorkflowTrainingConfig } from '../schema/workflowDocument'

const modelNodes: FlowHamsterNode[] = [
  {
    id: 'input',
    type: 'inputNode',
    position: { x: 0, y: 0 },
    data: { nodeType: 'input' as any, label: 'Input', params: { name: 'features' } },
  },
  {
    id: 'output',
    type: 'outputNode',
    position: { x: 120, y: 0 },
    data: { nodeType: 'output' as any, label: 'Output', params: {} },
  },
]

const modelEdges: FlowHamsterEdge[] = [
  { id: 'e1', source: 'input', target: 'output', sourceHandle: 'result', targetHandle: 'x' },
]

const dataNodes: DataFlowNode[] = [
  {
    id: 'out',
    type: 'dataPipelineNode',
    position: { x: 0, y: 0 },
    data: { nodeType: 'dataset_output' as any, label: 'Dataset Output', params: { fields: 'features,label' } },
  },
]

const dataEdges: DataFlowEdge[] = []

const bindings: WorkflowBinding[] = [
  {
    id: 'data:features->model_input:features',
    sourceGraph: 'data',
    sourceKey: 'features',
    target: 'model_input',
    targetKey: 'features',
  },
]

const trainingConfig: WorkflowTrainingConfig = {
  taskType: 'classification',
  loss: { type: 'cross_entropy', enabled: true, params: {} },
  optimizer: { type: 'adam', enabled: true, params: { lr: 0.001 } },
  scheduler: { type: 'none', enabled: false, params: {} },
  metrics: [],
  runtime: { device: 'cpu', epochs: 1, batchSize: 2, amp: false, gradClip: null, numWorkers: 0 },
  checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
}

function mockFetch(response: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    json: vi.fn().mockResolvedValue(response),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function postedBody(fetchMock: ReturnType<typeof mockFetch>) {
  return JSON.parse(String(fetchMock.mock.calls[0][1]?.body))
}

describe('api client workflow payloads', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('passes data graph and bindings to /api/generate', async () => {
    const fetchMock = mockFetch({ success: true, code: 'ok', warnings: [] })

    await generateCode(modelNodes, modelEdges, {}, trainingConfig, {
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    })

    const body = postedBody(fetchMock)
    expect(fetchMock.mock.calls[0][0]).toContain('/generate')
    expect(body.training_config).toEqual(trainingConfig)
    expect(body.data_graph).toEqual({ nodes: dataNodes, edges: dataEdges })
    expect(body.bindings).toEqual(bindings)
  })

  it('can execute the current workflow through /api/execute', async () => {
    const fetchMock = mockFetch({ success: true, output: 'epoch=1', error: null })

    await executeWorkflow(modelNodes, modelEdges, trainingConfig, {
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    }, 'cuda')

    const body = postedBody(fetchMock)
    expect(fetchMock.mock.calls[0][0]).toContain('/execute')
    expect(body.graph.nodes).toHaveLength(2)
    expect(body.training_config).toEqual(trainingConfig)
    expect(body.data_graph).toEqual({ nodes: dataNodes, edges: dataEdges })
    expect(body.bindings).toEqual(bindings)
    expect(body.target_device).toBe('cuda')
  })
})
