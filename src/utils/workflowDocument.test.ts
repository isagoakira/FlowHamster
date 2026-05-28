import { describe, it, expect } from 'vitest'
import {
  createWorkflowDocumentFromGraph,
  parseWorkflowDocument,
  getModelGraphFromWorkflowDocument,
} from './workflowDocument'
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { WORKFLOW_DOCUMENT_VERSION } from '../schema/workflowDocument'

function makeNode(id: string, type: string, params?: Record<string, unknown>): FlowHamsterNode {
  return {
    id,
    type: `${type}Node`,
    position: { x: 0, y: 0 },
    data: {
      nodeType: type as any,
      label: type,
      params: params || {},
    },
  }
}

function makeEdge(source: string, target: string): FlowHamsterEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
  }
}

describe('workflowDocument', () => {
  describe('createWorkflowDocumentFromGraph', () => {
    it('should create a document with model graph nodes and edges', () => {
      const nodes: FlowHamsterNode[] = [
        makeNode('input1', 'input', { shape: '1,28,28' }),
        makeNode('linear1', 'linear', { in_features: 784, out_features: 10 }),
        makeNode('output1', 'output'),
      ]
      const edges: FlowHamsterEdge[] = [
        makeEdge('input1', 'linear1'),
        makeEdge('linear1', 'output1'),
      ]

      const doc = createWorkflowDocumentFromGraph(nodes, edges, { name: 'Test' })

      expect(doc.version).toBe(WORKFLOW_DOCUMENT_VERSION)
      expect(doc.metadata.name).toBe('Test')
      expect(doc.modelGraph.nodes).toHaveLength(3)
      expect(doc.modelGraph.edges).toHaveLength(2)
      expect(doc.modelGraph.contract.inputs).toHaveLength(1)
      expect(doc.modelGraph.contract.outputs).toHaveLength(1)
    })

    it('should include data graph when provided', () => {
      const modelNodes: FlowHamsterNode[] = [makeNode('input1', 'input')]
      const modelEdges: FlowHamsterEdge[] = []
      const dataNodes = [{ id: 'csv1', type: 'csvSourceNode', position: { x: 0, y: 0 }, data: { nodeType: 'csv_source', label: 'CSV', params: { path: './data.csv' } } }] as any
      const dataEdges: any[] = []

      const doc = createWorkflowDocumentFromGraph(modelNodes, modelEdges, {
        dataGraphNodes: dataNodes,
        dataGraphEdges: dataEdges,
      })

      expect(doc.dataGraph.nodes).toHaveLength(1)
      expect(doc.dataGraph.kind).toBe('data')
    })

    it('should include bindings when provided', () => {
      const nodes: FlowHamsterNode[] = [makeNode('input1', 'input', { name: 'image' })]
      const edges: FlowHamsterEdge[] = []
      const bindings = [{ id: 'b1', sourceGraph: 'data' as const, sourceKey: 'image', target: 'model_input' as const, targetKey: 'image' }]

      const doc = createWorkflowDocumentFromGraph(nodes, edges, { bindings })

      expect(doc.bindings).toHaveLength(1)
      expect(doc.bindings[0].sourceKey).toBe('image')
    })

    it('should use default training config when none provided', () => {
      const doc = createWorkflowDocumentFromGraph([], [])
      expect(doc.trainingConfig.taskType).toBe('classification')
      expect(doc.trainingConfig.optimizer.type).toBe('adamw')
      expect(doc.trainingConfig.loss.type).toBe('cross_entropy')
    })

    it('should preserve custom training config', () => {
      const customConfig = {
        taskType: 'segmentation' as const,
        loss: { type: 'dice' as const, enabled: true, params: {} },
        optimizer: { type: 'sgd' as const, enabled: true, params: { lr: 0.01 } },
        scheduler: { type: 'step_lr' as const, enabled: false, params: {} },
        metrics: [],
        runtime: { device: 'cuda', epochs: 20, batchSize: 16, amp: true, gradClip: null, numWorkers: 4 },
        checkpoint: { enabled: true, saveTopK: 3, monitor: 'val_loss', mode: 'min' as const, earlyStopPatience: 5 },
      }
      const doc = createWorkflowDocumentFromGraph([], [], { trainingConfig: customConfig })
      expect(doc.trainingConfig.taskType).toBe('segmentation')
      expect(doc.trainingConfig.optimizer.type).toBe('sgd')
      expect(doc.trainingConfig.runtime.epochs).toBe(20)
    })
  })

  describe('parseWorkflowDocument', () => {
    it('should parse a complete workflow document payload', () => {
      const payload = {
        version: '2.0.0',
        metadata: { name: 'Parsed', description: 'Test' },
        modelGraph: {
          kind: 'model',
          nodes: [{ id: 'n1', type: 'inputNode', position: { x: 0, y: 0 }, data: { nodeType: 'input', label: 'Input', params: {} } }],
          edges: [],
          contract: { inputs: [], outputs: [] },
        },
        dataGraph: { kind: 'data', nodes: [], edges: [], contract: { inputs: [], outputs: [] } },
        trainingConfig: { taskType: 'classification', loss: { type: 'cross_entropy', enabled: true, params: {} }, optimizer: { type: 'adamw', enabled: true, params: {} }, scheduler: { type: 'cosine_annealing', enabled: false, params: {} }, metrics: [], runtime: { device: 'cpu', epochs: 5, batchSize: 32, amp: false, gradClip: null, numWorkers: 0 }, checkpoint: { enabled: false, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null } },
        bindings: [],
      }

      const doc = parseWorkflowDocument(payload)
      expect(doc.metadata.name).toBe('Parsed')
      expect(doc.modelGraph.nodes).toHaveLength(1)
    })

    it('should handle legacy project format', () => {
      const legacy = {
        nodes: [{ id: 'n1', type: 'inputNode', position: { x: 0, y: 0 }, data: { nodeType: 'input', label: 'Input', params: {} } }],
        edges: [],
      }

      const doc = parseWorkflowDocument(legacy)
      expect(doc.modelGraph.nodes).toHaveLength(1)
      expect(doc.version).toBe(WORKFLOW_DOCUMENT_VERSION)
    })

    it('should preserve data graph contract when present', () => {
      const payload = {
        modelGraph: { nodes: [], edges: [], contract: { inputs: [], outputs: [] } },
        dataGraph: {
          nodes: [{ id: 'd1', type: 'datasetOutputNode', position: { x: 0, y: 0 }, data: { nodeType: 'dataset_output', label: 'Out', params: { fields: 'image,label' } } }],
          edges: [],
          contract: { inputs: [], outputs: [{ name: 'image' }] },
        },
      }

      const doc = parseWorkflowDocument(payload)
      expect(doc.dataGraph.contract.outputs).toHaveLength(1)
      expect(doc.dataGraph.contract.outputs[0].name).toBe('image')
    })
  })

  describe('getModelGraphFromWorkflowDocument', () => {
    it('should extract model graph nodes and edges', () => {
      const nodes: FlowHamsterNode[] = [makeNode('n1', 'input')]
      const edges: FlowHamsterEdge[] = []
      const doc = createWorkflowDocumentFromGraph(nodes, edges)

      const graph = getModelGraphFromWorkflowDocument(doc)
      expect(graph.nodes).toHaveLength(1)
      expect(graph.edges).toHaveLength(0)
    })
  })
})
