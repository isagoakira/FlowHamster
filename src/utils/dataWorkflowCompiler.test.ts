import { describe, it, expect } from 'vitest'
import {
  compileDataWorkflow,
  collectDataOutputFields,
  validateBindingCompatibility,
  inferTaskTypeFromModel,
} from './dataWorkflowCompiler'
import { FlowHamsterNode } from '../types/graph'
import { DataFlowNode } from '../types/dataGraph'
import { WorkflowBinding } from '../schema/workflowDocument'

function makeModelNode(id: string, nodeType: string, params?: Record<string, unknown>): FlowHamsterNode {
  return {
    id,
    type: `${nodeType}Node`,
    position: { x: 0, y: 0 },
    data: { nodeType: nodeType as any, label: nodeType, params: params || {} },
  }
}

function makeDataNode(id: string, nodeType: string, params?: Record<string, unknown>): DataFlowNode {
  return {
    id,
    type: `${nodeType}Node`,
    position: { x: 0, y: 0 },
    data: { nodeType: nodeType as any, label: nodeType, params: params || {}, fieldSpecs: undefined },
  } as DataFlowNode
}

function makeBinding(sourceKey: string, targetKey: string, target: WorkflowBinding['target'] = 'model_input'): WorkflowBinding {
  return { id: `b-${sourceKey}`, sourceGraph: 'data', sourceKey, target, targetKey }
}

describe('dataWorkflowCompiler', () => {
  describe('compileDataWorkflow', () => {
    it('should compile a basic csv source pipeline', () => {
      const dataNodes: DataFlowNode[] = [
        makeDataNode('csv1', 'csv_source', { path: './data/train.csv', delimiter: ',' }),
        makeDataNode('out1', 'dataset_output', { fields: 'image,label' }),
      ]
      const dataEdges = [{ id: 'e1', source: 'csv1', target: 'out1', sourceHandle: null, targetHandle: 'x' }]
      const modelNodes: FlowHamsterNode[] = [makeModelNode('input1', 'input', { name: 'image' })]
      const bindings: WorkflowBinding[] = [makeBinding('image', 'image')]

      const result = compileDataWorkflow(modelNodes, dataNodes, dataEdges, bindings)

      expect(result.hasWorkflowRuntime).toBe(true)
      expect(result.outputFields).toEqual(['image', 'label'])
      expect(result.modelInputBindings).toHaveLength(1)
      expect(result.modelInputBindings[0]).toEqual({ targetKey: 'image', sourceKey: 'image' })
      expect(result.pythonScaffold).toContain('FlowHamsterDataset')
      expect(result.pythonScaffold).toContain('BOUND_MODEL_INPUTS')
    })

    it('should warn when model input is not bound', () => {
      const modelNodes: FlowHamsterNode[] = [makeModelNode('input1', 'input', { name: 'image' })]
      const result = compileDataWorkflow(modelNodes, [], [], [])

      expect(result.warnings.some((w) => w.includes('尚未绑定'))).toBe(true)
    })

    it('should warn when binding source field is not in dataset_output', () => {
      const dataNodes: DataFlowNode[] = [
        makeDataNode('out1', 'dataset_output', { fields: 'image' }),
      ]
      const bindings: WorkflowBinding[] = [makeBinding('nonexistent', 'image')]

      const result = compileDataWorkflow([], dataNodes, [], bindings)

      expect(result.warnings.some((w) => w.includes('未在 Dataset Output'))).toBe(true)
    })

    it('should handle empty data graph', () => {
      const result = compileDataWorkflow([], [], [], [])
      expect(result.hasWorkflowRuntime).toBe(false)
      expect(result.outputFields).toEqual([])
    })

    it('should detect validation errors for invalid node params', () => {
      const dataNodes: DataFlowNode[] = [
        makeDataNode('csv1', 'csv_source', { path: '', delimiter: ',' }),
      ]

      const result = compileDataWorkflow([], dataNodes, [], [])

      expect(result.validationErrors.length).toBeGreaterThan(0)
      expect(result.validationErrors.some((e) => e.field === 'path')).toBe(true)
    })

    it('should generate correct primary model input key', () => {
      const modelNodes: FlowHamsterNode[] = [makeModelNode('input1', 'input', { name: 'features' })]
      const bindings: WorkflowBinding[] = [makeBinding('data_field', 'features')]

      const result = compileDataWorkflow(modelNodes, [], [], bindings)
      expect(result.primaryModelInputKey).toBe('features')
    })

    it('should include target binding source when present', () => {
      const bindings: WorkflowBinding[] = [
        makeBinding('image', 'image'),
        { id: 'b2', sourceGraph: 'data', sourceKey: 'label', target: 'training_target', targetKey: '' },
      ]

      const result = compileDataWorkflow([], [], [], bindings)
      expect(result.targetBindingSource).toBe('label')
    })
  })

  describe('collectDataOutputFields', () => {
    it('should collect fields from dataset_output nodes', () => {
      const nodes: DataFlowNode[] = [
        makeDataNode('out1', 'dataset_output', { fields: 'image,label' }),
        makeDataNode('out2', 'dataset_output', { fields: 'mask' }),
      ]

      const fields = collectDataOutputFields(nodes)
      expect(fields).toEqual(['image', 'label', 'mask'])
    })

    it('should deduplicate fields', () => {
      const nodes: DataFlowNode[] = [
        makeDataNode('out1', 'dataset_output', { fields: 'image,label' }),
        makeDataNode('out2', 'dataset_output', { fields: 'image,mask' }),
      ]

      const fields = collectDataOutputFields(nodes)
      expect(fields).toEqual(['image', 'label', 'mask'])
    })

    it('should return empty array for no output nodes', () => {
      const fields = collectDataOutputFields([])
      expect(fields).toEqual([])
    })
  })

  describe('validateBindingCompatibility', () => {
    it('should mark compatible bindings', () => {
      const modelNodes: FlowHamsterNode[] = [makeModelNode('input1', 'input', { name: 'image' })]
      const dataNodes: DataFlowNode[] = [
        makeDataNode('out1', 'dataset_output', { fields: 'image' }),
      ]
      const bindings: WorkflowBinding[] = [makeBinding('image', 'image')]

      const validations = validateBindingCompatibility(bindings, dataNodes, modelNodes)
      expect(validations).toHaveLength(1)
      expect(validations[0].isCompatible).toBe(true)
    })

    it('should warn when source field spec is missing', () => {
      const modelNodes: FlowHamsterNode[] = [makeModelNode('input1', 'input', { name: 'image' })]
      const dataNodes: DataFlowNode[] = []
      const bindings: WorkflowBinding[] = [makeBinding('image', 'image')]

      const validations = validateBindingCompatibility(bindings, dataNodes, modelNodes)
      expect(validations[0].isCompatible).toBe(false)
      expect(validations[0].warning).toContain('No field spec found')
    })
  })

  describe('inferTaskTypeFromModel', () => {
    it('should infer classification from multi-class linear output', () => {
      const nodes: FlowHamsterNode[] = [
        makeModelNode('input1', 'input'),
        makeModelNode('linear1', 'linear', { out_features: 10 }),
        makeModelNode('output1', 'output'),
      ]
      const edges = [{ id: 'e1', source: 'linear1', target: 'output1' }]

      const inference = inferTaskTypeFromModel(nodes, edges)
      expect(inference.taskType).toBe('classification')
      expect(inference.confidence).toBe('high')
      expect(inference.outFeatures).toBe(10)
    })

    it('should infer regression from single-output linear', () => {
      const nodes: FlowHamsterNode[] = [
        makeModelNode('input1', 'input'),
        makeModelNode('linear1', 'linear', { out_features: 1 }),
        makeModelNode('output1', 'output'),
      ]
      const edges = [{ id: 'e1', source: 'linear1', target: 'output1' }]

      const inference = inferTaskTypeFromModel(nodes, edges)
      expect(inference.taskType).toBe('regression')
      expect(inference.confidence).toBe('high')
    })

    it('should return unknown when no linear layer found', () => {
      const nodes: FlowHamsterNode[] = [
        makeModelNode('input1', 'input'),
        makeModelNode('relu1', 'relu'),
        makeModelNode('output1', 'output'),
      ]
      const edges = [{ id: 'e1', source: 'relu1', target: 'output1' }]

      const inference = inferTaskTypeFromModel(nodes, edges)
      expect(inference.taskType).toBe('unknown')
      expect(inference.confidence).toBe('low')
    })
  })
})
