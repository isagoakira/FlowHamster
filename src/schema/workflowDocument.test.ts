import { describe, it, expect } from 'vitest'
import { WORKFLOW_DOCUMENT_VERSION, WorkflowDocument, WorkflowTrainingConfig } from './workflowDocument'

describe('workflowDocument schema', () => {
  it('should have correct document version', () => {
    expect(WORKFLOW_DOCUMENT_VERSION).toBe('2.0.0')
  })

  it('should construct a minimal valid WorkflowDocument', () => {
    const doc: WorkflowDocument = {
      version: WORKFLOW_DOCUMENT_VERSION,
      metadata: {
        name: 'Test Workflow',
        description: 'Test',
        schemaVersion: WORKFLOW_DOCUMENT_VERSION,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        exportSource: 'flowhamster',
      },
      modelGraph: {
        kind: 'model',
        nodes: [],
        edges: [],
        contract: { inputs: [], outputs: [] },
      },
      dataGraph: {
        kind: 'data',
        nodes: [],
        edges: [],
        contract: { inputs: [], outputs: [] },
      },
      trainingConfig: {
        taskType: 'classification',
        loss: { type: 'cross_entropy', enabled: true, params: {} },
        optimizer: { type: 'adamw', enabled: true, params: { lr: 0.001 } },
        scheduler: { type: 'step_lr', enabled: false, params: {} },
        metrics: [],
        runtime: { device: 'cpu', epochs: 10, batchSize: 32, amp: false, gradClip: null, numWorkers: 0 },
        checkpoint: { enabled: false, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
      },
      bindings: [],
    }

    expect(doc.version).toBe('2.0.0')
    expect(doc.metadata.name).toBe('Test Workflow')
    expect(doc.modelGraph.kind).toBe('model')
    expect(doc.dataGraph.kind).toBe('data')
  })

  it('should support all task types', () => {
    const taskTypes = ['classification', 'regression', 'segmentation', 'detection', 'nlp', 'custom'] as const
    for (const taskType of taskTypes) {
      const config: WorkflowTrainingConfig = {
        taskType,
        loss: { type: 'cross_entropy', enabled: true, params: {} },
        optimizer: { type: 'adamw', enabled: true, params: {} },
        scheduler: { type: 'cosine_annealing', enabled: false, params: {} },
        metrics: [],
        runtime: { device: 'auto', epochs: 10, batchSize: 32, amp: false, gradClip: null, numWorkers: 4 },
        checkpoint: { enabled: true, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
      }
      expect(config.taskType).toBe(taskType)
    }
  })

  it('should support composite loss components', () => {
    const config: WorkflowTrainingConfig = {
      taskType: 'segmentation',
      loss: {
        type: 'composite',
        enabled: true,
        params: {
          components: [
            { type: 'dice', weight: 0.3 },
            { type: 'cross_entropy', weight: 0.7 },
          ],
        },
      },
      optimizer: { type: 'adamw', enabled: true, params: {} },
      scheduler: { type: 'step_lr', enabled: false, params: {} },
      metrics: [{ type: 'accuracy', enabled: true, params: {} }],
      runtime: { device: 'auto', epochs: 10, batchSize: 32, amp: false, gradClip: null, numWorkers: 4 },
      checkpoint: { enabled: true, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
    }

    expect(config.loss.type).toBe('composite')
    const components = config.loss.params.components as Array<{ type: string; weight: number }>
    expect(components).toHaveLength(2)
    expect(components[0].weight).toBe(0.3)
    expect(components[1].weight).toBe(0.7)
  })

  it('should support workflow bindings', () => {
    const doc: WorkflowDocument = {
      version: WORKFLOW_DOCUMENT_VERSION,
      metadata: {
        name: 'Bound Workflow',
        description: '',
        schemaVersion: WORKFLOW_DOCUMENT_VERSION,
        createdAt: '',
        updatedAt: '',
        exportSource: 'flowhamster',
      },
      modelGraph: {
        kind: 'model',
        nodes: [],
        edges: [],
        contract: { inputs: [{ name: 'image', dtype: 'float32', shapeHint: '1,28,28' }], outputs: [] },
      },
      dataGraph: {
        kind: 'data',
        nodes: [],
        edges: [],
        contract: { inputs: [], outputs: [{ name: 'image', semanticRole: 'data_output' }] },
      },
      trainingConfig: {
        taskType: 'classification',
        loss: { type: 'cross_entropy', enabled: true, params: {} },
        optimizer: { type: 'adamw', enabled: true, params: {} },
        scheduler: { type: 'step_lr', enabled: false, params: {} },
        metrics: [],
        runtime: { device: 'auto', epochs: 10, batchSize: 32, amp: false, gradClip: null, numWorkers: 4 },
        checkpoint: { enabled: false, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
      },
      bindings: [
        { id: 'b1', sourceGraph: 'data', sourceKey: 'image', target: 'model_input', targetKey: 'image' },
        { id: 'b2', sourceGraph: 'data', sourceKey: 'label', target: 'training_target', targetKey: '' },
      ],
    }

    expect(doc.bindings).toHaveLength(2)
    expect(doc.bindings[0].sourceKey).toBe('image')
    expect(doc.bindings[0].targetKey).toBe('image')
    expect(doc.bindings[1].target).toBe('training_target')
  })
})
