import { describe, expect, it } from 'vitest'
import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'
import { DataFlowEdge, DataFlowNode } from '../types/dataGraph'
import { WorkflowBinding, WorkflowTrainingConfig } from '../schema/workflowDocument'
import { generateLocalCode } from './codeGenerator'
import { compileDataWorkflow } from './dataWorkflowCompiler'

function modelNode(id: string, nodeType: string, params: Record<string, unknown> = {}): FlowHamsterNode {
  return {
    id,
    type: `${nodeType}Node`,
    position: { x: 0, y: 0 },
    data: {
      nodeType: nodeType as any,
      label: nodeType,
      params: params as Record<string, number | string | boolean>,
    },
  }
}

function dataNode(id: string, nodeType: string, params: Record<string, unknown> = {}): DataFlowNode {
  return {
    id,
    type: 'dataPipelineNode',
    position: { x: 0, y: 0 },
    data: {
      nodeType: nodeType as any,
      label: nodeType,
      params: params as any,
    },
  } as DataFlowNode
}

const modelNodes: FlowHamsterNode[] = [
  modelNode('input', 'input', { name: 'image', shape: '3,224,224', dtype: 'float32' }),
  modelNode('conv', 'conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3, padding: 1 }),
  modelNode('pool', 'adaptiveavgpool2d', { output_size: 1 }),
  modelNode('flat', 'flatten', {}),
  modelNode('fc', 'linear', { in_features: 8, out_features: 2, bias: true }),
  modelNode('output', 'output', {}),
]

const modelEdges: FlowHamsterEdge[] = [
  { id: 'm1', source: 'input', target: 'conv', sourceHandle: 'result', targetHandle: 'x' },
  { id: 'm2', source: 'conv', target: 'pool', sourceHandle: 'result', targetHandle: 'x' },
  { id: 'm3', source: 'pool', target: 'flat', sourceHandle: 'result', targetHandle: 'x' },
  { id: 'm4', source: 'flat', target: 'fc', sourceHandle: 'result', targetHandle: 'x' },
  { id: 'm5', source: 'fc', target: 'output', sourceHandle: 'result', targetHandle: 'x' },
]

describe('augmentation pipeline code generation', () => {
  it('generates code with random_horizontal_flip and random_crop', () => {
    const dataNodes: DataFlowNode[] = [
      dataNode('folder', 'folder_source', { path: './data/images', pattern: '*.jpg' }),
      dataNode('flip', 'random_horizontal_flip', { p: 0.5 }),
      dataNode('crop', 'random_crop', { size: 224, padding: 4 }),
      dataNode('norm', 'normalize', { mean: '0.485,0.456,0.406', std: '0.229,0.224,0.225' }),
      dataNode('loader', 'dataloader', { batch_size: 4, shuffle: false, num_workers: 0 }),
      dataNode('out', 'dataset_output', { fields: 'image,label' }),
    ]
    const dataEdges: DataFlowEdge[] = [
      { id: 'd1', source: 'folder', target: 'flip' },
      { id: 'd2', source: 'flip', target: 'crop' },
      { id: 'd3', source: 'crop', target: 'norm' },
      { id: 'd4', source: 'norm', target: 'loader' },
      { id: 'd5', source: 'loader', target: 'out' },
    ] as DataFlowEdge[]

    const bindings: WorkflowBinding[] = [
      {
        id: 'data:image->model_input:image',
        sourceGraph: 'data',
        sourceKey: 'image',
        target: 'model_input',
        targetKey: 'image',
      },
      {
        id: 'data:label->training_target:target',
        sourceGraph: 'data',
        sourceKey: 'label',
        target: 'training_target',
        targetKey: 'target',
      },
    ]

    const trainingConfig: WorkflowTrainingConfig = {
      taskType: 'classification',
      loss: { type: 'cross_entropy', enabled: true, params: {} },
      optimizer: { type: 'adam', enabled: true, params: { lr: 0.001 } },
      scheduler: { type: 'none', enabled: false, params: {} },
      metrics: [],
      runtime: { device: 'cpu', epochs: 1, batchSize: 4, amp: false, gradClip: null, numWorkers: 0 },
      checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
    }

    const { code } = generateLocalCode(modelNodes, modelEdges, undefined, trainingConfig, {
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    })

    expect(code).toContain('Random Horizontal Flip')
    expect(code).toContain('Random Crop')
    expect(code).toContain('Normalize')
    expect(code).toContain('torch.flip(_field_image, dims=[2])')
    expect(code).toContain('build_flowhamster_dataloader')
    expect(code).toContain('resolve_bound_inputs')
  })

  it('generates code with color_jitter and grayscale', () => {
    const dataNodes: DataFlowNode[] = [
      dataNode('csv', 'csv_source', {
        path: './data/train.csv',
        delimiter: ',',
        feature_columns: 'f0,f1,f2,f3',
        label_column: 'label',
      }),
      dataNode('jitter', 'color_jitter', { brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1 }),
      dataNode('gray', 'grayscale', {}),
      dataNode('loader', 'dataloader', { batch_size: 2, shuffle: false, num_workers: 0 }),
      dataNode('out', 'dataset_output', { fields: 'features,label' }),
    ]
    const dataEdges: DataFlowEdge[] = [
      { id: 'd1', source: 'csv', target: 'jitter' },
      { id: 'd2', source: 'jitter', target: 'gray' },
      { id: 'd3', source: 'gray', target: 'loader' },
      { id: 'd4', source: 'loader', target: 'out' },
    ] as DataFlowEdge[]

    const bindings: WorkflowBinding[] = [
      {
        id: 'data:features->model_input:features',
        sourceGraph: 'data',
        sourceKey: 'features',
        target: 'model_input',
        targetKey: 'features',
      },
      {
        id: 'data:label->training_target:target',
        sourceGraph: 'data',
        sourceKey: 'label',
        target: 'training_target',
        targetKey: 'target',
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

    const { code } = generateLocalCode(modelNodes, modelEdges, undefined, trainingConfig, {
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    })

    expect(code).toContain('Color Jitter')
    expect(code).toContain('Grayscale')
    expect(code).toContain('torch.clamp')
    expect(code).toContain('0.299 * _field_image[0:1]')
  })

  it('reports warnings for unhandled augmentation nodes in backend path', () => {
    const dataNodes: DataFlowNode[] = [
      dataNode('folder', 'folder_source', { path: './data', pattern: '*.jpg' }),
      dataNode('flip', 'random_horizontal_flip', { p: 0.5 }),
      dataNode('crop', 'random_crop', { size: 224, padding: 4 }),
      dataNode('out', 'dataset_output', { fields: 'image,label' }),
    ]
    const dataEdges: DataFlowEdge[] = [
      { id: 'd1', source: 'folder', target: 'flip' },
      { id: 'd2', source: 'flip', target: 'crop' },
      { id: 'd3', source: 'crop', target: 'out' },
    ] as DataFlowEdge[]

    const bindings: WorkflowBinding[] = [
      {
        id: 'data:image->model_input:image',
        sourceGraph: 'data',
        sourceKey: 'image',
        target: 'model_input',
        targetKey: 'image',
      },
    ]

    const compiled = compileDataWorkflow(modelNodes, dataNodes, dataEdges, bindings, {
      taskType: 'classification',
      loss: { type: 'cross_entropy', enabled: true, params: {} },
      optimizer: { type: 'adam', enabled: true, params: { lr: 0.001 } },
      scheduler: { type: 'none', enabled: false, params: {} },
      metrics: [],
      runtime: { device: 'cpu', epochs: 1, batchSize: 4, amp: false, gradClip: null, numWorkers: 0 },
      checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
    })

    expect(compiled.hasWorkflowRuntime).toBe(true)
    expect(compiled.pythonScaffold).toContain('Random Horizontal Flip')
    expect(compiled.pythonScaffold).toContain('Random Crop')
    expect(compiled.warnings).toHaveLength(0)
  })
})
