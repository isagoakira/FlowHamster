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
  modelNode('linear', 'linear', { in_features: 10, out_features: 3, bias: true }),
  modelNode('output', 'output', {}),
]

const modelEdges: FlowHamsterEdge[] = [
  { id: 'm1', source: 'input', target: 'linear', sourceHandle: 'result', targetHandle: 'x' },
  { id: 'm2', source: 'linear', target: 'output', sourceHandle: 'result', targetHandle: 'input' },
]

const dataNodes: DataFlowNode[] = [
  dataNode('csv', 'csv_source', {
    path: './data/train.csv',
    delimiter: ',',
    feature_columns: 'f0,f1,f2,f3',
    label_column: 'label',
  }),
  dataNode('loader', 'dataloader', { batch_size: 4, shuffle: false, num_workers: 0, pin_memory: false }),
  dataNode('out', 'dataset_output', { fields: 'features,label' }),
]

const dataEdges: DataFlowEdge[] = [
  { id: 'd1', source: 'csv', target: 'loader' },
  { id: 'd2', source: 'loader', target: 'out' },
] as DataFlowEdge[]

const bindings: WorkflowBinding[] = [
  {
    id: 'data:image->model_input:image',
    sourceGraph: 'data',
    sourceKey: 'features',
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
  optimizer: { type: 'adamw', enabled: true, params: { lr: 0.001 } },
  scheduler: { type: 'step_lr', enabled: true, params: { step_size: 1, gamma: 0.5 } },
  metrics: [],
  runtime: { device: 'cpu', epochs: 1, batchSize: 4, amp: false, gradClip: null, numWorkers: 0 },
  checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
}

describe('data graph to training pipeline', () => {
  it('turns a dataloader node into executable DataLoader scaffold', () => {
    const compiled = compileDataWorkflow(modelNodes, dataNodes, dataEdges, bindings, trainingConfig)

    expect(compiled.hasWorkflowRuntime).toBe(true)
    expect(compiled.pythonScaffold).toContain('class FlowHamsterDataset(torch.utils.data.Dataset):')
    expect(compiled.pythonScaffold).toContain('def build_flowhamster_dataloader():')
    expect(compiled.pythonScaffold).toContain('torch.utils.data.DataLoader(')
    expect(compiled.pythonScaffold).toContain("'batch_size': 4")
    expect(compiled.pythonScaffold).toContain("_field_features")
    expect(compiled.pythonScaffold).toContain('self._csv_feature_columns = ["f0", "f1", "f2", "f3"]')
    expect(compiled.pythonScaffold).toContain('self._csv_label_column = "label"')
    expect(compiled.pythonScaffold).toContain('batch = next(iter(loader))')
  })

  it('generates a bound training script that trains over the data workflow', () => {
    const { code } = generateLocalCode(modelNodes, modelEdges, undefined, trainingConfig, {
      dataGraphNodes: dataNodes,
      dataGraphEdges: dataEdges,
      bindings,
    })

    expect(code).toContain('loader = build_flowhamster_dataloader()')
    expect(code).toContain('for epoch in range(1):')
    expect(code).toContain('for step, batch in enumerate(loader, start=1):')
    expect(code).toContain('model_feed, target = resolve_bound_inputs(batch, runtime_device)')
    expect(code).toContain('def build_flowhamster_dataloader():')
    expect(code).toContain('def flowhamster_prepare_loss_inputs')
    expect(code.match(/loss = loss_fn\(loss_input, loss_target\)/g) ?? []).toHaveLength(1)
    expect(code).toContain('loss.backward()')
    expect(code).toContain('optimizer.step()')
    expect(code).toContain('epoch={epoch + 1}')
    expect(code).not.toContain('task=classification, epochs=1')
  })

  it('reports binding mismatches before runtime', () => {
    const compiled = compileDataWorkflow(
      [modelNode('input', 'input', { name: 'features' })],
      [dataNode('out', 'dataset_output', { fields: 'features,label' })],
      [],
      [
        {
          id: 'data:features->model_input:typo',
          sourceGraph: 'data',
          sourceKey: 'features',
          target: 'model_input',
          targetKey: 'typo',
        },
        {
          id: 'data:missing->training_target:target',
          sourceGraph: 'data',
          sourceKey: 'missing',
          target: 'training_target',
          targetKey: 'target',
        },
      ],
      trainingConfig
    )

    expect(compiled.warnings.join('\n')).toContain('模型输入 features 尚未绑定数据字段')
    expect(compiled.warnings.join('\n')).toContain('绑定源字段 missing 未在 Dataset Output 中声明')
  })
})
