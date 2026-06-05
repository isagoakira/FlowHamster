import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { generateLocalCode } from '../src/utils/codeGenerator'
import { FlowHamsterEdge, FlowHamsterNode } from '../src/types/graph'
import { DataFlowEdge, DataFlowNode } from '../src/types/dataGraph'
import { WorkflowBinding, WorkflowTrainingConfig } from '../src/schema/workflowDocument'

const maybeDescribe = process.env.FLOWHAMSTER_RUN_AUGMENTATION_SMOKE === '1' ? describe : describe.skip

function modelNode(
  id: string,
  nodeType: string,
  label: string,
  params: Record<string, string | number | boolean>,
  x: number,
  y: number
): FlowHamsterNode {
  return {
    id,
    type: `${nodeType}Node`,
    position: { x, y },
    data: { nodeType: nodeType as any, label, params },
  }
}

function dataNode(
  id: string,
  nodeType: string,
  label: string,
  params: Record<string, string | number | boolean>,
  x: number,
  y: number
): DataFlowNode {
  return {
    id,
    type: `${nodeType}Node`,
    position: { x, y },
    data: { nodeType: nodeType as any, label, params },
  }
}

maybeDescribe('augmentation pipeline training smoke', () => {
  it('generates and runs a training loop with random_horizontal_flip + normalize', () => {
    const here = path.dirname(fileURLToPath(import.meta.url))
    const repoRoot = path.resolve(here, '..')
    const outDir = path.join(repoRoot, '.codex-logs', 'augmentation-smoke')
    mkdirSync(outDir, { recursive: true })

    const csvPath = path.join(outDir, 'train.csv')
    const rows = ['f0,f1,f2,f3,label']
    for (let i = 0; i < 16; i++) {
      const f0 = i / 10
      const f1 = (i % 3) / 3
      const f2 = (i % 5) / 5
      const f3 = (i % 7) / 7
      const label = f0 + f1 + f2 + f3 > 1.5 ? 1 : 0
      rows.push([f0, f1, f2, f3, label].join(','))
    }
    writeFileSync(csvPath, `${rows.join('\n')}\n`, 'utf8')

    // Simple MLP model
    const modelNodes: FlowHamsterNode[] = [
      modelNode('input', 'input', 'Input', { name: 'features', shape: '4', dtype: 'float32' }, 0, 0),
      modelNode('fc1', 'linear', 'Linear 4->8', { in_features: 4, out_features: 8, bias: true }, 180, 0),
      modelNode('relu', 'relu', 'ReLU', {}, 360, 0),
      modelNode('fc2', 'linear', 'Linear 8->2', { in_features: 8, out_features: 2, bias: true }, 540, 0),
      modelNode('output', 'output', 'Output', {}, 720, 0),
    ]
    const modelEdges: FlowHamsterEdge[] = [
      { id: 'e1', source: 'input', target: 'fc1', sourceHandle: 'result', targetHandle: 'x' },
      { id: 'e2', source: 'fc1', target: 'relu', sourceHandle: 'result', targetHandle: 'x' },
      { id: 'e3', source: 'relu', target: 'fc2', sourceHandle: 'result', targetHandle: 'x' },
      { id: 'e4', source: 'fc2', target: 'output', sourceHandle: 'result', targetHandle: 'x' },
    ]

    // Data pipeline with augmentation nodes
    const dataGraphNodes: DataFlowNode[] = [
      dataNode('csv', 'csv_source', 'CSV', {
        path: csvPath,
        delimiter: ',',
        feature_columns: 'f0,f1,f2,f3',
        label_column: 'label',
      }, 0, 0),
      dataNode('flip', 'random_horizontal_flip', 'Random H-Flip', { p: 0.5 }, 220, 0),
      dataNode('norm', 'normalize', 'Normalize', { mean: '0.5,0.5,0.5', std: '0.5,0.5,0.5' }, 440, 0),
      dataNode('loader', 'dataloader', 'DataLoader', {
        batch_size: 4,
        shuffle: false,
        num_workers: 0,
        pin_memory: false,
        drop_last: false,
      }, 660, 0),
      dataNode('out', 'dataset_output', 'Dataset Output', { fields: 'features,label' }, 880, 0),
    ]
    const dataGraphEdges: DataFlowEdge[] = [
      { id: 'd1', source: 'csv', target: 'flip' },
      { id: 'd2', source: 'flip', target: 'norm' },
      { id: 'd3', source: 'norm', target: 'loader' },
      { id: 'd4', source: 'loader', target: 'out' },
    ]
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
      optimizer: { type: 'adam', enabled: true, params: { lr: 0.01 } },
      scheduler: { type: 'none', enabled: false, params: {} },
      metrics: [],
      runtime: { device: 'cpu', epochs: 1, batchSize: 4, amp: false, gradClip: null, numWorkers: 0 },
      checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
    }

    const generated = generateLocalCode(modelNodes, modelEdges, {}, trainingConfig, {
      dataGraphNodes,
      dataGraphEdges,
      bindings,
    })

    expect(generated.code).toContain('build_flowhamster_dataloader')
    expect(generated.code).toContain('Random Horizontal Flip')
    expect(generated.code).toContain('Normalize')

    const scriptPath = path.join(outDir, 'generated_train.py')
    writeFileSync(scriptPath, generated.code, 'utf8')

    const python = process.env.FLOWHAMSTER_PYTHON || 'python'
    const stdout = execFileSync(python, [scriptPath], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 120_000,
    })

    expect(stdout).toContain('epoch=1')
    expect(stdout).toContain('step=1')
    expect(stdout).toContain('batch_size=4')
    expect(stdout).toContain('loss=')
  }, 120_000)
})
