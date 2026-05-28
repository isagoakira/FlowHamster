import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { generateLocalCode } from '../src/utils/codeGenerator'
import { FlowHamsterEdge, FlowHamsterNode } from '../src/types/graph'
import { DataFlowEdge, DataFlowNode } from '../src/types/dataGraph'
import { WorkflowBinding, WorkflowTrainingConfig } from '../src/schema/workflowDocument'

const maybeDescribe = process.env.FLOWHAMSTER_RUN_GPU_SMOKE === '1' ? describe : describe.skip

function hasCuda(python: string, cwd: string): boolean {
  try {
    const stdout = execFileSync(
      python,
      ['-c', 'import torch; print("1" if torch.cuda.is_available() else "0")'],
      { cwd, encoding: 'utf8', timeout: 30_000 }
    )
    return stdout.trim() === '1'
  } catch {
    return false
  }
}

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

maybeDescribe('GPU node graph training smoke', () => {
  it('runs a generated CSV DataLoader training loop on CUDA when available', () => {
    const here = path.dirname(fileURLToPath(import.meta.url))
    const repoRoot = path.resolve(here, '..')
    const python = process.env.FLOWHAMSTER_PYTHON || 'python'

    if (!hasCuda(python, repoRoot)) {
      console.warn('Skipping GPU smoke because torch.cuda.is_available() is false.')
      return
    }

    const outDir = path.join(repoRoot, '.codex-logs', 'gpu-training-smoke')
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

    const dataGraphNodes: DataFlowNode[] = [
      dataNode('csv', 'csv_source', 'Random CSV', {
        path: csvPath,
        delimiter: ',',
        feature_columns: 'f0,f1,f2,f3',
        label_column: 'label',
      }, 0, 0),
      dataNode('loader', 'dataloader', 'DataLoader', {
        batch_size: 4,
        shuffle: false,
        num_workers: 0,
        pin_memory: false,
        drop_last: false,
      }, 220, 0),
      dataNode('out', 'dataset_output', 'Dataset Output', { fields: 'features,label' }, 440, 0),
    ]
    const dataGraphEdges: DataFlowEdge[] = [
      { id: 'd1', source: 'csv', target: 'loader' },
      { id: 'd2', source: 'loader', target: 'out' },
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
      runtime: { device: 'cuda', epochs: 1, batchSize: 4, amp: false, gradClip: null, numWorkers: 0 },
      checkpoint: { enabled: false, saveTopK: 1, monitor: 'loss', mode: 'min', earlyStopPatience: null },
    }

    const generated = generateLocalCode(modelNodes, modelEdges, {}, trainingConfig, {
      dataGraphNodes,
      dataGraphEdges,
      bindings,
    })
    expect(generated.code).toContain('build_flowhamster_dataloader')
    expect(generated.code).toContain('device = "cuda"')

    const scriptPath = path.join(outDir, 'generated_cuda_train.py')
    writeFileSync(scriptPath, generated.code, 'utf8')

    const stdout = execFileSync(python, [scriptPath], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 120_000,
    })

    expect(stdout).toContain('epoch=1')
    expect(stdout).toContain('step=1')
    expect(stdout).toContain('batch_size=4')
    expect(stdout).toContain('output=(4, 2)')
    expect(stdout).toContain('loss=')
    expect(stdout).toContain('device=cuda')
  }, 120_000)
})
