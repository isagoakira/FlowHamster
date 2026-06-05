import { generateLocalCode } from '../src/utils/codeGenerator'
import type { FlowHamsterEdge, FlowHamsterNode } from '../src/types/graph'
import type { DataFlowEdge, DataFlowNode } from '../src/types/dataGraph'
import type { WorkflowBinding, WorkflowTrainingConfig } from '../src/schema/workflowDocument'

function node(id: string, nodeType: string, params: Record<string, unknown> = {}): FlowHamsterNode {
  return {
    id,
    type: nodeType,
    position: { x: 0, y: 0 },
    data: { nodeType: nodeType as any, label: nodeType, params: params as any },
  } as FlowHamsterNode
}

function edge(source: string, target: string, targetHandle = 'x', sourceHandle = 'result'): FlowHamsterEdge {
  return {
    id: `${source}_${target}_${targetHandle}`,
    source,
    target,
    sourceHandle,
    targetHandle,
  } as FlowHamsterEdge
}

function dataNode(id: string, nodeType: string, params: Record<string, unknown> = {}): DataFlowNode {
  return {
    id,
    type: nodeType,
    position: { x: 0, y: 0 },
    data: { nodeType: nodeType as any, label: nodeType, params: params as any },
  } as DataFlowNode
}

function dataEdge(source: string, target: string): DataFlowEdge {
  return { id: `${source}_${target}`, source, target } as DataFlowEdge
}

function linearGraph(): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  const nodes = [
    node('input', 'input', { name: 'features' }),
    node('fc1', 'linear', { in_features: 4, out_features: 8, bias: true }),
    node('relu', 'relu'),
    node('fc2', 'linear', { in_features: 8, out_features: 2, bias: true }),
    node('output', 'output'),
  ]
  const edges = [
    edge('input', 'fc1'),
    edge('fc1', 'relu'),
    edge('relu', 'fc2'),
    edge('fc2', 'output'),
  ]
  return { nodes, edges }
}

function lenet5Graph(): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  const nodes = [
    node('input', 'input', { name: 'image' }),
    node('conv1', 'conv2d', { in_channels: 1, out_channels: 6, kernel_size: 5, stride: 1, padding: 0, bias: true }),
    node('relu1', 'relu'),
    node('pool1', 'maxpool2d', { kernel_size: 2, stride: 2, padding: 0 }),
    node('conv2', 'conv2d', { in_channels: 6, out_channels: 16, kernel_size: 5, stride: 1, padding: 0, bias: true }),
    node('relu2', 'relu'),
    node('pool2', 'maxpool2d', { kernel_size: 2, stride: 2, padding: 0 }),
    node('flat', 'flatten', { start_dim: 1 }),
    node('fc1', 'linear', { in_features: 256, out_features: 120, bias: true }),
    node('fc2', 'linear', { in_features: 120, out_features: 84, bias: true }),
    node('fc3', 'linear', { in_features: 84, out_features: 10, bias: true }),
    node('output', 'output'),
  ]
  const edges = [
    edge('input', 'conv1'),
    edge('conv1', 'relu1'),
    edge('relu1', 'pool1'),
    edge('pool1', 'conv2'),
    edge('conv2', 'relu2'),
    edge('relu2', 'pool2'),
    edge('pool2', 'flat'),
    edge('flat', 'fc1'),
    edge('fc1', 'fc2'),
    edge('fc2', 'fc3'),
    edge('fc3', 'output'),
  ]
  return { nodes, edges }
}

function trainingConfig(): WorkflowTrainingConfig {
  return {
    taskType: 'classification',
    loss: { type: 'cross_entropy', enabled: true, params: {} },
    optimizer: { type: 'adam', enabled: true, params: { lr: 0.01 } },
    scheduler: { type: 'none', enabled: false, params: {} },
    metrics: [],
    runtime: { device: 'cpu', epochs: 1, batchSize: 2, amp: false, gradClip: null, numWorkers: 0 },
    checkpoint: { enabled: false, saveTopK: 3, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
  }
}

function dataWorkflow(): { nodes: DataFlowNode[]; edges: DataFlowEdge[]; bindings: WorkflowBinding[] } {
  return {
    nodes: [
      dataNode('csv', 'csv_source', { path: './data/train.csv', feature_columns: 'f0,f1,f2,f3', label_column: 'label' }),
      dataNode('loader', 'dataloader', { batch_size: 2, shuffle: false, num_workers: 0, pin_memory: false }),
      dataNode('out', 'dataset_output', { fields: 'features,label' }),
    ],
    edges: [dataEdge('csv', 'loader'), dataEdge('loader', 'out')],
    bindings: [
      { id: 'data:features->model_input:features', sourceGraph: 'data', sourceKey: 'features', target: 'model_input', targetKey: 'features' },
      { id: 'data:label->training_target:target', sourceGraph: 'data', sourceKey: 'label', target: 'training_target', targetKey: 'target' },
    ],
  }
}

function graphFor(name: string): {
  nodes: FlowHamsterNode[]
  edges: FlowHamsterEdge[]
  trainingConfig?: WorkflowTrainingConfig
  workflowOptions?: { dataGraphNodes?: DataFlowNode[]; dataGraphEdges?: DataFlowEdge[]; bindings?: WorkflowBinding[] }
} {
  if (name === 'lenet5') return lenet5Graph()
  if (name === 'linear_with_training') {
    const graph = linearGraph()
    const workflow = dataWorkflow()
    return {
      ...graph,
      trainingConfig: trainingConfig(),
      workflowOptions: {
        dataGraphNodes: workflow.nodes,
        dataGraphEdges: workflow.edges,
        bindings: workflow.bindings,
      },
    }
  }
  if (name === 'composite_linear') return linearGraph()
  return linearGraph()
}

const requested = process.argv[2] || 'linear'
const graph = graphFor(requested)
const result = generateLocalCode(
  graph.nodes,
  graph.edges,
  undefined,
  graph.trainingConfig,
  graph.workflowOptions,
)

console.log(JSON.stringify({ code: result.code }))
