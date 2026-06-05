import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'

export const WORKFLOW_DOCUMENT_VERSION = '2.0.0'

export type WorkflowGraphKind = 'model' | 'data'
export type WorkflowBindingTarget = 'model_input' | 'model_output' | 'training_input' | 'training_target' | 'loss_target'

export interface WorkflowPortContract {
  name: string
  dtype?: string
  shapeHint?: string
  semanticRole?: string
  description?: string
}

export interface WorkflowGraphSnapshot<NodeType = FlowHamsterNode, EdgeType = FlowHamsterEdge> {
  kind: WorkflowGraphKind
  nodes: NodeType[]
  edges: EdgeType[]
  contract: {
    inputs: WorkflowPortContract[]
    outputs: WorkflowPortContract[]
  }
}

export interface WorkflowComponentConfig {
  type: string
  enabled: boolean
  params: Record<string, unknown>
}

// Loss function types
export type LossType =
  // Basic / Classic
  | 'cross_entropy' | 'mse' | 'bce' | 'bce_logits'
  // CV - Segmentation
  | 'dice' | 'focal' | 'lovasz' | 'tversky' | 'iou' | 'giou' | 'dice_ce'
  // CV - Metric Learning
  | 'msssim' | 'perceptual' | 'content' | 'style'
  // CV - Detection
  | 'smooth_l1' | 'focal_loss' | 'class_balanced'
  // Audio Enhancement
  | 'stft' | 'sdr' | 'sisdr' | 'mel_spec' | 'waveform' | 'multi_res' | 'phase'
  // NLP / Other
  | 'label_smoothing' | 'contrastive' | 'custom'

// Composite loss component: single loss with weight
export interface LossComponent {
  type: LossType
  weight: number
  customCode?: string
}

export type TaskType = 'classification' | 'regression' | 'segmentation' | 'detection' | 'nlp' | 'custom'

export interface WorkflowTrainingConfig {
  taskType: TaskType
  loss: WorkflowComponentConfig
  optimizer: WorkflowComponentConfig
  scheduler: WorkflowComponentConfig
  metrics: WorkflowComponentConfig[]
  runtime: {
    device: string
    epochs: number
    batchSize: number
    amp: boolean
    gradClip: number | null
    numWorkers: number
  }
  checkpoint: {
    enabled: boolean
    saveTopK: number
    monitor: string
    mode: 'min' | 'max'
    earlyStopPatience: number | null
  }
}

export interface WorkflowDocumentExportOptions {
  name?: string
  description?: string
  trainingConfig?: WorkflowTrainingConfig
}

export interface WorkflowBinding {
  id: string
  sourceGraph: WorkflowGraphKind
  sourceKey: string
  target: WorkflowBindingTarget
  targetKey: string
  description?: string
}

export interface WorkflowMetadata {
  name: string
  description: string
  schemaVersion: string
  createdAt: string
  updatedAt: string
  exportSource: 'flowhamster'
}

export type WorkflowSerializableNode = Record<string, unknown>
export type WorkflowSerializableEdge = Record<string, unknown>

export interface WorkflowDocument {
  version: string
  metadata: WorkflowMetadata
  modelGraph: WorkflowGraphSnapshot<FlowHamsterNode, FlowHamsterEdge>
  dataGraph: WorkflowGraphSnapshot<WorkflowSerializableNode, WorkflowSerializableEdge>
  trainingConfig: WorkflowTrainingConfig
  bindings: WorkflowBinding[]
}

export interface LegacyWorkflowProject {
  version?: string
  nodes?: unknown[]
  edges?: unknown[]
  timestamp?: string
}
