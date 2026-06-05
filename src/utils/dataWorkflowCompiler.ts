import { FlowHamsterNode } from '../types/graph'
import { DataFlowEdge, DataFlowNode, DataNodeType, FieldDtype, FieldSpec } from '../types/dataGraph'
import { WorkflowBinding, WorkflowTrainingConfig } from '../schema/workflowDocument'

export interface CompiledDataWorkflow {
  outputFields: string[]
  modelInputBindings: Array<{ targetKey: string; sourceKey: string }>
  targetBindingSource: string | null
  lossTargetBindings: Array<{ lossNodeId: string; sourceKey: string }>
  primaryModelInputKey: string | null
  warnings: string[]
  summaryLines: string[]
  pythonScaffold: string
  hasWorkflowRuntime: boolean
  validationErrors: ValidationError[]
}

export interface ValidationError {
  nodeId: string
  field: string
  message: string
  severity: 'error' | 'warning'
}

export interface FieldContract {
  name: string
  dtype: FieldDtype
  shapeHint?: string
  required: boolean
}

// ─────────────────────────────────────────────────────────────────
// Dtype compatibility matrix
// ─────────────────────────────────────────────────────────────────

// Define which dtypes are compatible with which model input expectations
const DTYPE_COMPATIBILITY: Record<FieldDtype, { compatible: FieldDtype[], description: string }> = {
  tensor: { compatible: ['tensor', 'image', 'label', 'mask', 'text', 'audio', 'spectrogram', 'scalar', 'string'], description: 'Generic tensor - compatible with most inputs' },
  image: { compatible: ['image', 'tensor'], description: 'Image tensor (N,C,H,W format)' },
  label: { compatible: ['label', 'scalar', 'tensor'], description: 'Classification label (integer class index)' },
  mask: { compatible: ['mask', 'tensor'], description: 'Segmentation mask (N,1,H,W or N,H,W)' },
  text: { compatible: ['text', 'tensor'], description: 'Text/token sequences' },
  audio: { compatible: ['audio', 'tensor', 'spectrogram'], description: 'Audio waveform or features' },
  spectrogram: { compatible: ['spectrogram', 'tensor'], description: 'Spectrogram or audio features' },
  scalar: { compatible: ['scalar', 'label', 'tensor'], description: 'Scalar value (float or int)' },
  string: { compatible: ['string', 'text', 'tensor'], description: 'String value' },
}

// Check if source dtype is compatible with target dtype
function isDtypeCompatible(sourceDtype: FieldDtype, targetDtype: FieldDtype): boolean {
  if (sourceDtype === targetDtype) return true
  return DTYPE_COMPATIBILITY[sourceDtype]?.compatible.includes(targetDtype) ?? false
}

// Get incompatibility reasons
function getDtypeIncompatibility(sourceDtype: FieldDtype, targetDtype: FieldDtype): string | null {
  if (isDtypeCompatible(sourceDtype, targetDtype)) return null
  return `dtype mismatch: source '${sourceDtype}' may not be compatible with target '${targetDtype}'`
}

// Parse fieldSpecs from dataset_output node
function parseFieldSpecs(fieldSpecsStr: string | undefined): FieldSpec[] {
  if (!fieldSpecsStr) return []
  try {
    const parsed = JSON.parse(fieldSpecsStr)
    if (Array.isArray(parsed)) return parsed
    return []
  } catch {
    return []
  }
}

// Infer dtype from field name
function inferDtypeFromFieldName(name: string): FieldDtype {
  const lowered = name.toLowerCase()
  if (lowered.includes('image') || lowered.includes('img')) return 'image'
  if (lowered.includes('label') || lowered.includes('target') || lowered.includes('class')) return 'label'
  if (lowered.includes('mask')) return 'mask'
  if (lowered.includes('text') || lowered.includes('token')) return 'text'
  if (lowered.includes('audio') || lowered.includes('waveform')) return 'audio'
  if (lowered.includes('spectrogram') || lowered.includes('mel') || lowered.includes('mfcc')) return 'spectrogram'
  return 'tensor'
}

// Collect field specs from all dataset_output nodes
export function collectDataFieldSpecs(dataNodes: DataFlowNode[]): Map<string, FieldSpec> {
  const fieldSpecs = new Map<string, FieldSpec>()
  const outputNodes = dataNodes.filter(n => n.data.nodeType === 'dataset_output')

  for (const node of outputNodes) {
    const specs = parseFieldSpecs(node.data.fieldSpecs)
    if (specs.length > 0) {
      // Use provided specs
      for (const spec of specs) {
        if (!fieldSpecs.has(spec.name)) {
          fieldSpecs.set(spec.name, spec)
        }
      }
    } else {
      // Infer from field names
      const fields = String(node.data.params.fields || '').split(',').map(f => f.trim()).filter(Boolean)
      for (const field of fields) {
        if (!fieldSpecs.has(field)) {
          fieldSpecs.set(field, {
            name: field,
            dtype: inferDtypeFromFieldName(field),
            shapeHint: field === 'image' ? 'N,C,H,W' : undefined,
          })
        }
      }
    }
  }
  return fieldSpecs
}

// Validation result for a single binding
export interface BindingValidation {
  sourceKey: string
  targetKey: string
  sourceDtype?: FieldDtype
  targetDtype?: FieldDtype
  isCompatible: boolean
  warning?: string
  error?: string
}

// Validate all bindings for dtype/shape compatibility
export function validateBindingCompatibility(
  bindings: WorkflowBinding[],
  dataNodes: DataFlowNode[],
  modelNodes: FlowHamsterNode[]
): BindingValidation[] {
  const results: BindingValidation[] = []
  const fieldSpecs = collectDataFieldSpecs(dataNodes)

  // Get model input expectations (we'll use a basic inference for now)
  const modelInputMap = new Map<string, FieldDtype>()
  for (const node of modelNodes) {
    if (node.data.nodeType === 'input') {
      const name = String(node.data.params.name || '')
      if (name) {
        // Infer dtype from input name - default to tensor
        modelInputMap.set(name, inferDtypeFromFieldName(name))
      }
    }
  }

  for (const binding of bindings) {
    if (binding.sourceGraph !== 'data') continue

    const sourceSpec = fieldSpecs.get(binding.sourceKey)
    // Determine expected target dtype based on binding target type
    let targetDtype: FieldDtype
    if (binding.target === 'loss_target' || binding.target === 'training_target') {
      targetDtype = 'label'
    } else {
      targetDtype = modelInputMap.get(binding.targetKey) || 'tensor'
    }

    const validation: BindingValidation = {
      sourceKey: binding.sourceKey,
      targetKey: binding.targetKey,
      sourceDtype: sourceSpec?.dtype,
      targetDtype,
      isCompatible: true,
    }

    if (sourceSpec) {
      const incompatibility = getDtypeIncompatibility(sourceSpec.dtype, targetDtype)
      if (incompatibility) {
        validation.isCompatible = false
        validation.error = incompatibility
      }
      // Additional check: loss targets should preferably be label/scalar/tensor
      if ((binding.target === 'loss_target' || binding.target === 'training_target') &&
          !['label', 'scalar', 'tensor', 'mask'].includes(sourceSpec.dtype)) {
        validation.isCompatible = false
        validation.error = `Loss target requires label/scalar/mask dtype, got '${sourceSpec.dtype}'`
      }
    } else {
      validation.isCompatible = false
      validation.warning = `No field spec found for '${binding.sourceKey}', dtype unknown`
    }

    results.push(validation)
  }

  return results
}

// ─────────────────────────────────────────────────────────────────
// Task type inference from model architecture
// ─────────────────────────────────────────────────────────────────

export type InferredTaskType = 'classification' | 'regression' | 'segmentation' | 'detection' | 'nlp' | 'unknown'

export interface ModelInferenceResult {
  taskType: InferredTaskType
  suggestedLoss: string
  suggestedMetrics: string[]
  confidence: 'high' | 'medium' | 'low'
  reason: string
  outFeatures?: number
}

// Find upstream Linear layer from a node by following edges backwards
function findUpstreamLinear(nodes: FlowHamsterNode[], edges: any[], nodeId: string): FlowHamsterNode | null {
  // Find edges that target this node
  const incomingEdges = edges.filter(e => e.target === nodeId || e.targetHandle === nodeId)
  if (incomingEdges.length === 0) return null

  for (const edge of incomingEdges) {
    const sourceNode = nodes.find(n => n.id === edge.source)
    if (!sourceNode) continue
    if (sourceNode.data.nodeType === 'linear') return sourceNode
    // Recursively search upstream
    const upstream = findUpstreamLinear(nodes, edges, sourceNode.id)
    if (upstream) return upstream
  }
  return null
}

// Infer task type from model output layer
export function inferTaskTypeFromModel(
  modelNodes: FlowHamsterNode[],
  modelEdges: any[]
): ModelInferenceResult {
  // Find output nodes
  const outputNodes = modelNodes.filter(n => n.data.nodeType === 'output')
  if (outputNodes.length === 0) {
    return {
      taskType: 'unknown',
      suggestedLoss: 'cross_entropy',
      suggestedMetrics: ['accuracy'],
      confidence: 'low',
      reason: 'No output node found in model',
    }
  }

  // For each output, find the upstream linear layer
  let maxOutFeatures = 0
  let foundLinear = false

  for (const outputNode of outputNodes) {
    const linearNode = findUpstreamLinear(modelNodes, modelEdges, outputNode.id)
    if (linearNode && linearNode.data.params) {
      foundLinear = true
      const outFeatures = Number(linearNode.data.params.out_features) || 0
      if (outFeatures > maxOutFeatures) {
        maxOutFeatures = outFeatures
      }
    }
  }

  if (!foundLinear || maxOutFeatures === 0) {
    return {
      taskType: 'unknown',
      suggestedLoss: 'cross_entropy',
      suggestedMetrics: ['accuracy'],
      confidence: 'low',
      reason: 'Could not find Linear layer with out_features in model',
    }
  }

  // Infer task type based on out_features
  if (maxOutFeatures === 1) {
    return {
      taskType: 'regression',
      suggestedLoss: 'mse',
      suggestedMetrics: ['mse', 'mae'],
      confidence: 'high',
      reason: `Linear layer with out_features=1 detected → regression task`,
      outFeatures: maxOutFeatures,
    }
  } else if (maxOutFeatures > 1) {
    // Could be classification or detection/segmentation
    // For now, default to classification for multi-class
    return {
      taskType: 'classification',
      suggestedLoss: 'cross_entropy',
      suggestedMetrics: ['accuracy', 'top5'],
      confidence: 'high',
      reason: `Linear layer with out_features=${maxOutFeatures} detected → classification (${maxOutFeatures} classes)`,
      outFeatures: maxOutFeatures,
    }
  }

  return {
    taskType: 'unknown',
    suggestedLoss: 'cross_entropy',
    suggestedMetrics: ['accuracy'],
    confidence: 'low',
    reason: 'Could not determine task type from model architecture',
  }
}

// ─────────────────────────────────────────────────────────────────
// 节点参数校验
// ─────────────────────────────────────────────────────────────────

type ParamValidatorFn = (value: unknown) => ValidationError | null

const PARAM_VALIDATORS: Partial<Record<DataNodeType, Record<string, ParamValidatorFn>>> = {
  folder_source: {
    path: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'path', message: 'path must be a non-empty string', severity: 'error' } as ValidationError
      return null
    },
    pattern: (v) => {
      if (typeof v !== 'string') return { nodeId: '', field: 'pattern', message: 'pattern must be a string', severity: 'error' } as ValidationError
      return null
    },
  },
  csv_source: {
    path: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'path', message: 'path must be a non-empty string', severity: 'error' } as ValidationError
      return null
    },
    delimiter: (v) => {
      if (typeof v !== 'string' || v.length !== 1) return { nodeId: '', field: 'delimiter', message: 'delimiter must be a single character', severity: 'warning' } as ValidationError
      return null
    },
  },
  jsonl_source: {
    path: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'path', message: 'path must be a non-empty string', severity: 'error' } as ValidationError
      return null
    },
  },
  parquet_source: {
    path: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'path', message: 'path must be a non-empty string', severity: 'error' } as ValidationError
      return null
    },
  },
  huggingface_source: {
    dataset_name: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'dataset_name', message: 'dataset_name must be a non-empty string', severity: 'error' } as ValidationError
      return null
    },
  },
  read_image: {
    mode: (v) => {
      if (typeof v !== 'string' || !['RGB', 'L', 'RGBA'].includes(v)) return { nodeId: '', field: 'mode', message: 'mode must be RGB, L, or RGBA', severity: 'warning' } as ValidationError
      return null
    },
  },
  resize: {
    size: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'size', message: 'size must be W,H format', severity: 'error' } as ValidationError
      return null
    },
  },
  normalize: {
    mean: (v) => {
      if (typeof v !== 'string') return { nodeId: '', field: 'mean', message: 'mean must be comma-separated values', severity: 'error' } as ValidationError
      return null
    },
    std: (v) => {
      if (typeof v !== 'string') return { nodeId: '', field: 'std', message: 'std must be comma-separated values', severity: 'error' } as ValidationError
      return null
    },
  },
  train_val_split: {
    train_ratio: (v) => {
      const ratio = Number(v)
      if (isNaN(ratio) || ratio <= 0 || ratio >= 1) return { nodeId: '', field: 'train_ratio', message: 'train_ratio must be between 0 and 1 (exclusive)', severity: 'error' } as ValidationError
      return null
    },
  },
  batch: {
    batch_size: (v) => {
      const size = Number(v)
      if (isNaN(size) || size < 1) return { nodeId: '', field: 'batch_size', message: 'batch_size must be a positive integer', severity: 'error' } as ValidationError
      return null
    },
  },
  shuffle: {
    enabled: (v) => {
      if (typeof v !== 'boolean') return { nodeId: '', field: 'enabled', message: 'enabled must be a boolean', severity: 'warning' } as ValidationError
      return null
    },
    seed: (v) => {
      const seed = Number(v)
      if (isNaN(seed) || seed < 0) return { nodeId: '', field: 'seed', message: 'seed must be a non-negative integer', severity: 'warning' } as ValidationError
      return null
    },
  },
  dataloader: {
    batch_size: (v) => {
      const size = Number(v)
      if (isNaN(size) || size < 1) return { nodeId: '', field: 'batch_size', message: 'batch_size must be a positive integer', severity: 'error' } as ValidationError
      return null
    },
    num_workers: (v) => {
      const n = Number(v)
      if (isNaN(n) || n < 0) return { nodeId: '', field: 'num_workers', message: 'num_workers must be non-negative', severity: 'warning' } as ValidationError
      return null
    },
  },
  tokenizer: {
    tokenizer_type: (v) => {
      if (typeof v !== 'string' || !v) return { nodeId: '', field: 'tokenizer_type', message: 'tokenizer_type is required', severity: 'error' } as ValidationError
      return null
    },
    max_length: (v) => {
      const len = Number(v)
      if (isNaN(len) || len < 1) return { nodeId: '', field: 'max_length', message: 'max_length must be positive', severity: 'warning' } as ValidationError
      return null
    },
  },
  random_mask: {
    mask_prob: (v) => {
      const p = Number(v)
      if (isNaN(p) || p < 0 || p > 1) return { nodeId: '', field: 'mask_prob', message: 'mask_prob must be between 0 and 1', severity: 'error' } as ValidationError
      return null
    },
  },
}

function validateNodeParams(node: DataFlowNode): ValidationError[] {
  const errors: ValidationError[] = []
  const nodeType = node.data.nodeType as DataNodeType
  const validators = PARAM_VALIDATORS[nodeType]
  if (!validators) return errors

  const params = node.data.params || {}
  for (const [param, validator] of Object.entries(validators)) {
    const error = validator(params[param])
    if (error) {
      error.nodeId = node.id
      errors.push(error)
    }
  }

  return errors
}

// ─────────────────────────────────────────────────────────────────
// 契约校验
// ─────────────────────────────────────────────────────────────────

function validateDataFlowContract(
  nodes: DataFlowNode[],
  _edges: DataFlowEdge[],
  bindings: WorkflowBinding[]
): ValidationError[] {
  const errors: ValidationError[] = []
  const outputNodes = nodes.filter(n => n.data.nodeType === 'dataset_output')
  const outputFields = new Set<string>()

  for (const node of outputNodes) {
    const fields = String(node.data.params.fields || '').split(',').map(f => f.trim()).filter(Boolean)
    for (const field of fields) {
      if (outputFields.has(field)) {
        errors.push({
          nodeId: node.id,
          field: 'fields',
          message: `Duplicate field "${field}" in dataset_output`,
          severity: 'warning',
        })
      }
      outputFields.add(field)
    }
  }

  // 检查 binding 源字段是否在 output 中声明
  for (const binding of bindings) {
    if (binding.sourceGraph === 'data' && binding.sourceKey && !outputFields.has(binding.sourceKey)) {
      errors.push({
        nodeId: binding.id,
        field: 'sourceKey',
        message: `Binding source field "${binding.sourceKey}" not declared in dataset_output`,
        severity: binding.target === 'training_target' ? 'error' : 'warning',
      })
    }
  }

  // 检查是否有数据源节点
  const hasSource = nodes.some(n =>
    ['folder_source', 'csv_source', 'jsonl_source', 'parquet_source', 'huggingface_source'].includes(n.data.nodeType)
  )
  if (!hasSource && nodes.length > 0) {
    errors.push({
      nodeId: 'dataflow',
      field: 'source',
      message: 'No data source node found (folder_source, csv_source, jsonl_source, parquet_source, or huggingface_source)',
      severity: 'warning',
    })
  }

  return errors
}

function safeToken(value: string): string {
  return value.replace(/[^a-zA-Z0-9_]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase() || 'field'
}

function pythonValue(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (value === null || value === undefined) return 'None'
  return String(value)
}

function pythonBool(value: unknown, fallback: boolean): string {
  if (value === undefined || value === null) return fallback ? 'True' : 'False'
  return value === true || value === 'true' ? 'True' : 'False'
}

function inferTensorExpression(field: string, taskType?: string): string {
  const lowered = field.toLowerCase()
  if (lowered.includes('label') || lowered.includes('target') || lowered.includes('class')) {
    return taskType === 'regression'
      ? 'torch.randn(1, 1, device=device)'
      : 'torch.randint(0, 10, (1,), dtype=torch.long, device=device)'
  }
  if (lowered.includes('mask')) return 'torch.randint(0, 2, (1, 1, 224, 224), dtype=torch.long, device=device)'
  if (lowered.includes('token') || lowered.includes('text') || lowered.includes('ids')) {
    return 'torch.randint(0, 1000, (1, 32), dtype=torch.long, device=device)'
  }
  if (lowered.includes('image') || lowered.includes('input') || lowered.includes('feature')) {
    return 'torch.randn(1, 3, 224, 224, device=device)'
  }
  if (lowered.includes('audio') || lowered.includes('waveform') || lowered.includes('speech')) {
    return 'torch.randn(1, 16000, device=device)'  // 1 second of audio at 16kHz
  }
  if (lowered.includes('spectrogram') || lowered.includes('melspec') || lowered.includes('mfcc')) {
    return 'torch.randn(1, 80, 101, device=device)'  // (n_mels, n_frames)
  }
  if (lowered.includes('pitch') || lowered.includes('f0')) {
    return 'torch.randn(1, 101, device=device)'
  }
  return 'torch.randn(1, 8, device=device)'
}

// Fallback expression for __getitem__ - no device= since it runs in DataLoader worker
function getItemFallbackExpression(field: string, taskType?: string): string {
  const lowered = field.toLowerCase()
  if (lowered.includes('label') || lowered.includes('target') || lowered.includes('class')) {
    return taskType === 'regression'
      ? 'torch.tensor(0.0, dtype=torch.float32)'
      : 'torch.tensor(0, dtype=torch.long)'
  }
  if (lowered.includes('mask')) return 'torch.randint(0, 2, (1, 1, 224, 224), dtype=torch.long)'
  if (lowered.includes('token') || lowered.includes('text') || lowered.includes('ids')) {
    return 'torch.randint(0, 1000, (1, 32), dtype=torch.long)'
  }
  if (lowered.includes('image') || lowered.includes('input') || lowered.includes('feature')) {
    return 'torch.randn(1, 3, 224, 224)'
  }
  if (lowered.includes('audio') || lowered.includes('waveform') || lowered.includes('speech')) {
    return 'torch.randn(16000)'  // 1 second at 16kHz
  }
  if (lowered.includes('spectrogram') || lowered.includes('melspec') || lowered.includes('mfcc')) {
    return 'torch.randn(80, 101)'  // (n_mels, n_frames)
  }
  if (lowered.includes('pitch') || lowered.includes('f0')) {
    return 'torch.randn(101)'
  }
  return 'torch.randn(1, 8)'
}

// ─────────────────────────────────────────────────────────────────
// 数据节点代码生成器
// ─────────────────────────────────────────────────────────────────

interface DataNodeCodeGen {
  initCode: string[]
  getitemCode: string[]
  fieldTracking: Record<string, string>  // logical_field -> var_name
}

function generateDataNodeCode(node: DataFlowNode, _nodeIndex: number): DataNodeCodeGen {
  const nodeType = node.data.nodeType as DataNodeType
  const params = node.data.params || {}
  const result: DataNodeCodeGen = {
    initCode: [],
    getitemCode: [],
    fieldTracking: {},
  }

  switch (nodeType) {
    // === Sources ===
    case 'folder_source': {
      const path = pythonValue(params.path || './data/images')
      const pattern = pythonValue(params.pattern || '*.jpg')
      const recursive = params.recursive === true
      const id = safeToken(node.id)
      if (recursive) {
        result.initCode.push(
          `        self._${id}_paths = sorted(Path(${path}).rglob(${pattern}))`
        )
      } else {
        result.initCode.push(
          `        self._${id}_paths = sorted(Path(${path}).glob(${pattern}))`
        )
      }
      result.initCode.push(
        `        self._${id}_cache = {}`
      )
      result.getitemCode.push(
        `        # Folder Source: ${path} (recursive=${recursive})`,
        `        img_path = self._${id}_paths[index % len(self._${id}_paths)]`,
        `        img_tensor = torchvision.io.read_image(str(img_path)).float() / 255.0`,
        `        _field_image = img_tensor`,
        `        _field_path = str(img_path)`,
        `        _field_filename = img_path.name`
      )
      result.fieldTracking['image'] = '_field_image'
      result.fieldTracking['path'] = '_field_path'
      result.fieldTracking['filename'] = '_field_filename'
      break
    }

    case 'csv_source': {
      const path = pythonValue(params.path || './data/train.csv')
      const delimiter = pythonValue(params.delimiter || ',')
      const id = safeToken(node.id)
      const labelColumn = String(params.label_column ?? 'label').trim() || 'label'
      const featureColumns = String(params.feature_columns ?? '')
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean)
      const featureColumnsExpr = featureColumns.length > 0
        ? `[${featureColumns.map((column) => pythonValue(column)).join(', ')}]`
        : `[col for col in self._${id}_columns if col != self._${id}_label_column and col.lower() not in ('label', 'class', 'target')]`
      result.initCode.push(
        `        self._${id}_df = pd.read_csv(${path}, delimiter=${delimiter})`,
        `        self._${id}_columns = list(self._${id}_df.columns)`,
        `        self._${id}_label_column = ${pythonValue(labelColumn)}`,
        `        self._${id}_feature_columns = ${featureColumnsExpr}`,
        `        self._${id}_cache = {}`
      )
      result.getitemCode.push(
        `        # CSV Source: ${path}`,
        `        row = self._${id}_df.iloc[index % len(self._${id}_df)] if len(self._${id}_df) > 0 else {}`,
        `        _field_data = row.to_dict() if hasattr(row, 'to_dict') else dict(row)`,
        `        _feature_values = []`,
        `        if self._${id}_label_column in row and not pd.isna(row[self._${id}_label_column]):`,
        `            _field_label = int(float(row[self._${id}_label_column]))`,
        `        for col in self._${id}_feature_columns:`,
        `            numeric_val = 0.0`,
        `            val = row[col] if col in row else None`,
        `            if val is None or pd.isna(val):`,
        `                _feature_values.append(numeric_val)`,
        `                continue`,
        `            try:`,
        `                numeric_val = float(val)`,
        `            except:`,
        `                numeric_val = 0.0`,
        `            _feature_values.append(numeric_val)`,
        `            locals()[f'_field_{col}'] = torch.tensor(numeric_val, dtype=torch.float32)`,
        `            _field_names.append(f'_field_{col}')`,
        `        _field_features = torch.tensor(_feature_values, dtype=torch.float32) if _feature_values else torch.zeros(1, dtype=torch.float32)`
      )
      result.fieldTracking['label'] = '_field_label'
      result.fieldTracking['features'] = '_field_features'
      result.fieldTracking['csv_row'] = '_field_data'
      break
    }

    case 'jsonl_source': {
      const path = pythonValue(params.path || './data/train.jsonl')
      const id = safeToken(node.id)
      result.initCode.push(
        `        self._${id}_path = ${path}`,
        `        self._${id}_cache = {}`
      )
      result.getitemCode.push(
        `        # JSONL Source: ${path}`,
        `        with open(self._${id}_path, 'r') as f:`,
        `            lines = f.readlines()`,
        `        record = json.loads(lines[index % len(lines)])`,
        `        _field_record = record`
      )
      result.fieldTracking['record'] = '_field_record'
      break
    }

    case 'parquet_source': {
      const path = pythonValue(params.path || './data/train.parquet')
      const id = safeToken(node.id)
      result.initCode.push(
        `        self._${id}_df = pd.read_parquet(${path})`,
        `        self._${id}_columns = list(self._${id}_df.columns)`,
        `        self._${id}_cache = {}`
      )
      result.getitemCode.push(
        `        # Parquet Source: ${path}`,
        `        row = self._${id}_df.iloc[index % len(self._${id}_df)]`,
        `        for col in self._${id}_columns:`,
        `            val = row[col]`,
        `            if pd.isna(val): continue`,
        `            try:`,
        `                numeric_val = float(val)`,
        `                if col.lower() in ('label', 'class', 'target'):`,
        `                    _field_label = int(numeric_val)`,
        `                else:`,
        `                    locals()[f'_field_{col}'] = torch.tensor(numeric_val) if abs(numeric_val - int(numeric_val)) < 1e-9 else torch.tensor(float(val))`,
        `            except:`,
        `                locals()[f'_field_{col}'] = str(val)`,
        `            _field_names.append(f'_field_{col}')`
      )
      result.fieldTracking['label'] = '_field_label'
      break
    }

    case 'huggingface_source': {
      const datasetName = pythonValue(params.dataset_name || 'mnist')
      const split = pythonValue(params.split || 'train')
      const id = safeToken(node.id)
      result.initCode.push(
        `        from datasets import load_dataset`,
        `        self._${id}_dataset = load_dataset(${datasetName}, split=${split})`
      )
      result.getitemCode.push(
        `        # HuggingFace Source: ${datasetName}`,
        `        item = self._${id}_dataset[index % len(self._${id}_dataset)]`,
        `        for k, v in item.items():`,
        `            locals()[f'_field_{k}'] = v`,
        `            _field_names.append(f'_field_{k}')`
      )
      break
    }

    // === Readers ===
    case 'read_image': {
      const mode = pythonValue(params.mode || 'RGB')
      result.getitemCode.push(
        `        # Read Image (mode=${mode})`,
        `        if '_field_image' in locals():`,
        `            _field_image = torchvision.io.read_image(str(_field_image)).float() / 255.0`,
        `            if ${mode} == 'RGB' and _field_image.shape[0] != 3:`,
        `                _field_image = _field_image.repeat(3, 1, 1)[:3, :, :]`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'read_text': {
      const encoding = pythonValue(params.encoding || 'utf-8')
      result.getitemCode.push(
        `        # Read Text (encoding=${encoding})`,
        `        if '_field_path' in locals():`,
        `            with open(_field_path, 'r', encoding=${encoding}) as f:`,
        `                _field_text = f.read()`,
        `        elif '_field_record' in locals():`,
        `            _field_text = str(_field_record)`
      )
      result.fieldTracking['text'] = '_field_text'
      break
    }

    case 'unpack': {
      const key = pythonValue(params.key || 'payload')
      result.getitemCode.push(
        `        # Unpack: ${key}`,
        `        if '_field_record' in locals():`,
        `            _unpacked = _field_record.get(${key}, {})`,
        `            for k, v in _unpacked.items():`,
        `                locals()[f'_field_{k}'] = v`,
        `                _field_names.append(f'_field_{k}')`
      )
      break
    }

    // === Audio / Speech Processing ===
    case 'read_audio': {
      const sampleRate = Number(params.sample_rate || 16000)
      const mono = params.mono !== false
      result.initCode.push(
        `        import torchaudio`,
        `        import torchaudio.transforms as T`
      )
      result.getitemCode.push(
        `        # Read Audio (sample_rate=${sampleRate}, mono=${mono})`,
        `        if '_field_audio_path' in locals():`,
        `            _waveform, _sr = torchaudio.load(_field_audio_path)`,
        `            if ${sampleRate} != _sr:`,
        `                _resampler = T.Resample(_sr, ${sampleRate})`,
        `                _waveform = _resampler(_waveform)`,
        `            if ${mono} and _waveform.shape[0] > 1:`,
        `                _waveform = _waveform.mean(dim=0, keepdim=True)`,
        `            _field_audio = _waveform.squeeze(0)`,
        `        elif '_field_audio' in locals():`,
        `            pass  # Already a tensor`
      )
      result.fieldTracking['audio'] = '_field_audio'
      break
    }

    case 'stft': {
      const n_fft = Number(params.n_fft || 2048)
      const hop_length = Number(params.hop_length || 512)
      const win_length = Number(params.win_length || n_fft)
      const window = pythonValue(params.window || 'hann')
      const center = params.center !== false
      result.initCode.push(
        `        self._stft_fn_${safeToken(node.id)} = T.Spectrogram(`,
        `            n_fft=${n_fft},`,
        `            hop_length=${hop_length},`,
        `            win_length=${win_length},`,
        `            window_fn=torch.${window === "'hann'" ? 'hann_window' : 'hamming_window'},`,
        `            center=${center}`,
        `        )`
      )
      result.getitemCode.push(
        `        # STFT (n_fft=${n_fft}, hop_length=${hop_length})`,
        `        if '_field_audio' in locals():`,
        `            _field_spectrogram = self._stft_fn_${safeToken(node.id)}(_field_audio.unsqueeze(0)).squeeze(0)`,
        `            _field_spectrogram = torch.stack([_field_spectrogram.real, _field_spectrogram.imag], dim=-1)`
      )
      result.fieldTracking['spectrogram'] = '_field_spectrogram'
      break
    }

    case 'istft': {
      const n_fft = Number(params.n_fft || 2048)
      const hop_length = Number(params.hop_length || 512)
      const win_length = Number(params.win_length || n_fft)
      const window = pythonValue(params.window || 'hann')
      const center = params.center !== false
      result.initCode.push(
        `        self._istft_fn_${safeToken(node.id)} = T.InverseSpectrogram(`,
        `            n_fft=${n_fft},`,
        `            hop_length=${hop_length},`,
        `            win_length=${win_length},`,
        `            window_fn=torch.${window === "'hann'" ? 'hann_window' : 'hamming_window'},`,
        `            center=${center}`,
        `        )`
      )
      result.getitemCode.push(
        `        # iSTFT (n_fft=${n_fft}, hop_length=${hop_length})`,
        `        if '_field_spectrogram' in locals():`,
        `            _spec = _field_spectrogram[..., 0] + 1j * _field_spectrogram[..., 1]`,
        `            _field_audio_reconstructed = self._istft_fn_${safeToken(node.id)}(_spec)`,
        `            _field_audio = _field_audio_reconstructed.squeeze(0)`
      )
      result.fieldTracking['audio'] = '_field_audio'
      break
    }

    case 'spectrogram': {
      const n_fft = Number(params.n_fft || 2048)
      const hop_length = Number(params.hop_length || 512)
      const win_length = Number(params.win_length || n_fft)
      const power = Number(params.power || 2)
      result.initCode.push(
        `        self._spectrogram_fn_${safeToken(node.id)} = T.Spectrogram(`,
        `            n_fft=${n_fft},`,
        `            hop_length=${hop_length},`,
        `            win_length=${win_length},`,
        `            power=${power}`,
        `        )`
      )
      result.getitemCode.push(
        `        # Spectrogram (n_fft=${n_fft}, hop_length=${hop_length}, power=${power})`,
        `        if '_field_audio' in locals():`,
        `            _field_melspec = self._spectrogram_fn_${safeToken(node.id)}(_field_audio.unsqueeze(0)).squeeze(0)`
      )
      result.fieldTracking['spectrogram'] = '_field_melspec'
      break
    }

    case 'melspectrogram': {
      const sample_rate = Number(params.sample_rate || 16000)
      const n_fft = Number(params.n_fft || 2048)
      const hop_length = Number(params.hop_length || 512)
      const n_mels = Number(params.n_mels || 80)
      const f_min = Number(params.f_min || 0)
      const f_max = Number(params.f_max || 8000)
      result.initCode.push(
        `        self._melspec_fn_${safeToken(node.id)} = T.MelScale(`,
        `            n_mels=${n_mels},`,
        `            sample_rate=${sample_rate},`,
        `            f_min=${f_min},`,
        `            f_max=${f_max}`,
        `        )`,
        `        self._spectrogram_fn_${safeToken(node.id)} = T.Spectrogram(`,
        `            n_fft=${n_fft},`,
        `            hop_length=${hop_length},`,
        `            win_length=${n_fft}`,
        `        )`
      )
      result.getitemCode.push(
        `        # Mel Spectrogram (n_fft=${n_fft}, hop_length=${hop_length}, n_mels=${n_mels}, sr=${sample_rate})`,
        `        if '_field_audio' in locals():`,
        `            _spec = self._spectrogram_fn_${safeToken(node.id)}(_field_audio.unsqueeze(0))`,
        `            _field_melspec = self._melspec_fn_${safeToken(node.id)}(_spec).squeeze(0)`
      )
      result.fieldTracking['melspec'] = '_field_melspec'
      break
    }

    case 'mfcc': {
      const sample_rate = Number(params.sample_rate || 16000)
      const n_mfcc = Number(params.n_mfcc || 13)
      const n_fft = Number(params.n_fft || 2048)
      const hop_length = Number(params.hop_length || 512)
      const n_mels = Number(params.n_mels || 40)
      result.initCode.push(
        `        self._mfcc_fn_${safeToken(node.id)} = T.MFCC(`,
        `            sample_rate=${sample_rate},`,
        `            n_mfcc=${n_mfcc},`,
        `            melkwargs={`,
        `                'n_fft': ${n_fft},`,
        `                'hop_length': ${hop_length},`,
        `                'n_mels': ${n_mels},`,
        `            }`,
        `        )`
      )
      result.getitemCode.push(
        `        # MFCC (n_mfcc=${n_mfcc}, n_fft=${n_fft}, hop_length=${hop_length}, sr=${sample_rate})`,
        `        if '_field_audio' in locals():`,
        `            _field_mfcc = self._mfcc_fn_${safeToken(node.id)}(_field_audio.unsqueeze(0)).squeeze(0)`
      )
      result.fieldTracking['mfcc'] = '_field_mfcc'
      break
    }

    case 'gammatone': {
      const sample_rate = Number(params.sample_rate || 16000)
      const n_filters = Number(params.n_filters || 64)
      const f_min = Number(params.f_min || 50)
      const f_max = Number(params.f_max || 8000)
      const num_taps = Number(params.num_taps || 256)
      result.getitemCode.push(
        `        # Gammatone Filterbank (n_filters=${n_filters}, f_min=${f_min}, f_max=${f_max})`,
        `        if '_field_audio' in locals():`,
        `            _gammatone_filterbank = torch.zeros(${n_filters}, _field_audio.shape[-1])`,
        `            for i in range(${n_filters}):`,
        `                _f_center = ${f_min} * (${f_max}/${f_min}) ** (i / ${n_filters})`,
        `                _t = torch.arange(${num_taps}) / ${sample_rate}`,
        `                _gammatone = ${num_taps} ** 0.5 * torch.exp(-2 * torch.pi * 1.0 * _f_center * _t) * torch.pow(_t, 3) * torch.cos(2 * torch.pi * _f_center * _t)`,
        `                _gammatone_filterbank[i, :${num_taps}] = _gammatone`,
        `            _field_gammatone = torch.nn.functional.conv1d(_field_audio.unsqueeze(0), _gammatone_filterbank.unsqueeze(1), padding='same').squeeze(0)`
      )
      result.fieldTracking['gammatone'] = '_field_gammatone'
      break
    }

    case 'gammatone_chroma': {
      const n_chroma = Number(params.n_chroma || 12)
      const n_octaves = Number(params.n_octaves || 7)
      const n_filters = Number(params.n_filters || 64)
      result.getitemCode.push(
        `        # Gammatone Chroma (n_chroma=${n_chroma}, n_octaves=${n_octaves})`,
        `        if '_field_gammatone' in locals():`,
        `            _n_frames = _field_gammatone.shape[-1]`,
        `            _field_gammatone_chroma = torch.zeros(${n_chroma}, _n_frames)`,
        `            _bins_per_chroma = ${n_filters} // ${n_chroma}`,
        `            for i in range(${n_filters}):`,
        `                _chroma_idx = i % ${n_chroma}`,
        `                _field_gammatone_chroma[_chroma_idx] += _field_gammatone[i]`
      )
      result.fieldTracking['gammatone_chroma'] = '_field_gammatone_chroma'
      break
    }

    case 'subband': {
      const n_bands = Number(params.n_bands || 4)
      const mode = pythonValue(params.mode || 'uniform')
      result.getitemCode.push(
        `        # Sub-band Split (n_bands=${n_bands}, mode=${mode})`,
        `        if '_field_audio' in locals():`,
        `            _n_samples = _field_audio.shape[-1]`,
        `            _samples_per_band = _n_samples // ${n_bands}`,
        `            _field_subbands = []`,
        `            for _i in range(${n_bands}):`,
        `                _start = _i * _samples_per_band`,
        `                _end = (_i + 1) * _samples_per_band if _i < ${n_bands} - 1 else _n_samples`,
        `                _field_subbands.append(_field_audio[_start:_end])`,
        `            _field_subbands = torch.stack(_field_subbands)`
      )
      result.fieldTracking['subbands'] = '_field_subbands'
      break
    }

    case 'filterbank': {
      const n_filters = Number(params.n_filters || 80)
      const n_fft = Number(params.n_fft || 2048)
      const sample_rate = Number(params.sample_rate || 16000)
      const f_min = Number(params.f_min || 0)
      const f_max = Number(params.f_max || 8000)
      const filter_type = pythonValue(params.filter_type || 'mel')
      result.initCode.push(
        `        self._fbank_${safeToken(node.id)} = T.MelScale(`,
        `            n_mels=${n_filters},`,
        `            sample_rate=${sample_rate},`,
        `            f_min=${f_min},`,
        `            f_max=${f_max}`,
        `        )`
      )
      result.getitemCode.push(
        `        # Filterbank (${filter_type}, n_filters=${n_filters}, n_fft=${n_fft})`,
        `        if '_field_spectrogram' in locals():`,
        `            _field_filterbank = self._fbank_${safeToken(node.id)}(_field_spectrogram)`
      )
      result.fieldTracking['filterbank'] = '_field_filterbank'
      break
    }

    case 'preemphasis': {
      const coef = Number(params.coef || 0.97)
      result.getitemCode.push(
        `        # Preemphasis (coef=${coef})`,
        `        if '_field_audio' in locals():`,
        `            _field_audio = torch.cat([_field_audio[:1], _field_audio[1:] - ${coef} * _field_audio[:-1]])`
      )
      break
    }

    case 'cmvn': {
      const norm_means = params.norm_means !== false
      const norm_vars = params.norm_vars !== false
      const eps = Number(params.eps || 1e-6)
      result.getitemCode.push(
        `        # CMVN (norm_means=${norm_means}, norm_vars=${norm_vars}, eps=${eps})`,
        `        if '_field_features' in locals():`,
        norm_means ? `            _field_features = _field_features - _field_features.mean(dim=-1, keepdim=True)` : ``,
        norm_vars ? `            _field_features = _field_features / (_field_features.std(dim=-1, keepdim=True) + ${eps})` : ``
      )
      break
    }

    case 'voice_activity_detection': {
      const threshold = Number(params.threshold || 0.5)
      const frame_length = Number(params.frame_length || 2048)
      const hop_length = Number(params.hop_length || 512)
      const energy_type = pythonValue(params.energy_type || 'rms')
      result.getitemCode.push(
        `        # VAD (threshold=${threshold}, frame_length=${frame_length})`,
        `        if '_field_audio' in locals():`,
        `            _n_frames = (_field_audio.shape[-1] - ${frame_length}) // ${hop_length} + 1`,
        `            _vad_mask = torch.zeros(_n_frames, dtype=torch.bool)`,
        `            for _i in range(_n_frames):`,
        `                _frame = _field_audio[_i * ${hop_length}:_i * ${hop_length} + ${frame_length}]`,
        `                _energy = torch.mean(_frame ** 2) if ${energy_type} == 'rms' else torch.mean(torch.abs(_frame))`,
        `                _vad_mask[_i] = _energy > ${threshold}`,
        `            _field_vad = _vad_mask`
      )
      result.fieldTracking['vad'] = '_field_vad'
      break
    }

    case 'pitch_extraction': {
      const sample_rate = Number(params.sample_rate || 16000)
      const frame_length = Number(params.frame_length || 2048)
      const hop_length = Number(params.hop_length || 512)
      const method = pythonValue(params.method || 'praat')
      const min_f0 = Number(params.min_f0 || 50)
      const max_f0 = Number(params.max_f0 || 500)
      result.getitemCode.push(
        `        # Pitch Extraction (method=${method}, min_f0=${min_f0}, max_f0=${max_f0})`,
        `        if '_field_audio' in locals():`,
        `            _n_frames = (_field_audio.shape[-1] - ${frame_length}) // ${hop_length} + 1`,
        `            _field_pitch = torch.zeros(_n_frames)`,
        `            for _i in range(_n_frames):`,
        `                _frame = _field_audio[_i * ${hop_length}:_i * ${hop_length} + ${frame_length}].numpy()`,
        `                # Simplified autocorrelation-based pitch extraction`,
        `                _corr = torch.correlate(torch.tensor(_frame), torch.tensor(_frame), 'full')`,
        `                _corr = _corr[_corr.shape[0] // 2:]`,
        `                _min_lag = int(${sample_rate} / ${max_f0})`,
        `                _max_lag = int(${sample_rate} / ${min_f0})`,
        `                if _max_lag < len(_corr):`,
        `                    _peak_corr, _peak_idx = torch.max(_corr[_min_lag:_max_lag], dim=0)`,
        `                    if _peak_corr > 0:`,
        `                        _field_pitch[_i] = ${sample_rate} / (_peak_idx + _min_lag)`
      )
      result.fieldTracking['pitch'] = '_field_pitch'
      break
    }

    case 'onset_detection': {
      const method = pythonValue(params.method || 'spectral_flux')
      const threshold = Number(params.threshold || 0.5)
      const backtrack = params.backtrack !== false
      result.getitemCode.push(
        `        # Onset Detection (method=${method}, threshold=${threshold}, backtrack=${backtrack})`,
        `        if '_field_spectrogram' in locals():`,
        `            _diff_spec = torch.abs(_field_spectrogram[:, 1:]) - torch.abs(_field_spectrogram[:, :-1])`,
        `            _flux = torch.sum(torch.clamp(_diff_spec, min=0), dim=0)`,
        `            _onset_energy = (_flux > ${threshold}).float()`,
        `            if ${backtrack}:`,
        `                # Backtrack to local minima before onset`,
        `                for _j in range(len(_onset_energy) - 2, 0, -1):`,
        `                    if _onset_energy[_j] == 1 and _onset_energy[_j-1] < _onset_energy[_j]:`,
        `                        _onset_energy[_j] = 0`,
        `            _field_onsets = _onset_energy`
      )
      result.fieldTracking['onsets'] = '_field_onsets'
      break
    }

    // === Tensor Operations ===
    case 'tensor_reshape': {
      const shape = pythonValue(params.shape || '-1')
      result.getitemCode.push(
        `        # Reshape to ${shape}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.reshape(${shape})`
      )
      break
    }

    case 'tensor_flatten': {
      const start_dim = Number(params.start_dim ?? 0)
      result.getitemCode.push(
        `        # Flatten from dim ${start_dim}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.flatten(start_dim=${start_dim})`
      )
      break
    }

    case 'tensor_transpose': {
      const dim0 = Number(params.dim0 || 0)
      const dim1 = Number(params.dim1 || 1)
      result.getitemCode.push(
        `        # Transpose dims ${dim0} and ${dim1}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.transpose(${dim0}, ${dim1})`
      )
      break
    }

    case 'tensor_permute': {
      const dims = String(params.dims || '0,2,1')
      result.getitemCode.push(
        `        # Permute with dims ${dims}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.permute(${dims})`
      )
      break
    }

    case 'tensor_squeeze': {
      const dim = params.dim !== undefined ? String(params.dim) : 'None'
      result.getitemCode.push(
        `        # Squeeze${dim !== 'None' ? ` dim=${dim}` : ''}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.squeeze(${dim})`
      )
      break
    }

    case 'tensor_expand': {
      const shape = pythonValue(params.shape || '-1')
      result.getitemCode.push(
        `        # Expand to ${shape}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor.expand(${shape})`
      )
      break
    }

    case 'tensor_slice': {
      const start = Number(params.start ?? 0)
      const end = String(params.end ?? '-1')
      const step = Number(params.step ?? 1)
      result.getitemCode.push(
        `        # Slice [${start}:${end}:${step}]`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensor = _field_tensor[${start}:${end}:${step}]`
      )
      break
    }

    case 'tensor_stack': {
      const dim = Number(params.dim ?? 0)
      result.getitemCode.push(
        `        # Stack tensors along dim ${dim}`,
        `        if '_field_tensor' in locals():`,
        `            _field_tensors_stack = torch.stack([_field_tensor], dim=${dim})`
      )
      result.fieldTracking['tensor'] = '_field_tensors_stack'
      break
    }

    case 'tensor_cat': {
      const dim = Number(params.dim ?? 0)
      result.getitemCode.push(
        `        # Concatenate tensors along dim ${dim}`,
        `        if '_field_tensor' in locals():`,
        `            pass  # Use tensor_stack for single tensor, tensor_cat needs multiple inputs`
      )
      break
    }

    // === Field Operations ===
    case 'select_fields': {
      const fields = String(params.fields || 'image,label').split(',').map(f => f.trim())
      result.getitemCode.push(
        `        # Select Fields: ${fields.join(', ')}`
      )
      for (const field of fields) {
        result.fieldTracking[field] = `_field_${field}`
      }
      break
    }

    case 'rename_fields': {
      const mapping = String(params.mapping || '').split(',')
      result.getitemCode.push(`        # Rename Fields`)
      for (const pair of mapping) {
        const [from, to] = pair.split(':').map(s => s.trim())
        if (from && to) {
          result.getitemCode.push(
            `        if '_field_${from}' in locals(): _field_${to} = _field_${from}`
          )
          result.fieldTracking[to] = `_field_${to}`
        }
      }
      break
    }

    case 'filter': {
      const field = String(params.field || 'label')
      const operator = String(params.operator || '>=')
      const value = String(params.value || 0)
      result.getitemCode.push(
        `        # Filter: ${field} ${operator} ${value}`
      )
      // Note: Filtering requires dataset-level logic, not per-sample
      result.getitemCode.push(
        `        # (Filtering is applied at dataset initialization level)`
      )
      break
    }

    // === Transforms ===
    case 'resize': {
      const size = String(params.size || '224,224')
      const [w, h] = size.split(',').map(Number)
      result.getitemCode.push(
        `        # Resize: ${size}`,
        `        if '_field_image' in locals():`,
        `            _field_image = torch.nn.functional.interpolate(_field_image.unsqueeze(0), size=(${h}, ${w})).squeeze(0)`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'crop': {
      const x = Number(params.x || 0)
      const y = Number(params.y || 0)
      const width = Number(params.width || 224)
      const height = Number(params.height || 224)
      const cropType = String(params.type || 'center')
      if (cropType === 'center') {
        result.getitemCode.push(
          `        # Crop: center at ${x},${y} size ${width}x${height}`,
          `        if '_field_image' in locals():`,
          `            _field_image = _field_image[:, ${y}:${y + height}, ${x}:${x + width}]`
        )
      } else {
        result.getitemCode.push(
          `        # Random Crop - use random_crop node for training`
        )
      }
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'flip': {
      result.getitemCode.push(
        `        # Flip`,
        `        if '_field_image' in locals():`,
        `            if ${params.horizontal}: _field_image = torch.flip(_field_image, dims=[2])`,
        `            if ${params.vertical}: _field_image = torch.flip(_field_image, dims=[1])`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'normalize': {
      const mean = String(params.mean || '0.5,0.5,0.5').split(',').map(Number)
      const std = String(params.std || '0.5,0.5,0.5').split(',').map(Number)
      result.getitemCode.push(
        `        # Normalize: mean=${mean}, std=${std}`,
        `        if '_field_image' in locals():`,
        `            mean_t = torch.tensor([${mean.join(', ')}]).view(-1, 1, 1)`,
        `            std_t = torch.tensor([${std.join(', ')}]).view(-1, 1, 1)`,
        `            _field_image = (_field_image - mean_t) / std_t`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'to_tensor': {
      result.getitemCode.push(
        `        # To Tensor`,
        `        if '_field_image' in locals() and not torch.is_tensor(_field_image):`,
        `            _field_image = torch.tensor(_field_image).float() / 255.0 if _field_image.max() > 1 else torch.tensor(_field_image).float()`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'scale': {
      const min = Number(params.min || 0)
      const max = Number(params.max || 1)
      result.getitemCode.push(
        `        # Scale: [${min}, ${max}]`,
        `        if '_field_image' in locals():`,
        `            _min, _max = _field_image.min(), _field_image.max()`,
        `            if _max > _min:`,
        `                _field_image = (_field_image - _min) / (_max - _min) * ${max - min} + ${min}`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    // === Augmentation ===
    case 'random_horizontal_flip': {
      const p = Number(params.p || 0.5)
      result.getitemCode.push(
        `        # Random Horizontal Flip (p=${p})`,
        `        if '_field_image' in locals() and torch.rand(1) < ${p}:`,
        `            _field_image = torch.flip(_field_image, dims=[2])`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'random_vertical_flip': {
      const p = Number(params.p || 0.5)
      result.getitemCode.push(
        `        # Random Vertical Flip (p=${p})`,
        `        if '_field_image' in locals() and torch.rand(1) < ${p}:`,
        `            _field_image = torch.flip(_field_image, dims=[1])`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'random_crop': {
      const size = Number(params.size || 224)
      const padding = Number(params.padding || 4)
      result.getitemCode.push(
        `        # Random Crop (size=${size}, padding=${padding})`,
        `        if '_field_image' in locals():`,
        `            c, h, w = _field_image.shape`,
        `            if h > ${size} and w > ${size}:`,
        `                top = torch.randint(0, h - ${size}, (1,)).item()`,
        `                left = torch.randint(0, w - ${size}, (1,)).item()`,
        `                _field_image = _field_image[:, top:top+${size}, left:left+${size}]`,
        `            elif padding > 0:`,
        `                pad = torch.nn.functional.pad(_field_image, [${padding}]*4, mode='reflect')`,
        `                top = torch.randint(0, pad.shape[1] - ${size}, (1,)).item()`,
        `                left = torch.randint(0, pad.shape[2] - ${size}, (1,)).item()`,
        `                _field_image = pad[:, top:top+${size}, left:left+${size}]`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'random_rotation': {
      const degrees = Number(params.degrees || 15)
      result.getitemCode.push(
        `        # Random Rotation (degrees=${degrees})`,
        `        if '_field_image' in locals():`,
        `            angle = torch.rand(1).item() * 2 * ${degrees} - ${degrees}`,
        `            theta = torch.tensor([[torch.cos(torch.tensor(angle * 3.14159/180)), -torch.sin(torch.tensor(angle * 3.14159/180)), 0],`,
        `                                 [torch.sin(torch.tensor(angle * 3.14159/180)), torch.cos(torch.tensor(angle * 3.14159/180)), 0],`,
        `                                 [0, 0, 1]], dtype=torch.float32)`,
        `            grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), _field_image.unsqueeze(0).shape, align_corners=False)`,
        `            _field_image = torch.nn.functional.grid_sample(_field_image.unsqueeze(0), grid, align_corners=False).squeeze(0)`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'color_jitter': {
      const b = Number(params.brightness || 0.2)
      const c = Number(params.contrast || 0.2)
      const s = Number(params.saturation || 0.2)
      const h = Number(params.hue || 0.1)
      result.getitemCode.push(
        `        # Color Jitter (b=${b}, c=${c}, s=${s}, h=${h})`,
        `        if '_field_image' in locals() and _field_image.shape[0] == 3:`,
        `            _field_image = torch.clamp(_field_image * (1 + torch.rand(1) * ${b * 2} - ${b}), 0, 1)`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    case 'grayscale': {
      result.getitemCode.push(
        `        # Grayscale`,
        `        if '_field_image' in locals() and _field_image.shape[0] == 3:`,
        `            _field_image = 0.299 * _field_image[0:1] + 0.587 * _field_image[1:2] + 0.114 * _field_image[2:3]`
      )
      result.fieldTracking['image'] = '_field_image'
      break
    }

    // === Data Organization ===
    case 'train_val_split': {
      const trainRatio = Number(params.train_ratio || 0.8)
      result.getitemCode.push(
        `        # Train/Val Split (train_ratio=${trainRatio}) - handled at DataLoader level`
      )
      break
    }

    case 'shuffle': {
      const enabled = params.enabled !== false
      const seed = Number(params.seed || 42)
      if (enabled) {
        result.getitemCode.push(
          `        # Shuffle (seed=${seed}) - apply at DataLoader level`
        )
      }
      break
    }

    case 'repeat': {
      const times = Number(params.times || 3)
      result.getitemCode.push(
        `        # Repeat (times=${times}) - handled at DataLoader level`
      )
      break
    }

    // === NLP ===
    case 'tokenizer': {
      const tokenizerType = pythonValue(params.tokenizer_type || 'bert')
      const maxLen = Number(params.max_length || 512)
      const padding = pythonValue(params.padding || 'max_length')
      result.getitemCode.push(
        `        # Tokenizer: ${tokenizerType} (max_length=${maxLen}, padding=${padding})`,
        `        if '_field_text' in locals():`,
        `            from transformers import AutoTokenizer`,
        `            tokenizer = AutoTokenizer.from_pretrained(${tokenizerType})`,
        `            tokens = tokenizer(_field_text, max_length=${maxLen}, padding=${padding}, truncation=True, return_tensors='pt')`,
        `            for k, v in tokens.items():`,
        `                locals()[f'_field_{k}'] = v.squeeze(0)`,
        `                _field_names.append(f'_field_{k}')`
      )
      break
    }

    case 'truncate': {
      const maxLen = Number(params.max_length || 512)
      result.getitemCode.push(
        `        # Truncate (max_length=${maxLen})`,
        `        if '_field_input_ids' in locals():`,
        `            _field_input_ids = _field_input_ids[:${maxLen}]`
      )
      break
    }

    // === Batch ===
    case 'batch': {
      const batchSize = Number(params.batch_size || 32)
      const dropLast = params.drop_last === true
      result.getitemCode.push(
        `        # Batch (batch_size=${batchSize}, drop_last=${dropLast}) - batch dimension handled by DataLoader collate`
      )
      // batch node marks that data is ready for batching - no direct field output
      // but signals that output will be batched tensors
      result.fieldTracking['batch_idx'] = 'index'  // track batch position
      break
    }

    case 'collate': {
      const strategy = String(params.strategy || 'default')
      result.getitemCode.push(
        `        # Collate (strategy=${strategy}) - custom collate_fn generated at DataLoader level`
      )
      // Generate collate function based on strategy
      if (strategy === 'default') {
        result.initCode.push(
          `        self._collate_fn_${safeToken(node.id)} = torch.utils.data.default_collate`
        )
      } else if (strategy === 'pad') {
        result.initCode.push(
          `        def _pad_collate_${safeToken(node.id)}(batch):`,
          `            # Pad sequences to same length`,
          `            max_len = max(item['input_ids'].shape[0] for item in batch if 'input_ids' in item)`,
          `            result = []`,
          `            for item in batch:`,
          `                padded = item.copy()`,
          `                for k, v in item.items():`,
          `                    if torch.is_tensor(v) and v.dim() == 1 and v.shape[0] < max_len:`,
          `                        padded[k] = torch.nn.functional.pad(v, (0, max_len - v.shape[0]))`,
          `                result.append(padded)`,
          `            return torch.utils.data.default_collate(result)`,
          `        self._collate_fn_${safeToken(node.id)} = _pad_collate_${safeToken(node.id)}`
        )
      } else if (strategy === 'image') {
        result.initCode.push(
          `        def _image_collate_${safeToken(node.id)}(batch):`,
          `            # Collate with image stacking`,
          `            images = torch.stack([item['image'] for item in batch])`,
          `            labels = torch.tensor([item['label'] for item in batch])`,
          `            return {'image': images, 'label': labels}`,
          `        self._collate_fn_${safeToken(node.id)} = _image_collate_${safeToken(node.id)}`
        )
      }
      break
    }

    case 'dataloader': {
      const batchSize = Number(params.batch_size ?? 32)
      const shuffle = params.shuffle !== false
      const numWorkers = Number(params.num_workers ?? 4)
      const pinMemory = params.pin_memory !== false
      result.getitemCode.push(
        `        # DataLoader (batch_size=${batchSize}, shuffle=${shuffle}, num_workers=${numWorkers}, pin_memory=${pinMemory})`
      )
      // DataLoader node stores config for external DataLoader creation
      result.initCode.push(
        `        self._dataloader_config_${safeToken(node.id)} = {`,
        `            'batch_size': ${batchSize},`,
        `            'shuffle': ${pythonBool(shuffle, true)},`,
        `            'num_workers': ${numWorkers},`,
        `            'pin_memory': ${pythonBool(pinMemory, true)},`,
        `            'drop_last': ${pythonBool(params.drop_last, false)}`,
        `        }`
      )
      // Track dataloader output type
      result.fieldTracking['dataloader'] = '_dataloader'
      break
    }

    // === Output ===
    case 'dataset_output': {
      const fields = String(params.fields || '').split(',').map(f => f.trim()).filter(Boolean)
      result.getitemCode.push(
        `        # Dataset Output: ${fields.join(', ')}`
      )
      for (const field of fields) {
        result.fieldTracking[field] = `_field_${field}`
      }
      break
    }

    case 'cache': {
      const cacheDir = pythonValue(params.cache_dir || './cache')
      result.getitemCode.push(
        `        # Cache to: ${cacheDir}`
      )
      break
    }

    default:
      result.getitemCode.push(`        # Unknown node type: ${nodeType}`)
  }

  return result
}

function formatNodeSummary(node: DataFlowNode, index: number): string {
  const params = Object.entries(node.data.params || {})
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(', ')

  return params
    ? `${index + 1}. ${node.data.label} [${node.data.nodeType}] — ${params}`
    : `${index + 1}. ${node.data.label} [${node.data.nodeType}]`
}

function topologicalSortDataGraph(nodes: DataFlowNode[], edges: DataFlowEdge[]): { sortedNodes: DataFlowNode[]; warnings: string[] } {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]))
  const inDegree = new Map(nodes.map((node) => [node.id, 0]))
  const adjacency = new Map(nodes.map((node) => [node.id, [] as string[]]))

  for (const edge of edges) {
    if (!nodeMap.has(edge.source) || !nodeMap.has(edge.target)) continue
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1)
    adjacency.get(edge.source)?.push(edge.target)
  }

  const queue = nodes
    .filter((node) => (inDegree.get(node.id) ?? 0) === 0)
    .map((node) => node.id)

  const orderedIds: string[] = []
  while (queue.length > 0) {
    const current = queue.shift()!
    orderedIds.push(current)
    for (const next of adjacency.get(current) ?? []) {
      const nextDegree = (inDegree.get(next) ?? 0) - 1
      inDegree.set(next, nextDegree)
      if (nextDegree === 0) queue.push(next)
    }
  }

  if (orderedIds.length === nodes.length) {
    return {
      sortedNodes: orderedIds.map((id) => nodeMap.get(id)!).filter(Boolean),
      warnings: [],
    }
  }

  const orderedSet = new Set(orderedIds)
  return {
    sortedNodes: [...orderedIds.map((id) => nodeMap.get(id)!).filter(Boolean), ...nodes.filter((node) => !orderedSet.has(node.id))],
    warnings: ['数据图存在环或断裂，已退回到“尽量稳定”的节点顺序生成脚手架。'],
  }
}

export function collectDataOutputFields(nodes: DataFlowNode[]): string[] {
  const seen = new Set<string>()
  const fields: string[] = []

  for (const node of nodes) {
    if (node.data.nodeType !== 'dataset_output') continue
    for (const rawField of String(node.data.params.fields || '').split(',')) {
      const field = rawField.trim()
      if (!field || seen.has(field)) continue
      seen.add(field)
      fields.push(field)
    }
  }

  return fields
}

function getSourceLengthExpression(node: DataFlowNode): string | null {
  const id = safeToken(node.id)
  switch (node.data.nodeType) {
    case 'folder_source':
      return `len(self._${id}_paths)`
    case 'csv_source':
    case 'parquet_source':
      return `len(self._${id}_df)`
    case 'huggingface_source':
      return `len(self._${id}_dataset)`
    default:
      return null
  }
}

function getDataLoaderConfig(nodes: DataFlowNode[], trainingConfig?: WorkflowTrainingConfig): string {
  const loaderNode = nodes.find((node) => node.data.nodeType === 'dataloader')
  const params = loaderNode?.data.params ?? {}
  const batchSize = Number(params.batch_size ?? trainingConfig?.runtime?.batchSize ?? 1)
  const shuffle = params.shuffle ?? true
  const numWorkers = Number(params.num_workers ?? trainingConfig?.runtime?.numWorkers ?? 0)
  const pinMemory = params.pin_memory ?? false
  const dropLast = params.drop_last ?? false

  return [
    'DATALOADER_CONFIG = {',
    `    'batch_size': ${batchSize},`,
    `    'shuffle': ${pythonBool(shuffle, true)},`,
    `    'num_workers': ${numWorkers},`,
    `    'pin_memory': ${pythonBool(pinMemory, false)},`,
    `    'drop_last': ${pythonBool(dropLast, false)},`,
    '}',
  ].join('\n')
}

function collectModelInputNames(nodes: FlowHamsterNode[]): string[] {
  return nodes
    .filter((node) => node.data.nodeType === 'input')
    .map((node, index) => {
      const name = typeof node.data.params?.name === 'string' ? String(node.data.params.name).trim() : ''
      return name || `input_${index + 1}`
    })
}

export function compileDataWorkflow(
  modelNodes: FlowHamsterNode[],
  dataNodes: DataFlowNode[],
  dataEdges: DataFlowEdge[],
  bindings: WorkflowBinding[],
  trainingConfig?: WorkflowTrainingConfig
): CompiledDataWorkflow {
  const { sortedNodes, warnings: graphWarnings } = topologicalSortDataGraph(dataNodes, dataEdges)
  const outputFields = collectDataOutputFields(dataNodes)
  const outputFieldSet = new Set(outputFields)
  const modelInputNames = collectModelInputNames(modelNodes)
  const modelInputBindings = bindings
    .filter((binding) => binding.target === 'model_input')
    .map((binding) => ({ targetKey: binding.targetKey, sourceKey: binding.sourceKey }))
  const targetBindingSource = bindings.find((binding) => binding.target === 'training_target')?.sourceKey ?? null
  const lossTargetBindings = bindings
    .filter((binding) => binding.target === 'loss_target')
    .map((binding) => ({ lossNodeId: binding.targetKey, sourceKey: binding.sourceKey }))
  const primaryModelInputKey = modelInputNames[0] ?? modelInputBindings[0]?.targetKey ?? null

  const warnings = [...graphWarnings]
  const validationErrors: ValidationError[] = []

  // ─────────────────────────────────────────────────────────────────
  // 节点参数校验
  // ─────────────────────────────────────────────────────────────────
  for (const node of sortedNodes) {
    const errors = validateNodeParams(node)
    validationErrors.push(...errors)
  }

  // ─────────────────────────────────────────────────────────────────
  // 契约校验
  // ─────────────────────────────────────────────────────────────────
  const contractErrors = validateDataFlowContract(dataNodes, dataEdges, bindings)
  validationErrors.push(...contractErrors)

  for (const inputName of modelInputNames) {
    if (!modelInputBindings.some((binding) => binding.targetKey === inputName)) {
      warnings.push(`模型输入 ${inputName} 尚未绑定数据字段，运行时将回退为随机张量。`)
    }
  }

  for (const binding of bindings) {
    if (binding.sourceGraph === 'data' && binding.sourceKey && !outputFieldSet.has(binding.sourceKey)) {
      warnings.push(`绑定源字段 ${binding.sourceKey} 未在 Dataset Output 中声明。`)
    }
  }

  if (bindings.some((binding) => binding.target === 'training_target') && !targetBindingSource) {
    warnings.push('训练目标绑定缺失，训练脚手架将自动生成占位标签。')
  }

  if (lossTargetBindings.length > 0) {
    for (const lb of lossTargetBindings) {
      if (!outputFieldSet.has(lb.sourceKey)) {
        warnings.push(`Loss target binding 源字段 ${lb.sourceKey} 未在 Dataset Output 中声明。`)
      }
    }
  }

  const summaryLines = sortedNodes.map((node, index) => formatNodeSummary(node, index))
  if (summaryLines.length === 0) summaryLines.push('尚未配置数据图节点，当前导出使用占位 batch。')

  // ─────────────────────────────────────────────────────────────────
  // 生成真正的数据加载代码
  // ─────────────────────────────────────────────────────────────────
  const hasRealDataPipeline = sortedNodes.length > 0

  // 收集所有 init 代码和 getitem 代码
  const allInitCode: string[] = [
    '        # Auto-generated data pipeline initialization'
  ]
  const allGetitemCode: string[] = [
    '        # Auto-generated data pipeline'
  ]

  // 为每个数据节点生成代码
  for (let i = 0; i < sortedNodes.length; i++) {
    const node = sortedNodes[i]
    const nodeCode = generateDataNodeCode(node, i)
    allInitCode.push(...nodeCode.initCode)
    allGetitemCode.push(...nodeCode.getitemCode)
  }

  // 生成输出字段映射
  const outputFieldSetFromNodes = new Set<string>()
  const outputFieldVars: string[] = []
  for (const node of sortedNodes) {
    if (node.data.nodeType === 'dataset_output') {
      const fields = String(node.data.params.fields || '').split(',').map(f => f.trim()).filter(Boolean)
      for (const field of fields) {
        outputFieldSetFromNodes.add(field)
        // 根据字段名推断变量名
        const varName = field === 'image' ? '_field_image'
          : field === 'label' ? '_field_label'
          : field === 'path' ? '_field_path'
          : field === 'text' ? '_field_text'
          : `_field_${field}`
        outputFieldVars.push(`            ${JSON.stringify(field)}: ${varName},`)
      }
    }
  }

  const sampleFields = outputFields.length > 0 ? outputFields : ['image', 'label']
  const sampleLines = sampleFields.map((field) => `        ${JSON.stringify(field)}: ${inferTensorExpression(field, trainingConfig?.taskType)},`)
  const bindingMapLines = modelInputBindings.length > 0
    ? modelInputBindings.map((binding) => `    ${JSON.stringify(binding.targetKey)}: ${JSON.stringify(binding.sourceKey)},`)
    : []

  // 构建真正的 Dataset 类
  // Use outputFieldVars (from dataset_output node) to properly reference field variables set by getitem_code
  const datasetOutputFields = outputFieldVars.length > 0
    ? outputFieldVars.map((line) => {
      const match = line.match(/^\s*"([^"]+)":\s*([^,]+),?\s*$/)
      if (!match) return `${line.trim().replace(/,\s*$/, '')},`
      const [, field, varName] = match
      return `            ${JSON.stringify(field)}: locals().get(${JSON.stringify(varName)}, ${getItemFallbackExpression(field, trainingConfig?.taskType)}),`
    }).join('\n')
    : sampleFields.map((field) => `            ${JSON.stringify(field)}: ${getItemFallbackExpression(field, trainingConfig?.taskType)},`).join('\n')

  const sourceLengthExpressions = sortedNodes
    .map(getSourceLengthExpression)
    .filter((expr): expr is string => Boolean(expr))
  const datasetSizeLine = sourceLengthExpressions.length > 0
    ? `        self._size = max(1, ${sourceLengthExpressions.join(', ')})`
    : '        self._size = 1'
  const dataLoaderConfigBlock = getDataLoaderConfig(sortedNodes, trainingConfig)

  // Collect required imports based on node types present
  const requiredImports = ['import torch']
  const seenNodeTypes = new Set(sortedNodes.map(n => n.data.nodeType as string))
  if (seenNodeTypes.has('folder_source') || seenNodeTypes.has('read_image')) {
    requiredImports.push('import torchvision')
  }
  if (seenNodeTypes.has('csv_source')) {
    requiredImports.push('import pandas as pd')
  }
  if (seenNodeTypes.has('jsonl_source')) {
    requiredImports.push('import json')
  }
  if (seenNodeTypes.has('folder_source')) {
    requiredImports.unshift('from pathlib import Path')
  }
  const importsBlock = requiredImports.join('\n')

  const realDatasetClass = `
class FlowHamsterDataset(torch.utils.data.Dataset):
    def __init__(self):
${allInitCode.join('\n')}
${datasetSizeLine}

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        _field_names = []
${allGetitemCode.join('\n')}

        # 构建输出字典，使用 node 填充的真实变量
        sample = {
${datasetOutputFields}
        }

        return sample
`

  const pythonScaffold = `
${importsBlock}

DATA_PIPELINE_SUMMARY = [
${summaryLines.map((line) => `    ${JSON.stringify(line)},`).join('\n')}
]

BOUND_MODEL_INPUTS = {
${bindingMapLines.join('\n')}
}
BOUND_TRAINING_TARGET = ${targetBindingSource ? JSON.stringify(targetBindingSource) : 'None'}
BOUND_LOSS_TARGETS = {
${lossTargetBindings.map((lb) => `    ${JSON.stringify(lb.lossNodeId)}: ${JSON.stringify(lb.sourceKey)},`).join('\n')}
}
PRIMARY_MODEL_INPUT_KEY = ${primaryModelInputKey ? JSON.stringify(primaryModelInputKey) : 'None'}
${dataLoaderConfigBlock}

${hasRealDataPipeline ? realDatasetClass : `
class FlowHamsterDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._size = 1

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        sample = {
${sampleFields.map((field) => `            ${JSON.stringify(field)}: ${getItemFallbackExpression(field, trainingConfig?.taskType)},`).join('\n')}
        }
        return sample
`}

def build_demo_batch(device):
    try:
        loader = build_flowhamster_dataloader()
        batch = next(iter(loader))
        if isinstance(batch, dict):
            return batch
    except Exception as exc:
        print(f"Data pipeline warning: {exc}")

    return build_fallback_batch(device)

def build_fallback_batch(device):
    return {
${sampleLines.join('\n')}
    }

def build_flowhamster_dataset():
    return FlowHamsterDataset()

def build_flowhamster_dataloader():
    dataset = build_flowhamster_dataset()
    return torch.utils.data.DataLoader(dataset, **DATALOADER_CONFIG)

def _coerce_bound_value(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (int, float)):
        return torch.tensor(value, device=device)
    return None

def resolve_bound_inputs(batch, device):
    model_feed = {}
    for target_key, source_key in BOUND_MODEL_INPUTS.items():
        value = _coerce_bound_value(batch.get(source_key), device)
        if value is not None:
            model_feed[target_key] = value

    target = None
    if BOUND_TRAINING_TARGET is not None:
        target = _coerce_bound_value(batch.get(BOUND_TRAINING_TARGET), device)

    loss_targets = {}
    for loss_node_id, source_key in BOUND_LOSS_TARGETS.items():
        loss_targets[loss_node_id] = _coerce_bound_value(batch.get(source_key), device)

    return model_feed, target, loss_targets

def select_primary_model_input(model_feed, device, batch_size=None):
    if PRIMARY_MODEL_INPUT_KEY and PRIMARY_MODEL_INPUT_KEY in model_feed:
        return model_feed[PRIMARY_MODEL_INPUT_KEY]
    for value in model_feed.values():
        if torch.is_tensor(value):
            return value
    fallback_batch_size = int(batch_size) if batch_size else 1
    return torch.randn(fallback_batch_size, 3, 224, 224, device=device)
`.trim()

  return {
    outputFields,
    modelInputBindings,
    targetBindingSource,
    lossTargetBindings,
    primaryModelInputKey,
    warnings,
    summaryLines,
    pythonScaffold,
    hasWorkflowRuntime: dataNodes.length > 0 || bindings.length > 0,
    validationErrors,
  }
}

export function summarizeBindings(compilation: CompiledDataWorkflow): string[] {
  const lines = compilation.modelInputBindings.map((binding) => `${binding.sourceKey} -> ${binding.targetKey}`)
  if (compilation.targetBindingSource) lines.push(`${compilation.targetBindingSource} -> training_target`)
  return lines
}
