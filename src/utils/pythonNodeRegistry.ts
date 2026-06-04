export interface PythonNodeContext {
  instanceName: string
  opType: string
  fields: Record<string, unknown>
  customClassName?: string
}

function pyBool(value: unknown, fallback: boolean): string {
  if (value === undefined || value === null) return fallback ? 'True' : 'False'
  return value === true || value === 'true' ? 'True' : 'False'
}

export function pyLiteral(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string') return JSON.stringify(value)
  if (value === null || value === undefined) return 'None'
  return String(value)
}

export function safePythonName(value: string): string {
  return value
    .replace(/[^A-Za-z0-9_]/g, '_')
    .replace(/^(\d)/, '_$1')
    .replace(/__+/g, '_')
}

function formatKwargs(params: Record<string, unknown>, omit: string[] = []): string {
  const omitted = new Set(omit)
  return Object.entries(params)
    .filter(([key, value]) => !omitted.has(key) && value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}=${pyLiteral(value)}`)
    .join(', ')
}

type InitFactory = (ctx: PythonNodeContext) => string | null

const MODULE_INIT: Record<string, InitFactory> = {
  conv1d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.Conv1d(in_channels=${fields.in_channels ?? 0}, out_channels=${fields.out_channels ?? 0}, kernel_size=${fields.kernel_size ?? 3}, stride=${fields.stride ?? 1}, padding=${fields.padding ?? 0}, bias=${pyBool(fields.bias, false)})`,
  conv2d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.Conv2d(in_channels=${fields.in_channels ?? 0}, out_channels=${fields.out_channels ?? 0}, kernel_size=${fields.kernel_size ?? 3}, stride=${fields.stride ?? 1}, padding=${fields.padding ?? 0}, bias=${pyBool(fields.bias, false)})`,
  conv3d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.Conv3d(in_channels=${fields.in_channels ?? 0}, out_channels=${fields.out_channels ?? 0}, kernel_size=${fields.kernel_size ?? 3}, stride=${fields.stride ?? 1}, padding=${fields.padding ?? 0}, bias=${pyBool(fields.bias, false)})`,
  linear: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.Linear(in_features=${fields.in_features ?? 0}, out_features=${fields.out_features ?? 0}, bias=${pyBool(fields.bias, true)})`,
  relu: ({ instanceName }) => `self.${instanceName} = nn.ReLU()`,
  gelu: ({ instanceName }) => `self.${instanceName} = nn.GELU()`,
  silu: ({ instanceName }) => `self.${instanceName} = nn.SiLU()`,
  sigmoid: ({ instanceName }) => `self.${instanceName} = nn.Sigmoid()`,
  tanh: ({ instanceName }) => `self.${instanceName} = nn.Tanh()`,
  leakyrelu: ({ instanceName, fields }) => `self.${instanceName} = nn.LeakyReLU(negative_slope=${fields.negative_slope ?? 0.01})`,
  maxpool2d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.MaxPool2d(kernel_size=${fields.kernel_size ?? 2}, stride=${fields.stride ?? 2}, padding=${fields.padding ?? 0})`,
  avgpool2d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.AvgPool2d(kernel_size=${fields.kernel_size ?? 2}, stride=${fields.stride ?? 2}, padding=${fields.padding ?? 0})`,
  adaptiveavgpool2d: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.AdaptiveAvgPool2d(output_size=${fields.output_size ?? 1})`,
  globalavgpool: ({ instanceName }) => `self.${instanceName} = nn.AdaptiveAvgPool2d(1)`,
  batchnorm2d: ({ instanceName, fields }) => `self.${instanceName} = nn.BatchNorm2d(num_features=${fields.num_features ?? 0})`,
  layernorm: ({ instanceName, fields }) => `self.${instanceName} = nn.LayerNorm(normalized_shape=${fields.normalized_shape ?? 0})`,
  groupnorm: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.GroupNorm(num_groups=${fields.num_groups ?? 1}, num_channels=${fields.num_channels ?? 0})`,
  dropout: ({ instanceName, fields }) => `self.${instanceName} = nn.Dropout(p=${fields.p ?? 0.5})`,
  softmax: ({ instanceName, fields }) => `self.${instanceName} = nn.Softmax(dim=${fields.dim ?? -1})`,
  flatten: ({ instanceName, fields }) => `self.${instanceName} = nn.Flatten(start_dim=${fields.start_dim ?? 1})`,
  embedding: ({ instanceName, fields }) =>
    `self.${instanceName} = nn.Embedding(num_embeddings=${fields.num_embeddings ?? 0}, embedding_dim=${fields.embedding_dim ?? 0})`,
  instnorm: ({ instanceName, fields }) => `self.${instanceName} = nn.InstanceNorm2d(num_features=${fields.num_channels ?? 64})`,
  parameter: ({ instanceName, fields }) => `self.${instanceName} = nn.Parameter(torch.zeros(${fields.shape ?? '1'}))`,
  custom: ({ instanceName, fields, customClassName }) => {
    const className = safePythonName(customClassName ?? String(fields.customClassId ?? 'CustomModule'))
    const params = formatKwargs(fields, [
      'nodeType', 'label', 'customClassId', 'internalStructure', 'internalEdges',
      'outputVar', 'isComposite', 'isCustomComposite', 'isExpanded',
      'childNodeIds', 'internalEdgeIds', 'inputs', 'outputs',
      'boundaryEdges', 'originClassId', 'customClassRegistryId',
    ])
    return `self.${instanceName} = ${className}(${params})`
  },
}

export function genPythonNodeInit(ctx: PythonNodeContext): string | null {
  return MODULE_INIT[ctx.opType]?.(ctx) ?? null
}

export function genPythonNodeForward(
  opType: string,
  outputVar: string,
  instanceName: string,
  inputVars: string[],
  fields: Record<string, unknown>
): string | null {
  const first = inputVars[0] ?? 'x'
  switch (opType) {
    case 'input':
      return `${outputVar} = x`
    case 'output':
      return `${outputVar} = ${first}`
    case 'parameter':
      return `${outputVar} = self.${instanceName}`
    case 'add':
      return `${outputVar} = ${inputVars[0] ?? 'x'} + ${inputVars[1] ?? 'x'}`
    case 'mul':
      return `${outputVar} = ${inputVars[0] ?? 'x'} * ${inputVars[1] ?? 'x'}`
    case 'concat':
    case 'cat':
      return `${outputVar} = torch.cat([${inputVars.join(', ')}], dim=${fields.dim ?? 1})`
    case 'stack':
      return `${outputVar} = torch.stack([${inputVars.join(', ')}], dim=${fields.dim ?? 1})`
    case 'reshape':
    case 'view':
      return `${outputVar} = ${first}.reshape(${fields.shape ?? '-1'})`
    case 'transpose':
      return `${outputVar} = ${first}.transpose(${fields.dim0 ?? 0}, ${fields.dim1 ?? 1})`
    case 'permute':
      return `${outputVar} = ${first}.permute(${fields.dims ?? '0,2,1'})`
    case 'squeeze':
      return fields.dim === undefined ? `${outputVar} = ${first}.squeeze()` : `${outputVar} = ${first}.squeeze(dim=${fields.dim})`
    case 'slice': {
      const start = Number(fields.start ?? 0)
      const end = Number(fields.end ?? start + 1)
      const step = Number(fields.step ?? 1)
      if (fields.dim !== undefined && step === 1 && end >= start) {
        return `${outputVar} = ${first}.narrow(dim=${fields.dim}, start=${start}, length=${end - start})`
      }
      return `${outputVar} = ${first}[${start}:${end}:${step}]`
    }
    default:
      return `${outputVar} = self.${instanceName}(${first})`
  }
}
