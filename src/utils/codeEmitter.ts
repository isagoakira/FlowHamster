/**
 * Code Emitter
 *
 * Generates Python code from AST blocks.
 * Handles init, forward, loss, and evaluation code generation.
 */

import { NodeBlock } from './astBuilder'
import { SubModule, getCompositeNodeDef } from './nodeRegistry'
import { getAllCustomClasses } from './customCompositeRegistry'
import { getSourceNodeId } from './astBuilder'
import { safeEvaluate, resolveStringTemplate } from './safeEval'
import { genPythonNodeInit } from './pythonNodeRegistry'

// Feature toggles interface
export interface FeatureToggles {
  tensorPreview?: boolean
  gradientViz?: boolean
  multiOutput?: boolean
  evaluationNodes?: boolean
}

// ─────────────────────────────────────────────
// Template Resolution
// ─────────────────────────────────────────────

function resolveTemplate(value: number | string | boolean, params: Record<string, any>): string {
  if (typeof value === 'string' && value.startsWith('${') && value.endsWith('}')) {
    const expr = value.slice(2, -1)
    if (params[expr] !== undefined) {
      return String(params[expr])
    }
    const result = safeEvaluate(expr, params)
    if (result !== undefined) {
      return String(result)
    }
    return expr
  }
  return String(value)
}

function resolveParams(params: Record<string, any>, defaults: Record<string, any>): Record<string, any> {
  const merged = { ...defaults, ...params }
  const resolved: Record<string, any> = {}
  for (const [key, value] of Object.entries(merged)) {
    resolved[key] = resolveTemplate(value, merged)
  }
  return resolved
}

function pyBool(value: unknown, fallback: boolean): string {
  return value === undefined || value === null
    ? (fallback ? 'True' : 'False')
    : (value === true || value === 'true' ? 'True' : 'False')
}

function pyLiteral(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string') return JSON.stringify(value)
  if (value === null || value === undefined) return 'None'
  return String(value)
}

// ─────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────

function resolveSrc(src: string | null, allBlocks: NodeBlock[]): string {
  if (!src) return 'x'
  const srcId = getSourceNodeId(src)
  return allBlocks.find(b => b.nodeId === srcId)?.outputVar ?? 'x'
}

// ─────────────────────────────────────────────
// SubModule Code Generation
// ─────────────────────────────────────────────

function genSubModuleInit(subModule: SubModule, params: Record<string, any>, prefix: string): string | null {
  const resolvedParams = resolveParams(subModule.params, params)
  const name = `${prefix}_${subModule.id}`

  switch (subModule.type) {
    case 'linear':
      return `self.${name} = nn.Linear(in_features=${resolvedParams.in_features}, out_features=${resolvedParams.out_features}, bias=${pyBool(resolvedParams.bias, false)})`
    case 'conv1d':
      return `self.${name} = nn.Conv1d(in_channels=${resolvedParams.in_channels}, out_channels=${resolvedParams.out_channels}, kernel_size=${resolvedParams.kernel_size}, stride=${resolvedParams.stride || 1}, padding=${resolvedParams.padding || 0}, bias=${pyBool(resolvedParams.bias, false)})`
    case 'conv2d':
      return `self.${name} = nn.Conv2d(in_channels=${resolvedParams.in_channels}, out_channels=${resolvedParams.out_channels}, kernel_size=${resolvedParams.kernel_size}, stride=${resolvedParams.stride || 1}, padding=${resolvedParams.padding || 0}, bias=${pyBool(resolvedParams.bias, false)})`
    case 'conv3d':
      return `self.${name} = nn.Conv3d(in_channels=${resolvedParams.in_channels}, out_channels=${resolvedParams.out_channels}, kernel_size=${resolvedParams.kernel_size}, stride=${resolvedParams.stride || 1}, padding=${resolvedParams.padding || 0}, bias=${pyBool(resolvedParams.bias, false)})`
    case 'layernorm':
      return `self.${name} = nn.LayerNorm(normalized_shape=${resolvedParams.normalized_shape})`
    case 'batchnorm2d':
      return `self.${name} = nn.BatchNorm2d(num_features=${resolvedParams.num_features})`
    case 'groupnorm':
      return `self.${name} = nn.GroupNorm(num_groups=${resolvedParams.num_groups}, num_channels=${resolvedParams.num_channels})`
    case 'dropout':
      return `self.${name} = nn.Dropout(p=${resolvedParams.p || 0.5})`
    case 'relu':
      return `self.${name} = nn.ReLU()`
    case 'gelu':
      return `self.${name} = nn.GELU()`
    case 'silu':
      return `self.${name} = nn.SiLU()`
    case 'sigmoid':
      return `self.${name} = nn.Sigmoid()`
    case 'tanh':
      return `self.${name} = nn.Tanh()`
    case 'leakyrelu':
      return `self.${name} = nn.LeakyReLU(negative_slope=${resolvedParams.negative_slope || 0.01})`
    case 'softmax':
      return `self.${name} = nn.Softmax(dim=${resolvedParams.dim || -1})`
    case 'embedding':
      return `self.${name} = nn.Embedding(num_embeddings=${resolvedParams.num_embeddings}, embedding_dim=${resolvedParams.embedding_dim})`
    case 'maxpool2d':
      return `self.${name} = nn.MaxPool2d(kernel_size=${resolvedParams.kernel_size || 2}, stride=${resolvedParams.stride || 2}, padding=${resolvedParams.padding || 0})`
    case 'avgpool2d':
      return `self.${name} = nn.AvgPool2d(kernel_size=${resolvedParams.kernel_size || 2}, stride=${resolvedParams.stride || 2}, padding=${resolvedParams.padding || 0})`
    case 'adaptiveavgpool2d':
      return `self.${name} = nn.AdaptiveAvgPool2d(output_size=${resolvedParams.output_size || 1})`
    case 'globalavgpool':
      return `self.${name} = nn.AdaptiveAvgPool2d(1)`
    // Attention and special ops don't create modules in init
    case 'q_proj':
    case 'k_proj':
    case 'v_proj':
    case 'out_proj':
    case 'attn_score':
    case 'attn_weight':
    case 'attn_apply':
    case 'residual_add':
    case 'split_heads':
    case 'merge_heads':
    case 'transpose':
    case 'view':
    case 'reshape':
    case 'cat':
    case 'stack':
    case 'add':
    case 'mul':
    case 'matmul':
    case 'div':
    case 'sqrt':
    case 'constant':
    case 'parameter':
      return null
    default:
      return null
  }
}

function supportsMultiLayer(opType: string): boolean {
  return ['transformerencoder', 'transformerdecoder'].includes(opType)
}

export function shouldEmitCompositeAsClass(opType: string): boolean {
  return ['transformerencoder', 'transformerdecoder'].includes(opType)
}

export function genReusableCompositeClasses(blocks: NodeBlock[]): string[] {
  const classDefs: string[] = []
  const opTypes = new Set(blocks.map((block) => block.opType))

  if (opTypes.has('transformerencoder')) {
    classDefs.push(`class FlowHamsterTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class FlowHamsterTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FlowHamsterTransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x`)
  }

  if (opTypes.has('transformerdecoder')) {
    classDefs.push(`class FlowHamsterTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)`)
  }

  return classDefs
}

// ─────────────────────────────────────────────
// Composite Node Init Generation
// ─────────────────────────────────────────────

export function genCompositeInit(block: NodeBlock): string[] {
  const compositeDef = getCompositeNodeDef(block.opType)
  if (!compositeDef || !compositeDef.internalStructure) {
    return []
  }

  const lines: string[] = []
  const prefix = block.instanceName
  const opType = block.opType

  // Calculate head_dim for multi-head attention
  const embed_dim = Number(block.fields.embed_dim || block.fields.query_dim || 512)
  const num_heads = Number(block.fields.num_heads || 8)
  const head_dim = Math.floor(embed_dim / num_heads)
  const num_layers = Number(block.fields.num_layers || 1)

  // Add head_dim and num_heads as instance variables
  lines.push(`        self.${prefix}_head_dim = ${head_dim}`)
  lines.push(`        self.${prefix}_num_heads = ${num_heads}`)
  lines.push(`        self.${prefix}_scale = ${head_dim} ** -0.5`)
  lines.push(`        self.${prefix}_num_layers = ${num_layers}`)

  if (supportsMultiLayer(opType) && num_layers > 1) {
    // Multi-layer stacking
    for (let layer = 0; layer < num_layers; layer++) {
      const layerPrefix = `${prefix}_layer${layer}`
      lines.push(`        # === Layer ${layer} ===`)
      for (const subModule of compositeDef.internalStructure) {
        const initLine = genSubModuleInit(subModule, block.fields, layerPrefix)
        if (initLine) {
          lines.push(`        ${initLine}`)
        }
      }
    }
  } else {
    // Single layer
    for (const subModule of compositeDef.internalStructure) {
      const initLine = genSubModuleInit(subModule, block.fields, prefix)
      if (initLine) {
        lines.push(`        ${initLine}`)
      }
    }
  }

  return lines
}

// ─────────────────────────────────────────────
// Topological Sort for SubModules
// ─────────────────────────────────────────────

function topologicalSortSubModules(
  subModules: SubModule[],
  edges: { from: string; to: string; toPort?: string }[]
): SubModule[] {
  const inDegree: Map<string, number> = new Map()
  const adj: Map<string, string[]> = new Map()

  for (const sm of subModules) {
    inDegree.set(sm.id, 0)
    adj.set(sm.id, [])
  }

  for (const edge of edges) {
    adj.get(edge.from)?.push(edge.to)
    inDegree.set(edge.to, (inDegree.get(edge.to) || 0) + 1)
  }

  const queue: string[] = []
  for (const [id, degree] of inDegree) {
    if (degree === 0) queue.push(id)
  }

  const sorted: SubModule[] = []
  while (queue.length > 0) {
    const current = queue.shift()!
    const sm = subModules.find(s => s.id === current)
    if (sm) sorted.push(sm)
    for (const next of adj.get(current) || []) {
      const newDegree = (inDegree.get(next) || 0) - 1
      inDegree.set(next, newDegree)
      if (newDegree === 0) queue.push(next)
    }
  }

  return sorted
}

// ─────────────────────────────────────────────
// Single Layer Forward Generation
// ─────────────────────────────────────────────

function genSingleLayerForward(
  block: NodeBlock,
  layerPrefix: string,
  inputVar: string,
  subModules: SubModule[],
  edges: { from: string; fromPort?: string; to: string; toPort?: string }[],
  resolvedParams: Record<string, string | number>
): string[] {
  const lines: string[] = []
  const prefix = block.instanceName

  // Variable name mapping: subModuleId -> variableName
  const varMap: Map<string, string> = new Map()

  // Build reverse adjacency (to -> [froms])
  const revAdj: Map<string, { from: string; fromPort?: string; toPort?: string }[]> = new Map()
  for (const edge of edges) {
    if (!revAdj.has(edge.to)) revAdj.set(edge.to, [])
    revAdj.get(edge.to)!.push({ from: edge.from, fromPort: edge.fromPort, toPort: edge.toPort })
  }

  // Topological sort
  const sorted = topologicalSortSubModules(subModules, edges)

  // Track residual connections
  const residualConnections = new Set<string>()
  for (const sm of subModules) {
    if (sm.type === 'residual_add' || sm.type === 'add') {
      residualConnections.add(sm.id)
    }
  }

  // Process each sorted submodule
  for (const sm of sorted) {
    const outVar = `${layerPrefix}_${sm.id}_out`
    varMap.set(sm.id, outVar)

    const incoming = revAdj.get(sm.id) || []

    switch (sm.type) {
      case 'linear':
      case 'conv1d':
      case 'conv2d':
      case 'conv3d':
      case 'layernorm':
      case 'batchnorm2d':
      case 'groupnorm':
      case 'dropout':
      case 'relu':
      case 'gelu':
      case 'silu':
      case 'sigmoid':
      case 'tanh':
      case 'leakyrelu':
      case 'softmax':
      case 'embedding':
      case 'maxpool2d':
      case 'avgpool2d':
      case 'adaptiveavgpool2d':
      case 'globalavgpool':
      case 'q_proj':
      case 'k_proj':
      case 'v_proj':
      case 'out_proj': {
        let inVar: string | undefined
        if (incoming.length > 0) {
          inVar = varMap.get(incoming[0].from)
        } else if (sm.sourceInput) {
          inVar = sm.sourceInput === 'x' ? inputVar : sm.sourceInput
        }
        if (inVar) {
          lines.push(`${outVar} = self.${layerPrefix}_${sm.id}(${inVar})`)
        }
        break
      }
      case 'attn_score': {
        const qEdge = incoming.find(e => e.toPort === 'q')
        const kEdge = incoming.find(e => e.toPort === 'k')
        if (qEdge && kEdge) {
          const qVar = varMap.get(qEdge.from)
          const kVar = varMap.get(kEdge.from)
          if (qVar && kVar) {
            lines.push(`${outVar} = torch.matmul(${qVar}, ${kVar}.transpose(-2, -1)) * self.${prefix}_scale`)
          }
        }
        break
      }
      case 'attn_weight': {
        if (incoming.length > 0) {
          const scoreVar = varMap.get(incoming[0].from)
          if (scoreVar) {
            lines.push(`${outVar} = F.softmax(${scoreVar}, dim=-1)`)
          }
        }
        break
      }
      case 'attn_apply': {
        const weightEdge = incoming.find(e => e.toPort === 'weight')
        const vEdge = incoming.find(e => e.toPort === 'v')
        if (weightEdge && vEdge) {
          const weightVar = varMap.get(weightEdge.from)
          const vVar = varMap.get(vEdge.from)
          if (weightVar && vVar) {
            lines.push(`${outVar} = torch.matmul(${weightVar}, ${vVar})`)
          }
        }
        break
      }
      case 'residual_add': {
        const xEdge = incoming.find(e => e.toPort === 'x')
        const sublayerEdge = incoming.find(e => e.toPort === 'sublayer')
        const sublayerVar = sublayerEdge ? varMap.get(sublayerEdge.from) : null
        if (!sublayerVar) break

        let xVar: string
        if (xEdge) {
          xVar = varMap.get(xEdge.from) || inputVar
        } else {
          xVar = inputVar
        }
        lines.push(`${outVar} = ${xVar} + ${sublayerVar}`)
        break
      }
      case 'add': {
        if (incoming.length >= 2) {
          const aVar = varMap.get(incoming[0].from)
          const bVar = varMap.get(incoming[1].from)
          if (aVar && bVar) {
            lines.push(`${outVar} = ${aVar} + ${bVar}`)
          }
        }
        break
      }
      case 'transpose': {
        if (incoming.length > 0) {
          const inV = varMap.get(incoming[0].from)
          const dim0 = sm.params.dim0 ?? 1
          const dim1 = sm.params.dim1 ?? 2
          if (inV) {
            lines.push(`${outVar} = ${inV}.transpose(${dim0}, ${dim1})`)
          }
        }
        break
      }
      case 'view':
      case 'reshape': {
        if (incoming.length > 0) {
          const inV = varMap.get(incoming[0].from)
          const shapeTemplate = sm.params.shape || '(-1,)'
          const resolvedShape = resolveStringTemplate(shapeTemplate as string, resolvedParams)
          if (inV) {
            if (resolvedShape === `(-1, ${resolvedParams.num_heads}, ${resolvedParams.head_dim})`) {
              lines.push(`B, N = ${inV}.shape[:2]`)
              lines.push(`${outVar} = ${inV}.reshape(B, N, ${resolvedParams.num_heads}, ${resolvedParams.head_dim})`)
            } else if (resolvedShape === `(-1, ${resolvedParams.embed_dim})`) {
              lines.push(`B, N = ${inV}.shape[:2]`)
              lines.push(`${outVar} = ${inV}.permute(0, 2, 1, 3).reshape(B, N, ${resolvedParams.embed_dim})`)
            } else {
              lines.push(`${outVar} = ${inV}.reshape(${resolvedShape})`)
            }
          }
        }
        break
      }
      case 'matmul': {
        if (incoming.length >= 2) {
          const aVar = varMap.get(incoming[0].from)
          const bVar = varMap.get(incoming[1].from)
          if (aVar && bVar) {
            lines.push(`${outVar} = torch.matmul(${aVar}, ${bVar})`)
          }
        }
        break
      }
      case 'mul': {
        if (incoming.length >= 2) {
          const aVar = varMap.get(incoming[0].from)
          const bVar = varMap.get(incoming[1].from)
          if (aVar && bVar) {
            lines.push(`${outVar} = ${aVar} * ${bVar}`)
          }
        }
        break
      }
      case 'cat':
      case 'stack': {
        const inputVars = incoming.map(e => varMap.get(e.from)).filter(Boolean)
        const dim = sm.params.dim ?? 1
        if (inputVars.length >= 2) {
          const op = sm.type === 'cat' ? 'torch.cat' : 'torch.stack'
          lines.push(`${outVar} = ${op}([${inputVars.join(', ')}], dim=${dim})`)
        }
        break
      }
      default: {
        if (incoming.length > 0 && !residualConnections.has(sm.id)) {
          const inV = varMap.get(incoming[0].from)
          if (inV) {
            lines.push(`${outVar} = ${inV}`)
          }
        }
        break
      }
    }
  }

  return lines
}

// ─────────────────────────────────────────────
// Composite Forward Generation
// ─────────────────────────────────────────────

export function genCompositeForward(block: NodeBlock): string | null {
  const compositeDef = getCompositeNodeDef(block.opType)
  if (!compositeDef || !compositeDef.internalStructure || !compositeDef.internalEdges) {
    return null
  }

  const prefix = block.instanceName
  const subModules = compositeDef.internalStructure
  const edges = compositeDef.internalEdges

  const embed_dim = Number(block.fields.embed_dim || block.fields.query_dim || 512)
  const num_heads = Number(block.fields.num_heads || 8)
  const head_dim = Math.floor(embed_dim / num_heads)
  const num_layers = Number(block.fields.num_layers || 1)

  const resolvedParams: Record<string, string | number> = {
    embed_dim,
    num_heads,
    head_dim,
    ...block.fields,
  }

  if (supportsMultiLayer(block.opType) && num_layers > 1) {
    // Multi-layer: generate loop with dynamic attribute access
    const lines: string[] = []
    lines.push(`        # === Multi-layer ${block.opType} (${num_layers} layers) ===`)
    lines.push(`        x = src  # Initialize with external input`)
    lines.push(`        for layer_idx in range(self.${prefix}_num_layers):`)
    lines.push(`            layer_prefix = f"layer{layer_idx}"`)
    lines.push(`            # Layer input: x (first layer = external input, subsequent layers = previous layer output)`)

    const layerLines = genSingleLayerForward(block, 'layer', 'x', subModules, edges, resolvedParams)
    const moduleIdPattern = subModules.map(s => s.id).join('|')
    const regex = new RegExp(`self\\.layer_(${moduleIdPattern})`, 'g')

    for (const line of layerLines) {
      const replacedLine = line.replace(regex, (_match, moduleId) => `getattr(self, f"${prefix}_" + layer_prefix + "_${moduleId}")`)
      lines.push(`            ${replacedLine}`)
    }

    lines.push(`            x = layer_${compositeDef.outputVar}_out  # Update x for next layer`)
    lines.push(`        ${block.outputVar} = x`)

    return `        # === Composite ${block.opType} forward ===\n${lines.join('\n')}`
  } else {
    const layerLines = genSingleLayerForward(block, prefix, 'x', subModules, edges, resolvedParams)
    const outputVar = `${prefix}_${compositeDef.outputVar}_out`
    if (layerLines.length === 0) return null
    return `        # === Composite ${block.opType} forward ===\n${layerLines.map(l => `        ${l}`).join('\n')}\n        ${block.outputVar} = ${outputVar}`
  }
}

// ─────────────────────────────────────────────
// Init Generation
// ─────────────────────────────────────────────

export function genInit(block: NodeBlock): string | null {
  const registryInit = genPythonNodeInit({
    instanceName: block.instanceName,
    opType: block.opType,
    fields: block.fields,
  })
  if (registryInit) {
    return registryInit
  }

  if (block.category !== 'module') return null

  const f = block.fields
  const name = block.instanceName

  switch (block.opType) {
    case 'conv1d': return `self.${name} = nn.Conv1d(in_channels=${f.in_channels ?? 0}, out_channels=${f.out_channels ?? 0}, kernel_size=${f.kernel_size ?? 3}, stride=${f.stride ?? 1}, padding=${f.padding ?? 0}, bias=${pyBool(f.bias, false)})`
    case 'conv2d': return `self.${name} = nn.Conv2d(in_channels=${f.in_channels ?? 0}, out_channels=${f.out_channels ?? 0}, kernel_size=${f.kernel_size ?? 3}, stride=${f.stride ?? 1}, padding=${f.padding ?? 0}, bias=${pyBool(f.bias, false)})`
    case 'conv3d': return `self.${name} = nn.Conv3d(in_channels=${f.in_channels ?? 0}, out_channels=${f.out_channels ?? 0}, kernel_size=${f.kernel_size ?? 3}, stride=${f.stride ?? 1}, padding=${f.padding ?? 0}, bias=${pyBool(f.bias, false)})`
    case 'linear': return `self.${name} = nn.Linear(in_features=${f.in_features ?? 0}, out_features=${f.out_features ?? 0}, bias=${pyBool(f.bias, true)})`
    case 'relu': return `self.${name} = nn.ReLU()`
    case 'gelu': return `self.${name} = nn.GELU()`
    case 'silu': return `self.${name} = nn.SiLU()`
    case 'sigmoid': return `self.${name} = nn.Sigmoid()`
    case 'tanh': return `self.${name} = nn.Tanh()`
    case 'leakyrelu': return `self.${name} = nn.LeakyReLU(negative_slope=${f.negative_slope ?? 0.01})`
    case 'maxpool2d': return `self.${name} = nn.MaxPool2d(kernel_size=${f.kernel_size ?? 2}, stride=${f.stride ?? 2}, padding=${f.padding ?? 0})`
    case 'avgpool2d': return `self.${name} = nn.AvgPool2d(kernel_size=${f.kernel_size ?? 2}, stride=${f.stride ?? 2}, padding=${f.padding ?? 0})`
    case 'adaptiveavgpool2d': return `self.${name} = nn.AdaptiveAvgPool2d(output_size=${f.output_size ?? 1})`
    case 'globalavgpool': return `self.${name} = nn.AdaptiveAvgPool2d(1)`
    case 'batchnorm2d': return `self.${name} = nn.BatchNorm2d(num_features=${f.num_features ?? 0})`
    case 'layernorm': return `self.${name} = nn.LayerNorm(normalized_shape=${f.normalized_shape ?? 'C'})`
    case 'groupnorm': return `self.${name} = nn.GroupNorm(num_groups=${f.num_groups ?? 1}, num_channels=${f.num_channels ?? 0})`
    case 'dropout': return `self.${name} = nn.Dropout(p=${f.p ?? 0.5}, inplace=True)`
    case 'softmax': return `self.${name} = nn.Softmax(dim=${f.dim ?? -1})`
    case 'flatten': return `self.${name} = nn.Flatten(start_dim=${f.start_dim ?? 1})`
    case 'embedding': return `self.${name} = nn.Embedding(num_embeddings=${f.num_embeddings ?? 0}, embedding_dim=${f.embedding_dim ?? 0})`
    case 'selfattention':
    case 'crossattention':
    case 'multiheadattention': {
      const compositeDef = getCompositeNodeDef(block.opType)
      if (compositeDef && compositeDef.internalStructure && compositeDef.internalStructure.length > 0) {
        return null // Handled by genCompositeInit
      }
      if (block.opType === 'crossattention') {
        return `self.${name} = nn.CrossAttention(embed_dim=${f.query_dim ?? 512}, num_heads=${f.num_heads ?? 8})`
      }
      return `self.${name} = nn.MultiheadAttention(embed_dim=${f.embed_dim ?? 512}, num_heads=${f.num_heads ?? 8}, dropout=${f.dropout ?? 0})`
    }
    case 'ffn': return `self.${name} = nn.Sequential(nn.Linear(${f.dim ?? 512}, ${f.hidden_dim ?? 2048}), nn.GELU(), nn.Dropout(${f.dropout ?? 0}), nn.Linear(${f.hidden_dim ?? 2048}, ${f.dim ?? 512}))`
    case 'mlp': return `self.${name} = nn.Sequential(nn.Linear(${f.in_features ?? 784}, ${f.hidden_features ?? 256}), nn.ReLU(), nn.Linear(${f.hidden_features ?? 256}, ${f.out_features ?? 10}))`
    case 'transformerencoder':
    case 'transformerdecoder': {
      const d = f.embed_dim ?? f.d_model ?? 512
      const nh = f.num_heads ?? f.nhead ?? 8
      const dl = f.num_layers ?? 6
      const dimFf = f.dim_feedforward ?? 2048
      const dropout = f.dropout ?? 0.1
      if (block.opType === 'transformerdecoder') {
        return `self.${name} = FlowHamsterTransformerDecoder(embed_dim=${d}, num_heads=${nh}, num_layers=${dl}, dim_feedforward=${dimFf}, dropout=${dropout})`
      }
      return `self.${name} = FlowHamsterTransformerEncoder(embed_dim=${d}, num_heads=${nh}, num_layers=${dl}, dim_feedforward=${dimFf}, dropout=${dropout})`
    }
    case 'mamba': {
      const compositeDef = getCompositeNodeDef(block.opType)
      if (compositeDef && compositeDef.internalStructure && compositeDef.internalStructure.length > 0) {
        return null
      }
      const d = f.d_model ?? 512
      const dState = f.d_state ?? 16
      const dConv = f.d_conv ?? 4
      const expand = f.expand ?? 2
      const dtRank = (f.dt_rank === 0 || f.dt_rank == null) ? 'auto' : String(f.dt_rank)
      const nLayers = f.n_layers ?? 1
      const dropout = f.dropout ?? 0.0
      return `self.${name} = Mamba(d_model=${d}, d_state=${dState}, d_conv=${dConv}, expand=${expand}, dt_rank="${dtRank}", dropout=${dropout}, n_layers=${nLayers})`
    }
    case 'lstm': {
      const inp = f.input_size ?? 512
      const hid = f.hidden_size ?? 512
      const lay = f.num_layers ?? 2
      return `self.${name} = nn.LSTM(input_size=${inp}, hidden_size=${hid}, num_layers=${lay}, batch_first=True)`
    }
    case 'instnorm': return `self.${name} = nn.InstanceNorm2d(num_features=${f.num_channels ?? 64})`
    default: {
      if (block.opType.startsWith('custom_') || block.opType === 'custom') {
        const customClasses = getAllCustomClasses()
        // 对于 'custom' 类型，使用 fields.customClassId（如果有）
        // 对于 'custom_xxx' 类型，block.opType 就是 customClassId
        const customClassId = block.opType === 'custom'
          ? (block.fields.customClassId || block.nodeId)
          : block.opType
        const customClass = customClasses.find(c =>
          c.id === customClassId ||
          c.name === customClassId ||
          `custom_${c.name}` === customClassId
        )
        const className = customClass?.name ?? block.nodeId ?? 'CustomModule'
        const params = Object.entries(f)
          .filter(([k]) => !['nodeType', 'label', 'customClassId', 'internalStructure', 'internalEdges'].includes(k))
          .map(([k, v]) => `${k}=${pyLiteral(v)}`)
          .join(', ')
        return `self.${name} = ${className}(${params})`
      }
      return null
    }
  }
}

// ─────────────────────────────────────────────
// Forward Generation
// ─────────────────────────────────────────────

export function genForward(block: NodeBlock, allBlocks: NodeBlock[]): string | null {
  const { opType, category, inputs, outputVar, fields } = block

  if (opType === 'input') {
    return `        ${outputVar} = x`
  }

  if (opType === 'output') {
    const upstreamRefs = Object.values(inputs).filter((v): v is string => typeof v === 'string' && v !== null)
    const upstreamVar = resolveSrc(upstreamRefs[0] ?? null, allBlocks)
    return `        ${outputVar} = ${upstreamVar}`
  }

  if (category === 'operation') {
    if (opType === 'add') {
      return `        ${outputVar} = ${resolveSrc((inputs['a'] ?? null) as string | null, allBlocks)} + ${resolveSrc((inputs['b'] ?? null) as string | null, allBlocks)}`
    }
    if (opType === 'mul') {
      return `        ${outputVar} = ${resolveSrc((inputs['a'] ?? null) as string | null, allBlocks)} * ${resolveSrc((inputs['b'] ?? null) as string | null, allBlocks)}`
    }
    if (opType === 'concat') {
      const srcSet = new Set<string>()
      for (const srcList of Object.values((inputs as unknown) as Record<string, string[]>)) {
        if (Array.isArray(srcList)) {
          for (const r of srcList) srcSet.add(resolveSrc(r, allBlocks))
        }
      }
      const srcs = Array.from(srcSet)
      if (srcs.length >= 2) {
        return "        " + outputVar + " = torch.cat([" + srcs.join(', ') + "], dim=" + (fields.dim ?? 1) + ")"
      }
      if (srcs.length === 1) {
        return "        " + outputVar + " = " + srcs[0]
      }
      return null
    }
    if (opType === '__iconcat__') {
      const srcs: string[] = []
      for (const srcList of Object.values((inputs as unknown) as Record<string, string[]>)) {
        if (Array.isArray(srcList)) {
          for (const r of srcList) srcs.push(resolveSrc(r, allBlocks))
        }
      }
      const mode = (fields as any).mergeMode || 'concat'
      if (srcs.length >= 2) {
        if (mode === 'add') return "        " + outputVar + " = " + srcs[0] + " + " + srcs[1]
        if (mode === 'mul') return "        " + outputVar + " = " + srcs[0] + " * " + srcs[1]
        if (mode === 'stack') return "        " + outputVar + " = torch.stack([" + srcs.join(', ') + "], dim=" + ((fields as any).dim ?? 1) + ")"
        return "        " + outputVar + " = torch.cat([" + srcs.join(', ') + "], dim=" + ((fields as any).dim ?? 1) + ")"
      }
      if (srcs.length === 1) return "        " + outputVar + " = " + srcs[0]
      return null
    }
    if (opType === 'reshape') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      return `        ${outputVar} = ${up}.view(${fields.shape ?? -1})`
    }
    if (opType === 'transpose') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      return `        ${outputVar} = ${up}.transpose(${fields.dim0 ?? 0}, ${fields.dim1 ?? 1})`
    }
    if (opType === 'split') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      return `        ${outputVar} = ${up}.split(${fields.split_size ?? 32}, dim=${fields.dim ?? 0})`
    }
    if (opType === 'slice') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      const s = fields.start ?? 0, e = fields.end ?? -1, st = fields.step ?? 1
      if (fields.dim !== undefined && st === 1 && Number(e) >= Number(s)) {
        return `        ${outputVar} = ${up}.narrow(dim=${fields.dim}, start=${s}, length=${Number(e) - Number(s)})`
      }
      return `        ${outputVar} = ${up}[${s}:${e}:${st}]`
    }
    if (opType === 'permute') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      const dimsStr = (fields.dims as string) || '0,2,1'
      return `        ${outputVar} = ${up}.permute(${dimsStr})`
    }
    if (opType === 'squeeze') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      const dim = fields.dim != null && fields.dim !== '' ? String(fields.dim) : ''
      return `        ${outputVar} = ${up}.squeeze(${dim})`
    }
    if (opType === 'expand') {
      const up = resolveSrc(Object.values(inputs).find((v): v is string => typeof v === 'string' && v !== null) ?? null, allBlocks)
      const shape = (fields.shape as string) || '-1'
      return `        ${outputVar} = ${up}.expand(${shape})`
    }
    if (opType === 'constant') {
      const shape = (fields.shape as string) || '1'
      const trainable = fields.trainable === true
      if (trainable) {
        return `        ${outputVar} = nn.Parameter(torch.zeros(${shape}))`
      }
      return `        ${outputVar} = torch.zeros(${shape})`
    }
    if (opType === 'parameter') {
      return `        ${outputVar} = self.${block.instanceName}`
    }
    return null
  }

  if (category === 'module') {
    const upstreamRefs = Object.values(inputs).filter((v): v is string => v !== null)
    const upVar = resolveSrc(upstreamRefs[0] ?? null, allBlocks)
    if (opType === 'reshape') return `        ${outputVar} = ${upVar}.view(${fields.shape ?? -1})`
    if (opType === 'transpose') return `        ${outputVar} = ${upVar}.transpose(${fields.dim0 ?? 0}, ${fields.dim1 ?? 1})`
    if (opType === 'permute') {
      const dimsStr = (fields.dims as string) || '0,2,1'
      return `        ${outputVar} = ${upVar}.permute(${dimsStr})`
    }
    if (opType === 'squeeze') {
      const dim = fields.dim != null && fields.dim !== '' ? String(fields.dim) : ''
      return `        ${outputVar} = ${upVar}.squeeze(${dim})`
    }
    if (opType === 'flatten') return `        ${outputVar} = ${upVar}.flatten(start_dim=${fields.start_dim ?? 1})`
    if (opType === 'transformerencoder') return `        ${outputVar} = self.${block.instanceName}(${upVar})`
    if (opType === 'transformerdecoder') {
      const tgtVar = resolveSrc((inputs['tgt'] ?? null) as string | null, allBlocks) || upVar
      const memVar = resolveSrc((inputs['memory'] ?? null) as string | null, allBlocks) || tgtVar
      return `        ${outputVar} = self.${block.instanceName}(${tgtVar}, ${memVar})`
    }
    if (opType === 'lstm') return `        ${outputVar}, _ = self.${block.instanceName}(${upVar})`
    if (opType === 'mamba') return `        ${outputVar} = self.${block.instanceName}(${upVar})`
    if (opType.startsWith('custom_')) return `        ${outputVar} = self.${block.instanceName}(${upVar})`
    return `        ${outputVar} = self.${block.instanceName}(${upVar})`
  }

  return null
}

// ─────────────────────────────────────────────
// Loss Generation
// ─────────────────────────────────────────────

export function genLossInit(block: NodeBlock): string | null {
  switch (block.opType) {
    case 'crossentropyloss': return 'loss_fn = nn.CrossEntropyLoss()'
    case 'mseloss': return 'mse_loss_fn = nn.MSELoss()'
    case 'focalloss': {
      const alpha = Number(block.fields.alpha ?? 0.25)
      const gamma = Number(block.fields.gamma ?? 2.0)
      return `class FocalLoss(nn.Module):\n        def __init__(self, alpha=${alpha}, gamma=${gamma}):\n            super().__init__()\n            self.alpha = ${alpha}\n            self.gamma = ${gamma}\n        def forward(self, pred, target):\n            ce_loss = F.cross_entropy(pred, target, reduction="none")\n            pt = torch.exp(-ce_loss)\n            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss\n            return focal_loss.mean()\n\n    focal_loss_fn = FocalLoss(alpha=${alpha}, gamma=${gamma})`
    }
    case 'labelsmoothing': {
      const smoothing = Number(block.fields.smoothing ?? 0.1)
      return `label_smoothing_loss_fn = nn.CrossEntropyLoss(label_smoothing=${smoothing})`
    }
    case 'adam': return 'optimizer = torch.optim.Adam(model.parameters(), lr=0.001)'
    case 'adamw': return 'optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)'
    case 'sgd': return 'optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)'
    case 'rmsprop': return 'optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)'
    default: return null
  }
}

export function genLossForward(block: NodeBlock, upstreamOutputVar: string, targetExpr?: string): string | null {
  const target = targetExpr ?? 'target'
  switch (block.opType) {
    case 'crossentropyloss':
      return `    loss = loss_fn(${upstreamOutputVar}, ${target})`
    case 'mseloss':
      return `    loss = mse_loss_fn(${upstreamOutputVar}, ${target})`
    case 'focalloss':
      return `    loss = focal_loss_fn(${upstreamOutputVar}, ${target})`
    case 'labelsmoothing':
      return `    loss = label_smoothing_loss_fn(${upstreamOutputVar}, ${target})`
    default: return null
  }
}

// ─────────────────────────────────────────────
// Evaluation Generation
// ─────────────────────────────────────────────

export function genEvaluationCode(block: NodeBlock, allBlocks: NodeBlock[]): string[] {
  const lines: string[] = []
  const f = block.fields
  const opType = block.opType

  const predictionsRef = (block.inputs['predictions'] as string | null) ?? null
  const targetsRef = (block.inputs['targets'] as string | null) ?? null

  const predSrcId = getSourceNodeId(predictionsRef)
  const tgtSrcId = getSourceNodeId(targetsRef)

  const predBlock = predSrcId ? allBlocks.find(b => b.nodeId === predSrcId) : null
  const tgtBlock = tgtSrcId ? allBlocks.find(b => b.nodeId === tgtSrcId) : null

  const predVar = predBlock?.outputVar ?? 'y_pred'
  const tgtVar = tgtBlock?.outputVar ?? 'y_true'

  const average = f.average ?? 'macro'
  const topK = f.top_k ?? 1

  switch (opType) {
    case 'accuracy': {
      lines.push(`    # Accuracy evaluation`)
      lines.push(`    from sklearn.metrics import accuracy_score`)
      lines.push(`    y_pred_np = ${predVar}.argmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      if (topK > 1) {
        lines.push(`    from sklearn.metrics import top_k_accuracy_score`)
        lines.push(`    acc = top_k_accuracy_score(y_true_np, ${predVar}.numpy(), k=${topK})`)
      } else {
        lines.push(`    acc = accuracy_score(y_true_np, y_pred_np)`)
      }
      lines.push(`    print(f"Accuracy: {acc:.4f}")`)
      break
    }
    case 'f1': {
      lines.push(`    # F1 Score evaluation`)
      lines.push(`    from sklearn.metrics import f1_score`)
      lines.push(`    y_pred_np = ${predVar}.argmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      lines.push(`    f1 = f1_score(y_true_np, y_pred_np, average='${average}')`)
      lines.push(`    print(f"F1 Score (${average}): {f1:.4f}")`)
      break
    }
    case 'precision': {
      lines.push(`    # Precision evaluation`)
      lines.push(`    from sklearn.metrics import precision_score`)
      lines.push(`    y_pred_np = ${predVar}.argmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      lines.push(`    prec = precision_score(y_true_np, y_pred_np, average='${average}')`)
      lines.push(`    print(f"Precision (${average}): {prec:.4f}")`)
      break
    }
    case 'recall': {
      lines.push(`    # Recall evaluation`)
      lines.push(`    from sklearn.metrics import recall_score`)
      lines.push(`    y_pred_np = ${predVar}.argmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      lines.push(`    rec = recall_score(y_true_np, y_pred_np, average='${average}')`)
      lines.push(`    print(f"Recall (${average}): {rec:.4f}")`)
      break
    }
    case 'confusion_matrix': {
      lines.push(`    # Confusion Matrix evaluation`)
      lines.push(`    from sklearn.metrics import confusion_matrix`)
      lines.push(`    y_pred_np = ${predVar}.argmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      lines.push(`    cm = confusion_matrix(y_true_np, y_pred_np)`)
      lines.push(`    print(f"Confusion Matrix:\n{cm}")`)
      break
    }
    case 'mean_iou': {
      lines.push(`    # Mean IoU evaluation (semantic segmentation)`)
      lines.push(`    # torchmetrics implementation`)
      lines.push(`    try:`)
      lines.push(`        from torchmetrics import MeanIoU`)
      lines.push(`        miou = MeanIoU(num_classes=${f.num_classes ?? 10})`)
      lines.push(`        y_pred_labels = ${predVar}.argmax(dim=1)  # (N, H, W)`)
      lines.push(`        y_true_labels = ${tgtVar} if ${tgtVar}.dim() == 3 else ${tgtVar}`)
      lines.push(`        iou = miou(y_pred_labels, y_true_labels.long())`)
      lines.push(`        print(f"Mean IoU: {iou:.4f}")`)
      lines.push(`    except Exception as e:`)
      lines.push(`        print(f"Mean IoU computation failed: {e}")`)
      break
    }
    case 'roc_auc': {
      lines.push(`    # ROC AUC evaluation`)
      lines.push(`    from sklearn.metrics import roc_auc_score`)
      lines.push(`    y_pred_prob = ${predVar}.softmax(dim=1).numpy() if ${predVar}.dim() > 1 else ${predVar}.numpy()`)
      lines.push(`    y_true_np = ${tgtVar}.numpy() if hasattr(${tgtVar}, 'numpy') else ${tgtVar}`)
      lines.push(`    auc = roc_auc_score(y_true_np, y_pred_prob, average='${average}', multi_class='ovr')`)
      lines.push(`    print(f"ROC AUC (${average}): {auc:.4f}")`)
      break
    }
  }

  return lines
}
