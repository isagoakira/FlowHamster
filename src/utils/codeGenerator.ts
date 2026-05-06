/**
 * FlowHamster Code Generator v4 — Modular Architecture
 *
 * This is the main entry point that orchestrates the modular code generation:
 * - nodeSignatures.ts: Node type signatures
 * - graphPruner.ts: Graph pruning for reachable nodes
 * - astBuilder.ts: AST construction from graph
 * - moduleRegistry.ts: Module import management
 * - codeEmitter.ts: Code emission (init, forward, loss, evaluation)
 */

import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { DataFlowEdge, DataFlowNode } from '../types/dataGraph'
import { getExecutableGraph } from './graphStructure'
import { compileDataWorkflow } from './dataWorkflowCompiler'
import { WorkflowTrainingConfig } from '../schema/workflowDocument'

// Import modular components
import { buildAST, getSourceNodeId } from './astBuilder'
import { FeatureToggles } from './graphPruner'
import { collectModuleImports, genModuleImports } from './moduleRegistry'
import {
  genInit,
  genForward,
  genCompositeInit,
  genCompositeForward,
  genLossInit,
  genLossForward,
  genEvaluationCode,
  genReusableCompositeClasses,
  shouldEmitCompositeAsClass,
} from './codeEmitter'

// Re-export for backwards compatibility
export { pruneGraph } from './graphPruner'
export type { FeatureToggles } from './graphPruner'

// ─────────────────────────────────────────────
// Training Config Code Generation
// ─────────────────────────────────────────────

function getLossClass(lossType: string): string {
  switch (lossType) {
    // Basic / Classic
    case 'cross_entropy':
      return 'nn.CrossEntropyLoss()'
    case 'mse':
      return 'nn.MSELoss()'
    case 'bce':
    case 'bce_logits':
      return 'nn.BCEWithLogitsLoss()'
    // CV - Segmentation
    case 'dice':
      return 'DiceLoss()'
    case 'focal':
      return 'FocalLoss()'
    case 'lovasz':
      return 'LovaszLoss()'
    case 'tversky':
      return 'TverskyLoss()'
    case 'iou':
      return 'IoULoss()'
    case 'giou':
      return 'GIoULoss()'
    case 'dice_ce':
      return 'DiceCELoss()'
    // CV - Metric / Perceptual
    case 'msssim':
      return 'MS_SSIMLoss()'
    case 'perceptual':
      return 'PerceptualLoss()'
    case 'content':
      return 'ContentLoss()'
    case 'style':
      return 'StyleLoss()'
    // CV - Detection
    case 'smooth_l1':
      return 'nn.SmoothL1Loss()'
    case 'focal_loss':
      return 'FocalLoss()'
    case 'class_balanced':
      return 'ClassBalancedLoss()'
    // Audio Enhancement
    case 'stft':
      return 'STFTLoss()'
    case 'sdr':
      return 'SDRLoss()'
    case 'sisdr':
      return 'SISDRLoss()'
    case 'mel_spec':
      return 'MelSpectrogramLoss()'
    case 'waveform':
      return 'WaveformMSELoss()'
    case 'multi_res':
      return 'MultiResolutionSTFTLoss()'
    case 'phase':
      return 'PhaseLoss()'
    // NLP / Other
    case 'label_smoothing':
      return 'nn.LabelSmoothingLoss()'
    case 'contrastive':
      return 'ContrastiveLoss()'
    default:
      return 'nn.CrossEntropyLoss()'
  }
}

// Returns additional setup code needed for complex losses
function getLossSetupCode(lossType: string): string[] {
  switch (lossType) {
    case 'dice':
    case 'tversky':
    case 'iou':
    case 'giou':
      return [
        '# Dice/IoU losses require custom implementation',
        'class DiceLoss(nn.Module):',
        '    def __init__(self, smooth=1e-6):',
        '        super().__init__()',
        '        self.smooth = smooth',
        '    def forward(self, pred, target):',
        '        pred = F.softmax(pred, dim=1)',
        '        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()',
        '        intersection = (pred * target_one_hot).sum(dim=(2,3))',
        '        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))',
        '        iou = (2 * intersection + self.smooth) / (union + self.smooth)',
        '        return 1 - iou.mean()',
      ]
    case 'focal':
    case 'focal_loss':
      return [
        '# Focal Loss for class imbalance',
        'class FocalLoss(nn.Module):',
        '    def __init__(self, alpha=1, gamma=2):',
        '        super().__init__()',
        '        self.alpha = alpha',
        '        self.gamma = gamma',
        '    def forward(self, pred, target):',
        '        ce_loss = F.cross_entropy(pred, target, reduction="none")',
        '        pt = torch.exp(-ce_loss)',
        '        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss',
        '        return focal_loss.mean()',
      ]
    case 'lovasz':
      return [
        '# Lovász-Softmax Loss',
        'def lovasz_grad(gt_sorted):',
        '    gts = gt_sorted.sum()',
        '    intersection = gts - gt_sorted.float().cumsum(0)',
        '    union = gts + (1 - gt_sorted).float().cumsum(0)',
        '    jaccard = 1. - intersection / union',
        '    if len(jaccard) > 1:',
        '        jaccard[1:] = jaccard[1:] - jaccard[:-1]',
        '    return jaccard',
        'class LovaszLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        pred = F.softmax(pred, dim=1)',
        '        loss = sum(lovasz_grad(gt_unique) * lovasz_softmax(pred_unique, gt_unique))',
        '        return loss',
      ]
    case 'msssim':
      return [
        '# MS-SSIM Loss',
        'class MS_SSIMLoss(nn.Module):',
        '    def __init__(self, alpha=0.84):',
        '        super().__init__()',
        '        self.alpha = alpha',
        '    def forward(self, pred, target):',
        '        msssim = self.compute_msssim(pred, target)',
        '        return 1 - msssim',
        '    def compute_msssim(self, pred, target):',
        '        # Simplified MS-SSIM computation',
        '        return F.mse_loss(pred, target)',
      ]
    case 'perceptual':
      return [
        '# Perceptual Loss using VGG',
        'class PerceptualLoss(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
        '        vgg = torchvision.models.vgg16(pretrained=True).features[:16]',
        '        self.vgg = vgg.eval()',
        '        for p in self.vgg.parameters():',
        '            p.requires_grad = False',
        '    def forward(self, pred, target):',
        '        vgg_pred = self.vgg(pred)',
        '        vgg_target = self.vgg(target)',
        '        return F.mse_loss(vgg_pred, vgg_target)',
      ]
    case 'stft':
      return [
        '# STFT Loss for audio',
        'class STFTLoss(nn.Module):',
        '    def __init__(self, n_fft=2048, hop_length=512):',
        '        super().__init__()',
        '        self.n_fft = n_fft',
        '        self.hop_length = hop_length',
        '    def forward(self, pred, target):',
        '        pred_stft = torch.stft(pred.flatten(), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)',
        '        target_stft = torch.stft(target.flatten(), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)',
        '        return F.l1_loss(torch.abs(pred_stft), torch.abs(target_stft))',
      ]
    case 'sdr':
      return [
        '# SDR (Signal-to-Distortion Ratio) Loss',
        'class SDRLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        signal_power = (target ** 2).sum()',
        '        noise_power = ((pred - target) ** 2).sum()',
        '        return -10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)',
      ]
    case 'sisdr':
      return [
        '# SISDR (Scale-Invariant SDR) Loss',
        'class SISDRLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        alpha = (pred * target).sum() / ((target ** 2).sum() + 1e-8)',
        '        sdr = ((alpha * target) ** 2).sum() / (((pred - alpha * target) ** 2).sum() + 1e-8)',
        '        return -10 * torch.log10(sdr + 1e-8)',
      ]
    case 'mel_spec':
      return [
        '# Mel-Spectrogram Loss',
        'class MelSpectrogramLoss(nn.Module):',
        '    def __init__(self, sample_rate=16000, n_mels=128):',
        '        super().__init__()',
        '        self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)',
        '    def forward(self, pred, target):',
        '        mel_pred = self.mel_spec(pred)',
        '        mel_target = self.mel_spec(target)',
        '        return F.l1_loss(mel_pred, mel_target)',
      ]
    case 'multi_res':
      return [
        '# Multi-Resolution STFT Loss',
        'class MultiResolutionSTFTLoss(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
        '        self.resolutions = [(2048, 512), (1024, 256), (512, 128)]',
        '    def forward(self, pred, target):',
        '        total = 0',
        '        for n_fft, hop in self.resolutions:',
        '            stft = STFTLoss(n_fft=n_fft, hop_length=hop)',
        '            total += stft(pred, target)',
        '        return total / len(self.resolutions)',
      ]
    case 'phase':
      return [
        '# Phase Loss',
        'class PhaseLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        pred_ang = torch.angle(torch.stft(pred.flatten(), n_fft=2048, hop_length=512))',
        '        target_ang = torch.angle(torch.stft(target.flatten(), n_fft=2048, hop_length=512))',
        '        return F.l1_loss(pred_ang, target_ang)',
      ]
    case 'class_balanced':
      return [
        '# Class-Balanced Loss',
        'class ClassBalancedLoss(nn.Module):',
        '    def __init__(self, num_classes=10):',
        '        super().__init__()',
        '        self.num_classes = num_classes',
        '    def forward(self, pred, target):',
        '        beta = 0.9999',
        '        cls_counts = torch.bincount(target)',
        '        cls_weights = (1 - beta) / (1 - torch.pow(beta, cls_counts.float()))',
        '        cls_weights = cls_weights / cls_weights.sum() * self.num_classes',
        '        return F.cross_entropy(pred, target, weight=cls_weights.to(pred.device))',
      ]
    default:
      return []
  }
}

function formatPythonValue(value: unknown): string {
  if (typeof value === 'string') {
    if (value === 'True' || value === 'False' || value === 'None') return value
    return `'${value}'`
  }
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (value === null || value === undefined) return 'None'
  return String(value)
}

function genTrainingConfigCode(config: WorkflowTrainingConfig): { setupLines: string[]; trainLines: string[] } {
  const setupLines: string[] = []
  const trainLines: string[] = []
  const optimizerParamEntries = Object.entries(config.optimizer.params || {})
    .map(([key, value]) => `${key}=${formatPythonValue(value)}`)
    .join(', ')
  const optimizerArgs = optimizerParamEntries ? `, ${optimizerParamEntries}` : ''

  // Determine the actual loss type to use
  const actualLossType = config.loss.type === 'single'
    ? String(config.loss.params?.lossType || 'cross_entropy')
    : config.loss.type

  // Handle composite loss
  if (config.loss.type === 'composite') {
    const components = config.loss.params?.components
    if (!Array.isArray(components) || components.length === 0) {
      // Fall back to single loss or default
      if (config.loss.params?.fallbackLoss) {
        const fallbackClass = getLossClass(String(config.loss.params.fallbackLoss))
        setupLines.push(`    # Composite loss has no components, using fallback`)
        setupLines.push(`    loss_fn = ${fallbackClass}`)
      }
      // Skip composite handling, will fall through to single loss
    } else {
    setupLines.push('    # Composite loss function')
    const needsSetupCode = new Set<string>()

    for (let i = 0; i < components.length; i++) {
      const comp = components[i]
      if (comp.type === 'custom' && comp.customCode) {
        for (const line of String(comp.customCode).split('\n')) {
          const trimmed = line.trim()
          if (trimmed) {
            setupLines.push(`    ${trimmed}`)
          }
        }
      } else {
        const setupCode = getLossSetupCode(comp.type)
        if (setupCode.length > 0) {
          needsSetupCode.add(comp.type)
          // Add class definition (indented for top-level)
          for (const line of setupCode) {
            setupLines.push(`    ${line}`)
          }
          setupLines.push(`    loss_fn_${i} = ${getLossClass(comp.type)}`)
        } else {
          setupLines.push(`    loss_fn_${i} = ${getLossClass(comp.type)}`)
        }
      }
    }
    // Build weighted sum expression
    const lossExprs = components.map((comp, i) => {
      if (comp.type === 'custom' && comp.customCode) {
        return `${comp.weight} * custom_loss_fn_${i}(primary_output, target)`
      }
      return `${comp.weight} * loss_fn_${i}(primary_output, target)`
    })
    setupLines.push(`    # Weighted: ${components.map(c => `${c.weight}×${c.type}`).join(' + ')}`)
    trainLines.push('    if target is None:')
    trainLines.push('        target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)')
    trainLines.push(`    loss = ${lossExprs.join(' + ')}`)
    }
  }
  // Handle single custom loss
  else if (actualLossType === 'custom' && config.loss.params?.code) {
    setupLines.push('    # Custom loss function')
    for (const line of String(config.loss.params.code).split('\n')) {
      const trimmed = line.trim()
      if (trimmed) {
        setupLines.push(`    ${trimmed}`)
      }
    }
    trainLines.push('    if target is None:')
    trainLines.push('        target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)')
    trainLines.push('    loss = loss_fn(primary_output, target)')
  }
  // Handle single predefined loss
  else {
    const taskType = config.taskType || 'classification'
    let lossClass = getLossClass(actualLossType)
    let targetLine = '        target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)'

    // Handle loss-specific target generation and setup
    switch (actualLossType) {
      case 'mse':
      case 'waveform':
        lossClass = 'nn.MSELoss()'
        targetLine = '        target = torch.randn_like(primary_output)'
        break
      case 'bce':
      case 'bce_logits':
        lossClass = 'nn.BCEWithLogitsLoss()'
        targetLine = '        target = torch.randint(0, 2, (primary_output.shape[0],), device=primary_output.device).float()'
        break
      case 'segmentation':
      case 'dice':
      case 'lovasz':
      case 'tversky':
      case 'iou':
      case 'giou':
      case 'dice_ce':
        lossClass = 'nn.CrossEntropyLoss()'
        targetLine = '        target = torch.randint(0, primary_output.shape[1] if primary_output.dim() > 1 else 10, (primary_output.shape[0], *primary_output.shape[2:]), dtype=torch.long, device=primary_output.device)'
        break
      case 'detection':
      case 'focal':
      case 'focal_loss':
      case 'class_balanced':
        lossClass = 'nn.CrossEntropyLoss()'
        targetLine = '        target = torch.randint(0, primary_output.shape[1] if primary_output.dim() > 1 else 10, (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)'
        break
      case 'nlp':
      case 'label_smoothing':
        lossClass = 'nn.CrossEntropyLoss()'
        targetLine = '        target = torch.randint(0, primary_output.shape[-1] if primary_output.dim() > 1 else 10, (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)'
        break
      case 'regression':
        lossClass = 'nn.MSELoss()'
        targetLine = '        target = torch.randn_like(primary_output)'
        break
      default:
        // Keep default loss class, use taskType-based target if classification
        if (taskType === 'regression') {
          lossClass = 'nn.MSELoss()'
          targetLine = '        target = torch.randn_like(primary_output)'
        } else if (taskType === 'segmentation') {
          targetLine = '        target = torch.randint(0, primary_output.shape[1] if primary_output.dim() > 1 else 10, (primary_output.shape[0], *primary_output.shape[2:]), dtype=torch.long, device=primary_output.device)'
        }
        break
    }

    // Add setup code for complex losses
    const setupCode = getLossSetupCode(actualLossType)
    if (setupCode.length > 0) {
      for (const line of setupCode) {
        setupLines.push(`    ${line}`)
      }
    }
    setupLines.push(`    loss_fn = ${lossClass}`)
    trainLines.push('    if target is None:')
    trainLines.push(targetLine)
    trainLines.push('    loss = loss_fn(primary_output, target)')
  }

  switch (config.optimizer.type) {
    case 'adam':
      setupLines.push(`    optimizer = torch.optim.Adam(model.parameters()${optimizerArgs})`)
      break
    case 'sgd':
      setupLines.push(`    optimizer = torch.optim.SGD(model.parameters()${optimizerArgs})`)
      break
    case 'rmsprop':
      setupLines.push(`    optimizer = torch.optim.RMSprop(model.parameters()${optimizerArgs})`)
      break
    case 'adamw':
    default:
      setupLines.push(`    optimizer = torch.optim.AdamW(model.parameters()${optimizerArgs})`)
      break
  }

  if (config.scheduler.enabled) {
    const schedulerParamEntries = Object.entries(config.scheduler.params || {})
      .map(([key, value]) => `${key}=${formatPythonValue(value)}`)
      .join(', ')
    const schedulerArgs = schedulerParamEntries ? `optimizer, ${schedulerParamEntries}` : 'optimizer'

    switch (config.scheduler.type) {
      case 'step_lr':
      case 'steplr':
        setupLines.push(`    scheduler = torch.optim.lr_scheduler.StepLR(${schedulerArgs})`)
        break
      case 'reduce_on_plateau':
      case 'reducelronplateau':
        setupLines.push(`    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(${schedulerArgs})`)
        break
      case 'cosine_annealing':
      case 'cosineannealinglr':
      default:
        setupLines.push(`    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(${schedulerArgs})`)
        break
    }
  }

  trainLines.push('    model.train()')
  trainLines.push('    optimizer.zero_grad()')
  trainLines.push('    loss = loss_fn(primary_output, target)')
  trainLines.push('    loss.backward()')
  if (config.runtime.gradClip !== null) {
    trainLines.push(`    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=${config.runtime.gradClip})`)
  }
  trainLines.push('    optimizer.step()')
  if (config.scheduler.enabled) {
    if (config.scheduler.type === 'reduce_on_plateau' || config.scheduler.type === 'reducelronplateau') {
      trainLines.push('    scheduler.step(loss)')
    } else {
      trainLines.push('    scheduler.step()')
    }
  }
  trainLines.push(`    print(f"task=${config.taskType}, epochs=${config.runtime.epochs}, batch_size=${config.runtime.batchSize}, device={device}")`)

  return { setupLines, trainLines }
}

// ─────────────────────────────────────────────
// Main Entry Point
// ─────────────────────────────────────────────

export interface GeneratedCode {
  code: string
  branchAssignments: Record<string, number>
}

export interface GenerateWorkflowOptions {
  dataGraphNodes?: DataFlowNode[]
  dataGraphEdges?: DataFlowEdge[]
  bindings?: WorkflowBinding[]
}

export function generateLocalCode(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  features?: FeatureToggles,
  trainingConfig?: WorkflowTrainingConfig,
  workflowOptions?: GenerateWorkflowOptions
): GeneratedCode {
  if (!nodes || nodes.length === 0) {
    return {
      code: '# Empty graph -- drag nodes from the sidebar to start',
      branchAssignments: {},
    }
  }

  let executableGraph
  try {
    executableGraph = getExecutableGraph(nodes, edges)
  } catch (error) {
    console.error('[codeGenerator] getExecutableGraph failed:', error)
    return {
      code: `# Code generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      branchAssignments: {},
    }
  }

  let dataWorkflow
  try {
    dataWorkflow = compileDataWorkflow(
      nodes,
      workflowOptions?.dataGraphNodes ?? [],
      workflowOptions?.dataGraphEdges ?? [],
      workflowOptions?.bindings ?? [],
      trainingConfig
    )
  } catch (error) {
    console.warn('[codeGenerator] dataWorkflow compilation failed, using empty workflow:', error)
    dataWorkflow = { hasWorkflowRuntime: false, warnings: [], pythonScaffold: '' }
  }

  // Build AST from graph
  let astResult
  try {
    astResult = buildAST(executableGraph.nodes, executableGraph.edges, features)
  } catch (error) {
    console.error('[codeGenerator] buildAST failed:', error)
    return {
      code: `# AST construction failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      branchAssignments: {},
    }
  }

  const { blocks, branchAssignments, outputBlocks } = astResult

  // Filter blocks by category
  const modelBlocks = blocks.filter(b => !['training', 'evaluation'].includes(b.category))
  const lossBlocks = blocks.filter(b => b.category === 'training')
  const evaluationBlocks = blocks.filter(b => b.category === 'evaluation')

  // Collect module imports
  const customModules = collectModuleImports(modelBlocks)
  const moduleImports = genModuleImports(customModules)

  // Generate init lines
  const initLines: string[] = []
  const forwardLines: string[] = []

  for (const block of modelBlocks) {
    const compositeDef = getCompositeNodeDef(block.opType)
    if (
      compositeDef &&
      compositeDef.internalStructure &&
      compositeDef.internalStructure.length > 0 &&
      !shouldEmitCompositeAsClass(block.opType)
    ) {
      // Composite node: generate expanded init and forward
      const compositeInits = genCompositeInit(block)
      initLines.push(...compositeInits)
      const fwdLine = genCompositeForward(block)
      if (fwdLine) forwardLines.push(fwdLine)
    } else {
      const initLine = genInit(block)
      if (initLine) initLines.push(`        ${initLine}`)
      const fwdLine = genForward(block, modelBlocks)
      if (fwdLine) forwardLines.push(fwdLine)
    }
  }

  // Multi-output: collect upstream variable names for each Output
  const outputReturnVars: string[] = []
  for (const outBlock of outputBlocks) {
    const outputName = outBlock.fields?.output_name as string | undefined
    if (outputName) {
      forwardLines.push(`        ${outputName}_var = ${outBlock.outputVar}`)
      outputReturnVars.push(`${outputName}_var`)
    } else {
      outputReturnVars.push(outBlock.outputVar)
    }
  }

  const initBlock = initLines.length > 0 ? initLines.join('\n') : '            pass'

  // Generate return statement
  let returnLine: string
  if (outputReturnVars.length === 0) {
    returnLine = '        return x'
  } else if (outputReturnVars.length === 1) {
    returnLine = `        return ${outputReturnVars[0]}`
  } else {
    returnLine = `        return (${outputReturnVars.join(', ')})`
  }

  // Insert return statement before the last non-return line
  const nonReturnIndexes = forwardLines.map((l, i) => l.trim().startsWith('return') ? -1 : i).filter(i => i >= 0)
  const lastFwdIdx = nonReturnIndexes.length > 0 ? nonReturnIndexes[nonReturnIndexes.length - 1] : -1
  if (lastFwdIdx >= 0) {
    forwardLines.splice(lastFwdIdx + 1, 0, returnLine)
  } else {
    forwardLines.push(returnLine)
  }

  const forwardBlock = forwardLines.join('\n')

  // ── Loss / Optimizer initialization
  const lossInitLines: string[] = []
  const lossFwdLines: string[] = []

  for (const block of lossBlocks) {
    const li = genLossInit(block)
    if (li) lossInitLines.push(`    ${li}`)
  }

  for (const block of lossBlocks) {
    const upstreamRefs = Object.values(block.inputs).filter((v): v is string => v !== null)
    if (upstreamRefs.length === 0) continue
    const upstreamSrcId = getSourceNodeId(upstreamRefs[0] ?? null)
    const upstreamBlock = modelBlocks.find(b => b.nodeId === upstreamSrcId)
    const upstreamOutputName = upstreamBlock?.fields?.output_name as string | undefined
    const upstreamVar = upstreamOutputName
      ? `${upstreamOutputName}_var`
      : (upstreamBlock?.outputVar ?? 'x')
    const lf = genLossForward(block, upstreamVar)
    if (lf) lossFwdLines.push(`  ${lf}`)
  }

  // Construct __main__ block
  const returnVarsDecl = outputReturnVars.length > 1
    ? `${outputReturnVars.join(', ')} = model(x)`
    : `output = model(x)`
  const primaryOutputLine = outputReturnVars.length > 1
    ? `    primary_output = ${outputReturnVars[0]}`
    : '    primary_output = output'
  const workflowRuntimePrelude = dataWorkflow.hasWorkflowRuntime
    ? `    runtime_device = device if device != "auto" else "cpu"
    batch = build_demo_batch(runtime_device)
    model_feed, target = resolve_bound_inputs(batch, runtime_device)
    x = select_primary_model_input(model_feed, runtime_device)`
    : `    runtime_device = device if device != "auto" else "cpu"
    target = None
    x = torch.randn(1, 3, 224, 224, device=runtime_device)`

  // Evaluation code
  const evalLines: string[] = []
  for (const block of evaluationBlocks) {
    const lines = genEvaluationCode(block, modelBlocks)
    evalLines.push(...lines)
  }

  const needsTargetsPlaceholder = evaluationBlocks.length > 0 && evalLines.some(l => l.includes('y_true_np'))
  const targetsPlaceholderLine = needsTargetsPlaceholder
    ? `    # Placeholder labels (replace with real dataset labels in production)\n    y_true = target if target is not None else torch.randint(0, 10, (1,), device=runtime_device)`
    : ''

  const configDrivenTraining = trainingConfig?.loss?.enabled && trainingConfig?.optimizer?.enabled
    ? genTrainingConfigCode(trainingConfig)
    : null

  let mainBlock = ''
  if (lossBlocks.length > 0 || evaluationBlocks.length > 0 || configDrivenTraining) {
    const lossSetup = lossInitLines.join('\n')
    const lossCompute = lossFwdLines.join('\n')
    const evalBlock = evalLines.length > 0 ? `\n${evalLines.join('\n')}` : ''
    const configSetup = configDrivenTraining?.setupLines.join('\n') ?? ''
    const configTrain = configDrivenTraining?.trainLines.join('\n') ?? ''
    mainBlock = `
if __name__ == "__main__":
    model = FlowHamsterModel()
    device = "${trainingConfig?.runtime?.device ?? 'cpu'}"
    model = model.to(device if device != "auto" else "cpu")
${workflowRuntimePrelude}
    ${returnVarsDecl}
${primaryOutputLine}
${configSetup ? `${configSetup}` : ''}${lossSetup ? `${configSetup ? '\n' : ''}${lossSetup}` : ''}
${configTrain ? `${configTrain}` : ''}${lossCompute ? `${configTrain ? '\n' : ''}${lossCompute}` : ''}
${targetsPlaceholderLine ? `${targetsPlaceholderLine}` : ''}${evalBlock}
    print(f"${outputReturnVars.length > 1 ? outputReturnVars.map(v => `${v}: {${v}.shape}`).join(', ') : `output: {output.shape}`}" + ${lossFwdLines.length > 0 || Boolean(configDrivenTraining) ? `f", loss: {loss.item()}"` : `""`})
`
  } else {
    mainBlock = `
if __name__ == "__main__":
    model = FlowHamsterModel()
    device = "${trainingConfig?.runtime?.device ?? 'cpu'}"
    model = model.to(device if device != "auto" else "cpu")
${workflowRuntimePrelude}
    ${returnVarsDecl}
    print(${outputReturnVars.length > 1
      ? `"(" + ", ".join([f"{v}: {v}.shape" for v in outputReturnVars]) + ")"`
      : `"output: {output.shape}"`})
`
  }

  const warningBlock = dataWorkflow.warnings.length > 0
    ? `${dataWorkflow.warnings.map((warning) => `# Workflow warning: ${warning}`).join('\n')}\n`
    : ''
  const workflowSupportBlock = dataWorkflow.hasWorkflowRuntime
    ? `\n\n${dataWorkflow.pythonScaffold}\n`
    : '\n'
  const reusableClassesBlock = genReusableCompositeClasses(modelBlocks).join('\n\n')

  // Collect custom composite class definitions
  const customClasses = getAllCustomClasses()
  const customClassIdsUsed = new Set<string>()

  // Recursively collect customClassId from internalStructure
  function collectFromInternalStructure(internalStructure: any[]) {
    if (!internalStructure) return
    for (const sub of internalStructure) {
      if (sub.customClassId) {
        customClassIdsUsed.add(sub.customClassId)
      }
      // Also recursively check nested internalStructure
      if (sub.internalStructure) {
        collectFromInternalStructure(sub.internalStructure)
      }
      // Check if sub.type is 'custom' and has customClassId in its own data
      if (sub.type === 'custom' && sub.customClassId) {
        customClassIdsUsed.add(sub.customClassId)
      }
    }
  }

  for (const block of modelBlocks) {
    if (block.opType === 'custom') {
      // Use fields.customClassId if available, otherwise fall back to nodeId
      const customClassId = block.fields?.customClassId || block.nodeId
      customClassIdsUsed.add(customClassId)

      // Also recursively collect from internalStructure
      if (block.fields?.internalStructure) {
        collectFromInternalStructure(block.fields.internalStructure)
      }
    } else if (block.opType.startsWith('custom_')) {
      customClassIdsUsed.add(block.opType)
    }
  }

  let customClassesBlock = ''
  for (const cls of customClasses) {
    // 检查 cls.id（原始注册ID）、cls.name（模块名）、custom_${cls.name}（前缀形式）
    if (customClassIdsUsed.has(cls.id) || customClassIdsUsed.has(cls.name) || customClassIdsUsed.has(`custom_${cls.name}`)) {
      customClassesBlock += `\n\n${generateCustomClassCode(cls)}`
    }
  }

  // Assemble final code
  const code = [
    `# Generated by FlowHamster`,
    `# DO NOT EDIT -- Regenerated from graph editor`,
    warningBlock,
    moduleImports,
    reusableClassesBlock,
    customClassesBlock,
    `class FlowHamsterModel(nn.Module):`,
    `    def __init__(self):`,
    `        super().__init__()`,
    initBlock,
    ``,
    `    def forward(self, x):`,
    forwardBlock,
    workflowSupportBlock,
    mainBlock,
  ].join('\n')

  return {
    code,
    branchAssignments,
  }
}

// Import these at the bottom to avoid circular dependencies
import { generateCustomClassCode, getAllCustomClasses } from './customCompositeRegistry'
import { getCompositeNodeDef } from './nodeRegistry'
import { WorkflowBinding } from '../schema/workflowDocument'
