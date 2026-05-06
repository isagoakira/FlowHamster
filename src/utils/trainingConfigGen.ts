/**
 * Training Config Generator — Generates training setup and loop code
 *
 * Handles loss, optimizer, scheduler, and training loop code generation
 * based on WorkflowTrainingConfig.
 */

import { WorkflowTrainingConfig } from '../schema/workflowDocument'
import { getLossClass, getLossSetupCode } from './lossRegistry'

/**
 * Formats a JavaScript value into a Python-compatible string representation
 */
export function formatPythonValue(value: unknown): string {
  if (typeof value === 'string') {
    if (value === 'True' || value === 'False' || value === 'None') return value
    return `'${value}'`
  }
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (value === null || value === undefined) return 'None'
  return String(value)
}

/**
 * Generates setup and training loop lines from a WorkflowTrainingConfig
 */
export function genTrainingConfigCode(
  config: WorkflowTrainingConfig
): { setupLines: string[]; trainLines: string[] } {
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
