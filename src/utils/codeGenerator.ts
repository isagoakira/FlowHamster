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
import { FLOWHAMSTER_LOSS_INPUT_HELPER, genTrainingConfigCode } from './trainingConfigGen'

// Re-export for backwards compatibility
export { pruneGraph } from './graphPruner'
export type { FeatureToggles } from './graphPruner'

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

function indentPythonLines(lines: string[], spaces: number): string {
  const prefix = ' '.repeat(spaces)
  return lines.map((line) => (line.trim() ? `${prefix}${line}` : line)).join('\n')
}

function removeLegacyTrainingSummary(lines: string[]): string[] {
  return lines.filter((line) => !line.trim().startsWith('print(f"task='))
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
  const hasLossTargets = (dataWorkflow.lossTargetBindings?.length ?? 0) > 0

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
    const lossBinding = dataWorkflow.lossTargetBindings?.find(lb => lb.lossNodeId === block.nodeId)
    const targetExpr = lossBinding
      ? `loss_targets.get(${JSON.stringify(block.nodeId)})`
      : (hasLossTargets ? `loss_targets.get(${JSON.stringify(block.nodeId)})` : 'target')
    const lf = genLossForward(block, upstreamVar, targetExpr)
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
    model_feed, target, loss_targets = resolve_bound_inputs(batch, runtime_device)
    x = select_primary_model_input(model_feed, runtime_device, target.shape[0] if torch.is_tensor(target) and target.dim() > 0 else None)`
    : `    runtime_device = device if device != "auto" else "cpu"
    target = None
    loss_targets = {}
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
    const configTrainLines = removeLegacyTrainingSummary(configDrivenTraining?.trainLines ?? [])
    const configTrain = configTrainLines.join('\n')
    if (configDrivenTraining && dataWorkflow.hasWorkflowRuntime) {
      const resolveLine = hasLossTargets
        ? '            model_feed, target, loss_targets = resolve_bound_inputs(batch, runtime_device)'
        : '            model_feed, target = resolve_bound_inputs(batch, runtime_device)'
      mainBlock = `
if __name__ == "__main__":
    model = FlowHamsterModel()
    device = "${trainingConfig?.runtime?.device ?? 'cpu'}"
    model = model.to(device if device != "auto" else "cpu")
    runtime_device = device if device != "auto" else "cpu"
${configSetup ? `${configSetup}\n` : ''}    loader = build_flowhamster_dataloader()
    for epoch in range(${trainingConfig?.runtime?.epochs ?? 1}):
        for step, batch in enumerate(loader, start=1):
${resolveLine}
            x = select_primary_model_input(model_feed, runtime_device, target.shape[0] if torch.is_tensor(target) and target.dim() > 0 else None)
            ${returnVarsDecl}
${indentPythonLines([primaryOutputLine], 8)}
${indentPythonLines(configTrainLines, 8)}
            print(f"epoch={epoch + 1}, step={step}, batch_size={x.shape[0] if hasattr(x, 'shape') and x.dim() > 0 else 1}, output={tuple(primary_output.shape) if hasattr(primary_output, 'shape') else type(primary_output).__name__}, loss={loss.item() if hasattr(loss, 'item') else loss}, device={runtime_device}")
`
    } else {
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
    }
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
  const lossInputHelperBlock = configDrivenTraining ? `\n${FLOWHAMSTER_LOSS_INPUT_HELPER}\n` : ''

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
      if (sub.data?.customClassId) {
        customClassIdsUsed.add(sub.data.customClassId)
      }
      // Also recursively check nested internalStructure
      if (sub.internalStructure) {
        collectFromInternalStructure(sub.internalStructure)
      }
      if (sub.data?.internalStructure) {
        collectFromInternalStructure(sub.data.internalStructure)
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
    lossInputHelperBlock,
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
