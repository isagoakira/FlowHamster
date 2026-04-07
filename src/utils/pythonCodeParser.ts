/**
 * Python Module Code Parser
 * Parses Python class definitions to extract module structure:
 * - __init__ parameters
 * - forward() inputs/outputs
 * - Sub-modules defined in __init__
 */

export interface ParsedParameter {
  name: string
  type: string
  default?: string
}

export interface ParsedModule {
  name: string
  className: string
  parameters: ParsedParameter[]
  forwardInputs: string[]
  forwardOutputs: string[]
  submodules: Array<{ name: string; type: string }>
  rawCode: string
}

export function parsePythonCode(code: string, className?: string): ParsedModule | null {
  try {
    // Find the class definition
    const classMatchPattern = className
      ? new RegExp(`class\\s+${className}\\s*\\([^)]*\\)\\s*:`)
      : /class\s+(\w+)\s*\([^)]*\)\s*:/
    const classMatch = classMatchPattern.exec(code)
    if (!classMatch) return null

    const matchedClassName = /class\s+(\w+)/.exec(classMatch[0])?.[1]
    if (!matchedClassName) return null

    // Extract class body (simplified - find the next class or end)
    const classStart = code.indexOf(`class ${matchedClassName}`)
    const nextClassIndex = code.indexOf('\nclass ', classStart + 1)
    const classBody = nextClassIndex > 0
      ? code.substring(classStart, nextClassIndex)
      : code.substring(classStart)

    // Parse __init__ parameters
    const initMatch = /def\s+__init__\s*\(\s*self\s*(?:,\s*(.*?))?\s*\)\s*:/s.exec(classBody)
    const parameters: ParsedParameter[] = []

    if (initMatch && initMatch[1]) {
      const paramsStr = initMatch[1]
      // Match parameters with or without type annotations
      const paramMatches = paramsStr.matchAll(/(\w+)\s*(?::\s*([^=]+?))?(?:\s*=\s*(.+?))?(?:,\s*$|$)/gm)
      for (const match of paramMatches) {
        if (match[1]) {
          parameters.push({
            name: match[1],
            type: match[2]?.trim() || 'any',
            default: match[3]?.trim()
          })
        }
      }
    }

    // Parse forward() signature
    const forwardMatch = /def\s+forward\s*\(\s*self\s*(?:,\s*(.*?))?\s*\)\s*->\s*(.+?)\s*:/s.exec(classBody)
    const forwardInputs: string[] = []
    const forwardOutputs: string[] = []

    if (forwardMatch) {
      if (forwardMatch[1]) {
        // Parse input parameters
        const inputParams = forwardMatch[1].split(',').map(p => p.trim()).filter(p => p)
        forwardInputs.push(...inputParams)
      }
      // Parse output type
      const outputType = forwardMatch[2]?.trim() || 'Tensor'
      forwardOutputs.push(outputType)
    }

    // Parse sub-modules
    const submodules: Array<{ name: string; type: string }> = []
    const submoduleMatches = classBody.matchAll(/self\.(\w+)\s*=\s*(\w+)/g)
    for (const match of submoduleMatches) {
      submodules.push({
        name: match[1],
        type: match[2]
      })
    }

    return {
      name: matchedClassName,
      className: matchedClassName,
      parameters,
      forwardInputs,
      forwardOutputs,
      submodules,
      rawCode: classBody
    }
  } catch (error) {
    console.error('Failed to parse Python code:', error)
    return null
  }
}

/**
 * Parse a complete Python file with multiple classes
 */
export function parsePythonFile(code: string): ParsedModule[] {
  const modules: ParsedModule[] = []
  const classMatches = code.matchAll(/class\s+(\w+)\s*\([^)]*\)\s*:/g)

  for (const match of classMatches) {
    const className = match[1]
    const parsed = parsePythonCode(code, className)
    if (parsed) {
      modules.push(parsed)
    }
  }

  return modules
}

/**
 * Generate a simple module template
 */
export function generateModuleTemplate(
  className: string,
  parameters: Array<{ name: string; type: string; default?: string }>
): string {
  const paramStr = parameters
    .map(p => p.default
      ? `${p.name}: ${p.type} = ${p.default}`
      : `${p.name}: ${p.type}`)
    .join(', ')

  return `class ${className}(nn.Module):
    def __init__(self, ${paramStr}):
        super().__init__()
        # TODO: Add sub-modules

    def forward(self, x):
        # TODO: Implement forward pass
        return x
`
}

/**
 * Infer parameter type from default value
 */
export function inferParamType(defaultValue: string | undefined): string {
  if (!defaultValue) return 'any'

  if (defaultValue === 'None') return 'Optional'
  if (defaultValue === 'True' || defaultValue === 'False') return 'bool'
  if (/^\d+\.\d+$/.test(defaultValue)) return 'float'
  if (/^\d+$/.test(defaultValue)) return 'int'
  if (defaultValue.startsWith("'") || defaultValue.startsWith('"')) return 'str'
  if (defaultValue.startsWith('[')) return 'List'
  if (defaultValue.startsWith('{')) return 'Dict'

  return 'any'
}
