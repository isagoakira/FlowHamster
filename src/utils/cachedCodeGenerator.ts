/**
 * Cached Code Generator
 *
 * Wrapper around generateLocalCode that uses the code cache
 * to avoid regenerating code for unchanged graphs.
 */

import { generateLocalCode } from './codeGenerator'
import { useCodeCacheStore, computeGraphSnapshot } from '../stores/codeCache'

const CODE_GENERATOR_VERSION = 'class-instance-v2'

interface CodeGeneratorParams {
  dataGraphNodes?: any[]
  dataGraphEdges?: any[]
  bindings?: any[]
}

/**
 * Generate code with caching support.
 * Returns cached code if the graph snapshot matches, otherwise generates and caches new code.
 */
export function getCachedOrGenerateCode(
  nodes: any[],
  edges: any[],
  features: any,
  trainingConfig: any,
  params: CodeGeneratorParams
): string {
  // Compute graph snapshot for cache key
  const graphSnapshot = `${CODE_GENERATOR_VERSION}:${computeGraphSnapshot(nodes, edges)}`

  // Try to get cached code
  const cached = useCodeCacheStore.getState().get('default', graphSnapshot)

  if (cached) {
    return cached
  }

  // Generate new code
  const result = generateLocalCode(nodes, edges, features, trainingConfig, {
    dataGraphNodes: params.dataGraphNodes,
    dataGraphEdges: params.dataGraphEdges,
    bindings: params.bindings,
  })

  const code = typeof result === 'string' ? result : (result as any).code ?? JSON.stringify(result)

  // Cache the generated code
  useCodeCacheStore.getState().set('default', graphSnapshot, code)

  return code
}

/**
 * Invalidate the code cache for a specific template.
 * Call this when you want to force regeneration.
 */
export function invalidateCodeCache(templateId: string = 'default'): void {
  useCodeCacheStore.getState().invalidate(templateId)
}

/**
 * Clear the entire code cache.
 */
export function clearCodeCache(): void {
  useCodeCacheStore.getState().clear()
}
