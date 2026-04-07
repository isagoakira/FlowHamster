/**
 * 代码生成结果缓存
 *
 * 缓存后端返回的代码，避免重复生成
 */

import { create } from 'zustand'

export interface CachedCode {
  code: string
  templateId: string
  timestamp: number
  graphSnapshot: string // 用于检测图是否变化
}

interface CodeCacheStore {
  // 缓存存储
  cache: Record<string, CachedCode>
  // 最后访问时间
  lastAccessed: Record<string, number>

  // Actions
  get: (templateId: string, graphSnapshot: string) => string | null
  set: (templateId: string, graphSnapshot: string, code: string) => void
  invalidate: (templateId: string) => void
  clear: () => void
  // 获取缓存元数据
  getMetadata: (templateId: string) => { cached: boolean; age?: number } | null
}

const CACHE_TTL = 5 * 60 * 1000 // 5分钟缓存过期

export const useCodeCacheStore = create<CodeCacheStore>((set, get) => ({
  cache: {},
  lastAccessed: {},

  get: (templateId: string, graphSnapshot: string): string | null => {
    const state = get()
    const cached = state.cache[templateId]

    if (!cached) {
      return null
    }

    // 检查是否过期
    const now = Date.now()
    if (now - cached.timestamp > CACHE_TTL) {
      // 缓存过期，删除
      set((s) => {
        const { [templateId]: _, ...rest } = s.cache
        return { cache: rest }
      })
      return null
    }

    // 检查图是否变化
    if (cached.graphSnapshot !== graphSnapshot) {
      return null
    }

    // 更新访问时间
    set((s) => ({
      lastAccessed: {
        ...s.lastAccessed,
        [templateId]: now,
      },
    }))

    return cached.code
  },

  set: (templateId: string, graphSnapshot: string, code: string) => {
    set((s) => ({
      cache: {
        ...s.cache,
        [templateId]: {
          code,
          templateId,
          graphSnapshot,
          timestamp: Date.now(),
        },
      },
      lastAccessed: {
        ...s.lastAccessed,
        [templateId]: Date.now(),
      },
    }))
  },

  invalidate: (templateId: string) => {
    set((s) => {
      const { [templateId]: _, ...restCache } = s.cache
      const { [templateId]: __, ...restAccessed } = s.lastAccessed
      return {
        cache: restCache,
        lastAccessed: restAccessed,
      }
    })
  },

  clear: () => {
    set({ cache: {}, lastAccessed: {} })
  },

  getMetadata: (templateId: string) => {
    const cached = get().cache[templateId]
    if (!cached) {
      return null
    }
    return {
      cached: true,
      age: Date.now() - cached.timestamp,
    }
  },
}))

/**
 * 计算图的快照哈希
 * 用于检测图结构是否发生变化
 */
export function computeGraphSnapshot(nodes: any[], edges: any[]): string {
  // 按 ID 排序确保顺序一致
  const nodeStr = nodes
    .map((n) => `${n.id}:${n.type}:${JSON.stringify(n.data?.params || {})}`)
    .sort()
    .join('|')
  const edgeStr = edges
    .map((e) => `${e.source}:${e.target}:${e.sourceHandle}:${e.targetHandle}`)
    .sort()
    .join('|')

  return `${nodeStr}#${edgeStr}`
}
