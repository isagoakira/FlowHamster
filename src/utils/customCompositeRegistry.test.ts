/**
 * Unit tests for customCompositeRegistry.ts
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import {
  registerCustomClass,
  updateCustomClass,
  unregisterCustomClass,
  getCustomClass,
  getAllCustomClasses,
  getCustomClassesByCategory,
  loadCustomClasses,
} from './customCompositeRegistry'
import { SubModule, InternalEdge } from './nodeRegistry'

// Storage key for localStorage mock
const STORAGE_KEY = 'flowhamster_custom_composites'

// Mock localStorage for jsdom environment
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
    get length() { return Object.keys(store).length },
    key: (i: number) => Object.keys(store)[i] || null,
  }
})()

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
})

// Helper to create a SubModule
function createSubModule(id: string, type: string, label: string): SubModule {
  return {
    id,
    type: type as any,
    label,
    params: {},
  }
}

// Helper to create an InternalEdge
function createInternalEdge(from: string, to: string): InternalEdge {
  return { from, to }
}

describe('customCompositeRegistry', () => {
  // Clear localStorage before each test
  beforeEach(() => {
    localStorageMock.clear()
  })

  afterEach(() => {
    localStorageMock.clear()
  })

  describe('registerCustomClass()', () => {
    it('should register a new custom class', () => {
      const internalStructure = [
        createSubModule('relu1', 'relu', 'ReLU1'),
        createSubModule('linear1', 'linear', 'Linear1'),
      ]
      const internalEdges = [
        createInternalEdge('relu1', 'linear1'),
      ]

      const result = registerCustomClass(
        'MyModule',
        'mlp',
        'other',
        '📦',
        'Test module',
        internalStructure,
        internalEdges,
        'linear1'
      )

      expect(result).toBeDefined()
      expect(result.id).toBeDefined()
      expect(result.name).toBe('MyModule')
      expect(result.baseType).toBe('mlp')
      expect(result.category).toBe('other')
      expect(result.internalStructure).toHaveLength(2)
    })

    it('should return existing class for duplicate name', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      const result1 = registerCustomClass(
        'DuplicateModule',
        'mlp',
        'other',
        '📦',
        'First registration',
        internalStructure,
        [],
        'relu1'
      )

      const result2 = registerCustomClass(
        'DuplicateModule',
        'mlp',
        'other',
        '📦',
        'Second registration',
        internalStructure,
        [],
        'relu1'
      )

      // Should be the same object (not a new registration)
      expect(result1.id).toBe(result2.id)
      expect(getAllCustomClasses()).toHaveLength(1)
    })
  })

  describe('updateCustomClass()', () => {
    it('should update an existing class', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      const registered = registerCustomClass(
        'UpdateTest',
        'mlp',
        'other',
        '📦',
        'Original description',
        internalStructure,
        [],
        'relu1'
      )

      const updated = updateCustomClass(registered.id, {
        description: 'Updated description',
        name: 'UpdatedName',
      })

      expect(updated).toBeDefined()
      expect(updated?.description).toBe('Updated description')
      expect(updated?.name).toBe('UpdatedName')
    })

    it('should return null for non-existent id', () => {
      const result = updateCustomClass('non-existent-id', { description: 'test' })
      expect(result).toBeNull()
    })

    it('should regenerate code template when structure changes', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      const registered = registerCustomClass(
        'CodeRegen',
        'mlp',
        'other',
        '📦',
        'Test',
        internalStructure,
        [],
        'relu1'
      )

      const originalCode = registered.codeTemplate

      const updated = updateCustomClass(registered.id, {
        internalStructure: [
          createSubModule('relu1', 'relu', 'ReLU1'),
          createSubModule('linear1', 'linear', 'Linear1'),
        ],
      })

      // Code template should be regenerated
      expect(updated?.codeTemplate).not.toBe(originalCode)
    })
  })

  describe('unregisterCustomClass()', () => {
    it('should unregister a class by id', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      const registered = registerCustomClass(
        'RemoveTest',
        'mlp',
        'other',
        '📦',
        'Test',
        internalStructure,
        [],
        'relu1'
      )

      const result = unregisterCustomClass(registered.id)

      expect(result).toBe(true)
      expect(getAllCustomClasses()).toHaveLength(0)
    })

    it('should return false for non-existent id', () => {
      const result = unregisterCustomClass('non-existent-id')
      expect(result).toBe(false)
    })
  })

  describe('getCustomClass()', () => {
    it('should get a class by id', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      const registered = registerCustomClass(
        'GetTest',
        'mlp',
        'other',
        '📦',
        'Test',
        internalStructure,
        [],
        'relu1'
      )

      const result = getCustomClass(registered.id)

      expect(result).toBeDefined()
      expect(result?.name).toBe('GetTest')
    })

    it('should return undefined for non-existent id', () => {
      const result = getCustomClass('non-existent-id')
      expect(result).toBeUndefined()
    })
  })

  describe('getAllCustomClasses()', () => {
    it('should return all registered classes', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      registerCustomClass('Class1', 'mlp', 'other', '📦', 'Test1', internalStructure, [], 'relu1')
      registerCustomClass('Class2', 'ffn', 'nlp', '📦', 'Test2', internalStructure, [], 'relu1')

      const result = getAllCustomClasses()

      expect(result).toHaveLength(2)
    })

    it('should return empty array when no classes registered', () => {
      const result = getAllCustomClasses()
      expect(result).toHaveLength(0)
    })
  })

  describe('getCustomClassesByCategory()', () => {
    it('should filter classes by category', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      registerCustomClass('CVClass', 'mlp', 'cv', '🖼', 'CV', internalStructure, [], 'relu1')
      registerCustomClass('NLClass', 'mlp', 'nlp', '📝', 'NLP', internalStructure, [], 'relu1')
      registerCustomClass('OtherClass', 'mlp', 'other', '📦', 'Other', internalStructure, [], 'relu1')

      const cvClasses = getCustomClassesByCategory('cv')
      const nlpClasses = getCustomClassesByCategory('nlp')
      const otherClasses = getCustomClassesByCategory('other')

      expect(cvClasses).toHaveLength(1)
      expect(cvClasses[0].name).toBe('CVClass')
      expect(nlpClasses).toHaveLength(1)
      expect(otherClasses).toHaveLength(1)
    })
  })

  describe('localStorage persistence', () => {
    it('should persist classes to localStorage', () => {
      const internalStructure = [createSubModule('relu1', 'relu', 'ReLU1')]

      registerCustomClass('PersistTest', 'mlp', 'other', '📦', 'Test', internalStructure, [], 'relu1')

      // Check localStorage directly
      const stored = localStorageMock.getItem(STORAGE_KEY)
      expect(stored).toBeTruthy()

      const parsed = JSON.parse(stored!)
      expect(parsed).toHaveLength(1)
      expect(parsed[0].name).toBe('PersistTest')
    })

    it('should load classes from localStorage', () => {
      // Pre-populate localStorage
      const storedData = [{
        id: 'test-id',
        name: 'StoredClass',
        baseType: 'mlp',
        category: 'other',
        emoji: '📦',
        description: 'Stored',
        internalStructure: [{ id: 'relu1', type: 'relu', label: 'ReLU1', params: {} }],
        internalEdges: [],
        outputVar: 'relu1',
        codeTemplate: 'class StoredClass',
        createdAt: Date.now(),
      }]
      localStorageMock.setItem(STORAGE_KEY, JSON.stringify(storedData))

      // Should load from localStorage
      const result = loadCustomClasses()

      expect(result).toHaveLength(1)
      expect(result[0].name).toBe('StoredClass')
    })

    it('should return empty array on invalid localStorage data', () => {
      localStorageMock.setItem(STORAGE_KEY, 'invalid json')

      const result = loadCustomClasses()

      expect(result).toHaveLength(0)
    })
  })

  describe('generateCustomClassCode()', () => {
    it('should generate valid Python class code', () => {
      const internalStructure = [
        createSubModule('relu1', 'relu', 'ReLU1'),
        createSubModule('linear1', 'linear', 'Linear1'),
      ]
      const internalEdges = [createInternalEdge('relu1', 'linear1')]

      const registered = registerCustomClass(
        'CodeGenTest',
        'mlp',
        'other',
        '📦',
        'Test',
        internalStructure,
        internalEdges,
        'linear1'
      )

      const code = registered.codeTemplate

      expect(code).toContain('class CodeGenTest')
      expect(code).toContain('nn.Module')
      expect(code).toContain('def forward')
      expect(code).toContain('self.relu1')
      expect(code).toContain('self.linear1')
    })

    it('should handle sequential nodes', () => {
      const internalStructure = [
        createSubModule('input', 'input', 'Input'),
        createSubModule('relu1', 'relu', 'ReLU1'),
        createSubModule('relu2', 'relu', 'ReLU2'),
        createSubModule('output', 'linear', 'Output'),
      ]
      const internalEdges = [
        createInternalEdge('input', 'relu1'),
        createInternalEdge('relu1', 'relu2'),
        createInternalEdge('relu2', 'output'),
      ]

      const registered = registerCustomClass(
        'SequentialTest',
        'mlp',
        'other',
        '📦',
        'Test',
        internalStructure,
        internalEdges,
        'output'
      )

      const code = registered.codeTemplate

      // Should process nodes in topological order
      expect(code).toContain('self.input')
      expect(code).toContain('self.relu1')
      expect(code).toContain('self.relu2')
      expect(code).toContain('self.output')
    })
  })
})
