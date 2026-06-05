import { beforeEach, describe, expect, it } from 'vitest'
import { useBindingStore } from './bindingStore'

describe('bindingStore', () => {
  beforeEach(() => {
    useBindingStore.setState({ bindings: [] })
  })

  it('adds a loss_target binding', () => {
    const id = useBindingStore.getState().addLossBinding('loss_1', 'label')
    expect(id).toBeDefined()

    const binding = useBindingStore.getState().getLossBindingForNode('loss_1')
    expect(binding).toBeDefined()
    expect(binding?.target).toBe('loss_target')
    expect(binding?.targetKey).toBe('loss_1')
    expect(binding?.sourceKey).toBe('label')
  })

  it('updates existing loss_target binding instead of duplicating', () => {
    const store = useBindingStore.getState()
    store.addLossBinding('loss_1', 'label')
    store.addLossBinding('loss_1', 'mask')

    const bindings = store.getLossBindings()
    expect(bindings).toHaveLength(1)
    expect(bindings[0].sourceKey).toBe('mask')
  })

  it('removes a loss_target binding', () => {
    const store = useBindingStore.getState()
    store.addLossBinding('loss_1', 'label')
    store.removeLossBinding('loss_1')

    expect(store.getLossBindingForNode('loss_1')).toBeUndefined()
    expect(store.getLossBindings()).toHaveLength(0)
  })

  it('supports multiple loss targets (multi-loss scenario)', () => {
    const store = useBindingStore.getState()
    store.addLossBinding('loss_ce', 'label')
    store.addLossBinding('loss_mse', 'bbox')

    const bindings = store.getLossBindings()
    expect(bindings).toHaveLength(2)
    expect(store.getLossBindingForNode('loss_ce')?.sourceKey).toBe('label')
    expect(store.getLossBindingForNode('loss_mse')?.sourceKey).toBe('bbox')
  })

  it('queries bindings by target type', () => {
    const store = useBindingStore.getState()
    store.addBinding({ sourceGraph: 'data', sourceKey: 'image', target: 'model_input', targetKey: 'x' })
    store.addLossBinding('loss_1', 'label')

    expect(store.getBindingsByTarget('model_input')).toHaveLength(1)
    expect(store.getBindingsByTarget('loss_target')).toHaveLength(1)
  })
})
