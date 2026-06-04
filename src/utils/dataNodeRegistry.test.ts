import { describe, expect, it } from 'vitest'
import { DATA_NODE_CATEGORIES, ALL_DATA_NODES } from './dataNodeRegistry'

describe('data node registry', () => {
  it('exposes CSV feature and label column parameters in the UI defaults', () => {
    const csvNode = DATA_NODE_CATEGORIES.sources.nodes.find((node) => node.type === 'csv_source')

    expect(csvNode).toBeDefined()
    expect(csvNode?.defaultParams).toMatchObject({
      path: './data/train.csv',
      delimiter: ',',
      feature_columns: 'f0,f1,f2,f3',
      label_column: 'label',
    })
    expect(csvNode?.fieldOrder).toEqual(['path', 'delimiter', 'feature_columns', 'label_column'])
  })

  it('includes advanced augmentation nodes', () => {
    const augNodes = DATA_NODE_CATEGORIES.advanced_aug.nodes
    expect(augNodes.map((n) => n.type)).toEqual([
      'mixup',
      'cutmix',
      'autoaugment',
      'randaugment',
      'cutout',
      'posterize',
      'solarize',
    ])
  })

  it('includes multi-source synthesis nodes', () => {
    const msNodes = DATA_NODE_CATEGORIES.multi_source.nodes
    expect(msNodes.map((n) => n.type)).toEqual([
      'zip_datasets',
      'interleave_datasets',
      'sample_from_datasets',
    ])
  })

  it('includes feature engineering nodes', () => {
    const feNodes = DATA_NODE_CATEGORIES.feature_engineering.nodes
    expect(feNodes.map((n) => n.type)).toEqual([
      'standard_scaler',
      'minmax_scaler',
      'pca',
      'normalize_features',
      'fill_missing_values',
      'one_hot_encode',
    ])
  })

  it('has unique node types across all categories', () => {
    const types = ALL_DATA_NODES.map((n) => n.type)
    const uniqueTypes = new Set(types)
    expect(uniqueTypes.size).toBe(types.length)
  })

  it('includes all existing augmentation nodes', () => {
    const augNodes = DATA_NODE_CATEGORIES.augmentation.nodes
    expect(augNodes.map((n) => n.type)).toEqual([
      'random_horizontal_flip',
      'random_vertical_flip',
      'random_crop',
      'random_rotation',
      'color_jitter',
      'random_erasing',
      'gaussian_blur',
      'grayscale',
    ])
  })
})
