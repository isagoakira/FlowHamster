import { DataNodeType } from '../types/dataGraph'

export interface DataNodeDefinition {
  type: DataNodeType
  label: string
  description: string
  category: string
  defaultParams: Record<string, string | number | boolean>
  fieldOrder?: string[]
}

export interface DataNodeCategory {
  label: string
  nodes: DataNodeDefinition[]
}

export const DATA_NODE_CATEGORIES: { [key: string]: DataNodeCategory } = {
  sources: {
    label: '📁 Sources',
    nodes: [
      {
        type: 'folder_source',
        label: 'Folder Source',
        description: 'Load samples from a folder path',
        category: 'sources',
        defaultParams: { path: './data/images', pattern: '*.jpg', recursive: false },
        fieldOrder: ['path', 'pattern', 'recursive'],
      },
      {
        type: 'csv_source',
        label: 'CSV Source',
        description: 'Read tabular features and labels from CSV',
        category: 'sources',
        defaultParams: { path: './data/train.csv', delimiter: ',', feature_columns: 'f0,f1,f2,f3', label_column: 'label' },
        fieldOrder: ['path', 'delimiter', 'feature_columns', 'label_column'],
      },
      {
        type: 'jsonl_source',
        label: 'JSONL Source',
        description: 'Read line-wise JSON records',
        category: 'sources',
        defaultParams: { path: './data/train.jsonl' },
        fieldOrder: ['path'],
      },
      {
        type: 'parquet_source',
        label: 'Parquet Source',
        description: 'Read Apache Parquet files (columnar format)',
        category: 'sources',
        defaultParams: { path: './data/train.parquet' },
        fieldOrder: ['path'],
      },
      {
        type: 'huggingface_source',
        label: 'HuggingFace Dataset',
        description: 'Load from HuggingFace datasets hub',
        category: 'sources',
        defaultParams: { dataset_name: 'mnist', split: 'train', config: '' },
        fieldOrder: ['dataset_name', 'split', 'config'],
      },
    ],
  },

  readers: {
    label: '📖 Readers',
    nodes: [
      {
        type: 'read_image',
        label: 'Read Image',
        description: 'Decode image file into tensor',
        category: 'readers',
        defaultParams: { mode: 'RGB' },
        fieldOrder: ['mode'],
      },
      {
        type: 'read_video',
        label: 'Read Video',
        description: 'Extract frames from video file',
        category: 'readers',
        defaultParams: { fps: 30, max_frames: 16, start_time: 0 },
        fieldOrder: ['fps', 'max_frames', 'start_time'],
      },
      {
        type: 'read_audio',
        label: 'Read Audio',
        description: 'Load audio file as waveform',
        category: 'readers',
        defaultParams: { sample_rate: 16000, mono: true },
        fieldOrder: ['sample_rate', 'mono'],
      },
      {
        type: 'read_text',
        label: 'Read Text',
        description: 'Read text file by lines',
        category: 'readers',
        defaultParams: { encoding: 'utf-8' },
        fieldOrder: ['encoding'],
      },
      {
        type: 'unpack',
        label: 'Unpack',
        description: 'Expand nested fields from record',
        category: 'readers',
        defaultParams: { key: 'payload' },
        fieldOrder: ['key'],
      },
    ],
  },

  field_ops: {
    label: '🔀 Field Operations',
    nodes: [
      {
        type: 'select_fields',
        label: 'Select Fields',
        description: 'Keep only selected fields',
        category: 'field_ops',
        defaultParams: { fields: 'image,label' },
        fieldOrder: ['fields'],
      },
      {
        type: 'rename_fields',
        label: 'Rename Fields',
        description: 'Rename fields using source:target pairs',
        category: 'field_ops',
        defaultParams: { mapping: 'img:image,cls:label' },
        fieldOrder: ['mapping'],
      },
      {
        type: 'filter',
        label: 'Filter',
        description: 'Filter samples by field condition',
        category: 'field_ops',
        defaultParams: { field: 'label', operator: '>=', value: 0 },
        fieldOrder: ['field', 'operator', 'value'],
      },
    ],
  },

  transforms: {
    label: '🔧 Transforms',
    nodes: [
      {
        type: 'resize',
        label: 'Resize',
        description: 'Resize image to target size',
        category: 'transforms',
        defaultParams: { size: '224,224' },
        fieldOrder: ['size'],
      },
      {
        type: 'crop',
        label: 'Crop',
        description: 'Crop image at specified region',
        category: 'transforms',
        defaultParams: { x: 0, y: 0, width: 224, height: 224, type: 'center' },
        fieldOrder: ['x', 'y', 'width', 'height', 'type'],
      },
      {
        type: 'flip',
        label: 'Flip',
        description: 'Flip image horizontally or vertically',
        category: 'transforms',
        defaultParams: { horizontal: true, vertical: false },
        fieldOrder: ['horizontal', 'vertical'],
      },
      {
        type: 'rotate',
        label: 'Rotate',
        description: 'Rotate image by degrees',
        category: 'transforms',
        defaultParams: { degrees: 0 },
        fieldOrder: ['degrees'],
      },
      {
        type: 'pad',
        label: 'Pad',
        description: 'Pad image edges',
        category: 'transforms',
        defaultParams: { padding: 4, mode: 'constant' },
        fieldOrder: ['padding', 'mode'],
      },
      {
        type: 'normalize',
        label: 'Normalize',
        description: 'Normalize tensor with mean and std',
        category: 'transforms',
        defaultParams: { mean: '0.5,0.5,0.5', std: '0.5,0.5,0.5' },
        fieldOrder: ['mean', 'std'],
      },
      {
        type: 'to_tensor',
        label: 'To Tensor',
        description: 'Convert PIL/numpy to PyTorch tensor',
        category: 'transforms',
        defaultParams: {},
        fieldOrder: [],
      },
      {
        type: 'scale',
        label: 'Scale',
        description: 'Scale values to range [min, max]',
        category: 'transforms',
        defaultParams: { min: 0, max: 1 },
        fieldOrder: ['min', 'max'],
      },
    ],
  },

  augmentation: {
    label: '🎲 Augmentation',
    nodes: [
      {
        type: 'random_horizontal_flip',
        label: 'Random H-Flip',
        description: 'Randomly flip image horizontally',
        category: 'augmentation',
        defaultParams: { p: 0.5 },
        fieldOrder: ['p'],
      },
      {
        type: 'random_vertical_flip',
        label: 'Random V-Flip',
        description: 'Randomly flip image vertically',
        category: 'augmentation',
        defaultParams: { p: 0.5 },
        fieldOrder: ['p'],
      },
      {
        type: 'random_crop',
        label: 'Random Crop',
        description: 'Random crop with optional padding',
        category: 'augmentation',
        defaultParams: { size: 224, padding: 4 },
        fieldOrder: ['size', 'padding'],
      },
      {
        type: 'random_rotation',
        label: 'Random Rotation',
        description: 'Random rotation within degrees range',
        category: 'augmentation',
        defaultParams: { degrees: 15 },
        fieldOrder: ['degrees'],
      },
      {
        type: 'color_jitter',
        label: 'Color Jitter',
        description: 'Randomly change brightness, contrast, saturation, hue',
        category: 'augmentation',
        defaultParams: { brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1 },
        fieldOrder: ['brightness', 'contrast', 'saturation', 'hue'],
      },
      {
        type: 'random_erasing',
        label: 'Random Erasing',
        description: 'Randomly erase a rectangle region',
        category: 'augmentation',
        defaultParams: { p: 0.5, scale: '0.02,0.33', ratio: '0.3,3.3' },
        fieldOrder: ['p', 'scale', 'ratio'],
      },
      {
        type: 'gaussian_blur',
        label: 'Gaussian Blur',
        description: 'Apply Gaussian blur',
        category: 'augmentation',
        defaultParams: { kernel_size: 5, sigma: '1.0,2.0' },
        fieldOrder: ['kernel_size', 'sigma'],
      },
      {
        type: 'grayscale',
        label: 'Grayscale',
        description: 'Convert image to grayscale',
        category: 'augmentation',
        defaultParams: {},
        fieldOrder: [],
      },
    ],
  },

  compose: {
    label: '🔗 Compose',
    nodes: [
      {
        type: 'compose',
        label: 'Compose',
        description: 'Compose multiple transforms (pipe-separated)',
        category: 'compose',
        defaultParams: { transforms: 'resize|normalize|to_tensor' },
        fieldOrder: ['transforms'],
      },
    ],
  },

  organization: {
    label: '📊 Data Organization',
    nodes: [
      {
        type: 'train_val_split',
        label: 'Train/Val Split',
        description: 'Split dataset into train and validation subsets',
        category: 'organization',
        defaultParams: { train_ratio: 0.8, val_ratio: 0.2, seed: 42 },
        fieldOrder: ['train_ratio', 'val_ratio', 'seed'],
      },
      {
        type: 'shuffle',
        label: 'Shuffle',
        description: 'Shuffle samples before batching',
        category: 'organization',
        defaultParams: { enabled: true, seed: 42 },
        fieldOrder: ['enabled', 'seed'],
      },
      {
        type: 'concat',
        label: 'Concat',
        description: 'Concatenate multiple data sources',
        category: 'organization',
        defaultParams: {},
        fieldOrder: [],
      },
      {
        type: 'repeat',
        label: 'Repeat',
        description: 'Repeat dataset N times',
        category: 'organization',
        defaultParams: { times: 3 },
        fieldOrder: ['times'],
      },
    ],
  },

  batch: {
    label: '📦 Batch Processing',
    nodes: [
      {
        type: 'batch',
        label: 'Batch',
        description: 'Group samples into batches',
        category: 'batch',
        defaultParams: { batch_size: 32, drop_last: false },
        fieldOrder: ['batch_size', 'drop_last'],
      },
      {
        type: 'collate',
        label: 'Collate',
        description: 'Custom collate function for batching',
        category: 'batch',
        defaultParams: { strategy: 'default' },
        fieldOrder: ['strategy'],
      },
      {
        type: 'dataloader',
        label: 'DataLoader',
        description: 'PyTorch DataLoader with workers',
        category: 'batch',
        defaultParams: { batch_size: 32, shuffle: true, num_workers: 4, pin_memory: true },
        fieldOrder: ['batch_size', 'shuffle', 'num_workers', 'pin_memory'],
      },
    ],
  },

  nlp: {
    label: '📝 NLP',
    nodes: [
      {
        type: 'tokenizer',
        label: 'Tokenizer',
        description: 'Tokenize text with specified tokenizer',
        category: 'nlp',
        defaultParams: { tokenizer_type: 'bert', max_length: 512, padding: 'max_length' },
        fieldOrder: ['tokenizer_type', 'max_length', 'padding'],
      },
      {
        type: 'truncate',
        label: 'Truncate',
        description: 'Truncate sequences to max length',
        category: 'nlp',
        defaultParams: { max_length: 512 },
        fieldOrder: ['max_length'],
      },
      {
        type: 'add_special_tokens',
        label: 'Add Special Tokens',
        description: 'Add special tokens (CLS, SEP, PAD, etc.)',
        category: 'nlp',
        defaultParams: {},
        fieldOrder: [],
      },
      {
        type: 'random_mask',
        label: 'Random Mask (MLM)',
        description: 'Randomly mask tokens for MLM pre-training',
        category: 'nlp',
        defaultParams: { mask_prob: 0.15 },
        fieldOrder: ['mask_prob'],
      },
    ],
  },

  // ============================================================
  // Audio / Speech Processing
  // ============================================================
  audio_time_freq: {
    label: '🎵 Time-Freq Transform',
    nodes: [
      {
        type: 'stft',
        label: 'STFT',
        description: 'Short-Time Fourier Transform for time-frequency representation',
        category: 'audio_time_freq',
        defaultParams: { n_fft: 2048, hop_length: 512, win_length: 2048, window: 'hann', center: true },
        fieldOrder: ['n_fft', 'hop_length', 'win_length', 'window', 'center'],
      },
      {
        type: 'istft',
        label: 'iSTFT',
        description: 'Inverse STFT for waveform reconstruction',
        category: 'audio_time_freq',
        defaultParams: { n_fft: 2048, hop_length: 512, win_length: 2048, window: 'hann', center: true },
        fieldOrder: ['n_fft', 'hop_length', 'win_length', 'window', 'center'],
      },
      {
        type: 'spectrogram',
        label: 'Spectrogram',
        description: 'Compute magnitude spectrogram (STFT + mag)',
        category: 'audio_time_freq',
        defaultParams: { n_fft: 2048, hop_length: 512, win_length: 2048, power: 2 },
        fieldOrder: ['n_fft', 'hop_length', 'win_length', 'power'],
      },
      {
        type: 'melspectrogram',
        label: 'Mel Spectrogram',
        description: 'Compute mel-scale spectrogram for perceptual audio features',
        category: 'audio_time_freq',
        defaultParams: { sample_rate: 16000, n_fft: 2048, hop_length: 512, n_mels: 80, f_min: 0, f_max: 8000 },
        fieldOrder: ['sample_rate', 'n_fft', 'hop_length', 'n_mels', 'f_min', 'f_max'],
      },
      {
        type: 'mfcc',
        label: 'MFCC',
        description: 'Mel-frequency cepstral coefficients (speech recognition features)',
        category: 'audio_time_freq',
        defaultParams: { sample_rate: 16000, n_mfcc: 13, n_fft: 2048, hop_length: 512, n_mels: 40, dct_type: 2 },
        fieldOrder: ['sample_rate', 'n_mfcc', 'n_fft', 'hop_length', 'n_mels', 'dct_type'],
      },
    ],
  },

  gammatone: {
    label: '🎼 Gammatone',
    nodes: [
      {
        type: 'gammatone',
        label: 'Gammatone Filterbank',
        description: 'Gammatone filterbank for auditory-inspired spectral analysis',
        category: 'gammatone',
        defaultParams: { sample_rate: 16000, n_filters: 64, f_min: 50, f_max: 8000, num_taps: 256 },
        fieldOrder: ['sample_rate', 'n_filters', 'f_min', 'f_max', 'num_taps'],
      },
      {
        type: 'gammatone_chroma',
        label: 'Gammatone Chroma',
        description: 'Gammatone chroma features for audio similarity',
        category: 'gammatone',
        defaultParams: { sample_rate: 16000, n_chroma: 12, n_octaves: 7, n_filters: 64 },
        fieldOrder: ['sample_rate', 'n_chroma', 'n_octaves', 'n_filters'],
      },
    ],
  },

  speech_subband: {
    label: '🔊 Sub-band Processing',
    nodes: [
      {
        type: 'subband',
        label: 'Sub-band Split',
        description: 'Split audio into multiple frequency sub-bands',
        category: 'speech_subband',
        defaultParams: { n_bands: 4, mode: 'uniform' },
        fieldOrder: ['n_bands', 'mode'],
      },
      {
        type: 'filterbank',
        label: 'Filterbank',
        description: 'Apply mel/gammatone filterbank to spectrogram',
        category: 'speech_subband',
        defaultParams: { n_filters: 80, n_fft: 2048, sample_rate: 16000, f_min: 0, f_max: 8000, filter_type: 'mel' },
        fieldOrder: ['n_filters', 'n_fft', 'sample_rate', 'f_min', 'f_max', 'filter_type'],
      },
      {
        type: 'preemphasis',
        label: 'Preemphasis',
        description: 'Apply pre-emphasis filter (enhance high frequencies)',
        category: 'speech_subband',
        defaultParams: { coef: 0.97 },
        fieldOrder: ['coef'],
      },
      {
        type: 'cmvn',
        label: 'CMVN',
        description: 'Cepstral mean and variance normalization',
        category: 'speech_subband',
        defaultParams: { norm_means: true, norm_vars: true, eps: 1e-6 },
        fieldOrder: ['norm_means', 'norm_vars', 'eps'],
      },
      {
        type: 'voice_activity_detection',
        label: 'VAD',
        description: 'Voice activity detection (energy-based)',
        category: 'speech_subband',
        defaultParams: { threshold: 0.5, frame_length: 2048, hop_length: 512, energy_type: 'rms' },
        fieldOrder: ['threshold', 'frame_length', 'hop_length', 'energy_type'],
      },
      {
        type: 'pitch_extraction',
        label: 'Pitch Extraction',
        description: 'Extract fundamental frequency (F0) contour',
        category: 'speech_subband',
        defaultParams: { sample_rate: 16000, frame_length: 2048, hop_length: 512, method: 'praat', min_f0: 50, max_f0: 500 },
        fieldOrder: ['sample_rate', 'frame_length', 'hop_length', 'method', 'min_f0', 'max_f0'],
      },
      {
        type: 'onset_detection',
        label: 'Onset Detection',
        description: 'Detect note/phoneme onset times',
        category: 'speech_subband',
        defaultParams: { method: 'spectral_flux', threshold: 0.5, backtrack: true },
        fieldOrder: ['method', 'threshold', 'backtrack'],
      },
    ],
  },

  // ============================================================
  // Tensor Operations (shared with model graph)
  // ============================================================
  tensor_ops: {
    label: '🔢 Tensor Ops',
    nodes: [
      {
        type: 'tensor_reshape',
        label: 'Reshape',
        description: 'Reshape tensor to new dimensions',
        category: 'tensor_ops',
        defaultParams: { shape: '-1' },
        fieldOrder: ['shape'],
      },
      {
        type: 'tensor_flatten',
        label: 'Flatten',
        description: 'Flatten tensor to 1D or 2D',
        category: 'tensor_ops',
        defaultParams: { start_dim: 0 },
        fieldOrder: ['start_dim'],
      },
      {
        type: 'tensor_transpose',
        label: 'Transpose',
        description: 'Swap two dimensions of tensor',
        category: 'tensor_ops',
        defaultParams: { dim0: 0, dim1: 1 },
        fieldOrder: ['dim0', 'dim1'],
      },
      {
        type: 'tensor_permute',
        label: 'Permute',
        description: 'Permute tensor dimensions arbitrarily',
        category: 'tensor_ops',
        defaultParams: { dims: '0,2,1' },
        fieldOrder: ['dims'],
      },
      {
        type: 'tensor_squeeze',
        label: 'Squeeze',
        description: 'Remove dimensions of size 1',
        category: 'tensor_ops',
        defaultParams: { dim: -1 },
        fieldOrder: ['dim'],
      },
      {
        type: 'tensor_expand',
        label: 'Expand',
        description: 'Expand tensor to new shape (broadcast)',
        category: 'tensor_ops',
        defaultParams: { shape: '-1' },
        fieldOrder: ['shape'],
      },
      {
        type: 'tensor_slice',
        label: 'Slice',
        description: 'Slice tensor with start:end:step',
        category: 'tensor_ops',
        defaultParams: { start: 0, end: -1, step: 1 },
        fieldOrder: ['start', 'end', 'step'],
      },
      {
        type: 'tensor_stack',
        label: 'Stack',
        description: 'Stack tensors along new dimension',
        category: 'tensor_ops',
        defaultParams: { dim: 0 },
        fieldOrder: ['dim'],
      },
      {
        type: 'tensor_cat',
        label: 'Concatenate',
        description: 'Concatenate tensors along existing dimension',
        category: 'tensor_ops',
        defaultParams: { dim: 0 },
        fieldOrder: ['dim'],
      },
    ],
  },

  output: {
    label: '📤 Output',
    nodes: [
      {
        type: 'dataset_output',
        label: 'Dataset Output',
        description: 'Declare exported fields for model binding',
        category: 'output',
        defaultParams: { fields: 'image,label', fieldSpecs: '' },
        fieldOrder: ['fields', 'fieldSpecs'],
      },
      {
        type: 'cache',
        label: 'Cache',
        description: 'Cache processed data to disk',
        category: 'output',
        defaultParams: { cache_dir: './cache' },
        fieldOrder: ['cache_dir'],
      },
    ],
  },
}

// Flatten all categories for easy lookup
export const ALL_DATA_NODES: DataNodeDefinition[] = Object.values(DATA_NODE_CATEGORIES).flatMap(
  (cat) => cat.nodes
)

export function getDataNodeDef(type: DataNodeType): DataNodeDefinition | undefined {
  return ALL_DATA_NODES.find((n) => n.type === type)
}
