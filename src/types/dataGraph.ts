import { Edge, Node } from 'reactflow'

export type DataNodeType =
  // === Sources ===
  | 'folder_source'
  | 'csv_source'
  | 'jsonl_source'
  | 'parquet_source'
  | 'huggingface_source'
  // === Readers ===
  | 'read_image'
  | 'read_video'
  | 'read_audio'
  | 'read_text'
  | 'unpack'
  // === Field Operations ===
  | 'select_fields'
  | 'rename_fields'
  | 'filter'
  // === Image Transforms ===
  | 'resize'
  | 'crop'
  | 'flip'
  | 'rotate'
  | 'pad'
  | 'normalize'
  | 'to_tensor'
  | 'scale'
  // === Audio Augmentation ===
  | 'random_horizontal_flip'
  | 'random_vertical_flip'
  | 'random_crop'
  | 'random_rotation'
  | 'color_jitter'
  | 'random_erasing'
  | 'gaussian_blur'
  | 'grayscale'
  // === Advanced Augmentation ===
  | 'mixup'
  | 'cutmix'
  | 'autoaugment'
  | 'randaugment'
  | 'cutout'
  | 'posterize'
  | 'solarize'
  // === Compose ===
  | 'compose'
  // === Data Organization ===
  | 'train_val_split'
  | 'shuffle'
  | 'concat'
  | 'repeat'
  // === Batch ===
  | 'batch'
  | 'collate'
  | 'dataloader'
  // === NLP ===
  | 'tokenizer'
  | 'truncate'
  | 'add_special_tokens'
  | 'random_mask'
  // === Audio / Speech Processing ===
  | 'stft'
  | 'istft'
  | 'spectrogram'
  | 'melspectrogram'
  | 'mfcc'
  | 'gammatone'
  | 'gammatone_chroma'
  | 'subband'
  | 'filterbank'
  | 'preemphasis'
  | 'cmvn'
  | 'voice_activity_detection'
  | 'pitch_extraction'
  | 'onset_detection'
  // === Tensor Operations ===
  | 'tensor_reshape'
  | 'tensor_flatten'
  | 'tensor_transpose'
  | 'tensor_permute'
  | 'tensor_squeeze'
  | 'tensor_expand'
  | 'tensor_slice'
  | 'tensor_stack'
  | 'tensor_cat'
  // === Multi-source Synthesis ===
  | 'zip_datasets'
  | 'interleave_datasets'
  | 'sample_from_datasets'
  // === Feature Engineering ===
  | 'standard_scaler'
  | 'minmax_scaler'
  | 'pca'
  | 'normalize_features'
  | 'fill_missing_values'
  | 'one_hot_encode'
  // === Output ===
  | 'dataset_output'
  | 'cache'

export type FieldDtype = 'tensor' | 'scalar' | 'string' | 'image' | 'label' | 'mask' | 'text' | 'audio' | 'spectrogram'

export interface FieldSpec {
  name: string
  dtype: FieldDtype
  shapeHint?: string
}

export interface DataNodeData {
  nodeType: DataNodeType
  label: string
  params: Record<string, string | number | boolean>
  fieldOrder?: string[]
  /** Optional field specifications for dataset_output nodes (JSON string of FieldSpec[]) */
  fieldSpecs?: string
}

export type DataFlowNode = Node<DataNodeData>
export type DataFlowEdge = Edge

