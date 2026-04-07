/**
 * Handle ID 枚举 - 统一管理所有节点端口 ID
 * 消除字符串字面量，防止拼写错误
 */

// 基础 Handle IDs
export const HandleIds = {
  // 标准输入输出
  INPUT: 'a',           // 单输入节点
  OUTPUT: 'result',      // 标准输出

  // 复合节点
  ENCODER_X: 'x',       // TransformerEncoder 输入
  ENCODER_TGT: 'tgt',   // TransformerDecoder tgt 输入
  ENCODER_MEM: 'mem',   // TransformerDecoder memory 输入

  // Concat 节点
  CONCAT_IN_0: 'in_0',
  CONCAT_IN_1: 'in_1',
  CONCAT_IN_2: 'in_2',
  CONCAT_IN_3: 'in_3',

  // Add/Mul 节点
  ADD_A: 'a',
  ADD_B: 'b',

  // Attention 节点
  ATTN_Q: 'q',
  ATTN_K: 'k',
  ATTN_V: 'v',
  ATTN_WEIGHT: 'weight',

  // Slice 节点
  SLICE_INPUT: 'a',
} as const

// 类型别名
export type HandleId = typeof HandleIds[keyof typeof HandleIds]

// 所有 Handle ID 列表（用于验证）
export const ALL_HANDLE_IDS = Object.values(HandleIds)
