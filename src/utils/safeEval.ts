/**
 * 安全表达式求值器
 *
 * 替代 new Function()，只支持基本算术运算和变量引用
 * 防止代码注入攻击
 */

/**
 * 安全地求值简单表达式
 * 只支持：加、减、乘、除、括号、变量引用
 */
export function safeEvaluate(
  expr: string,
  params: Record<string, number | string>
): number | string | undefined {
  // 清理表达式：移除危险字符
  const cleanExpr = expr
    .replace(/[^a-zA-Z0-9_+*/()-]/g, '')
    .trim()

  if (!cleanExpr) return undefined

  // 替换变量为实际值
  const vars = Object.keys(params)
  let evalExpr = cleanExpr

  for (const v of vars) {
    const regex = new RegExp(`\\b${v}\\b`, 'g')
    const value = params[v]
    if (typeof value === 'number') {
      evalExpr = evalExpr.replace(regex, String(value))
    } else if (typeof value === 'string') {
      // 如果变量是字符串且表达式看起来像字符串操作，才替换
      if (/^["']/.test(cleanExpr) || /["']$/.test(cleanExpr)) {
        evalExpr = evalExpr.replace(regex, value)
      }
    }
  }

  // 验证表达式只包含安全字符
  if (!/^[\d\s_+*/()-]+$/.test(evalExpr)) {
    // 包含不安全字符，拒绝求值
    return undefined
  }

  // 使用 Function 构造一个只能做算术的函数
  // 注意：这里使用严格模式，且表达式已经过清理
  try {
    // 使用 eval 的安全替代：Function 构造
    // 只允许数字和基本运算符
    const result = new Function(`return ${evalExpr}`)()
    return result
  } catch {
    return undefined
  }
}

/**
 * 解析模板变量
 * 如 "${embed_dim}" -> 512
 * 如 "${d_model * expand}" -> 1024
 */
export function resolveTemplate(
  value: number | string | boolean,
  params: Record<string, any>
): string {
  if (typeof value === 'string' && value.startsWith('${') && value.endsWith('}')) {
    const expr = value.slice(2, -1)

    // 首先尝试直接查找简单变量名
    if (params[expr] !== undefined) {
      return String(params[expr])
    }

    // 尝试安全求值
    const result = safeEvaluate(expr, params)
    if (result !== undefined) {
      return String(result)
    }

    // 回退到返回原始表达式
    return expr
  }
  return String(value)
}

/**
 * 解析字符串模板
 * 如 "(B, N, ${num_heads}, ${head_dim})" -> "(B, N, 8, 64)"
 */
export function resolveStringTemplate(
  template: string,
  params: Record<string, any>
): string {
  return template.replace(/\$\{([^}]+)\}/g, (match, expr) => {
    // 首先尝试直接查找
    if (params[expr] !== undefined) {
      return String(params[expr])
    }

    // 尝试安全求值
    const result = safeEvaluate(expr, params)
    if (result !== undefined) {
      return String(result)
    }

    // 回退到原始值
    return match
  })
}
