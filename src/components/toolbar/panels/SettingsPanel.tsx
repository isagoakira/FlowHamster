/**
 * Settings Panel - Feature toggles
 */

import { useGraphStore } from '../../../hooks/useGraphStore'
import { FeatureToggles } from '../../../hooks/useGraphStore'
import { panelContainerStyle } from '../styles/toolbarSharedStyles'

const FEATURE_FLAGS: { key: keyof FeatureToggles; label: string; desc: string }[] = [
  { key: 'tensorPreview', label: '张量预览（Preview Output）', desc: '节点右键菜单 + Toolbar ▶ Run Preview，实时显示输出 shape/mean/std' },
  { key: 'gradientViz', label: '梯度可视化（Gradient Analysis）', desc: '分析梯度流向，识别梯度消失/爆炸风险' },
  { key: 'multiOutput', label: '多任务/多输出分支着色', desc: '在 Canvas 上用颜色区分不同输出分支路径' },
  { key: 'evaluationNodes', label: '评估节点（Loss / Optimizer）', desc: '在节点面板显示 CrossEntropyLoss / Adam / SGD 等评估相关节点' },
]

interface SettingsPanelProps {
  onClose: () => void
}

export function SettingsPanel({ onClose }: SettingsPanelProps) {
  const store = useGraphStore()

  return (
    <div
      style={panelContainerStyle}
      onClick={(e) => e.stopPropagation()}
    >
      <div style={{ fontSize: '13px', fontWeight: 700, color: '#a0c0ff', marginBottom: '12px', letterSpacing: '0.5px' }}>
        ⚙️ 功能开关 — Feature Toggles
      </div>
      <div style={{ fontSize: '11px', color: '#555', marginBottom: '12px' }}>
        默认全部关闭，按需开启
      </div>
      {FEATURE_FLAGS.map((flag) => (
        <div
          key={flag.key}
          onClick={() => store.toggleFeature(flag.key)}
          style={{
            padding: '10px 12px', borderRadius: '8px', cursor: 'pointer',
            marginBottom: '6px', background: '#222', border: '1px solid #333',
            transition: 'border-color 0.15s',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '16px', height: '16px', borderRadius: '4px',
              background: store.features[flag.key] ? '#4488ff' : '#333',
              border: `1px solid ${store.features[flag.key] ? '#88aaff' : '#555'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '10px', color: '#fff', flexShrink: 0,
            }}>
              {store.features[flag.key] ? '☑' : '☐'}
            </div>
            <span style={{ fontSize: '12px', fontWeight: 600, color: store.features[flag.key] ? '#c0d8ff' : '#888' }}>
              {flag.label}
            </span>
          </div>
          <div style={{ fontSize: '10px', color: '#555', marginTop: '4px', marginLeft: '24px', lineHeight: '1.4' }}>
            {flag.desc}
          </div>
        </div>
      ))}
      <button
        onClick={onClose}
        style={{ marginTop: '8px', width: '100%', background: '#2a2a3a', border: '1px solid #444',
          borderRadius: '6px', color: '#ccc', padding: '6px', cursor: 'pointer', fontSize: '12px' }}
      >关闭</button>
    </div>
  )
}
