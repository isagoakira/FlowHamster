/**
 * Training Config Panel - Training configuration settings
 */

import { useCallback, useState } from 'react'
import { useGraphStore } from '../../../hooks/useGraphStore'
import { WorkflowTrainingConfig, TaskType } from '../../../schema/workflowDocument'
import { inferTaskTypeFromModel, ModelInferenceResult } from '../../../utils/dataWorkflowCompiler'
import { trainingPanelStyle, fieldGroupStyle, labelStyle, inputStyle, codeTextareaStyle, btnStyle, dangerBtn } from '../styles/toolbarSharedStyles'

interface TrainingConfigPanelProps {
  onClose: () => void
}

export function TrainingConfigPanel({ onClose }: TrainingConfigPanelProps) {
  const trainingConfig = useGraphStore((s) => s.trainingConfig)
  const setTrainingConfig = useGraphStore((s) => s.setTrainingConfig)
  const resetTrainingConfig = useGraphStore((s) => s.resetTrainingConfig)
  const modelNodes = useGraphStore((s) => s.nodes)
  const modelEdges = useGraphStore((s) => s.edges)
  const [inferenceResult, setInferenceResult] = useState<ModelInferenceResult | null>(null)

  const updateConfig = (updater: (current: WorkflowTrainingConfig) => WorkflowTrainingConfig) => {
    setTrainingConfig(updater(trainingConfig))
  }

  const handleInferFromModel = useCallback(() => {
    const result = inferTaskTypeFromModel(modelNodes, modelEdges)
    setInferenceResult(result)
    // Auto-apply the suggested settings if confidence is high
    if (result.confidence === 'high' && result.taskType !== 'unknown') {
      updateConfig((current) => ({
        ...current,
        taskType: result.taskType as TaskType,
        loss: { ...current.loss, type: result.suggestedLoss, enabled: true },
      }))
    }
  }, [modelNodes, modelEdges])

  return (
    <div style={trainingPanelStyle} onClick={(e) => e.stopPropagation()}>
      <div style={{ fontSize: '13px', fontWeight: 700, color: '#a0c0ff', marginBottom: '12px', letterSpacing: '0.5px' }}>
        🏋️ 训练配置
      </div>
      <div style={{ fontSize: '11px', color: '#666', marginBottom: '12px', lineHeight: 1.5 }}>
        Loss / Optimizer / Scheduler 从节点面板迁移为独立配置；模型图继续只负责网络结构。
      </div>

      {/* Infer from Model button */}
      <div style={{ marginBottom: '12px' }}>
        <button
          onClick={handleInferFromModel}
          style={{
            ...btnStyle,
            width: '100%',
            background: '#1a2a3a',
            borderColor: '#3355aa',
            color: '#88aaff',
          }}
        >
          🔍 从模型推断配置
        </button>
        {inferenceResult && (
          <div style={{
            marginTop: '8px',
            padding: '8px',
            borderRadius: '6px',
            background: inferenceResult.confidence === 'high' ? '#1a2a2a' : '#231a12',
            border: `1px solid ${inferenceResult.confidence === 'high' ? '#335533' : '#5c4728'}`,
            fontSize: '11px',
          }}>
            <div style={{ color: inferenceResult.confidence === 'high' ? '#88cc88' : '#f0c27a', marginBottom: '4px' }}>
              {inferenceResult.confidence === 'high' ? '✅ 高置信度' : '⚠️ 低置信度'}: {inferenceResult.reason}
            </div>
            <div style={{ color: '#888' }}>
              推断类型: <span style={{ color: '#88aaff' }}>{inferenceResult.taskType}</span>
              {inferenceResult.outFeatures && <span> (out_features={inferenceResult.outFeatures})</span>}
            </div>
            <div style={{ color: '#888' }}>
              建议Loss: <span style={{ color: '#88ff88' }}>{inferenceResult.suggestedLoss}</span>
            </div>
            <div style={{ color: '#888' }}>
              建议Metrics: <span style={{ color: '#ffff88' }}>{inferenceResult.suggestedMetrics.join(', ')}</span>
            </div>
          </div>
        )}
      </div>

      <div style={fieldGroupStyle}>
        <label style={labelStyle}>任务类型</label>
        <select
          style={inputStyle}
          value={trainingConfig.taskType}
          onChange={(e) => updateConfig((current) => ({ ...current, taskType: e.target.value as any }))}
        >
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
          <option value="segmentation">Segmentation</option>
          <option value="detection">Detection</option>
          <option value="nlp">NLP</option>
          <option value="custom">Custom</option>
        </select>
      </div>

      <div style={fieldGroupStyle}>
        <label style={labelStyle}>Loss</label>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
          <button
            onClick={() => updateConfig((current) => ({
              ...current,
              loss: { ...current.loss, type: 'single', enabled: true, params: { lossType: current.loss.params?.lossType || 'cross_entropy', code: '' } },
            }))}
            style={{
              ...btnStyle,
              flex: 1,
              background: trainingConfig.loss.type === 'single' || !['single', 'composite'].includes(trainingConfig.loss.type) ? '#1a2a3a' : '#111',
              borderColor: trainingConfig.loss.type === 'single' || !['single', 'composite'].includes(trainingConfig.loss.type) ? '#3355aa' : '#333',
              color: trainingConfig.loss.type === 'single' || !['single', 'composite'].includes(trainingConfig.loss.type) ? '#88aaff' : '#666',
            }}
          >
            Single
          </button>
          <button
            onClick={() => updateConfig((current) => ({
              ...current,
              loss: {
                ...current.loss,
                type: 'composite',
                enabled: true,
                params: {
                  ...current.loss.params,
                  components: (current.loss.params?.components as any[])?.length ? current.loss.params.components : [
                    { type: 'cross_entropy', weight: 0.7 },
                    { type: 'mse', weight: 0.3 },
                  ],
                },
              },
            }))}
            style={{
              ...btnStyle,
              flex: 1,
              background: trainingConfig.loss.type === 'composite' ? '#1a2a3a' : '#111',
              borderColor: trainingConfig.loss.type === 'composite' ? '#3355aa' : '#333',
              color: trainingConfig.loss.type === 'composite' ? '#88aaff' : '#666',
            }}
          >
            Composite (加权多项)
          </button>
        </div>

        {/* Single Loss Mode */}
        {(trainingConfig.loss.type === 'single' || !['single', 'composite'].includes(trainingConfig.loss.type)) && (
          <div>
            <select
              style={inputStyle}
              value={String(trainingConfig.loss.params?.lossType || trainingConfig.loss.type === 'custom' ? 'custom' : 'cross_entropy')}
              onChange={(e) => updateConfig((current) => ({
                ...current,
                loss: { ...current.loss, type: e.target.value, params: { ...current.loss.params, lossType: e.target.value } },
              }))}
            >
              <optgroup label="── Classic ──">
                <option value="cross_entropy">CrossEntropyLoss</option>
                <option value="mse">MSELoss</option>
                <option value="bce">BCEWithLogitsLoss</option>
                <option value="bce_logits">BCE (sigmoid only)</option>
              </optgroup>
              <optgroup label="── CV: Segmentation ──">
                <option value="dice">DiceLoss</option>
                <option value="focal">FocalLoss</option>
                <option value="lovasz">Lovász-Softmax</option>
                <option value="tversky">TverskyLoss</option>
                <option value="iou">IoULoss</option>
                <option value="giou">GIoULoss</option>
                <option value="dice_ce">Dice + CE (Soft)</option>
              </optgroup>
              <optgroup label="── CV: Metric ──">
                <option value="msssim">MS-SSIM</option>
                <option value="perceptual">Perceptual (VGG)</option>
                <option value="content">Content Loss</option>
                <option value="style">Style Loss</option>
              </optgroup>
              <optgroup label="── CV: Detection ──">
                <option value="smooth_l1">SmoothL1 (Huber)</option>
                <option value="focal_loss">FocalLoss (RetinaNet)</option>
                <option value="class_balanced">ClassBalanced</option>
              </optgroup>
              <optgroup label="── Audio Enhancement ──">
                <option value="stft">STFT Loss</option>
                <option value="sdr">SDR (Signal Dist.)</option>
                <option value="sisdr">SISDR</option>
                <option value="mel_spec">Mel-Spectrogram</option>
                <option value="waveform">Waveform MSE</option>
                <option value="multi_res">Multi-Resolution STFT</option>
                <option value="phase">Phase Loss</option>
              </optgroup>
              <optgroup label="── Other ──">
                <option value="label_smoothing">Label Smoothing</option>
                <option value="contrastive">Contrastive</option>
                <option value="custom">Custom (自定义)</option>
              </optgroup>
            </select>
            {trainingConfig.loss.params?.lossType === 'custom' && (
              <textarea
                value={String(trainingConfig.loss.params?.code || '')}
                onChange={(e) => updateConfig((current) => ({
                  ...current,
                  loss: { ...current.loss, params: { ...current.loss.params, code: e.target.value } },
                }))}
                placeholder={"# 自定义损失函数\nloss_fn = nn.CrossEntropyLoss()  # 示例"}
                style={{ ...codeTextareaStyle, marginTop: '6px' }}
              />
            )}
          </div>
        )}

        {/* Composite Loss Mode */}
        {trainingConfig.loss.type === 'composite' && (() => {
          const components = (trainingConfig.loss.params?.components || []) as Array<{ type: string; weight: number; customCode?: string }>
          return (
            <div style={{ border: '1px solid #333', borderRadius: '6px', padding: '8px', background: '#0a0a0a' }}>
              {components.map((comp, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '6px', marginBottom: '6px', alignItems: 'center' }}>
                  <select
                    style={{ ...inputStyle, width: '120px' }}
                    value={comp.type}
                    onChange={(e) => {
                      const newComponents = [...components]
                      newComponents[idx] = { ...newComponents[idx], type: e.target.value }
                      updateConfig((current) => ({
                        ...current,
                        loss: { ...current.loss, params: { ...current.loss.params, components: newComponents } },
                      }))
                    }}
                  >
                    <optgroup label="Classic">
                      <option value="cross_entropy">CrossEntropy</option>
                      <option value="mse">MSE</option>
                      <option value="bce">BCE</option>
                    </optgroup>
                    <optgroup label="CV Seg">
                      <option value="dice">Dice</option>
                      <option value="focal">Focal</option>
                      <option value="lovasz">Lovász</option>
                      <option value="tversky">Tversky</option>
                      <option value="iou">IoU</option>
                      <option value="giou">GIoU</option>
                    </optgroup>
                    <optgroup label="CV Metric">
                      <option value="msssim">MS-SSIM</option>
                      <option value="perceptual">Perceptual</option>
                    </optgroup>
                    <optgroup label="Audio">
                      <option value="stft">STFT</option>
                      <option value="sdr">SDR</option>
                      <option value="sisdr">SISDR</option>
                      <option value="mel_spec">Mel-Spec</option>
                    </optgroup>
                    <optgroup label="Other">
                      <option value="label_smoothing">Label Smoothing</option>
                      <option value="custom">Custom</option>
                    </optgroup>
                  </select>
                  <input
                    style={{ ...inputStyle, width: '70px', textAlign: 'center' }}
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={comp.weight}
                    onChange={(e) => {
                      const newComponents = [...components]
                      newComponents[idx] = { ...newComponents[idx], weight: parseFloat(e.target.value) || 0 }
                      updateConfig((current) => ({
                        ...current,
                        loss: { ...current.loss, params: { ...current.loss.params, components: newComponents } },
                      }))
                    }}
                  />
                  <span style={{ fontSize: '11px', color: '#666', width: '20px' }}>×</span>
                  <button
                    onClick={() => {
                      const newComponents = components.filter((_, i) => i !== idx)
                      updateConfig((current) => ({
                        ...current,
                        loss: { ...current.loss, params: { ...current.loss.params, components: newComponents } },
                      }))
                    }}
                    style={{ ...dangerBtn, padding: '4px 8px', fontSize: '11px' }}
                  >
                    ✕
                  </button>
                  {comp.type === 'custom' && (
                    <input
                      style={{ ...inputStyle, flex: 1, fontSize: '10px' }}
                      placeholder="loss_fn = ..."
                      value={comp.customCode || ''}
                      onChange={(e) => {
                        const newComponents = [...components]
                        newComponents[idx] = { ...newComponents[idx], customCode: e.target.value }
                        updateConfig((current) => ({
                          ...current,
                          loss: { ...current.loss, params: { ...current.loss.params, components: newComponents } },
                        }))
                      }}
                    />
                  )}
                </div>
              ))}
              <button
                onClick={() => updateConfig((current) => ({
                  ...current,
                  loss: {
                    ...current.loss,
                    params: {
                      ...current.loss.params,
                      components: [...components, { type: 'cross_entropy', weight: 0.5 }],
                    },
                  },
                }))}
                style={{ ...btnStyle, width: '100%', marginTop: '4px', fontSize: '11px', background: '#1a1a2a', borderColor: '#444' }}
              >
                + 添加损失项
              </button>
              <div style={{ marginTop: '6px', fontSize: '10px', color: '#555', fontFamily: 'Monaco, Consolas, monospace' }}>
                total_loss = Σ(weight_i × loss_i)
              </div>
            </div>
          )
        })()}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Optimizer</label>
          <select
            style={inputStyle}
            value={trainingConfig.optimizer.type}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              optimizer: { ...current.optimizer, type: e.target.value, enabled: true },
            }))}
          >
            <option value="adamw">AdamW</option>
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
            <option value="rmsprop">RMSprop</option>
          </select>
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Scheduler</label>
          <select
            style={inputStyle}
            value={trainingConfig.scheduler.enabled ? trainingConfig.scheduler.type : 'none'}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              scheduler: {
                ...current.scheduler,
                enabled: e.target.value !== 'none',
                type: e.target.value === 'none' ? current.scheduler.type : e.target.value,
              },
            }))}
          >
            <option value="none">None</option>
            <option value="cosine_annealing">CosineAnnealing</option>
            <option value="step_lr">StepLR</option>
            <option value="reduce_on_plateau">ReduceLROnPlateau</option>
          </select>
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Device</label>
          <select
            style={inputStyle}
            value={trainingConfig.runtime.device}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              runtime: { ...current.runtime, device: e.target.value },
            }))}
          >
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
          </select>
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Epochs</label>
          <input
            style={inputStyle}
            type="number"
            min={1}
            value={trainingConfig.runtime.epochs}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              runtime: { ...current.runtime, epochs: Math.max(1, Number(e.target.value) || 1) },
            }))}
          />
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Batch Size</label>
          <input
            style={inputStyle}
            type="number"
            min={1}
            value={trainingConfig.runtime.batchSize}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              runtime: { ...current.runtime, batchSize: Math.max(1, Number(e.target.value) || 1) },
            }))}
          />
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Learning Rate</label>
          <input
            style={inputStyle}
            type="number"
            step="0.0001"
            min={0}
            value={Number(trainingConfig.optimizer.params.lr ?? 0.001)}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              optimizer: {
                ...current.optimizer,
                params: { ...current.optimizer.params, lr: Number(e.target.value) || 0.001 },
              },
            }))}
          />
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Weight Decay</label>
          <input
            style={inputStyle}
            type="number"
            step="0.0001"
            min={0}
            value={Number(trainingConfig.optimizer.params.weight_decay ?? 0)}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              optimizer: {
                ...current.optimizer,
                params: { ...current.optimizer.params, weight_decay: Number(e.target.value) || 0 },
              },
            }))}
          />
        </div>
      </div>

      <div style={{ display: 'flex', gap: '10px', marginTop: '6px', marginBottom: '14px' }}>
        <label style={{ fontSize: '11px', color: '#999', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <input
            type="checkbox"
            checked={trainingConfig.runtime.amp}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              runtime: { ...current.runtime, amp: e.target.checked },
            }))}
          />
          AMP Mixed Precision
        </label>
        <label style={{ fontSize: '11px', color: '#999', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <input
            type="checkbox"
            checked={trainingConfig.checkpoint.enabled}
            onChange={(e) => updateConfig((current) => ({
              ...current,
              checkpoint: { ...current.checkpoint, enabled: e.target.checked },
            }))}
          />
          Checkpoint
        </label>
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <button onClick={() => resetTrainingConfig()} style={{ ...btnStyle, flex: 1 }}>
          重置默认值
        </button>
        <button onClick={onClose} style={{ ...btnStyle, flex: 1, background: '#2a2a3a', borderColor: '#444' }}>
          关闭
        </button>
      </div>

      {/* Loss 构建说明 */}
      <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #333' }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#a0c0ff', marginBottom: '8px' }}>
          📖 Loss 构建说明
        </div>
        <div style={{ fontSize: '10px', color: '#666', lineHeight: 1.6, marginBottom: '10px' }}>
          <b>统一 Loss 配置系统：</b>Loss 函数通过此面板配置，不使用图中的 Loss 节点。
          配置的 Loss 会自动集成到训练循环中。
        </div>
        <div style={{ fontSize: '10px', color: '#555', lineHeight: 1.5 }}>
          <div style={{ marginBottom: '4px' }}><span style={{ color: '#88ff88' }}>●</span> <b>Single:</b> 使用单一损失函数</div>
          <div style={{ marginBottom: '4px' }}><span style={{ color: '#ffaa44' }}>●</span> <b>Composite:</b> 加权组合多个损失函数 (Σ weight_i × loss_i)</div>
          <div><span style={{ color: '#88aaff' }}>●</span> <b>自定义:</b> 需要在代码中定义 CustomLoss 类</div>
        </div>
      </div>

      {/* Loss 构建预览 */}
      <LossBuildPreview trainingConfig={trainingConfig} />
    </div>
  )
}

// Loss 构建预览组件
function LossBuildPreview({ trainingConfig }: { trainingConfig: WorkflowTrainingConfig }) {
  const lossType: string = trainingConfig.loss.type === 'composite'
    ? 'composite'
    : (trainingConfig.loss.params?.lossType as string || trainingConfig.loss.type as string || 'cross_entropy')

  const codePreviewStyle: React.CSSProperties = {
    background: '#0a0a0a',
    border: '1px solid #222',
    borderRadius: '6px',
    padding: '10px',
    fontFamily: 'Monaco, Consolas, "Courier New", monospace',
    fontSize: '10px',
    lineHeight: 1.5,
    color: '#888',
  }

  const functionStyle = { color: '#82aaff' as const }
  const commentStyle = { color: '#546e7a' as const }

  const renderCode = () => {
    if (lossType === 'composite') {
      const components = (trainingConfig.loss.params?.components || []) as Array<{ type: string; weight: number }>
      return (
        <>
          <div><span style={commentStyle}># 1. 初始化各个 Loss 函数</span></div>
          {components.map((comp, idx) => (
            <div key={idx}>
              <span style={functionStyle}>loss_fn_{idx}</span> = {getLossClassPreview(comp.type)}
            </div>
          ))}
          <div style={{ marginTop: '8px' }}><span style={commentStyle}># 2. 前向传播</span></div>
          <div><span style={functionStyle}>output</span> = model(input)</div>
          <div style={{ marginTop: '8px' }}><span style={commentStyle}># 3. 计算加权组合 Loss</span></div>
          <div>
            loss ={' '}
            {components.map((comp, idx) => (
              <span key={idx}>
                {comp.weight} × <span style={functionStyle}>loss_fn_{idx}</span>(output, target)
                {idx < components.length - 1 ? ' + ' : ''}
              </span>
            ))}
          </div>
          <div style={{ marginTop: '8px' }}><span style={commentStyle}># 4. 梯度反传 & 优化</span></div>
          <div><span style={functionStyle}>loss.backward</span>()</div>
          <div><span style={functionStyle}>optimizer.step</span>()</div>
        </>
      )
    }

    return (
      <>
        <div><span style={commentStyle}># 1. Loss 初始化</span></div>
        <div><span style={functionStyle}>loss_fn</span> = {getLossClassPreview(lossType)}</div>
        <div style={{ marginTop: '8px' }}><span style={commentStyle}># 2. 前向传播</span></div>
        <div><span style={functionStyle}>output</span> = model(input)</div>
        <div style={{ marginTop: '8px' }}><span style={commentStyle}># 3. 计算 Loss</span></div>
        <div><span style={functionStyle}>loss</span> = <span style={functionStyle}>loss_fn</span>(output, target)</div>
        <div style={{ marginTop: '8px' }}><span style={commentStyle}># 4. 梯度反传 & 优化</span></div>
        <div><span style={functionStyle}>loss.backward</span>()</div>
        <div><span style={functionStyle}>optimizer.step</span>()</div>
      </>
    )
  }

  return (
    <div style={{ marginTop: '12px' }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: '#888', marginBottom: '6px' }}>
        🔍 Loss 构建预览
      </div>
      <div style={codePreviewStyle}>
        {renderCode()}
      </div>
      {lossType === 'custom' && (
        <div style={{ marginTop: '6px', fontSize: '10px', color: '#ffaa44' }}>
          ⚠️ 自定义 Loss 需要在右侧编辑器中编写完整的损失函数类
        </div>
      )}
    </div>
  )
}

// Loss 类名预览
function getLossClassPreview(lossType: string): string {
  const lossMap: Record<string, string> = {
    cross_entropy: 'nn.CrossEntropyLoss()',
    mse: 'nn.MSELoss()',
    bce: 'nn.BCEWithLogitsLoss()',
    bce_logits: 'nn.BCEWithLogitsLoss()',
    dice: 'DiceLoss()',
    focal: 'FocalLoss()',
    lovasz: 'LovaszLoss()',
    tversky: 'TverskyLoss()',
    iou: 'IoULoss()',
    giou: 'GIoULoss()',
    dice_ce: 'DiceCELoss()',
    msssim: 'MS_SSIMLoss()',
    perceptual: 'PerceptualLoss()',
    content: 'ContentLoss()',
    style: 'StyleLoss()',
    smooth_l1: 'nn.SmoothL1Loss()',
    focal_loss: 'FocalLoss()',
    class_balanced: 'ClassBalancedLoss()',
    stft: 'STFTLoss()',
    sdr: 'SDRLoss()',
    sisdr: 'SISDRLoss()',
    mel_spec: 'MelSpectrogramLoss()',
    waveform: 'WaveformMSELoss()',
    multi_res: 'MultiResolutionSTFTLoss()',
    phase: 'PhaseLoss()',
    label_smoothing: 'nn.LabelSmoothingLoss()',
    contrastive: 'ContrastiveLoss()',
  }
  return lossMap[lossType] || 'nn.CrossEntropyLoss()  # 默认'
}
