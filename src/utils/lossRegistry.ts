/**
 * Loss Registry — Centralized loss function class mapping and setup code
 *
 * Provides lookup for loss function classes and their custom implementations.
 */

// Loss types are defined in workflowDocument but not used directly here

/**
 * Returns the PyTorch loss class instantiation string for the given loss type
 */
export function getLossClass(lossType: string): string {
  switch (lossType) {
    // Basic / Classic
    case 'cross_entropy':
      return 'nn.CrossEntropyLoss()'
    case 'mse':
      return 'nn.MSELoss()'
    case 'bce':
    case 'bce_logits':
      return 'nn.BCEWithLogitsLoss()'
    // CV - Segmentation
    case 'dice':
      return 'DiceLoss()'
    case 'focal':
      return 'FocalLoss()'
    case 'lovasz':
      return 'LovaszLoss()'
    case 'tversky':
      return 'TverskyLoss()'
    case 'iou':
      return 'IoULoss()'
    case 'giou':
      return 'GIoULoss()'
    case 'dice_ce':
      return 'DiceCELoss()'
    // CV - Metric / Perceptual
    case 'msssim':
      return 'MS_SSIMLoss()'
    case 'perceptual':
      return 'PerceptualLoss()'
    case 'content':
      return 'ContentLoss()'
    case 'style':
      return 'StyleLoss()'
    // CV - Detection
    case 'smooth_l1':
      return 'nn.SmoothL1Loss()'
    case 'focal_loss':
      return 'FocalLoss()'
    case 'class_balanced':
      return 'ClassBalancedLoss()'
    // Audio Enhancement
    case 'stft':
      return 'STFTLoss()'
    case 'sdr':
      return 'SDRLoss()'
    case 'sisdr':
      return 'SISDRLoss()'
    case 'mel_spec':
      return 'MelSpectrogramLoss()'
    case 'waveform':
      return 'WaveformMSELoss()'
    case 'multi_res':
      return 'MultiResolutionSTFTLoss()'
    case 'phase':
      return 'PhaseLoss()'
    // NLP / Other
    case 'label_smoothing':
      return 'nn.LabelSmoothingLoss()'
    case 'contrastive':
      return 'ContrastiveLoss()'
    default:
      return 'nn.CrossEntropyLoss()'
  }
}

/**
 * Returns additional setup code (class definitions) needed for complex losses
 */
export function getLossSetupCode(lossType: string): string[] {
  switch (lossType) {
    case 'dice':
    case 'tversky':
    case 'iou':
    case 'giou':
      return [
        '# Dice/IoU losses require custom implementation',
        'class DiceLoss(nn.Module):',
        '    def __init__(self, smooth=1e-6):',
        '        super().__init__()',
        '        self.smooth = smooth',
        '    def forward(self, pred, target):',
        '        pred = F.softmax(pred, dim=1)',
        '        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()',
        '        intersection = (pred * target_one_hot).sum(dim=(2,3))',
        '        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))',
        '        iou = (2 * intersection + self.smooth) / (union + self.smooth)',
        '        return 1 - iou.mean()',
      ]
    case 'focal':
    case 'focal_loss':
      return [
        '# Focal Loss for class imbalance',
        'class FocalLoss(nn.Module):',
        '    def __init__(self, alpha=1, gamma=2):',
        '        super().__init__()',
        '        self.alpha = alpha',
        '        self.gamma = gamma',
        '    def forward(self, pred, target):',
        '        ce_loss = F.cross_entropy(pred, target, reduction="none")',
        '        pt = torch.exp(-ce_loss)',
        '        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss',
        '        return focal_loss.mean()',
      ]
    case 'lovasz':
      return [
        '# Lovász-Softmax Loss',
        'def lovasz_grad(gt_sorted):',
        '    gts = gt_sorted.sum()',
        '    intersection = gts - gt_sorted.float().cumsum(0)',
        '    union = gts + (1 - gt_sorted).float().cumsum(0)',
        '    jaccard = 1. - intersection / union',
        '    if len(jaccard) > 1:',
        '        jaccard[1:] = jaccard[1:] - jaccard[:-1]',
        '    return jaccard',
        'class LovaszLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        pred = F.softmax(pred, dim=1)',
        '        loss = sum(lovasz_grad(gt_unique) * lovasz_softmax(pred_unique, gt_unique))',
        '        return loss',
      ]
    case 'msssim':
      return [
        '# MS-SSIM Loss',
        'class MS_SSIMLoss(nn.Module):',
        '    def __init__(self, alpha=0.84):',
        '        super().__init__()',
        '        self.alpha = alpha',
        '    def forward(self, pred, target):',
        '        msssim = self.compute_msssim(pred, target)',
        '        return 1 - msssim',
        '    def compute_msssim(self, pred, target):',
        '        # Simplified MS-SSIM computation',
        '        return F.mse_loss(pred, target)',
      ]
    case 'perceptual':
      return [
        '# Perceptual Loss using VGG',
        'class PerceptualLoss(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
        '        vgg = torchvision.models.vgg16(pretrained=True).features[:16]',
        '        self.vgg = vgg.eval()',
        '        for p in self.vgg.parameters():',
        '            p.requires_grad = False',
        '    def forward(self, pred, target):',
        '        vgg_pred = self.vgg(pred)',
        '        vgg_target = self.vgg(target)',
        '        return F.mse_loss(vgg_pred, vgg_target)',
      ]
    case 'stft':
      return [
        '# STFT Loss for audio',
        'class STFTLoss(nn.Module):',
        '    def __init__(self, n_fft=2048, hop_length=512):',
        '        super().__init__()',
        '        self.n_fft = n_fft',
        '        self.hop_length = hop_length',
        '    def forward(self, pred, target):',
        '        pred_stft = torch.stft(pred.flatten(), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)',
        '        target_stft = torch.stft(target.flatten(), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)',
        '        return F.l1_loss(torch.abs(pred_stft), torch.abs(target_stft))',
      ]
    case 'sdr':
      return [
        '# SDR (Signal-to-Distortion Ratio) Loss',
        'class SDRLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        signal_power = (target ** 2).sum()',
        '        noise_power = ((pred - target) ** 2).sum()',
        '        return -10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)',
      ]
    case 'sisdr':
      return [
        '# SISDR (Scale-Invariant SDR) Loss',
        'class SISDRLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        alpha = (pred * target).sum() / ((target ** 2).sum() + 1e-8)',
        '        sdr = ((alpha * target) ** 2).sum() / (((pred - alpha * target) ** 2).sum() + 1e-8)',
        '        return -10 * torch.log10(sdr + 1e-8)',
      ]
    case 'mel_spec':
      return [
        '# Mel-Spectrogram Loss',
        'class MelSpectrogramLoss(nn.Module):',
        '    def __init__(self, sample_rate=16000, n_mels=128):',
        '        super().__init__()',
        '        self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)',
        '    def forward(self, pred, target):',
        '        mel_pred = self.mel_spec(pred)',
        '        mel_target = self.mel_spec(target)',
        '        return F.l1_loss(mel_pred, mel_target)',
      ]
    case 'multi_res':
      return [
        '# Multi-Resolution STFT Loss',
        'class MultiResolutionSTFTLoss(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
        '        self.resolutions = [(2048, 512), (1024, 256), (512, 128)]',
        '    def forward(self, pred, target):',
        '        total = 0',
        '        for n_fft, hop in self.resolutions:',
        '            stft = STFTLoss(n_fft=n_fft, hop_length=hop)',
        '            total += stft(pred, target)',
        '        return total / len(self.resolutions)',
      ]
    case 'phase':
      return [
        '# Phase Loss',
        'class PhaseLoss(nn.Module):',
        '    def forward(self, pred, target):',
        '        pred_ang = torch.angle(torch.stft(pred.flatten(), n_fft=2048, hop_length=512))',
        '        target_ang = torch.angle(torch.stft(target.flatten(), n_fft=2048, hop_length=512))',
        '        return F.l1_loss(pred_ang, target_ang)',
      ]
    case 'class_balanced':
      return [
        '# Class-Balanced Loss',
        'class ClassBalancedLoss(nn.Module):',
        '    def __init__(self, num_classes=10):',
        '        super().__init__()',
        '        self.num_classes = num_classes',
        '    def forward(self, pred, target):',
        '        beta = 0.9999',
        '        cls_counts = torch.bincount(target)',
        '        cls_weights = (1 - beta) / (1 - torch.pow(beta, cls_counts.float()))',
        '        cls_weights = cls_weights / cls_weights.sum() * self.num_classes',
        '        return F.cross_entropy(pred, target, weight=cls_weights.to(pred.device))',
      ]
    default:
      return []
  }
}

/**
 * Returns all supported loss type strings
 */
export function getAllLossTypes(): string[] {
  return [
    // Basic / Classic
    'cross_entropy',
    'mse',
    'bce',
    'bce_logits',
    // CV - Segmentation
    'dice',
    'focal',
    'lovasz',
    'tversky',
    'iou',
    'giou',
    'dice_ce',
    // CV - Metric Learning
    'msssim',
    'perceptual',
    'content',
    'style',
    // CV - Detection
    'smooth_l1',
    'focal_loss',
    'class_balanced',
    // Audio Enhancement
    'stft',
    'sdr',
    'sisdr',
    'mel_spec',
    'waveform',
    'multi_res',
    'phase',
    // NLP / Other
    'label_smoothing',
    'contrastive',
    'custom',
  ]
}
