"""
Vision Transformer (ViT) - 视觉Transformer
Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2021)
"""
import torch
import torch.nn as nn


# ============================================================================
# 辅助函数
# ============================================================================
def drop_path(x, drop_prob=0.0, training=False):
    """随机深度 (Stochastic Depth)"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """随机深度层"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# 图像块嵌入层
# ============================================================================
class PatchEmbed(nn.Module):
    """
    将图像分割成固定大小的块，并线性嵌入

    输入: (B, C, H, W)
    输出: (B, N, D) 其中 N = (H*W)/(P*P), P 为块大小
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 线性投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================================================
# 多头自注意力 (MSA)
# ============================================================================
class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV: (B, N, 3*embed_dim) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力应用
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# MLP (FFN) 块
# ============================================================================
class MLP(nn.Module):
    """Transformer 中的前馈网络"""
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Transformer Encoder Layer
# ============================================================================
class EncoderLayer(nn.Module):
    """单个 Transformer 编码器层"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # 注意力残差连接
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        # MLP 残差连接
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Vision Transformer 主网络
# ============================================================================
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)

    结构：
        Patch Embedding -> [CLS] token -> Positional Embedding -> Dropout
        -> Transformer Encoder (L layers)
        -> Norm -> [CLS] token -> MLP Head -> Output
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过 Transformer 编码器
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 取 [CLS] token 进行分类
        x = x[:, 0]
        x = self.head(x)

        return x


# ============================================================================
# 预设模型
# ============================================================================
def vit_tiny_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12,
        num_heads=3, num_classes=num_classes
    )


def vit_small_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12,
        num_heads=6, num_classes=num_classes
    )


def vit_base_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12,
        num_heads=12, num_classes=num_classes
    )


def vit_large_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24,
        num_heads=16, num_classes=num_classes
    )
