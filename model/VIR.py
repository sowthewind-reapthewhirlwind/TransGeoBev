import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# RealFormer components, adapted and simplified for integration
class ResidualMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        d_head, remainder = divmod(d_model, num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(d_head)
        self.kqv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev=None):
        batch_size, seq_len, _ = x.shape
        kqv = self.kqv_proj(x)
        key, query, value = torch.chunk(kqv, 3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        energy = self.scale * torch.matmul(query, key)
        if prev is not None:
            energy = energy + prev
        attn = F.softmax(energy, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.reshape(batch_size, seq_len, -1)
        out = self.dropout(self.out_proj(context))
        return out, energy

class TransformerBlockWithResidualAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ResidualMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, prev_attn=None):
        residual = x
        x, attn_scores = self.attn(self.norm1(x), prev_attn)
        x = x + residual
        x = self.mlp(self.norm2(x)) + x
        return x, attn_scores


# Assuming we are modifying an existing DistilledVisionTransformer class to incorporate TransformerBlockWithResidualAttention
class DistilledVisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_classes, depth, num_heads, mlp_ratio, qkv_bias, norm_layer, dropout=0.1):
        super().__init__()
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Class and distillation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks with residual attention
        self.blocks = nn.ModuleList([
            TransformerBlockWithResidualAttention(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Prediction heads
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # convert images to patches
        x = x.flatten(2).transpose(1, 2)  # flatten and transpose

        # expand class and distillation token and concatenate
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        # Pass through transformer blocks with residual attention handling
        prev_attn = None
        for blk in self.blocks:
            x, prev_attn = blk(x, prev_attn)

        x = self.norm(x)
        x_cls = self.head(x[:, 0])
        x_dist = self.head_dist(x[:, 1])

        # Returning the average of the class and distillation token outputs
        return (x_cls + x_dist) / 2

# Correcting the register_model decorator to properly handle the model function
def register_model(func):
    def wrapper(*args, **kwargs):
        print(f"Model {func.__name__} registered successfully.")
        return func(*args, **kwargs)
    return wrapper

@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=(224, 224), patch_size=16, embed_dim=384, num_classes=1000, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs
    )
    if pretrained:
        # Load pretrained weights, omitted for brevity
        pass
    return model

# Example of registering and instantiating the model
model_instance = deit_small_distilled_patch16_224()  # Register and instantiate the model
model_instance

