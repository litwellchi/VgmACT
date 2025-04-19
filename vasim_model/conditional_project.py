import torch
import torch.nn as nn
from einops import rearrange



class ModalityCompressor(nn.Module):
    def __init__(self, input_dim, output_dim, method='mean'):
        super().__init__()
        self.method = method
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 投影器：D_i → D
        self.projector = nn.Linear(input_dim, output_dim)

        # token 压缩方式
        if method == 'attention':
            self.query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        elif method == 'mlp':
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )
        else:
            self.pool = lambda x: x.mean(dim=1, keepdim=True)  # mean pooling

    def forward(self, x):
        """
        x: (B, T_i, D_i)
        return: (B, 1, D)
        """
        if self.method == 'attention':
            B = x.size(0)
            q = self.query.expand(B, -1, -1)          # (B, 1, D_i)
            pooled, _ = self.attn(q, x, x)            # (B, 1, D_i)
        elif self.method == 'mlp':
            x = x.transpose(1, 2)                     # (B, D_i, T_i)
            pooled = self.pool(x).squeeze(-1)         # (B, D_i)
            pooled = self.mlp(pooled).unsqueeze(1)    # (B, 1, D_i)
        else:
            pooled = self.pool(x)                     # (B, 1, D_i)

        return self.projector(pooled)                 # (B, 1, D)

import torch
import torch.nn as nn
from einops import rearrange
import math

class TemporalTransformerCondition(nn.Module):
    def __init__(self, in_channels, frame_size, max_frames, hidden_size, patch_embed_dim, proj_dim, num_heads, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embed_dim = patch_embed_dim
        self.proj_dim = proj_dim
        self.max_frames = max_frames
        self.frame_size = frame_size  # (H, W)

        C, H, W = in_channels, *frame_size

        # Frame embedding: flatten each frame and map to hidden_size
        self.frame_embed = nn.Sequential(
            nn.Linear(C * H * W, patch_embed_dim),
            nn.ReLU(),
            nn.Linear(patch_embed_dim, hidden_size)
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_size))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, proj_dim)

    def forward(self, video_samples):
        """
        video_samples: (B, C, T, H, W)
        Returns: (B, 1, proj_dim)
        """
        B, C, T, H, W = video_samples.shape
        assert T <= self.max_frames, f"Input has {T} frames, but max_frames={self.max_frames}"

        x = rearrange(video_samples, 'b c t h w -> b t (c h w)')  # (B, T, C*H*W)
        x = self.frame_embed(x)  # (B, T, hidden_size)

        # Add position encoding
        pos = self.pos_embed[:, :T, :]
        x = x + pos

        x = self.temporal_transformer(x)  # (B, T, hidden_size)
        x = x.mean(dim=1)  # (B, hidden_size)
        return self.output_proj(x).unsqueeze(1)  # (B, 1, proj_dim)

    @classmethod
    def from_input_shape(cls, input_shape, proj_dim=256, target_model_size_mb=100):
        """
        Creates a model based on input shape: (C, T, H, W)
        Automatically adjusts hidden size and layers to match target size.
        """
        C, T, H, W = input_shape

        # Estimate hidden_size based on target size
        base_hidden = 512
        size_multiplier = int(math.sqrt(target_model_size_mb / 20))  # heuristic
        hidden_size = base_hidden * size_multiplier  # scale up/down
        hidden_size = min(hidden_size, 1024)  # cap to avoid overgrowth
        hidden_size = max(hidden_size, 256)

        patch_embed_dim = hidden_size // 2
        num_layers = max(2, min(8, target_model_size_mb // 20))
        num_heads = max(2, hidden_size // 128)
        max_frames = T

        return cls(
            in_channels=C,
            frame_size=(H, W),
            max_frames=max_frames,
            hidden_size=hidden_size,
            patch_embed_dim=patch_embed_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )