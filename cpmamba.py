"""Center-Prioritized Mamba for hyperspectral image classification."""
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm import Mamba


class SimilarityBranch_Lite(nn.Module):
    """Aggregate local features using center similarity in a reduced channel space."""

    def __init__(self, dim, kernel_size, reduction):
        super(SimilarityBranch_Lite, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        mid_dim = dim // reduction

        self.proj = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        center_h, center_w = H // 2, W // 2

        center_feat_raw = x[:, :, center_h, center_w].unsqueeze(2).unsqueeze(3)

        x_low = self.proj(x)
        c_low = x_low[:, :, center_h, center_w].unsqueeze(2).unsqueeze(3)

        x_norm = F.normalize(x_low, p=2, dim=1)
        c_norm = F.normalize(c_low, p=2, dim=1)

        sim_map = torch.sum(x_norm * c_norm, dim=1, keepdim=True)

        attention = F.softmax(sim_map.view(B, -1), dim=-1).view(B, 1, H, W)

        out = torch.sum(x * attention, dim=(2, 3))

        scale_weight = self.gate(out)
        return out * scale_weight


class CascadedSimilarityAggregation_Lite(nn.Module):
    """Combine center and multi-scale summaries into channel weights."""

    def __init__(self, dim, patch_size=11):
        super(CascadedSimilarityAggregation_Lite, self).__init__()
        self.dim = dim
        self.patch_size = patch_size

        self.branch_large = SimilarityBranch_Lite(dim, kernel_size=11, reduction=8)
        self.branch_mid = SimilarityBranch_Lite(dim, kernel_size=7, reduction=8)
        self.branch_small = SimilarityBranch_Lite(dim, kernel_size=3, reduction=8)

        self.center_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.channel_weight_gen = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(dim // 4, dim, bias=False),
            nn.Sigmoid()
        )

    def crop_center(self, x, crop_size):
        B, C, H, W = x.shape
        start = (H - crop_size) // 2
        return x[:, :, start:start + crop_size, start:start + crop_size]

    def forward(self, x):
        B, C, H, W = x.shape
        center_h, center_w = H // 2, W // 2

        feat_11 = x
        feat_7 = self.crop_center(x, 7)
        feat_3 = self.crop_center(x, 3)
        feat_c = x[:, :, center_h, center_w]

        out_11 = self.branch_large(feat_11)
        out_7 = self.branch_mid(feat_7)
        out_3 = self.branch_small(feat_3)
        out_c = self.center_proj(feat_c)

        stacked = torch.cat([out_11, out_7, out_3, out_c], dim=1)
        fused_feat = self.fusion(stacked)

        channel_weights = self.channel_weight_gen(fused_feat)

        return channel_weights.unsqueeze(2).unsqueeze(3)


class CP_Mamba(nn.Module):
    """Classify the center pixel of a hyperspectral patch."""

    def __init__(self, num_classes=9, input_channels=30, patch_size=11, dim=64, depth=1):
        super(CP_Mamba, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_pixels = patch_size * patch_size

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.spectral_stem = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

        self.cascade_sim_pre = CascadedSimilarityAggregation_Lite(dim=dim, patch_size=patch_size)

        self.pre_mamba_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.mamba_spatial = nn.ModuleList([
            Mamba(d_model=dim, d_state=16, d_conv=4, expand=2) for _ in range(depth)
        ])
        # Keep the original checkpoint keys for this inactive block.
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
            for _ in range(depth)
        ])
        self.pegs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
            for _ in range(depth)
        ])

        self.norm_layer = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def get_similarity_indices(self, x):
        B, C, H, W = x.shape
        center_h, center_w = H // 2, W // 2
        center_pixel = x[:, :, center_h, center_w].unsqueeze(2).unsqueeze(3)
        x_norm = F.normalize(x, p=2, dim=1)
        c_norm = F.normalize(center_pixel, p=2, dim=1)
        similarity = torch.sum(x_norm * c_norm, dim=1).view(B, -1)
        _, sort_idx_raw = torch.sort(similarity, descending=True, dim=1)
        _, unsort_idx_raw = torch.sort(sort_idx_raw, dim=1)
        sort_idx = sort_idx_raw.unsqueeze(-1)
        unsort_idx = unsort_idx_raw.unsqueeze(-1)
        return sort_idx, unsort_idx

    def forward(self, x):
        B, _, H, W = x.shape

        # Adaptive fusion of spatial and spectral features.
        feat_spatial = self.spatial_stem(x)
        feat_spectral = self.spectral_stem(x)
        cat_feat = torch.cat([feat_spatial, feat_spectral], dim=1)
        alpha = self.gate_conv(cat_feat)
        x_fused = alpha * feat_spatial + (1 - alpha) * feat_spectral

        # Generate channel weights from the fused patch.
        weight_pre = self.cascade_sim_pre(x_fused)

        x_mamba_in = self.pre_mamba_conv(x_fused)

        # Scan in descending center-similarity order.
        sort_idx, unsort_idx = self.get_similarity_indices(x_mamba_in)
        sort_idx_expand = sort_idx.expand(-1, -1, self.dim)
        unsort_idx_expand = unsort_idx.expand(-1, -1, self.dim)
        x_flat = x_mamba_in.flatten(2).transpose(1, 2)
        x_flat = self.norm_layer(x_flat)
        x_spatial_seq = torch.gather(x_flat, 1, sort_idx_expand)
        x_img = None

        for i in range(len(self.mamba_spatial)):
            out_spatial_sorted = self.mamba_spatial[i](x_spatial_seq)
            # Restore the patch grid before positional encoding.
            out_spatial_restored = torch.gather(out_spatial_sorted, 1, unsort_idx_expand)
            x_img = out_spatial_restored.transpose(1, 2).view(B, self.dim, H, W)
            x_img = self.pegs[i](x_img)

            if i < len(self.mamba_spatial) - 1:
                x_flat_next = x_img.flatten(2).transpose(1, 2)
                x_flat_next = self.norm_layer(x_flat_next)
                x_spatial_seq = torch.gather(x_flat_next, 1, sort_idx_expand)

        x_enhanced = x_img * weight_pre
        feat_gap = x_enhanced.flatten(2).mean(dim=2)
        logits = self.classifier(feat_gap)
        return logits
