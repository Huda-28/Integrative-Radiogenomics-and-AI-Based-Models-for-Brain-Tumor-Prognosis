"""M6 SOTA Architecture: Brain MRI Foundation Backbone (LoRA) + 5-Habitat Spatial Tokens + Mamba-GMU + Discrete-Time Conformal Survival."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultimodalUnit(nn.Module):
    """Dynamic sigmoid modality gating (GMU) replacing fixed scalar auxiliary fusion."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj_img = nn.Linear(embed_dim, embed_dim)
        self.proj_mol = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, h_img: torch.Tensor, h_mol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.gate(torch.cat([h_img, h_mol], dim=-1)))
        fused = z * torch.tanh(self.proj_img(h_img)) + (1.0 - z) * torch.tanh(self.proj_mol(h_mol))
        return fused, z


class SelectiveStateSpaceBlock(nn.Module):
    """Linear-complexity O(N) Mamba-style gated state-space mixer over habitat and modality tokens."""

    def __init__(self, embed_dim: int, expand: int = 2, dropout: float = 0.15):
        super().__init__()
        inner_dim = embed_dim * expand
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim)
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, inner_dim + 1, dtype=torch.float32)))
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        u, gate = self.in_proj(x_norm).chunk(2, dim=-1)
        u_conv = F.silu(self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2))
        delta = F.softplus(self.dt_proj(u_conv))
        decay = torch.exp(-delta * torch.exp(self.A_log))

        # Causal selective recurrence across ordered habitat/modality tokens
        state = torch.zeros_like(u_conv[:, 0, :])
        outputs = []
        for t in range(u_conv.size(1)):
            state = decay[:, t, :] * state + (1.0 - decay[:, t, :]) * u_conv[:, t, :]
            outputs.append(state)
        h = torch.stack(outputs, dim=1) * F.silu(gate)
        return residual + self.drop(self.out_proj(h))


class HomoscedasticMultiTaskLoss(nn.Module):
    """Kendall-style learnable log-variance weighting balancing survival NLL and auxiliary molecular BCE."""

    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for idx, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[idx])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[idx]
        return total


class M6FoundationMambaConformalNet(nn.Module):
    """
    Upgrades M5 along 6 axes:
    1. Accepts 5 separate BraTS compartment tokens (ET, NCR, ED, TC, WT) from a Brain MRI Foundation Backbone (LoRA-adapted).
    2. Adds learnable modality-type embeddings + optional WSI Pathomics foundation token.
    3. Combines linear-time Selective State-Space mixing with Pre-Norm Cross-Attention and Gated Multimodal Unit (GMU).
    4. Outputs Discrete-Time Logistic-Hazard bins (12 intervals across 0-36m) for non-proportional survival curves.
    5. Supports MC-Dropout + split conformal calibration quantiles for 90% survival prediction intervals.
    """

    def __init__(
        self,
        foundation_habitat_dim: int = 768,
        radiomic_dim: int = 18,
        molecular_dim: int = 16,
        clinical_dim: int = 6,
        longitudinal_dim: int = 4,
        pathomics_dim: int = 512,
        embed_dim: int = 128,
        num_time_bins: int = 12,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.habitat_proj = nn.Sequential(
            nn.Linear(foundation_habitat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.radiomic_proj = nn.Sequential(nn.Linear(radiomic_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.molecular_proj = nn.Sequential(nn.Linear(molecular_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.clinical_proj = nn.Sequential(nn.Linear(clinical_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.longitudinal_proj = nn.Sequential(nn.Linear(longitudinal_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.pathomics_proj = nn.Sequential(nn.Linear(pathomics_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        # Learnable embeddings distinguishing 5 habitat regions + tabular/temporal/pathology tokens (10 tokens max)
        self.modality_type_embed = nn.Parameter(torch.randn(1, 10, embed_dim) * 0.02)

        # Virtual-biopsy heads from pooled 5-habitat imaging representation
        self.aux_mgmt_head = nn.Linear(embed_dim * 2, 1)
        self.aux_idh_head = nn.Linear(embed_dim * 2, 1)
        self.aux_pathways_head = nn.Linear(embed_dim * 2, 3)
        self.aux_proj = nn.Linear(5, embed_dim)

        self.gmu = GatedMultimodalUnit(embed_dim)
        self.mamba_block = SelectiveStateSpaceBlock(embed_dim, dropout=dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Discrete-time non-proportional hazard head (12 x 3-month bins = 36 months)
        self.discrete_hazard_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_time_bins),
        )
        self.register_buffer("conformal_quantile_q90", torch.tensor(0.12))

    def forward(
        self,
        habitat_foundation_tokens: torch.Tensor,  # (B, 5, 768) for ET, NCR, ED, TC, WT
        radiomics: torch.Tensor,
        molecular: torch.Tensor,
        clinical: torch.Tensor,
        longitudinal_delta: torch.Tensor,
        pathomics_wsi: torch.Tensor | None = None,
        modality_missing_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t_habs = self.habitat_proj(habitat_foundation_tokens)  # (B, 5, D)
        t_rad = self.radiomic_proj(radiomics)
        t_mol = self.molecular_proj(molecular)
        t_clin = self.clinical_proj(clinical)
        t_long = self.longitudinal_proj(longitudinal_delta)

        pooled_hab = t_habs.mean(dim=1)
        img_ctx = torch.cat([pooled_hab, t_rad], dim=-1)
        mgmt_logit = self.aux_mgmt_head(img_ctx)
        idh_logit = self.aux_idh_head(img_ctx)
        pathway_logits = self.aux_pathways_head(img_ctx)
        aux_emb = self.aux_proj(torch.sigmoid(torch.cat([mgmt_logit, idh_logit, pathway_logits], dim=-1)))

        # Dynamic GMU fusion replaces fixed 0.35 scaling
        t_mol_gated, gmu_gate = self.gmu(aux_emb, t_mol)

        token_list = [
            t_habs[:, 0], t_habs[:, 1], t_habs[:, 2], t_habs[:, 3], t_habs[:, 4],
            t_rad, t_mol_gated, t_clin, t_long,
        ]
        if pathomics_wsi is not None:
            token_list.append(self.pathomics_proj(pathomics_wsi))

        tokens = torch.stack(token_list, dim=1)
        tokens = tokens + self.modality_type_embed[:, : tokens.size(1), :]

        # Hybrid Mamba State-Space + Pre-Norm Cross-Attention + FFN
        tokens = self.mamba_block(tokens)
        normed = self.pre_norm(tokens)
        attn_out, attn_weights = self.cross_attn(normed, normed, normed, key_padding_mask=modality_missing_mask)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)

        # Aggregate weighted summary + clinical-molecular synergy representation
        summary_vec = torch.cat([tokens[:, :5].mean(dim=1), tokens[:, 6:].mean(dim=1)], dim=-1)
        interval_logits = self.discrete_hazard_head(summary_vec)
        interval_hazards = torch.sigmoid(interval_logits)
        survival_curve = torch.cumprod(1.0 - interval_hazards + 1e-6, dim=-1)

        q = self.conformal_quantile_q90
        return {
            "interval_hazards": interval_hazards,
            "survival_curve": survival_curve,
            "survival_lower_90": torch.clamp(survival_curve - q, 0.005, 1.0),
            "survival_upper_90": torch.clamp(survival_curve + q, 0.0, 0.995),
            "gmu_modality_gate": gmu_gate,
            "mgmt_logit": mgmt_logit.squeeze(-1),
            "idh_logit": idh_logit.squeeze(-1),
            "pathway_logits": pathway_logits,
            "attention_weights": attn_weights,
        }
