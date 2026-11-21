"""Phases 4-6: Habitat Radiomics + Deep 3D MRI + Genomics Cross-Modal Attention Survival Network."""
import torch
import torch.nn as nn


class CrossModalAttentionSurvivalNet(nn.Module):
    """Fuses habitat radiomics, 3D MRI encoder tokens, molecular biomarkers, and clinical covariates."""

    def __init__(
        self,
        radiomic_dim: int = 96,
        deep_mri_dim: int = 256,
        genomic_dim: int = 32,
        clinical_dim: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.radiomic_proj = nn.Sequential(
            nn.Linear(radiomic_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.deep_mri_proj = nn.Sequential(
            nn.Linear(deep_mri_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.genomic_proj = nn.Sequential(
            nn.Linear(genomic_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Cox proportional hazards log-risk survival head
        self.survival_head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        radiomics: torch.Tensor,
        deep_mri: torch.Tensor,
        genomics: torch.Tensor,
        clinical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            [
                self.radiomic_proj(radiomics),
                self.deep_mri_proj(deep_mri),
                self.genomic_proj(genomics),
                self.clinical_proj(clinical),
            ],
            dim=1,
        )  # (B, 4, D)
        attended, attn_weights = self.cross_attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attended).flatten(start_dim=1)
        log_hazard = self.survival_head(fused).squeeze(-1)
        return log_hazard, attn_weights


def cox_partial_log_likelihood_loss(
    log_hazard: torch.Tensor,
    event_time: torch.Tensor,
    event_observed: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood loss for right-censored survival data."""
    order = torch.argsort(event_time, descending=True)
    sorted_hazard = log_hazard[order]
    sorted_events = event_observed[order].float()
    log_cumsum_exp = torch.logcumsumexp(sorted_hazard, dim=0)
    uncensored_likelihood = (sorted_hazard - log_cumsum_exp) * sorted_events
    num_events = torch.clamp(sorted_events.sum(), min=1.0)
    return -uncensored_likelihood.sum() / num_events
