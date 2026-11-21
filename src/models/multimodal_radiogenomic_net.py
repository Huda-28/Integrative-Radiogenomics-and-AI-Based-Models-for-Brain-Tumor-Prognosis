"""Phase 6 (M5): Deep Multimodal Token Fusion Network with Missing-Modality Masking & Auxiliary MGMT/IDH Heads."""
import torch
import torch.nn as nn


class M5CrossModalRadiogenomicNet(nn.Module):
    """
    M5 architecture from the study modeling ladder:
    - Input tokens for MRI habitat encoder, handcrafted radiomics, clinical, and molecular streams
    - Cross-modal multi-head attention with key_padding_mask for missing molecular modalities
    - Primary Cox proportional hazards head for Overall Survival (OS)
    - Auxiliary classification heads for MGMT methylation and IDH mutation to preserve biological signal
    """

    def __init__(
        self,
        radiomic_dim: int = 64,
        deep_mri_dim: int = 128,
        molecular_dim: int = 16,
        clinical_dim: int = 6,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.radiomic_token = nn.Sequential(nn.Linear(radiomic_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.deep_mri_token = nn.Sequential(nn.Linear(deep_mri_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.molecular_token = nn.Sequential(nn.Linear(molecular_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.clinical_token = nn.Sequential(nn.Linear(clinical_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Primary Cox OS log-hazard head
        self.cox_os_head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Optional auxiliary biological heads (MGMT promoter methylation & IDH mutation)
        self.aux_mgmt_head = nn.Linear(embed_dim * 2, 1)
        self.aux_idh_head = nn.Linear(embed_dim * 2, 1)

    def forward(
        self,
        radiomics: torch.Tensor,
        deep_mri: torch.Tensor,
        molecular: torch.Tensor,
        clinical: torch.Tensor,
        modality_missing_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t_rad = self.radiomic_token(radiomics)
        t_mri = self.deep_mri_token(deep_mri)
        t_mol = self.molecular_token(molecular)
        t_clin = self.clinical_token(clinical)

        tokens = torch.stack([t_rad, t_mri, t_mol, t_clin], dim=1)  # (B, 4, D)
        attended, attn_weights = self.cross_attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=modality_missing_mask,
        )
        fused_tokens = self.norm(tokens + attended)
        fused_flat = fused_tokens.flatten(start_dim=1)

        imaging_context = torch.cat([fused_tokens[:, 0], fused_tokens[:, 1]], dim=-1)
        return {
            "log_hazard": self.cox_os_head(fused_flat).squeeze(-1),
            "mgmt_logit": self.aux_mgmt_head(imaging_context).squeeze(-1),
            "idh_logit": self.aux_idh_head(imaging_context).squeeze(-1),
            "attention_weights": attn_weights,
        }
