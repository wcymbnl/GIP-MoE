import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionMask(nn.Module):

    def __init__(
        self,
        img_dim: int = 256,
        text_dim: int = 768,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_residual_gate: bool = True,
    ):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Conv2d(img_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim)
        )

        self.log_temp = nn.Parameter(torch.zeros(1))  # temp = exp(log_temp), init = 1.0

 
        self.mask_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.mask_refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.use_residual_gate = use_residual_gate

    def forward(self, image_features, text_embeddings, text_mask=None):
        """
        Args:
            image_features:  (B, C, H, W)
            text_embeddings: (B, L, D_text)
            text_mask:       (B, L), optional
                             有效 token=1, 无效 token=0

        Returns:
            guided_vision_feature: (B, C, H, W)
            routing_mask:          (B, 1, H, W)
            token_attention:       (B, H*W, L)
        """
        B, C, H, W = image_features.shape
        _, L, _ = text_embeddings.shape


        proj_img = self.img_proj(image_features)                  
        proj_img_flat = proj_img.flatten(2).transpose(1, 2)      

        proj_text = self.text_proj(text_embeddings)              

        proj_img_flat = F.normalize(proj_img_flat, p=2, dim=-1)  
        proj_text = F.normalize(proj_text, p=2, dim=-1)          

        temp = self.log_temp.exp().clamp(min=1e-4, max=100.0)
        sim_matrix = torch.einsum("bne,ble->bnl", proj_img_flat, proj_text) / temp  # (B, N, L)


        if text_mask is not None:
            valid_mask = text_mask.unsqueeze(1).bool()
            sim_matrix = sim_matrix.masked_fill(~valid_mask, -1e4)

        token_attention = F.softmax(sim_matrix, dim=-1)          


        text_context = torch.einsum("bnl,ble->bne", token_attention, proj_text)  

        fusion_feat = torch.cat([proj_img_flat, text_context], dim=-1)          

        mask_logits = self.mask_mlp(fusion_feat).squeeze(-1)    

        routing_mask = mask_logits.view(B, 1, H, W)              


        routing_mask = self.mask_refine(routing_mask)            
        routing_mask = torch.sigmoid(routing_mask)              

        gate = self.gate_scale * routing_mask

        if self.use_residual_gate:
            guided_vision_feature = image_features * (1.0 + gate)
        else:
            guided_vision_feature = image_features * gate

        return guided_vision_feature, routing_mask, token_attention


