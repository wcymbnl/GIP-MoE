import torch
import torch.nn as nn
import torch.nn.functional as F


class MoAELLMPromptAdapter(nn.Module):
    def __init__(
        self,
        vis1_dim,
        vis2_dim,
        text_dim,
        llm_dim,
        moe_n_experts=4,
        dynamic_prompt_len=8,
        instruct_prompt_len=4,
        d_vis=128,
        d_text=128,
        temperature=0.1,
        freeze_expert_keys=True,
        use_noise=True,
        use_dynamic_prompt=True,
        use_instruct_prompt=True,
    ):
        super().__init__()
        self.temperature = temperature
        self.moe_n_experts = moe_n_experts
        self.use_noise = use_noise
        self.dynamic_prompt_len = dynamic_prompt_len
        self.instruct_prompt_len = instruct_prompt_len
        self.llm_dim = llm_dim

        self.use_dynamic_prompt = use_dynamic_prompt
        self.use_instruct_prompt = use_instruct_prompt

        self.vis1_proj = nn.Linear(vis1_dim, d_vis)
        self.vis2_proj = nn.Linear(vis2_dim, d_vis)
        self.text_router_proj = nn.Linear(text_dim, d_text)

        router_dim = d_vis + d_vis + d_text
        self.expert_keys = nn.Parameter(torch.empty(router_dim, moe_n_experts))
        nn.init.orthogonal_(self.expert_keys)
        self.expert_keys.requires_grad = not freeze_expert_keys

        self.all_prompt_experts = nn.Parameter(
            torch.empty(moe_n_experts, dynamic_prompt_len, llm_dim)
        )
        nn.init.normal_(self.all_prompt_experts, mean=0.0, std=0.02)

        self.instruct_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, instruct_prompt_len * llm_dim),
        )

    def forward(self, vis_feat1, vis_feat2, text_feat):
        batch_size = text_feat.size(0)
        device = text_feat.device
        dtype = text_feat.dtype

        dynamic_prompt = None
        instruct_prompt_embd = None
        moe_scores = None
        importance_loss = torch.zeros((), device=device, dtype=dtype)

        if self.use_dynamic_prompt:
            v1 = self.vis1_proj(vis_feat1)
            v2 = self.vis2_proj(vis_feat2)
            t = self.text_router_proj(text_feat)

            router_state = torch.cat([v1, v2, t], dim=-1)
            moe_logits = router_state @ self.expert_keys
            moe_logits = moe_logits / self.temperature

            if self.use_noise:
                noise = torch.randn_like(moe_logits) / (self.moe_n_experts ** 2)
                moe_logits = moe_logits + noise

            moe_scores = F.softmax(moe_logits, dim=-1)

            dynamic_prompt = torch.einsum(
                "bk,knh->bnh",
                moe_scores,
                self.all_prompt_experts,
            )

            sum_scores = moe_scores.sum(dim=0)
            importance_loss = (sum_scores.std() / sum_scores.mean().clamp_min(1e-6)) ** 2
            threshold = 0.1
            importance_loss = torch.where(
                importance_loss > threshold,
                importance_loss,
                torch.zeros_like(importance_loss),
            )

        if self.use_instruct_prompt:
            instruct_prompt_embd = self.instruct_proj(text_feat)
            instruct_prompt_embd = instruct_prompt_embd.view(
                batch_size, self.instruct_prompt_len, self.llm_dim
            )

        prompt_parts = []
        if dynamic_prompt is not None:
            prompt_parts.append(dynamic_prompt)
        if instruct_prompt_embd is not None:
            prompt_parts.append(instruct_prompt_embd)

        if len(prompt_parts) == 0:
            final_prompt = torch.zeros(
                batch_size, 0, self.llm_dim, device=device, dtype=dtype
            )
        else:
            final_prompt = torch.cat(prompt_parts, dim=1)

        return {
            "final_prompt": final_prompt,
            "dynamic_prompt": dynamic_prompt,
            "instruct_prompt_embd": instruct_prompt_embd,
            "moe_scores": moe_scores,
            "importance_loss": importance_loss,
        }
