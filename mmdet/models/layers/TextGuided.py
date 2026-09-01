from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


from .vmamba import SS2D_mamba2
from .csm_triton2 import cross_scan_fn, cross_merge_fn


class TextGuidedSS2D2(SS2D_mamba2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.guide_in_proj = (
            nn.Linear(self.d_model, self.d_inner, bias=self.bias)
            if self.d_model != self.d_inner else nn.Identity()
        )

        guide_hidden_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.k_group)
        ]
        self.guide_hidden_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in guide_hidden_proj], dim=0)
        )

        forward_types = {
            "m0": partial(self.forward_corem0, force_fp32=True, dstate=self.d_state),
        }
        self.forward_core = forward_types.get(self.forward_type, None)
        self.forward = self.forward_mm

    def forward_corem0(
        self,
        x=None,
        guide=None,
        force_fp32=False,
        chunk_size=64,
        dstate=64,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]

        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _, guide_hidden_state_dim, _ = guide.shape

        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)

        xs = cross_scan_fn(
            x.view(B, H, W, RD),
            in_channel_first=False,
            out_channel_first=False,
            scans=_scan_mode,
            force_torch=scan_force_torch,
        )

        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        guide_k = guide.unsqueeze(2).repeat(1, 1, K, 1)
        guide_dbl = torch.einsum(
            "b l k d, k c d -> b l k c",
            guide_k,
            self.guide_hidden_proj_weight,
        )
        x_dbl = torch.cat((guide_dbl, x_dbl), dim=1)

        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)

        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)

        xs = torch.cat((guide_k, xs), dim=1)
        xs = xs.contiguous().view(B, L + guide_hidden_state_dim, KR, D)
        dts = dts.contiguous().view(B, L + guide_hidden_state_dim, KR)
        Bs = Bs.contiguous().view(B, L + guide_hidden_state_dim, K, N)
        Cs = Cs.contiguous().view(B, L + guide_hidden_state_dim, K, N)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        ys, final_state = mamba_chunk_scan_combined(
            xs,
            dts,
            As,
            Bs,
            Cs,
            chunk_size=chunk_size,
            D=Ds,
            dt_bias=dt_bias,
            initial_states=initial_state,
            dt_softplus=True,
            return_final_states=True,
        )

        ys = ys[:, guide_hidden_state_dim:, ...]
        final_state = final_state.view(B, K * RD, N)

        y = cross_merge_fn(
            ys.view(B, H, W, K, RD),
            in_channel_first=False,
            out_channel_first=False,
            scans=_scan_mode,
            force_torch=scan_force_torch,
        )

        if self.initial_state is not None:
            self.initial_state = nn.Parameter(
                final_state.detach().sum(0, keepdim=True),
                requires_grad=False,
            )

        y = y.view(B, H, W, -1)
        norm_dtype = self.out_norm.weight.dtype
        y = self.out_norm(y.to(norm_dtype))
        return y.to(x.dtype), final_state.detach()

    def forward_mm(self, x, guide, **kwargs):
        x = self.in_proj(x)
        guide = self.guide_in_proj(guide)

        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)

        if self.with_dconv:
            x = self.conv2d(x)

        x = self.act(x)
        guide = self.act(guide)

        y, final_state = self.forward_core(x, guide)

        y = self.out_act(y)
        final_state = self.out_act(final_state)

        if not self.disable_z:
            y = y * z

        out = self.dropout(self.out_proj(y))
        return out, final_state
