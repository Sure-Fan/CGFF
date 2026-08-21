import torch
import torch.nn as nn
import torch.nn.functional as F


class KV_Extension_ClearCLIP(nn.Module):
    """
    ClearCLIP-style K-V extension for multi-head attention.

    Expected input:
        q_ext: [B, num_heads, S, head_dim]
        k_ext: [B, num_heads, S, head_dim]
        v_ext: [B, num_heads, S, head_dim]

    Attention:
        attn = q_ext @ k_ext_global.T
        out  = attn @ v_ext_global

    Default output:
        [B, num_heads, S, head_dim]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_ext=None,
        k_ext=None,
        v_ext=None,
        ex_feats_grid=None,   # 兼容旧调用；如果没传 q_ext，就用 ex_feats_grid 当 q_ext
        num_heads=16,
        scale=1.0,
        lbl_grid=None,
        beta=1.2,
        gamma=3.0,
        indices=None,
        model_cfg=None,
        H=None,
        W=None,
        cutting_hp=0.0,
        temperature=1.0,
        return_heads=True,
    ):
        # ------------------------------------------------------------
        # 0. Backward compatibility
        # ------------------------------------------------------------
        if q_ext is None:
            q_ext = ex_feats_grid

        if q_ext is None:
            raise ValueError("q_ext or ex_feats_grid must be provided.")

        if k_ext is None:
            raise ValueError(
                "k_ext must be provided for real KV extension. "
                "Do not use q as key unless you explicitly want Q-Q similarity."
            )

        if v_ext is None:
            raise ValueError("v_ext must be provided.")

        if q_ext.dim() != 4:
            raise ValueError(
                f"Expected q_ext shape [B, num_heads, S, head_dim], got {q_ext.shape}"
            )

        B, H_heads, S, Dh = q_ext.shape

        if H_heads != num_heads:
            raise ValueError(
                f"num_heads mismatch: q_ext has {H_heads}, but num_heads={num_heads}"
            )

        if k_ext.shape != q_ext.shape:
            raise ValueError(
                f"k_ext shape must match q_ext. Got k_ext={k_ext.shape}, q_ext={q_ext.shape}"
            )

        if v_ext.shape != q_ext.shape:
            raise ValueError(
                f"v_ext shape must match q_ext. Got v_ext={v_ext.shape}, q_ext={q_ext.shape}"
            )

        if H is None or W is None:
            side = int(S ** 0.5)
            if side * side != S:
                raise ValueError(
                    f"S={S} cannot be reshaped as square grid. Please pass H and W."
                )
            H, W = side, side

        if H * W != S:
            raise ValueError(f"H * W must equal S. Got H={H}, W={W}, S={S}")

        if model_cfg is not None:
            beta = getattr(model_cfg, "beta", beta)
            gamma = getattr(model_cfg, "gamma", gamma)
            cutting_hp = getattr(model_cfg, "cutting_hp", cutting_hp)
            temperature = getattr(model_cfg, "temperature", temperature)

        # ------------------------------------------------------------
        # 1. Normalize q and k
        # ------------------------------------------------------------
        q = F.normalize(q_ext, dim=-1)
        k = F.normalize(k_ext, dim=-1)

        # ------------------------------------------------------------
        # 2. Reshape to per-head global memory
        #
        # q_per_head: [num_heads, B, S, Dh]
        # k_memory:   [num_heads, B*S, Dh]
        # v_memory:   [num_heads, B*S, Dh]
        # ------------------------------------------------------------
        q_per_head = q.permute(1, 0, 2, 3).contiguous()
        k_memory = k.permute(1, 0, 2, 3).contiguous().reshape(num_heads, B * S, Dh)
        v_memory = v_ext.permute(1, 0, 2, 3).contiguous().reshape(num_heads, B * S, Dh)

        # ------------------------------------------------------------
        # 3. Q-K attention
        #
        # attn_weights: [num_heads, B, S, B*S]
        # ------------------------------------------------------------
        attn_weights = torch.einsum(
            "hbsd,hnd->hbsn",
            q_per_head,
            k_memory,
        )

        attn_weights = attn_weights * scale

        # ClearCLIP-style reweighting
        attn_weights = (attn_weights - attn_weights.mean(dim=-1, keepdim=True) * beta) * gamma

        # ------------------------------------------------------------
        # 4. Optional threshold masking
        # ------------------------------------------------------------
        if cutting_hp is not None:
            max_per_row = attn_weights.max(dim=-1, keepdim=True).values

            cutting_hp_tensor = torch.as_tensor(
                cutting_hp,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )

            dynamic_cutting_hp = torch.minimum(max_per_row, cutting_hp_tensor)

            attn_weights = attn_weights.masked_fill(
                attn_weights < dynamic_cutting_hp,
                float("-inf"),
            )

        # ------------------------------------------------------------
        # 5. Softmax
        # ------------------------------------------------------------
        attn_weights = F.softmax(attn_weights / temperature, dim=-1)

        attn_weights = torch.nan_to_num(
            attn_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ------------------------------------------------------------
        # 6. Aggregate values
        #
        # attn_output: [num_heads, B, S, Dh]
        # ------------------------------------------------------------
        attn_output = torch.einsum(
            "hbsn,hnd->hbsd",
            attn_weights,
            v_memory,
        )

        # ------------------------------------------------------------
        # 7. Return
        # ------------------------------------------------------------
        if return_heads:
            return attn_output.permute(1, 0, 2, 3).contiguous()

        attn_output = attn_output.permute(1, 2, 0, 3).contiguous()
        attn_output = attn_output.reshape(B, S, num_heads * Dh)

        return attn_output