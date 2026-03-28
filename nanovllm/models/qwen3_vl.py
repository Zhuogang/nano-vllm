import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func

from nanovllm.models.qwen3 import Qwen3Model
from nanovllm.layers.rotary_embedding import apply_rotary_emb
from nanovllm.layers.embed_head import ParallelLMHead


# ===== Vision Encoder Components =====


def apply_rotary_pos_emb_vision(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotary position embeddings to vision features.
    x: [seq, heads, head_dim], cos/sin: [seq, head_dim/2]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def compute_vision_rotary_emb(grid_thw, head_dim, device, dtype):
    """Compute 2D rotary position embeddings for all vision patches."""
    pos_h_list, pos_w_list = [], []
    for t, h, w in grid_thw:
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w).flatten().repeat(t)
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1).flatten().repeat(t)
        pos_h_list.append(hpos)
        pos_w_list.append(wpos)

    pos_h = torch.cat(pos_h_list).float()
    pos_w = torch.cat(pos_w_list).float()

    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, 2, device=device, dtype=torch.float32) / half_dim))

    h_freqs = pos_h.unsqueeze(1) * inv_freq.unsqueeze(0)
    w_freqs = pos_w.unsqueeze(1) * inv_freq.unsqueeze(0)
    freqs = torch.cat([h_freqs, w_freqs], dim=-1)

    return freqs.cos().to(dtype), freqs.sin().to(dtype)


class Qwen3VLPatchEmbed(nn.Module):

    def __init__(self, patch_size, temporal_patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.proj = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        pixel_values = pixel_values.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        return self.proj(pixel_values.to(dtype=target_dtype)).reshape(-1, self.hidden_size)


class Qwen3VLVisionMLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = nn.GELU(approximate="tanh")
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)

        cos, sin = rotary_pos_emb
        q = apply_rotary_pos_emb_vision(q, cos, sin)
        k = apply_rotary_pos_emb_vision(k, cos, sin)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        o = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        )
        return self.proj(o.reshape(seq_len, -1))


class Qwen3VLVisionBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(self, x, cu_seqlens, rotary_pos_emb):
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLPatchMerger(nn.Module):
    """Merges spatial_merge_size x spatial_merge_size adjacent patches into one token."""

    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size, use_postshuffle_norm=False):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size * spatial_merge_size ** 2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor, grid_thw: list) -> torch.Tensor:
        sm = self.spatial_merge_size
        merged = []
        offset = 0
        for t, h, w in grid_thw:
            n = t * h * w
            patches = x[offset:offset + n].view(t, h, w, -1)
            patches = patches.view(t, h // sm, sm, w // sm, sm, -1)
            patches = patches.permute(0, 1, 3, 2, 4, 5).flatten(3)
            patches = patches.flatten(0, 2)
            merged.append(patches)
            offset += n
        x = torch.cat(merged, dim=0)
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = Qwen3VLPatchEmbed(
            config.patch_size, config.temporal_patch_size,
            config.in_channels, config.hidden_size,
        )
        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.depth)
        ])
        self.merger = Qwen3VLPatchMerger(
            config.hidden_size, config.out_hidden_size, config.spatial_merge_size,
        )

        self.deepstack_visual_indexes = getattr(config, "deepstack_visual_indexes", [])
        if self.deepstack_visual_indexes:
            self.deepstack_merger_list = nn.ModuleList([
                Qwen3VLPatchMerger(
                    config.hidden_size, config.out_hidden_size, config.spatial_merge_size, use_postshuffle_norm=True,
                )
                for _ in self.deepstack_visual_indexes
            ])

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embed(pixel_values)

        rotary_pos_emb = compute_vision_rotary_emb(grid_thw, self.head_dim, x.device, x.dtype)

        seqlens = [t * h * w for t, h, w in grid_thw]
        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=x.device)
        torch.cumsum(torch.tensor(seqlens, device=x.device, dtype=torch.int32), 0, out=cu_seqlens[1:])

        deepstack_features = []
        ds_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x, cu_seqlens, rotary_pos_emb)
            if ds_idx < len(self.deepstack_visual_indexes) and i == self.deepstack_visual_indexes[ds_idx]:
                deepstack_features.append(self.deepstack_merger_list[ds_idx](x, grid_thw))
                ds_idx += 1

        visual_embeds = self.merger(x, grid_thw)
        return visual_embeds, deepstack_features


# ===== Combined Vision-Language Model =====


class Qwen3VLModel(Qwen3Model):
    """Extends Qwen3Model with a vision encoder and DeepStack injection."""

    def __init__(self, text_config, vision_config, image_token_id: int):
        super().__init__(text_config)
        self.visual = Qwen3VLVisionModel(vision_config)
        self.image_token_id = image_token_id
        self.deepstack_visual_indexes = getattr(vision_config, "deepstack_visual_indexes", [])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: list | None = None,
    ) -> torch.Tensor:
        # 1. Run vision encoder (only during prefill with images)
        if pixel_values is not None:
            visual_embeds, deepstack_features = self.visual(pixel_values, image_grid_thw)
            image_mask = (input_ids == self.image_token_id)
        else:
            visual_embeds = deepstack_features = None
            image_mask = None

        # 2. Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # 3. Scatter visual embeddings into placeholder positions
        if visual_embeds is not None:
            hidden_states[image_mask] = visual_embeds.to(hidden_states.dtype)

        # 4. Decoder layers with DeepStack injection
        residual = None
        ds_idx = 0
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if (deepstack_features and ds_idx < len(deepstack_features)
                    and i == self.deepstack_visual_indexes[ds_idx]):
                hidden_states[image_mask] = hidden_states[image_mask] + deepstack_features[ds_idx].to(hidden_states.dtype)
                ds_idx += 1

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        text_config = config.text_config
        vision_config = config.vision_config
        image_token_id = getattr(config, "image_token_id", 151655)

        self.model = Qwen3VLModel(text_config, vision_config, image_token_id)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", True):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: list | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, pixel_values, image_grid_thw)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
