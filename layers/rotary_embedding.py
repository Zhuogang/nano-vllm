import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class MRoPEEmbedding(nn.Module):
    """Multimodal RoPE: splits head dimensions across 3 axes (temporal, height, width)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: list[int],
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.mrope_section = mrope_section
        self.interleaved = interleaved
        assert rotary_dim == head_size
        assert sum(mrope_section) == rotary_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # 1D cache for decode path (identical to standard RoPE)
        cache_1d = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache_1d, persistent=False)

        # Separate cos/sin caches for 3D M-RoPE path
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

        # Interleave permutation: chunked [TTT...HHH...WWW...] -> interleaved [THWTHWTHW...]
        if interleaved:
            perm = self._compute_interleave_perm(mrope_section)
            self.register_buffer("_perm", torch.tensor(perm, dtype=torch.long), persistent=False)

    @staticmethod
    def _compute_interleave_perm(sections: list[int]) -> list[int]:
        s0, s1, s2 = sections
        min_s = min(s1, s2)
        perm = []
        for i in range(min_s):
            perm.append(i)                   # T_i
            perm.append(s0 + i)              # H_i
            perm.append(s0 + s1 + i)         # W_i
        for i in range(min_s, s0):
            perm.append(i)                   # remaining T dims
        return perm

    @torch.compile
    def forward_1d(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

    def forward_3d(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [3, seq_len] -- (temporal, height, width)
        sec = self.mrope_section
        s0, s1 = sec[0], sec[1]

        # Each axis looks up its section of frequencies using its own positions
        cos_t = self.cos_cache[positions[0]][:, :s0]
        sin_t = self.sin_cache[positions[0]][:, :s0]
        cos_h = self.cos_cache[positions[1]][:, s0:s0 + s1]
        sin_h = self.sin_cache[positions[1]][:, s0:s0 + s1]
        cos_w = self.cos_cache[positions[2]][:, s0 + s1:]
        sin_w = self.sin_cache[positions[2]][:, s0 + s1:]

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        if self.interleaved:
            cos = cos[:, self._perm]
            sin = sin[:, self._perm]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 1:
            return self.forward_1d(positions, query, key)
        return self.forward_3d(positions, query, key)


_ROPE_CACHE: dict = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is not None:
        mrope_section = rope_scaling.get("mrope_section")
        if mrope_section:
            interleaved = rope_scaling.get("mrope_interleaved", False)
            key = (head_size, rotary_dim, max_position, base, tuple(mrope_section), interleaved)
            if key not in _ROPE_CACHE:
                _ROPE_CACHE[key] = MRoPEEmbedding(
                    head_size, rotary_dim, max_position, base, mrope_section, interleaved,
                )
            return _ROPE_CACHE[key]

    key = (head_size, rotary_dim, max_position, base)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _ROPE_CACHE[key]
