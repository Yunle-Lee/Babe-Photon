"""Photon-AMD end-to-end demo with a real PyTorch Transformer.

Builds a small but realistic TransformerLM and runs it through Photon's
pipeline, comparing blocking vs pipelined decode throughput on AMD GPU.

The model is a genuine PyTorch transformer with:
  - Rotary position embeddings (RoPE)
  - Multi-head self-attention with causal mask
  - SwiGLU feed-forward network
  - RMS layer norm
  - Residual connections

All operations run asynchronously on the GPU compute stream, allowing
Photon's pipeline to hide CPU bookkeeping beneath GPU forward passes.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

torch.manual_seed(42)

# ---- Mini TransformerLM ---------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # [max_seq_len, dim/2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Apply RoPE to the last dimension of x."""
    seq_len = x.shape[1]
    cos_slice = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin_slice = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos_slice - x2 * sin_slice
    rot2 = x1 * sin_slice + x2 * cos_slice
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """x: [batch, seq=1, dim]  positions: [batch]"""
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)
        # Apply RoPE
        q = apply_rope(q, self.cos.to(q.device), self.sin.to(q.device), positions[0].item())
        k = apply_rope(k, self.cos.to(k.device), self.sin.to(k.device), positions[0].item())
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int, max_seq_len: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """A minimal but realistic transformer language model."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 2048,
        n_layers: int = 12,
        n_heads: int = 16,
        ffn_dim: int = 5632,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Single-token decode forward.

        Args:
            input_ids: [batch, 1] token ids
            positions: [batch] position indices

        Returns:
            logits: [batch, vocab_size]
        """
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return self.lm_head(x).squeeze(1)  # [batch, vocab]


# ---- Photon Integration ---------------------------------------------------


class PhotonTransformerAdapter:
    """Wraps a TransformerLM for use as Photon's do_decode callback."""

    def __init__(self, model: TransformerLM, device: torch.device):
        self.model = model
        self._device = device

    @torch.no_grad()
    def do_decode(self, slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
        """Run one decode forward for the batch."""
        bs = batch.batch_size
        if bs == 0:
            return
        with torch.cuda.stream(stream):
            input_ids = slot.decode_token_ids[:bs].unsqueeze(1)  # [bs, 1]
            positions = torch.tensor(
                [seq.seq_pos for seq in batch.sequences],
                dtype=torch.long, device=self._device,
            )
            logits = self.model(input_ids, positions)  # [bs, vocab]
            slot.logits[:bs, : logits.shape[1]] = logits


# ---- Benchmark ------------------------------------------------------------


def build_model(
    dim: int = 2048,
    n_layers: int = 8,
    n_heads: int = 16,
    ffn_dim: int = 5632,
    vocab_size: int = 1024,
) -> TransformerLM:
    model = TransformerLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=4096,
    )
    model = model.to(dtype=torch.float16, device="cuda:0")
    model.eval()
    return model


def warmup_model(model: TransformerLM, iters: int = 5):
    """Warm up all GPU kernels."""
    B = 4
    ids = torch.randint(0, 1024, (B, 1), device="cuda:0")
    pos = torch.arange(B, device="cuda:0")
    s = torch.cuda.Stream()
    for _ in range(iters):
        with torch.cuda.stream(s):
            _ = model(ids, pos)
    s.synchronize()


def run_engine(
    model: TransformerLM,
    num_requests: int,
    output_len: int,
    num_slots: int,
    enable_zombies: bool,
    vocab_size: int = 1024,
) -> tuple[float, int]:
    adapter = PhotonTransformerAdapter(model, torch.device("cuda:0"))
    config = PhotonConfig(
        max_batch_size=num_requests,
        num_slots=num_slots,
        vocab_size=vocab_size,
        enable_zombies=enable_zombies,
        capture_graphs=False,
        use_async_copy=(num_slots >= 2),
        eos_token_id=0,
    )
    callbacks = PipelineCallbacks(do_decode=adapter.do_decode)
    engine = PhotonEngine(config, callbacks)

    for _ in range(num_requests):
        engine.submit(prompt=[1, 2, 3, 4, 5], max_new_tokens=output_len)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(v) for v in engine.collect_results().values())
    return elapsed, total_tokens


def main():
    dim = 2048
    n_layers = 8
    n_heads = 16
    ffn_dim = 5632
    vocab_size = 1024
    num_requests = 2
    output_len = 128

    params = (dim * dim * n_layers * 8)  # rough param count
    print(f"Building Transformer ({n_layers} layers, dim={dim}, {n_heads} heads, ~{params/1e6:.0f}M params)")
    model = build_model(
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        ffn_dim=ffn_dim, vocab_size=vocab_size,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {model.token_embedding.weight.device}")

    print(f"Warming up kernels ({num_requests} × {output_len} tokens)...", end=" ", flush=True)
    warmup_model(model)
    print("done.")

    gpu_name = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "GPU"
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name} ({mem:.0f} GB VRAM)")
    print(f"Config: {num_requests} requests × {output_len} tokens each")
    print(f"Model: {n_layers} layers transformer, FP16\n")

    results = []
    for label, ns, z in [
        ("Blocking  (1 slot, serial)",  1, False),
        ("Pipelined (2 slots + overlap)", 2, True),
    ]:
        elapsed, tokens = run_engine(model, num_requests, output_len, ns, z, vocab_size)
        tps = tokens / elapsed if elapsed > 0 else 0
        ms = elapsed / tokens * 1000 if tokens > 0 else 0
        print(f"  {label:<35s} {elapsed:7.4f}s  {tokens:4d} tok  {tps:8.1f} tok/s  {ms:6.2f} ms/tok")
        results.append((label, elapsed, tokens, tps, ms))

    if results[0][3] > 0:
        speedup = (results[1][3] / results[0][3] - 1) * 100
        time_saved = results[0][1] - results[1][1]
        print(f"\n{'─' * 70}")
        print(f"  Pipeline speedup:   {speedup:+.1f}%")
        print(f"  Time saved:          {time_saved:.4f}s")
        print(f"  Blocking throughput: {results[0][3]:.1f} tok/s")
        print(f"  Pipelined throughput:{results[1][3]:.1f} tok/s")
        print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
