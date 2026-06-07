"""Photon-AMD end-to-end pipeline benchmark.

Compares blocking (1 slot) vs pipelined (2 slots + async copy + zombies)
using real GPU matmuls to simulate decode forward work.

All kernels are warmed up before measurement to exclude first-time
compilation overhead from the steady-state comparison.
"""

from __future__ import annotations

import time
import torch

from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

torch.manual_seed(42)


# ---- Pre-allocated GPU decode callback -----------------------------------


class GpuDecodeCallback:
    """Allocation-free do_decode callable with pre-computed weights."""

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_layers: int = 4,
        vocab_size: int = 1024,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        # Pre-allocate layer weights.
        self.layer_weights = [
            torch.randn(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device)
            for _ in range(num_layers)
        ]

        # Pre-allocate a scratch buffer for hidden states.
        self.scratch = torch.empty(
            (128, hidden_dim), dtype=self.dtype, device=self.device
        )

    def warmup(self, batch_size: int = 4, iters: int = 5):
        """Warm up all kernels used in the hot path."""
        stream = torch.cuda.Stream(device=self.device)
        for _ in range(iters):
            with torch.cuda.stream(stream):
                self.scratch[:batch_size].normal_(0, 1)
                h = self.scratch[:batch_size]
                for w in self.layer_weights:
                    h = h @ w
        stream.synchronize()

    def __call__(self, slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
        bs = batch.batch_size
        if bs == 0:
            return
        with torch.cuda.stream(stream):
            self.scratch[:bs].normal_(0, 1)
            h = self.scratch[:bs]
            for w in self.layer_weights:
                h = h @ w
            # Fill logits: first vocab_size dims from matmul, rest stay as-is.
            # The disallow mask (cleared in launch) ensures no spurious constraints.
            slot.logits[:bs, : self.vocab_size] = h[:, : self.vocab_size]


# ---- Benchmark -----------------------------------------------------------


def run_bench(
    label: str,
    num_slots: int,
    gpu_decode: GpuDecodeCallback,
    num_requests: int,
    output_len: int,
    enable_zombies: bool,
) -> dict:
    config = PhotonConfig(
        max_batch_size=num_requests,
        num_slots=num_slots,
        vocab_size=gpu_decode.vocab_size,
        hidden_dim=gpu_decode.hidden_dim,
        enable_zombies=enable_zombies,
        capture_graphs=False,
        use_async_copy=(num_slots >= 2),
    )
    callbacks = PipelineCallbacks(do_decode=gpu_decode)
    engine = PhotonEngine(config, callbacks)

    for _ in range(num_requests):
        engine.submit(prompt=[1, 2, 3, 4, 5], max_new_tokens=output_len)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(v) for v in engine.collect_results().values())
    tok_s = total_tokens / elapsed if elapsed > 0 else 0
    avg_ms = elapsed / total_tokens * 1000 if total_tokens > 0 else 0

    return {
        "label": label,
        "num_slots": num_slots,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
        "tok_per_sec": tok_s,
        "avg_ms_per_tok": avg_ms,
    }


def main():
    hidden_dim = 4096
    num_layers = 4
    vocab_size = 1024
    num_requests = 4
    output_len = 64

    print("Initializing GPU decode callback...", end=" ", flush=True)
    cb = GpuDecodeCallback(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    print(f"({len(cb.layer_weights)} layers, {hidden_dim}×{hidden_dim} FP16)")

    print("Warming up kernels...", end=" ", flush=True)
    cb.warmup(batch_size=num_requests, iters=5)
    print("done.\n")

    # GPU info
    try:
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}  ({mem:.0f} GB VRAM)")
    except Exception:
        print("GPU: AMD (detection via torch.cuda failed)")

    print(f"\nBenchmark: {num_requests} requests × {output_len} tokens each")
    print(f"Model: {num_layers} layers × ({hidden_dim}×{hidden_dim}) FP16 matmuls\n")

    results = []

    for label, ns, z in [
        ("Blocking              (1 slot, no overlap)", 1, False),
        ("Pipelined             (2 slots + async copy)", 2, True),
    ]:
        r = run_bench(label, ns, cb, num_requests, output_len, z)
        results.append(r)
        print(f"  {r['label'][:35]:<35s} "
              f"{r['elapsed_s']:7.4f}s  "
              f"{r['tok_per_sec']:8.1f} tok/s  "
              f"{r['avg_ms_per_tok']:6.2f} ms/tok  "
              f"({r['total_tokens']} tokens)")

    # ---- Speedup -----------------------------------------------------------
    if results[0]["tok_per_sec"] > 0:
        speedup = (results[1]["tok_per_sec"] / results[0]["tok_per_sec"] - 1) * 100
        print(f"\n{'─' * 70}")
        print(f"  Pipeline speedup:  {speedup:+.1f}%")
        print(f"  Time saved:        {results[0]['elapsed_s'] - results[1]['elapsed_s']:.4f}s")
        print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
