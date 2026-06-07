"""Demo: blocking vs pipelined Photon inference.

This example shows the GPU bubble and how Photon eliminates it using
simulated model decode steps.  Run on an AMD GPU with ROCm to see the
difference between single-slot (blocking) and dual-slot (pipelined) modes.
"""

from __future__ import annotations

import sys
import time

import torch

from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch


def make_simulated_decode_forward(gpu_step_ms: float = 20.0):
    """Create a do_decode callback that simulates GPU work with torch ops.

    Uses matrix-multiply on GPU to simulate the memory-bandwidth-bound
    behaviour of a real decode step, scaled to match *gpu_step_ms*.
    """
    # Pre-allocate a weight matrix on GPU to simulate weight-streaming.
    H = 4096
    weight = torch.randn(H, H, dtype=torch.float16, device="cuda:0")

    def do_decode(slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
        with torch.cuda.stream(stream):
            bs = batch.batch_size
            if bs == 0:
                return
            # Simulate attention: Q @ K^T (memory-bound)
            x = torch.randn(bs, H, dtype=torch.float16, device="cuda:0")
            y = x @ weight
            # Write results to slot's logits buffer.
            slot.logits[:bs, : y.shape[-1]] = y

    return do_decode


def make_simulated_decode_sleep(gpu_step_ms: float = 20.0):
    """Create a do_decode callback that simulates GPU work with sleep.

    This is a minimal simulation; it does not actually use the GPU.
    Use for quick smoke tests of the pipeline logic.
    """

    def do_decode(slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
        bs = batch.batch_size
        if bs == 0:
            return
        with torch.cuda.stream(stream):
            time.sleep(gpu_step_ms / 1000)
            # Fill logits with dummy data so sampling works.
            slot.logits[:bs] = torch.randn(
                bs, slot.logits.shape[1],
                dtype=slot.logits.dtype, device=slot.logits.device,
            )

    return do_decode


def run_demo(
    use_sleep: bool = True,
    num_requests: int = 4,
    output_len: int = 16,
    gpu_step_ms: float = 20.0,
    cpu_overhead_ms: float = 1.5,
) -> dict:
    """Run a comparison of blocking vs pipelined Photon inference.

    Parameters
    ----------
    use_sleep : bool
        If True, use sleep-based simulation (no real GPU work).
        If False, use torch matmul on GPU.
    num_requests : int
        Number of concurrent generation requests.
    output_len : int
        Number of tokens to generate per request.
    gpu_step_ms : float
        Simulated decode step time in ms.
    cpu_overhead_ms : float
        CPU-side overhead per tick in ms.

    Returns
    -------
    dict
        Timing results for blocking and pipelined modes.
    """
    do_decode = (
        make_simulated_decode_sleep(gpu_step_ms)
        if use_sleep
        else make_simulated_decode_forward(gpu_step_ms)
    )

    results = {}

    for mode, num_slots, enable_zombies in [
        ("blocking", 1, False),
        ("pipelined", 2, True),
    ]:
        config = PhotonConfig(
            max_batch_size=num_requests,
            num_slots=num_slots,
            enable_zombies=enable_zombies,
            capture_graphs=False,  # No graphs for demo clarity.
        )
        callbacks = PipelineCallbacks(do_decode=do_decode)
        engine = PhotonEngine(config, callbacks)

        # Submit requests.
        for _ in range(num_requests):
            engine.submit(
                prompt=[1, 2, 3, 4, 5],  # Dummy prompt.
                max_new_tokens=output_len,
            )

        t0 = time.perf_counter()
        engine.run()
        elapsed = time.perf_counter() - t0

        results[mode] = {
            "elapsed_s": elapsed,
            "total_tokens": output_len * num_requests,
            "tokens_per_second": output_len * num_requests / elapsed,
            "avg_step_ms": elapsed / output_len * 1000,
        }

    # Print comparison.
    print("=" * 60)
    print("Photon Pipeline Demo: Blocking vs Pipelined")
    print("=" * 60)
    print(f"  Requests: {num_requests}, Output length: {output_len}")
    print(f"  GPU step: {gpu_step_ms:.1f} ms, CPU overhead: {cpu_overhead_ms:.1f} ms")
    print()

    for mode in ("blocking", "pipelined"):
        r = results[mode]
        print(f"  [{mode:>10s}]  {r['tokens_per_second']:8.1f} tok/s  "
              f"{r['avg_step_ms']:8.2f} ms/step  "
              f"{r['elapsed_s']:8.3f} s total")

    speedup = (
        results["pipelined"]["tokens_per_second"]
        / results["blocking"]["tokens_per_second"]
        - 1
    ) * 100
    print(f"\n  Pipelined speedup: {speedup:+.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    use_sleep = "--real-gpu" not in sys.argv
    n_req = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out_len = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    gpu_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

    print(f"Using {'sleep-based' if use_sleep else 'real-GPU'} simulation.\n")
    run_demo(
        use_sleep=use_sleep,
        num_requests=n_req,
        output_len=out_len,
        gpu_step_ms=gpu_ms,
    )
