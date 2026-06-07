"""HIP graph capture and replay for the decode step.

Photon captures the decode forward as a HIP graph once per batch size,
then replays it on every tick.  This eliminates per-step kernel-launch
overhead — instead of the CPU issuing hundreds of kernel launches each
step, the GPU replays a single pre-recorded command buffer.

Motivation
----------
At decode time, each forward step may involve dozens of small GPU
kernels (attention, MLP projections, residual adds, layernorms).
Each kernel launch from the CPU incurs a fixed cost (~5–10 μs on
modern systems).  For a 40-layer model, that is hundreds of launches
per token, adding non-trivial latency to the CPU-side critical path.
HIP graphs collapse all launches into a single replay, removing the
launch overhead from the decode loop entirely.

Requirements for successful capture
------------------------------------
1. All GPU buffers accessed by the graph must be pre-allocated with
   fixed addresses — the ping-pong slots satisfy this.
2. The graph is captured using one slot's buffers; when we replay
   we swap which slot's buffers are "live" by using the slot's own
   ``decode_token_ids``, ``logits``, etc. as graph I/O.

AMD ROCm (HIP) considerations
-------------------------------
PyTorch's ``torch.cuda.CUDAGraph`` maps directly to HIP graphs on AMD
via the ROCm backend.  The API is identical — PyTorch translates CUDA
graph calls to HIP graph calls under the hood.

Known limitations on AMD gfx942 (MI300):
  - Dynamic shapes inside a captured graph are not supported.
    Workaround: capture separate graphs per batch size.
  - Host callbacks inside a capture are unsupported.
    Workaround: do all callback-style work before/after graph replay.
  - Memory used inside a graph MUST be allocated before capture.
    Workaround: all slot buffers are pre-allocated in decode_slot.py.

We avoid all three edge cases by design.
"""

from __future__ import annotations

from typing import Callable

import torch


class GraphManager:
    """Captures and replays HIP graphs for the decode forward pass.

    Usage
    -----
    During initialisation::

        gm = GraphManager()
        for bs in config.graph_batch_sizes:
            gm.capture(bs, forward_fn, stream, warmup_iters=2)

    During decode::

        gm.replay(batch_size)     # instead of launching raw kernels

    The forward function passed to ``capture`` MUST use only the slot's
    pre-allocated buffers — no allocations, no shape changes.
    """

    def __init__(self) -> None:
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def capture(
        self,
        batch_size: int,
        do_forward: Callable[[], None],
        stream: torch.cuda.Stream,
        warmup_iters: int = 2,
    ) -> None:
        """Capture a HIP graph for *batch_size*.

        Args:
            batch_size: Number of active rows in the decode batch.
            do_forward: Callable that runs one decode forward on
                the compute stream.  Must be a closure that captures
                all necessary slot buffer references.
            stream: The compute stream to capture on.
            warmup_iters: Number of warmup runs before capture to
                trigger any lazy initialisation (e.g. CUDA kernel
                autotuning, cuBLAS handle creation).
        """
        g = torch.cuda.CUDAGraph()

        # Warmup: run a few iterations to trigger any lazy initialisation
        # (kernel autotuning, allocator warmup, etc.).
        for _ in range(warmup_iters):
            do_forward()

        # Capture.
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g, stream=stream):
                do_forward()

        self._graphs[batch_size] = g

    def replay(self, batch_size: int) -> None:
        """Replay the captured graph for *batch_size* on the default
        stream.

        The caller is responsible for ensuring the correct slot's input
        buffers have been filled before calling replay, and must handle
        stream synchronisation.
        """
        g = self._graphs.get(batch_size)
        if g is None:
            raise KeyError(
                f"No captured graph for batch_size={batch_size}. "
                f"Available sizes: {sorted(self._graphs.keys())}"
            )
        g.replay()

    def has(self, batch_size: int) -> bool:
        """Check whether a graph is captured for *batch_size*."""
        return batch_size in self._graphs

    @property
    def available_sizes(self) -> list[int]:
        """Sorted list of batch sizes with captured graphs."""
        return sorted(self._graphs.keys())
