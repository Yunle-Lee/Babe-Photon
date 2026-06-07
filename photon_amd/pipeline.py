"""Photon pipeline: the main launch/commit/finalize orchestration.

This is where the three Photon mechanisms converge:

PING-PONG SLOTS (Mechanism 1)
  Two ``DecodeSlot`` instances alternate every tick.  While slot A's
  forward runs on the compute stream, slot B's D2H copy lands on the
  copy stream and the CPU commits its results.

FORWARD NOW, SAMPLE LATER (Mechanism 2)
  The forward for step *t+1* is launched before step *t* is committed.
  This works because the forward only needs the previous token id
  (already on GPU) — it doesn't depend on the CPU's bookkeeping.
  Sampling waits on the constrained-decode mask which is built during
  commit, establishing "commit-before-finalize" ordering for structured
  outputs.

ZOMBIES (Mechanism 3)
  A request that finished at step *t* was already baked into step
  *t+1*'s batch.  Instead of cancelling mid-flight, it rides as a
  zombie for one more forward.  The commit phase detects that it is
  already finalized and skips it.

Stream model
------------
::

    compute_stream ──┬── Forward (step N) ──┬── Forward (step N+1) ──┬── ...
                     │                      │                        │
    copy_stream    ──┴── (idle) ────────────┴── D2H copy (step N) ──┴── ...
                        wait on                wait on
                        step_done_event        step_done_event

The copy stream is independent of the compute stream, so the GPU can
start the next forward immediately — the bubble is removed.

AMD ROCm implementation notes
-------------------------------
  - ``torch.cuda.Stream`` maps to ``hipStream_t`` on AMD.
  - ``torch.cuda.Event`` maps to ``hipEvent_t``.
  - ``pin_memory=True`` maps to ``hipHostMalloc``.
  - ``torch.cuda.CUDAGraph`` maps to HIP graph APIs.
  - All GPU-side buffers are pre-allocated at engine init to avoid
    runtime ``hipMalloc`` calls (which can trigger device-wide sync).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from .config import PhotonConfig
from .decode_slot import DecodeSlot, create_decode_slot
from .graph import GraphManager
from .scheduler import Batch, RequestState, Scheduler
from .stream_manager import StreamManager

# Sentinel token ID written to sampled_ids by zombie forward steps.
# The commit phase skips rows with this value (no token appended).
_ZOMBIE_SENTINEL = -1


@dataclass
class PipelineCallbacks:
    """User-provided callbacks that define the model-specific forward pass.

    The Photon pipeline calls these at the right points in each tick;
    the user's model code fills in the actual computation.  All callbacks
    must operate on pre-allocated slot buffers — no allocations, no
    shape changes.

    Attributes
    ----------
    do_decode : Callable[[DecodeSlot, Batch, torch.cuda.Stream], None]
        REQUIRED. Run one decode forward for *batch* on *stream*,
        writing results into *slot*'s logits/hidden_last buffers.
        The function MUST only use the slot's pre-allocated buffers.

    do_prefill : Callable | None
        Optional. Run one prefill forward for *batch* on *stream*.
        Uses the same compute stream as decode so the pipeline can
        transparently interleave them.

    do_sample : Callable | None
        Optional. Sample the next token from *slot*'s logits for
        *batch*, writing to sampled_ids and sampled_logprobs.
        If not provided, argmax sampling with optional disallow_mask
        support is used by default.
    """

    do_decode: Callable[[DecodeSlot, Batch, torch.cuda.Stream], None]
    do_prefill: Optional[Callable[[DecodeSlot, Batch, torch.cuda.Stream], None]] = None
    do_sample: Optional[Callable[[DecodeSlot, Batch, torch.cuda.Stream], None]] = None

    def __post_init__(self):
        if self.do_sample is None:
            # Default to argmax with disallow_mask support.
            self.do_sample = _default_sample


def _default_sample(
    slot: DecodeSlot,
    batch: Batch,
    stream: torch.cuda.Stream,
) -> None:
    """Default token sampling: argmax with optional disallow mask."""
    with torch.cuda.stream(stream):
        bs = batch.batch_size
        if bs == 0:
            return
        # Apply disallow mask: reshape flat mask to [bs, vocab] and force
        # True positions to -inf before argmax.
        mask = slot.meta.disallow_mask.device[:bs * slot.logits.shape[1]]
        if mask.any():
            mask2d = mask.view(bs, -1)
            slot.logits[:bs].masked_fill_(mask2d, float("-inf"))
        # Argmax sample.
        slot.sampled_ids[:bs] = slot.logits[:bs].argmax(dim=-1)
        slot.sampled_logprobs[:bs] = 0.0


class PhotonEngine:
    """The main Photon inference pipeline.

    Usage
    -----
    >>> from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
    >>>
    >>> config = PhotonConfig(max_batch_size=32)
    >>> callbacks = PipelineCallbacks(do_decode=my_decode_fn)
    >>> engine = PhotonEngine(config, callbacks)
    >>> seq_id = engine.submit(prompt_tokens)
    >>> engine.run()
    >>> results = engine.collect_results()

    The engine owns:
    - Two HIP streams (compute + copy) via :class:`StreamManager`.
    - Two decode slots (ping-pong) via :func:`create_decode_slot`.
    - Optional HIP graphs via :class:`GraphManager`.
    - A :class:`Scheduler` for batch assembly and tick orchestration.

    Each call to :meth:`step` advances the pipeline by one scheduler
    tick, which runs the launch → commit → finalize phases.
    """

    def __init__(
        self,
        config: PhotonConfig,
        callbacks: PipelineCallbacks,
    ) -> None:
        self.config = config
        self.callbacks = callbacks
        self.device = config.device
        self.dtype = config.dtype

        # ---- Streams -----------------------------------------------------
        self.streams = StreamManager(self.device)

        # ---- Decode slots -------------------------------------------------
        self.slots: list[DecodeSlot] = []
        for slot_id in range(config.num_slots):
            slot = create_decode_slot(
                slot_id=slot_id,
                device=self.device,
                dtype=config.dtype,
                max_batch=config.max_batch_size,
                kv_cache_pages=config.kv_cache_pages,
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_dim,
                compute_stream=self.streams.compute_stream,
            )
            self.slots.append(slot)

        # ---- HIP graphs ---------------------------------------------------
        self.graph_manager: Optional[GraphManager] = None
        if config.capture_graphs:
            self.graph_manager = GraphManager()
            self._capture_graphs()

        # ---- Scheduler ----------------------------------------------------
        self.scheduler = Scheduler(
            config=config,
            on_launch=self._launch,
            on_commit=self._commit,
            on_finalize=self._finalize,
        )

        # ---- Metrics ------------------------------------------------------
        self._total_tokens: int = 0
        self._total_steps: int = 0
        self._total_time_s: float = 0.0
        self._results: dict[int, list[int]] = {}

    # ---- Public API -------------------------------------------------------

    def submit(self, prompt: list[int], max_new_tokens: int = 256) -> int:
        """Enqueue a generation request.

        Args:
            prompt: Token ids of the prompt.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The assigned sequence id (use with :meth:`collect_results`).
        """
        return self.scheduler.submit(prompt, max_new_tokens)

    def step(self) -> None:
        """Advance the pipeline by one scheduler tick."""
        t0 = time.perf_counter()
        self.scheduler.tick()
        self._total_time_s += time.perf_counter() - t0
        self._total_steps += 1

    def has_work(self) -> bool:
        """Return ``True`` if there are pending, active sequences, or
        in-flight batches still waiting in the queue to be committed."""
        return (
            self.scheduler.active_count() > 0
            or self.scheduler.pending_count() > 0
            or len(self.scheduler.state.batch_queue) > 0
        )

    def run(self) -> None:
        """Run the pipeline until no work remains, then drain.

        This is the convenience method for batch workloads.  For
        streaming/interactive use, call :meth:`step` manually in a loop.
        """
        while self.has_work():
            self.step()
        # Drain any remaining steps still in the batch queue.
        self.scheduler.finish()
        # Wait for all GPU work to finish.
        self.streams.sync_all()

    def collect_results(self) -> dict[int, list[int]]:
        """Return ``{seq_id: generated_token_ids}`` for all finished
        sequences."""
        return self._results

    # ---- Metrics ----------------------------------------------------------

    @property
    def tokens_per_second(self) -> float:
        """Average decode throughput in tokens/second."""
        if self._total_time_s <= 0:
            return 0.0
        return self._total_tokens / self._total_time_s

    @property
    def steps_per_second(self) -> float:
        """Average scheduler tick frequency."""
        if self._total_time_s <= 0:
            return 0.0
        return self._total_steps / self._total_time_s

    # ---- Internal: scheduler callbacks ------------------------------------

    def _launch(self, slot_id: int, batch: Batch) -> None:
        """Phase 1: launch the forward for *batch* on *slot_id*.

        1. Copy metadata H2D (batch_idx, input_pos, token ids).
        2. Run decode forward (graph replay or raw).
        3. Sample tokens.
        4. Record step_done_event on compute stream.
        """
        slot = self.slots[slot_id]
        meta = slot.meta

        # Copy batch metadata to pinned host buffers.
        for i, seq in enumerate(batch.sequences):
            meta.batch_idx.host[i] = seq.seq_id
            meta.input_pos.host[i] = seq.seq_pos
            slot.decode_token_ids[i] = seq.next_token_id()

        # Clear the disallow mask for this slot (zero → no constraints).
        # Without this, uninitialised mask bytes act as spurious constraints.
        mask_host = slot.meta.disallow_mask.host
        mask_len = batch.batch_size * self.config.vocab_size
        mask_host[:mask_len].zero_()
        slot.meta.disallow_mask.copy_host_to_device(
            mask_len, self.streams.copy_stream,
        )

        # Run the forward — graph replay when available, raw otherwise.
        if (
            self.graph_manager is not None
            and self.graph_manager.has(batch.batch_size)
        ):
            with torch.cuda.stream(slot.compute_stream):
                self.graph_manager.replay(batch.batch_size)
        else:
            self.callbacks.do_decode(slot, batch, slot.compute_stream)

        # Sample from forward output.
        self.callbacks.do_sample(slot, batch, slot.compute_stream)

        # Mark this step's outputs as ready for D2H copy.
        slot.record_step_done(slot.compute_stream)

    def _commit(self, slot_id: int, batch: Batch) -> None:
        """Phase 2: commit the sampled tokens to sequences.

        1. Wait for the D2H copy of sampled_ids to land.
        2. Commit tokens to each sequence.
        3. Record commit_done_event on copy stream.
        """
        slot = self.slots[slot_id]

        # Wait for the copy of sampled outputs to complete.
        self.streams.copy_stream.wait_event(slot.step_done_event)

        # Copy sampled_ids D2H via copy stream.
        with torch.cuda.stream(self.streams.copy_stream):
            host_buf = torch.empty(
                (batch.batch_size,), dtype=torch.long, device="cpu",
                pin_memory=True,
            )
            host_buf.copy_(
                slot.sampled_ids[:batch.batch_size], non_blocking=True
            )

        # Synchronise copy stream so the CPU can safely read host_buf.
        self.streams.copy_stream.synchronize()

        # Commit tokens to each sequence.
        for i, seq in enumerate(batch.sequences):
            token_id = int(host_buf[i].item())

            # Sentinel: zombie slot output → skip commit (no append, no state change).
            if token_id == _ZOMBIE_SENTINEL:
                self.scheduler.zombie_tracker.release(i)
                continue

            # Check zombie state.  If this sequence already finalized
            # (zombie), skip committing.
            zombie_result = self.scheduler.zombie_tracker.commit(
                i, eos_detected=(token_id == self.config.eos_token_id)
            )

            if zombie_result == "skip":
                self.scheduler.zombie_tracker.release(i)
                continue

            if zombie_result == "finalized" or token_id == self.config.eos_token_id:
                seq.state = RequestState.FINISHED
                seq.pending_token = None
            else:
                seq.generated_tokens.append(token_id)
                seq.seq_pos += 1
                seq.pending_token = None

            # Decrement the inflight ref for this row (was incremented at launch).
            self.scheduler.zombie_tracker.release(i)

            # Store final results for the caller when a sequence finishes.
            if seq.is_done() and seq.seq_id not in self._results:
                self._results[seq.seq_id] = list(seq.generated_tokens)

            self._total_tokens += 1

        slot.record_commit_done(self.streams.copy_stream)

    def _finalize(self, slot_id: int, batch: Batch) -> None:
        """Phase 3: post-commit finalisation.

        For plain text generation this is a no-op.  For constrained
        decoding, this is where the disallow mask for the NEXT step
        would be uploaded, after commit has made the per-sequence
        state current (commit-before-finalize ordering).
        """
        # Placeholder for constrained-decode mask preparation.
        # When integrated, the mask is computed from the just-committed
        # token state and uploaded to the slot's disallow_mask buffer
        # for use in the next step's sampling.
        pass

    # ---- Internal: graph capture ------------------------------------------

    def _capture_graphs(self) -> None:
        """Capture HIP graphs for each configured batch size.

        Graphs are captured on slot 0's buffers.  The forward function
        is a closure that captures references to slot 0's pre-allocated
        buffers, ensuring the graph's memory addresses are fixed.
        """
        if self.graph_manager is None:
            return

        for bs in self.config.graph_batch_sizes:
            # Create a dummy batch for capture.
            dummy_batch = Batch()
            slot = self.slots[0]

            def make_forward(batch_size: int):
                """Create a forward closure that references slot 0 buffers."""
                # We capture the batch_size but use slot 0's fixed buffers.
                _bs = batch_size
                _slot = slot
                _callbacks = self.callbacks

                def do_forward():
                    _callbacks.do_decode(_slot, dummy_batch, _slot.compute_stream)

                return do_forward

            self.graph_manager.capture(
                bs, make_forward(bs), slot.compute_stream, warmup_iters=2,
            )
