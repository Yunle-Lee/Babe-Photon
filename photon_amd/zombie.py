"""Zombie lifecycle for pipelined decoding — Mechanism 3.

Problem
-------
When a sequence hits EOS (or its length cap) at step *t*, the scheduler
marks it ``finalized`` and emits its result.  But step *t+1* was already
launched with that sequence in its batch — it is still alive on the GPU
as a **zombie**.

Solution
--------
The zombie stays in the batch for one extra forward (+1 ``inflight_ref``).
When step *t+1* commits, the sequence is already ``finalized``, so the
commit is **skipped** — the zombie harmlessly rode along, occupying a
batch slot and writing some KV cache entries that will never be read.

Only when ``inflight_refs`` hits 0 are its resources (KV pages, LoRA
slot) recycled back to the free pool.

Zombie tax
----------
Per Photon's cost model, a zombie wastes one forward per finished
request.  At batch=1 that is ~1% overhead for L≈110.  At batch > 1
it is one extra row in a step already paying the full weight-streaming
cost, so it costs almost nothing.

The tax term *z* in the speedup model:

.. math::

    \\text{speedup} = \\frac{T_\\text{block}}{T_\\text{pipe}} \\times (1 - z)

where *z* ≈ 1/L at batch=1 and approaches 0 at batch ≫ 1.

Reference
---------
  Moondream Blog: "Popping the GPU Bubble" (2026-06-04), Mechanism 3:
  "Zombies: finalize early, release late"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ZombieState:
    """Per-sequence state for the zombie lifecycle.

    This lives alongside the scheduler's per-sequence metadata and is
    updated by the scheduler's commit and release phases.

    Attributes
    ----------
    finalized : bool
        After commit detects EOS or length-cap, this flips ``True``.
        The result is emitted to the caller but the sequence stays
        alive on the GPU.
    inflight_refs : int
        How many in-flight forwards still reference this sequence.
        Values:
          - 0 : idle or fully released.
          - 1 : one forward (the current step) references it.
          - 2 : two forwards reference it (step *t* and step *t+1*).
    kv_page_start : int
        Start index of allocated KV cache pages.
    kv_page_count : int
        Number of allocated KV cache pages.
    lora_slot : int
        LoRA adapter slot index (0 if no LoRA).
    """

    finalized: bool = False
    inflight_refs: int = 0
    kv_page_start: int = 0
    kv_page_count: int = 0
    lora_slot: int = 0


@dataclass
class ZombieTracker:
    """Manages zombie lifecycle across the batch.

    Called at three points in the scheduler tick:

    ===============  ====================================================
    Hook             Action
    ===============  ====================================================
    ``launch_prepare``  Increment ``inflight_refs`` for every sequence
                        in the batch that is about to be launched.
    ``commit``          If EOS detected, set ``finalized=True`` and emit
                        result.  If already finalized, skip commit.
    ``release``         Decrement ``inflight_refs``.  If it hits zero
                        and ``finalized`` is ``True``, recycle resources.
    ===============  ====================================================

    The tracker is stateless w.r.t. individual sequences — the
    :class:`ZombieState` objects are owned externally (by the scheduler)
    and the tracker manages them by reference.
    """

    states: list[ZombieState] = field(default_factory=list)

    def register(self, state: ZombieState) -> None:
        """Register a new sequence's zombie state for tracking."""
        self.states.append(state)

    def launch_ref(self, row_idx: int) -> None:
        """Increment inflight refs for the sequence at batch row *row_idx*.

        Called at the start of each launch when the sequence is included
        in a new decode batch.
        """
        if row_idx < len(self.states):
            self.states[row_idx].inflight_refs += 1

    def commit(self, row_idx: int, eos_detected: bool) -> Optional[str]:
        """Commit step for batch row *row_idx*.

        Args:
            row_idx: Batch row index.
            eos_detected: Whether the sampled token indicates end-of-sequence.

        Returns:
            ``"finalized"`` if this step caused EOS (caller should emit
            result).
            ``"skip"`` if the sequence is already a zombie — the caller
            should skip committing this row.
            ``None`` for normal, non-EOS commit — the caller should
            append the token and advance sequence position.
        """
        if row_idx >= len(self.states):
            return None

        st = self.states[row_idx]

        if st.finalized:
            # This sequence is already a zombie.  Skip the commit.
            return "skip"

        if eos_detected:
            st.finalized = True
            return "finalized"

        return None

    def release(self, row_idx: int) -> None:
        """Decrement refcount; recycle resources if the sequence is done.

        Called during the scheduler's recycle phase, after a commit
        has resolved the zombie state.
        """
        if row_idx >= len(self.states):
            return

        st = self.states[row_idx]
        st.inflight_refs = max(0, st.inflight_refs - 1)

        if st.inflight_refs <= 0 and st.finalized:
            self._recycle(st)
            # Reset for potential reuse by a future sequence assigned
            # to the same row index.
            st.finalized = False
            st.kv_page_count = 0
            st.lora_slot = 0

    def should_skip(self, row_idx: int) -> bool:
        """Return ``True`` if this row is a finalized zombie that should
        be skipped during commit."""
        if row_idx >= len(self.states):
            return False
        return self.states[row_idx].finalized

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _recycle(st: ZombieState) -> None:
        """Return KV pages and LoRA slot to free pools.

        In a production engine this would call into the KV cache manager
        and LoRA slot allocator.  Stubbed here for the reference
        implementation.
        """
        # TODO: Integrate with KV cache page allocator and LoRA slot
        # allocator when those subsystems are implemented.
        pass
