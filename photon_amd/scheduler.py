"""Request scheduling and batch assembly.

The scheduler is the CPU-side brain of the pipeline.  Each tick it
executes three phases:

1. **LAUNCH** — Assemble the batch for step *t+1*, upload metadata to GPU,
   dispatch the forward.  This does NOT depend on step *t*'s commit, so
   it can run immediately.
2. **COMMIT** — Wait for step *t*'s D2H copy to land.  Read sampled
   tokens, append to sequences, detect EOS.  Build the constrained-decode
   mask for step *t+1* from the updated state.
3. **FINALIZE** — Apply the mask (if any), sample step *t+1*'s token
   from the forward output.  (The forward already ran during LAUNCH,
   overlapping with COMMIT.)

The three phases produce a pipeline depth of 2: at any moment, one step
is on the GPU (forward), one is being committed on the CPU, and the
next is being assembled.

For plain text (no constrained decoding), forward and sampling can both
run a step ahead.  For constrained sequences the forward still runs
ahead, but sampling waits on the previous commit — the
"commit-before-finalize" ordering described in the blog post.

States
------
.. code-block::

    QUEUED ──► DECODING ──► FINISHED
                  │
                  └──► (zombie for 1 extra step) ──► released
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from .config import PhotonConfig
from .zombie import ZombieState, ZombieTracker


class RequestState(Enum):
    """Per-sequence lifecycle states."""
    QUEUED = auto()       # Waiting for first decode
    DECODING = auto()     # Actively generating tokens
    FINISHED = auto()     # EOS hit or length cap


@dataclass
class Sequence:
    """Per-sequence state tracked by the scheduler.

    Attributes
    ----------
    seq_id : int
        Monotonically increasing sequence identifier.
    prompt_tokens : list[int]
        Prompt token ids (prefill handled externally).
    generated_tokens : list[int]
        Tokens generated so far.
    state : RequestState
        Current lifecycle state.
    max_new_tokens : int
        Hard cap on number of generated tokens.
    zombie : ZombieState
        Zombie lifecycle state (see :mod:`zombie`).
    batch_idx : int
        Row index in the current decode batch (-1 if not in a batch).
    seq_pos : int
        Position in the sequence for positional encoding.
    kv_page_start : int
        Start index of allocated KV cache pages.
    kv_page_count : int
        Number of allocated KV cache pages.
    """

    seq_id: int
    prompt_tokens: list[int]
    generated_tokens: list[int] = field(default_factory=list)
    state: RequestState = RequestState.QUEUED
    max_new_tokens: int = 256
    zombie: ZombieState = field(default_factory=ZombieState)
    pending_token: Optional[int] = None  # Sampled but not yet committed

    batch_idx: int = -1
    seq_pos: int = 0

    kv_page_start: int = 0
    kv_page_count: int = 0

    def next_token_id(self) -> int:
        """The last generated token.
        
        If ``pending_token`` is set (sampled but not yet committed via
        scheduler tick), returns it.  Otherwise returns the last
        generated token or the prompt's last token for the first step.
        """
        if self.pending_token is not None:
            return self.pending_token
        if self.generated_tokens:
            return self.generated_tokens[-1]
        if self.prompt_tokens:
            return self.prompt_tokens[-1]
        return 0

    def is_done(self) -> bool:
        """Return ``True`` if this sequence should not participate in
        further decode steps."""
        return (
            self.state == RequestState.FINISHED
            or len(self.generated_tokens) >= self.max_new_tokens
        )


@dataclass
class Batch:
    """A decode batch assembled for one forward step.

    Contains references to the sequences that participate in this forward.
    """

    sequences: list[Sequence] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    batch_size: int = 0

    def add(self, seq: Sequence) -> None:
        """Add a sequence to this batch.

        Sets the sequence's ``batch_idx`` to its row index and appends
        its current token and position to the batch arrays.
        """
        seq.batch_idx = len(self.sequences)
        self.sequences.append(seq)
        self.token_ids.append(seq.next_token_id())
        self.positions.append(seq.seq_pos)
        self.batch_size = len(self.sequences)


@dataclass
class PipelineState:
    """Mutable state carried across scheduler ticks.

    This is the in-flight bookkeeping that the launch / commit /
    finalize phases read and update.
    """

    # The two ping-pong slots.  slot_id = tick_count % 2.
    current_slot: int = 0

    # Sequences in the decode batch for the current and next steps.
    active_sequences: list[Sequence] = field(default_factory=list)

    # Batch queue: entries are (slot_id, batch) tuples waiting for commit.
    # Depth-2 pipeline → up to 2 entries.
    batch_queue: deque[tuple[int, Batch]] = field(default_factory=deque)

    # Tick counter (monotonic).
    tick: int = 0


class Scheduler:
    """Assembles batches and drives the launch/commit/finalize cycle.

    The scheduler does NOT own GPU resources — those are in
    :class:`PhotonEngine`.  It owns the logical per-sequence state
    and the batch-assembly policy.

    Parameters
    ----------
    config : PhotonConfig
        Engine configuration.
    on_launch : Callable[[int, Batch], None]
        Callback invoked when a batch is assembled and ready to launch.
        Receives ``(slot_id, batch)``.
    on_commit : Callable[[int, Batch], None]
        Callback invoked when a previous step's results have landed.
        Receives ``(slot_id, batch)``.
    on_finalize : Callable[[int, Batch], None]
        Callback invoked after commit, before the next launch.
        Receives ``(slot_id, batch)``.
    """

    def __init__(
        self,
        config: PhotonConfig,
        on_launch: Callable[[int, Batch], None],
        on_commit: Callable[[int, Batch], None],
        on_finalize: Callable[[int, Batch], None],
    ) -> None:
        self.config = config
        self._on_launch = on_launch
        self._on_commit = on_commit
        self._on_finalize = on_finalize

        # Pending requests waiting for their first decode step.
        self.pending: deque[Sequence] = deque()

        # Central zombie tracker for all sequences.
        self.zombie_tracker = ZombieTracker()

        # Mutable pipeline state.
        self.state = PipelineState()

        # All sequences ever registered (indexed for lookup and cleanup).
        self._all_sequences: dict[int, Sequence] = {}
        self._next_seq_id: int = 0

    # ---- Public API -------------------------------------------------------

    def submit(self, prompt: list[int], max_new_tokens: int = 256) -> int:
        """Enqueue a new generation request.

        Args:
            prompt: Token ids of the prompt.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The assigned sequence id.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        seq = Sequence(
            seq_id=seq_id,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
        )
        self._all_sequences[seq_id] = seq
        self.zombie_tracker.register(seq.zombie)
        self.pending.append(seq)
        return seq_id

    def tick(self) -> None:
        """Run one scheduler tick: launch → commit → finalize.

        .. code-block:: text

            Tick N:
              LAUNCH    batch for step N   (slot = N % 2)
              COMMIT    batch for step N-1 (slot = (N-1) % 2)
              FINALIZE  batch for step N   (sampling mask)
        """
        self.state.tick += 1
        slot_id = self.state.tick % self.config.num_slots

        # ---- PHASE 1: LAUNCH (step t+1) -----------------------------------
        batch = self._assemble_batch()
        if batch is not None and batch.batch_size > 0:
            self._on_launch(slot_id, batch)
            self.state.batch_queue.append((slot_id, batch))

        # ---- PHASE 2: COMMIT (oldest step) --------------------------------
        # In normal pipeline mode: commit only when 2+ entries exist
        # (pipeline depth).  When there are no active/pending sequences,
        # drain even with 1 entry (tail mode).
        has_remaining_work = (
            self.pending_count() > 0 or self.active_count() > 0
        )
        commit_threshold = 2 if has_remaining_work else 1
        if len(self.state.batch_queue) >= commit_threshold:
            oldest_slot, oldest_batch = self.state.batch_queue.popleft()
            self._on_commit(oldest_slot, oldest_batch)

        # ---- PHASE 3: FINALIZE --------------------------------------------
        if len(self.state.batch_queue) >= 1:
            newest_slot, newest_batch = self.state.batch_queue[-1]
            self._on_finalize(newest_slot, newest_batch)

        # Recycle finished sequences.
        self._recycle_done()

    def finish(self) -> None:
        """Drain remaining steps after all requests are done.

        Called by :meth:`PhotonEngine.run` when ``has_work()`` returns
        ``False`` to flush the pipeline.
        """
        while self.state.batch_queue:
            slot_id, batch = self.state.batch_queue.popleft()
            self._on_commit(slot_id, batch)
            self._on_finalize(slot_id, batch)

    def active_count(self) -> int:
        """Number of sequences that are currently in the decode loop
        (QUEUED or DECODING, not FINISHED)."""
        return sum(
            1 for s in self.state.active_sequences
            if not s.is_done()
        )

    def pending_count(self) -> int:
        """Number of sequences waiting in the pending queue."""
        return len(self.pending)

    # ---- Internal ---------------------------------------------------------

    def _assemble_batch(self) -> Optional[Batch]:
        """Build the next decode batch from pending and active sequences.

        Admission policy:
          1. Admit as many new sequences from the pending queue as fit.
          2. Continue decoding active sequences that are not finished.
          3. Fill batch up to ``max_batch_size``.

        Each sequence added to the batch gets a launch reference
        (``inflight_refs += 1``) via the zombie tracker.
        """
        batch = Batch()

        # Admit new sequences from the pending queue.
        while self.pending and batch.batch_size < self.config.max_batch_size:
            seq = self.pending.popleft()
            seq.state = RequestState.DECODING
            seq.seq_pos = len(seq.prompt_tokens)
            batch.add(seq)
            self.zombie_tracker.launch_ref(seq.batch_idx)

        # Continue decoding active sequences.
        for seq in self.state.active_sequences:
            if seq.is_done():
                continue
            if batch.batch_size >= self.config.max_batch_size:
                break
            batch.add(seq)
            self.zombie_tracker.launch_ref(seq.batch_idx)

        if batch.batch_size == 0:
            return None

        self.state.active_sequences = batch.sequences
        return batch

    def _recycle_done(self) -> None:
        """Remove finished sequences from the active set.

        Resource recycling (KV pages, LoRA slots) is handled by
        :meth:`ZombieTracker.release` during the commit phase.
        """
        self.state.active_sequences = [
            s for s in self.state.active_sequences if not s.is_done()
        ]
