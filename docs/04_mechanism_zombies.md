# Mechanism 3: Zombies — Finalize Early, Release Late

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "Mechanism 3: zombies: finalize early, release late"

---

## 1. The Problem

In Mechanism 2, we noted that *batch membership* for step t+1 depends on
step t's committed result.  If a sequence finishes at step t (EOS or
length cap), it should not be in step t+1's batch.

But step t+1 was already launched **before** step t was committed — the
whole point of pipelining.  So the finished sequence is *already baked
into step t+1's forward*.  You cannot un-launch GPU work.

## 2. The Naive Option: Mid-Flight Cancellation

One could imagine cancelling the row mid-flight:
- Set the sequence's token to a "don't care" value.
- Or skip it in the attention kernel.

But this requires special-case logic in every kernel (attention, MLP,
sampling, commit), creating a "thicket of cancellation special cases."

## 3. The Elegant Solution: Let It Ride

Photon's approach is simpler: **let the finished sequence ride as a
zombie for one extra forward**.

```
Step t:     Sequence X is alive → forward computes real result
            Commit detects EOS → X is marked "finalized"
            Result is emitted to caller
            BUT: X's KV pages are NOT released

Step t+1:   X was already in the batch → forward includes X as "zombie"
            Commit sees X is finalized → SKIP, no token appended
            Release X's inflight ref

Refcount=0: X's KV pages and LoRA slot are recycled
```

The zombie occupies a batch slot and writes some KV cache entries that
nobody will ever read — but that's a small price to pay.

## 4. The Per-Sequence State Machine

Each sequence carries two extra fields:

| Field | Type | Meaning |
|-------|------|---------|
| ``finalized`` | ``bool`` | ``True`` after EOS or length cap detected. |
| ``inflight_refs`` | ``int`` | 0, 1, or 2 — how many in-flight forwards reference this sequence. |

State transitions:

```
IDLE (0 refs, not finalized)
  │
  ├── launch ──► ACTIVE (1 ref)
  │                │
  │                ├── launch ──► ACTIVE (2 refs, pipelined)
  │                │                │
  │                │                ├── commit (EOS) ──► ZOMBIE (2 refs, finalized)
  │                │                │                      │
  │                │                │                      ├── commit (skip)
  │                │                │                      └── release ──► ZOMBIE (1 ref)
  │                │                │
  │                │                ├── commit (normal) ──► ACTIVE (1 ref)
  │                │                └── release ──► ACTIVE (1 ref)
  │                │
  │                ├── commit (EOS) ──► ZOMBIE (1 ref, finalized)
  │                │                      │
  │                │                      └── release ──► IDLE (recycled)
  │                │
  │                └── commit (normal) ──► ACTIVE (1 ref)
  │
  └── release ──► IDLE (recycled if finalized, no-op otherwise)
```

## 5. Zombie Tax Analysis

From the blog post, the *zombie tax* $z$ is the fraction of wasted
forwards:

$$z \approx \frac{1}{L \cdot B}$$

where:
- $L$ = average generated tokens per request
- $B$ = batch size

### 5.1 At batch = 1

One wasted forward per $L$ tokens: $z \approx 1/L$.  For $L \approx 110$,
that is ~1% overhead — negligible.

### 5.2 At batch > 1

The zombie is just **one extra row** in a step that is already streaming
the full model weights through the compute units.  In the memory-bandwidth-bound
regime of decode, streaming the weights dominates, and the extra row adds
almost no cost.  The tax effectively vanishes at batch sizes ≫ 1.

### 5.3 Empirical validation

From the blog post:

| Hardware | Streams | L | Predicted | Observed |
|----------|---------|---|-----------|----------|
| 3090     | 1       | 104 | +5.7%     | +6.5%    |
| 3090     | 32      | 113 | +11.1%    | +11.6%   |
| B200     | 1       | 115 | +17.2%    | +17.6%   |
| B200     | 32      | 104 | +39.1%    | +35.4%   |

The predicted speedup (which accounts for zombie tax) closely matches
the observed values, confirming the model.

## 6. Resource Reclamation

Resources are **not released** when a sequence first finishes.  They are
only released when ``inflight_refs`` reaches 0:

- **KV cache pages**: Returned to the page allocator's free list.
- **LoRA slot**: Returned to the LoRA slot pool.
- **Batch slot**: Freed for a new sequence in the next schedule cycle.

This release-late policy is the price of not having mid-flight
cancellation logic.

## 7. AMD ROCm Implementation Notes

In ``photon_amd/zombie.py``:

- ``ZombieState``: The per-sequence fields (``finalized``, ``inflight_refs``,
  ``kv_page_start``, ``kv_page_count``, ``lora_slot``).
- ``ZombieTracker``: Central manager called at three hook points:
  - ``launch_ref()`` — increment refs when a sequence enters a batch.
  - ``commit()`` — detect EOS, return "finalized" or "skip".
  - ``release()`` — decrement refs, recycle when done.
- ``should_skip()`` — query whether a row is a zombie.

The scheduler calls these hooks in ``_assemble_batch()`` (launch ref),
``_commit()`` (commit), and ``_recycle_done()`` (release).

## 8. References

1. Blog post: [Mechanism 3 diagram](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-3-zombies-finalize-early-release-late)
2. Photon source: ``kestrel/models/moondream/zombie.py`` (approximate path)
3. Related: [Reference counting in garbage collection](https://en.wikipedia.org/wiki/Reference_counting)
