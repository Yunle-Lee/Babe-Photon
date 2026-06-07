# Mechanism 2: Forward Now, Sample Later

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "Mechanism 2: forward now, sample later"

---

## 1. The Dependency Puzzle

The next forward does not depend on anything the CPU does with the last
token — the forward only needs the token ID, which is already on the GPU.
So in principle, the forward can run ahead.

But *some* things about the next step do depend on the last step's
committed result:

1. **Batch membership**: If a request finished at step t, it should not
   be in step t+1. This is handled by Mechanism 3 (zombies).

2. **Constrained-decode mask**: Moondream's spatial skills (``point``,
   ``detect``, ``segment``) restrict which tokens the model may produce
   at each step.  The mask for step t+1 depends on the token sampled at
   step t — which the CPU hasn't committed yet when we want to launch
   step t+1.

## 2. Solution: Split Forward from Sampling

The key insight: the dependency is in **sampling**, not in the forward.
The forward itself needs no mask — it just computes logits for the
entire vocabulary.

The solution is to split each step into two phases that run at different
pipeline stages:

```
┌─────────────────────────────────────────────────────────┐
│  Tick N                                                  │
│                                                          │
│  LAUNCH:   Forward(step N) → [scores]                    │
│            (no mask needed — scores are unbiased)        │
│                                                          │
│  COMMIT:   Wait for token(step N-1) to land on CPU.      │
│            Advance state → build mask(step N).           │
│                                                          │
│  FINALIZE: Apply mask(step N) → Sample(step N).          │
│            (mask is now current because commit is done)   │
└─────────────────────────────────────────────────────────┘
```

This establishes **"commit-before-finalize" ordering**: the forward runs
immediately, but sampling waits until the previous step's commit has
made the per-sequence state current.

## 3. Unified Handling

The blog post describes how this elegantly handles both plain text and
constrained decoding with a single loop:

### 3.1 Plain text (no constrained decoding)

- No mask needed.
- Forward AND sampling can both run a step ahead.
- Both happen in the LAUNCH phase.

### 3.2 Constrained decoding (structured output)

- Forward runs a step ahead (in LAUNCH).
- Sampling waits on the previous commit (in FINALIZE).
- The commit phase builds the mask from the updated state,
  then FINALIZE applies it and samples.

### 3.3 Why this is elegant

No special-casing is needed.  The same scheduler code handles both:
- LAUNCH always runs the forward.
- COMMIT always processes the previous step's result.
- FINALIZE always does the mask-dependent work.

For plain text, FINALIZE is a no-op (mask is empty).  For constrained
sequences, FINALIZE waits on the commit that just happened.

## 4. The Disallow Mask

Moondream's constrained decoding uses a *disallow mask*: a boolean
tensor `[batch, vocab]` where `True` means "force this token's logit
to negative infinity before sampling."

The mask for each sequence is built from that sequence's current decode
state — which tokens it has emitted so far in the structured output.
For example:
- A `point` output always emits a coordinate `(x, y)`.
- A `detect` output walks an `x, y, width, height` cycle.

The mask for step t+1 is computed at the end of step t's commit, when
the sequence state is current.  It is then uploaded to the GPU slot
(via async H2D on the copy stream) so it is ready when FINALIZE runs.

## 5. Performance Implications

From the blog post's "It only pays once the bubble is actually hideable":

> We caught a bug where the pipelined numbers came out at *blocking*
> speed, traced to an accidental synchronous copy while building the
> constrained-decode mask.  Moving it to the copy stream was worth +11%
> on the 3090 and +34% on the B200.

The lesson: **any synchronisation on the hot path kills the pipelining
gain**.  Even a single synchronous copy within the critical path
destroys the overlap.

## 6. AMD ROCm Implementation Notes

In our ``photon_amd`` implementation:

- ``PipelineCallbacks.do_sample`` is called during ``_launch()`` for
  plain text (mask-independent sampling) or during ``_finalize()`` for
  constrained decoding.
- The disallow mask is stored in ``DecodeMetaBuffers.disallow_mask``
  and uploaded to GPU via the copy stream asynchronously.
- ``slot.mask_ready_event`` signals when the mask upload completes,
  so the sampling kernel can wait on it.

## 7. References

1. Photon source: ``kestrel/models/moondream/tokenizer.py`` (structured decode logic)
2. Blog post: [Mechanism 2 diagram](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-2-forward-now-sample-later)
3. NVIDIA: [Guided Decoding with Logits Processor](https://huggingface.co/docs/transformers/main_classes/logits)
