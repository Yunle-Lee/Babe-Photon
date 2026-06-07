# The GPU Bubble: A Mathematical Analysis

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "A cost model for the bubble"

---

## 1. Defining the Bubble

A decode step has three distinct pieces of work:

| Piece | Location | Description | Time |
|-------|----------|-------------|------|
| **Forward** | GPU | Weight-streaming matmuls (attention projections, MLP). Memory-bandwidth bound at decode. | $T_f$ |
| **Sampling** | GPU → CPU | Constrained-decode mask application, argmax/sample, spatial decode, D2H copy. | $T_s$ |
| **Bookkeeping** | CPU | Plan (batch assembly), launch (graph replay trigger), commit (token append, state advance, mask build). | $T_b$ |

### 1.1 Blocking Mode

```
Time →
CPU: [plan][launch] ──────────────── [sync] [commit][plan][launch] ────────────────
GPU:                [forward][sample] =idle=                  [forward][sample]
                    ◄──────── T_f + T_s ────────►
                                                 ◄── T_b ──►  ← GPU BUBBLE
```

Total step time: $T_{\text{block}} = T_f + T_s + T_b$

The GPU idle time is $T_b$ — the bubble.

### 1.2 Pipelined Mode

```
Time →
CPU: [plan][launch] ───────────────── [commit][plan][launch] ───────────────── [commit]
GPU:                [forward][sample]                       [forward][sample]
                    ◄────── T_f + T_s ──────►              ◄────── T_f + T_s ──►
```

Total step time: $T_{\text{pipe}} = \max(T_f + T_s, T_b)$

The GPU idle time is $0$ (assuming $T_f + T_s \ge T_b$).

## 2. The Speedup Formula

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

where **$z$** is the zombie tax from Mechanism 3:

$$z \approx \frac{1}{L} \quad \text{(at batch = 1)}$$

$$z \approx 0 \quad \text{(at batch ≫ 1)}$$

### 2.1 Two competing forces

1. **Bubble hiding** ($T_{\text{block}} / T_{\text{pipe}}$): The direct
   benefit of overlapping CPU bookkeeping with GPU computation.

2. **Zombie tax** ($z$): The indirect cost of letting finished requests
   ride one extra forward.  At batch=1 it is $1/L$ (~1%).  At large
   batch sizes, the cost amortises to near zero because the extra row
   is in a memory-bandwidth-bound step that already streams all the
   weights.

## 3. Why the Bubble Grows

The forward time $T_f$ scales with model weight size and GPU memory
bandwidth:

$$T_f \approx \frac{W}{B} \cdot \frac{1}{N_{\text{CU}}}$$

where:
- $W$ = total model weights in bytes
- $B$ = GPU memory bandwidth (bytes/s)
- $N_{\text{CU}}$ = number of active compute units

As GPU memory bandwidth increases (H100: 3.35 TB/s → B200: 8 TB/s) or
models get smaller (Moondream 2B → 0.5B), $T_f$ shrinks but $T_b$ stays
constant.  The bubble becomes a **larger fraction** of the total step
time.

### 3.1 Concrete example

| Model | GPU | W (GB) | B (TB/s) | $T_f$ (ms) | $T_b$ (ms) | Bubble % |
|-------|-----|--------|----------|------------|------------|----------|
| Moondream 2B | H100 | ~4 | 3.35 | ~1.2 | 1.5 | 55.6% |
| Moondream 2B | B200 | ~4 | 8.0 | ~0.5 | 1.5 | 75.0% |
| Moondream 0.5B | H100 | ~1 | 3.35 | ~0.3 | 1.5 | 83.3% |

The bubble is **dominant** for small models on fast GPUs — exactly the
regime where Moondream operates.

## 4. Prefill in the Same Pipeline

Prefill (processing a new request's prompt + image) is also a forward
pass, but over many tokens at once — it is compute-bound, not
memory-bandwidth-bound, so $T_f$ is larger and $T_s$ is more expensive.

Photon uses the **same two-slot pipeline for prefill and decode**.
A prefill is just a ``kind="prefill"`` launch that uses the same slot
and compute stream.  Because the pipeline only cares that a slot is
free, not what kind of work last used it, a prefill and a decode can
be transparently interleaved.

This matters most when outputs are short — a request that emits only
three tokens spends almost all its time in prefill, and a workload of
many short requests is effectively a stream of prefills with a little
decode sprinkled in.  Sharing one pipeline lets the prefill CPU
bookkeeping (admission, tokenisation) overlap with decode forwards
and vice versa.

## 5. Validation

From the blog post's empirical measurements on a B200:

| Mode | $T_f$ (ms) | $T_s$ (ms) | Period (ms) |
|------|-----------|-----------|-------------|
| 1 stream | 2.45 | 0.14 | 2.63 |
| 8 streams | 3.12 | 0.14 | 3.30 |
| 32 streams | 3.80 | 0.14 | 3.98 |

$T_f + T_s \approx \text{period}$, meaning the GPU is busy for
essentially the entire period — the bubble is eliminated.  The
remaining GPU idle is under 0.05 ms per step.

### 5.1 Blocking vs pipelined comparison

| Hardware | Streams | Blocking (ms) | Pipelined (ms) | Speedup |
|----------|---------|---------------|----------------|---------|
| RTX 3090 | 1 | 5.44 | 5.10 | +6.5% |
| RTX 3090 | 8 | 7.52 | 6.97 | +7.8% |
| RTX 3090 | 32 | 11.74 | 10.52 | +11.6% |
| B200 | 1 | 3.11 | 2.63 | +17.6% |
| B200 | 8 | 4.04 | 3.30 | +21.9% |
| B200 | 32 | 5.55 | 3.98 | +35.4% |

Three observations:
1. **The win grows with GPU speed** — +12% on 3090 but +35% on B200.
2. **Zombie tax is real but small** — ~1% at batch=1, essentially 0 at
   batch > 1.
3. **It only pays when the bubble is hideable** — any synchronous
   operation on the hot path kills the gain.

## 6. References

1. Blog post: [A cost model for the bubble](https://moondream.ai/blog/popping-the-gpu-bubble#a-cost-model-for-the-bubble)
2. NVIDIA: [GPU Memory Bandwidth](https://www.nvidia.com/en-us/data-center/a100/)
3. AMD: [MI300 Memory Bandwidth](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
