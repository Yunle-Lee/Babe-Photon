# Porting Photon from NVIDIA CUDA to AMD ROCm

> A practical guide to porting Photon's inference engine from NVIDIA GPUs
> to AMD Instinct accelerators (gfx942 / MI300).

---

## 1. Overview

Photon (and its reference implementation in the ``kestrel`` repository)
is written against NVIDIA CUDA APIs.  This document catalogues every
CUDA-specific surface area and maps it to the equivalent AMD ROCm/HIP
API, noting where the two diverge.

The good news: **PyTorch provides a transparent compatibility layer.**
Most PyTorch code that runs on CUDA runs unchanged on ROCm because
PyTorch's CUDA API functions translate to HIP calls under the hood.
The areas that require attention are:

1. Stream and event management
2. GPU graph capture
3. Pinned memory
4. Kernel launch overhead
5. Memory allocation patterns
6. Hardware-specific performance characteristics

---

## 2. CUDA → HIP/ROCm Mapping

### 2.1 Streams

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaStream_t`` | ``hipStream_t`` | Wrapped by ``torch.cuda.Stream`` |
| ``torch.cuda.Stream()`` | ``torch.cuda.Stream()`` | Identical API |
| ``stream.synchronize()`` | ``stream.synchronize()`` | Identical |
| Priority | Priority | Both support 0 (default) and -1 (higher) |

**Verdict**: No code changes needed.  The PyTorch Stream API is fully
compatible.

### 2.2 Events

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaEvent_t`` | ``hipEvent_t`` | Wrapped by ``torch.cuda.Event`` |
| ``event.record(stream)`` | ``event.record(stream)`` | Identical |
| ``stream.wait_event(event)`` | ``stream.wait_event(event)`` | Identical |
| ``event.synchronize()`` | ``event.synchronize()`` | Identical |
| ``enable_timing=False`` | ``enable_timing=False`` | Disables profiling, reduces overhead |

**Verdict**: No code changes needed.

### 2.3 GPU Graphs

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaGraph_t`` | ``hipGraph_t`` | Wrapped by ``torch.cuda.CUDAGraph`` |
| ``torch.cuda.CUDAGraph()`` | ``torch.cuda.CUDAGraph()`` | Identical API |
| ``graph.capture()`` | ``graph.capture()`` | Same context manager |
| ``graph.replay()`` | ``graph.replay()`` | Same API |

**Stricter limitations on AMD**:
1. ❌ Dynamic shapes inside a captured graph → unsupported.
   **Mitigation**: Capture separate graphs per batch size (already done
   by Photon).
2. ❌ Host callbacks inside a captured graph → unsupported.
   **Mitigation**: All callbacks run outside the graph (before/after replay).
3. ✅ Fixed-shape pre-allocated buffers → supported.
   **Mitigation**: Photon's ping-pong slots pre-allocate all buffers.

**Verdict**: Code is compatible but requires stricter adherence to
fixed-shape, no-callback patterns.  Photon already satisfies these.

### 2.4 Pinned Memory

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaHostAlloc`` | ``hipHostMalloc`` | ``torch.Tensor.pin_memory()`` |
| ``cudaMemcpyAsync`` | ``hipMemcpyAsync`` | ``tensor.copy_(..., non_blocking=True)`` |

**Verdict**: No code changes needed.

### 2.5 Memory Allocation

| CUDA | HIP/ROCm | Notes |
|------|-----------|-------|
| ``cudaMalloc`` | ``hipMalloc`` | ``torch.empty(device="cuda")`` |
| Allocation during hot path | **May trigger device-wide sync** | **Critical difference** |

**Key ROCm caveat**: ``hipMalloc`` on AMD GPUs can trigger a
device-wide synchronisation, unlike on NVIDIA where it is generally
non-blocking.  Photon's design of pre-allocating all buffers at init
time is therefore *even more important* for the AMD port.

**Verdict**: No code changes needed because Photon already avoids
runtime allocations.  But this is a correctness-critical invariant.

---

## 3. Performance Characteristics: NVIDIA vs AMD

### 3.1 Memory Bandwidth

| GPU | Peak BW | Architecture | Year |
|-----|---------|-------------|------|
| H100 | 3.35 TB/s | Hopper | 2023 |
| H200 | 4.8 TB/s | Hopper | 2024 |
| B200 | 8.0 TB/s | Blackwell | 2025 |
| MI300X | 5.3 TB/s | CDNA3 (gfx942) | 2024 |

The MI300X sits between an H100 and B200 in memory bandwidth, so the
GPU bubble is a meaningful problem for it.  Pipelining should yield
speedups comparable to the H100 row in the blog post's table (~15–25%
at batch).

### 3.2 Compute Characteristics

| Feature | NVIDIA | AMD |
|---------|--------|-----|
| Tensor Cores | Yes (FP16/BF16/FP8) | Matrix FP16 (equivalent) |
| FP16 throughput | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| BF16 throughput | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| Wavefront size | 32 (warp) | 64 (wavefront) |
| Max waves/CU | Varies | 32 per CU |

At decode time, the workload is memory-bandwidth-bound, so the FP16/BF16
throughput difference is less relevant.  The wavefront size difference
(32 vs 64) may affect occupancy in custom kernels.

### 3.3 Kernel Launch Overhead

AMD HIP has slightly higher kernel-launch overhead than CUDA (~5–10 μs
per launch vs 3–5 μs).  This makes HIP graph capture *even more
valuable* on AMD, as it collapses many launches into one.

---

## 4. Photon-AMD Implementation Notes

### 4.1 What works unchanged

The following Photon components map 1:1 to ROCm:

- **StreamManager** (``stream_manager.py``): Stream/Evet API is identical.
- **DecodeSlot** (``decode_slot.py``): Pinned memory and GPU buffer
  allocation via ``torch.empty`` is identical.
- **ZombieTracker** (``zombie.py``): Pure Python state machine, no GPU API.
- **Scheduler** (``scheduler.py``): Pure Python logic, no GPU API.
- **Pipeline** (``pipeline.py``): All GPU API calls map transparently.

### 4.2 What requires conditional handling

- **Graph capture** (``graph.py``): While the API is identical, AMD
  requires stricter adherence to the pre-allocation pattern.  We already
  satisfy this by using pre-allocated slot buffers.

### 4.3 What is AMD-custom

- **dtype preference**: ``float16`` is recommended over ``bfloat16`` on
  gfx942 for optimal memory-bandwidth utilisation, though both are supported.
- **Event overhead**: ``enable_timing=False`` is critical on AMD to avoid
  profiling overhead in the hot path.  Photon already does this.

---

## 5. Testing Strategy

### 5.1 Unit tests (CPU-able)

- ``test_zombie.py``: Zombie state machine transitions.
- ``test_scheduler.py``: Batch assembly and tick logic.

These run without a GPU and validate the pure-Python logic.

### 5.2 GPU tests (require AMD GPU + ROCm)

- ``test_slots.py``: Buffer allocation, pinned memory, H2D/D2H copies.
- ``test_benchmark.py``: Simulated pipeline throughput comparison.

Run with: ``pytest tests/ -v -k gpu --tb=short``

### 5.3 Integration benchmark

```bash
python -m photon_amd.benchmark
```

Compares blocking (1 slot) vs pipelined (2 slots + HIP graphs) modes
and reports the speedup.

---

## 6. Known Limitations

1. **HIP graph capture**: Only supported on gfx942 (MI300) and newer.
   Older CDNA2 (MI250) and RDNA3 (7900 XTX) GPUs do not support HIP
   graph capture.
2. **Multi-GPU**: This reference implementation supports single-GPU only.
   Multi-GPU inference (tensor parallelism, pipeline parallelism) would
   require additional HIP-aware inter-GPU communication primitives.
3. **Custom CUDA kernels**: Photon's production kernels (Moondream-specific
   attention, spatial decoder) are not included — this is a
   pipeline-level reference, not a full model runtime.

---

## 7. References

1. [ROCm Documentation](https://rocm.docs.amd.com/)
2. [HIP Porting Guide](https://rocm.docs.amd.com/en/latest/how-to/hipify/hip_porting_guide.html)
3. [PyTorch ROCm](https://pytorch.org/get-started/locally/)
4. Moondream Blog: [Photon Page](https://moondream.ai/p/photon)
5. Moondream Docs: [Supported Hardware](https://docs.moondream.ai/running-locally#supported-hardware)
