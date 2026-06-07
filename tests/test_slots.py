"""Unit tests for decode slots and CPU-GPU buffers.

These tests require a GPU (marked with 'gpu').  Skip with ``-k "not gpu"``.
"""

import pytest
import torch

from photon_amd.config import PhotonConfig
from photon_amd.decode_slot import (
    CpuGpuBuffer,
    DecodeMetaBuffers,
    create_decode_slot,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    return torch.device("cuda:0")


class TestCpuGpuBuffer:
    @pytest.mark.gpu
    def test_allocate(self, device):
        buf = CpuGpuBuffer.allocate(8, torch.int32, device)
        assert buf.host.shape == (8,)
        assert buf.device.shape == (8,)
        assert buf.host.is_pinned() is True

    @pytest.mark.gpu
    def test_allocate_with_extra(self, device):
        vocab = 128
        buf = CpuGpuBuffer.allocate(8, torch.bool, device, extra=vocab - 1)
        assert buf.host.shape == (8 + vocab - 1,)

    @pytest.mark.gpu
    def test_copy_h2d_and_d2h(self, device):
        buf = CpuGpuBuffer.allocate(4, torch.float32, device)
        buf.host[:] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        s = torch.cuda.Stream(device=device)
        buf.copy_host_to_device(4, s)
        s.synchronize()

        assert torch.allclose(
            buf.device[:4].cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0])
        )


class TestDecodeMetaBuffers:
    @pytest.mark.gpu
    def test_allocate(self, device):
        meta = DecodeMetaBuffers.allocate(
            max_batch=8, vocab_size=32, device=device
        )
        assert meta.batch_idx.host.shape == (8,)
        assert meta.input_pos.host.shape == (8,)
        # disallow_mask is [max_batch * vocab] (flattened).
        assert meta.disallow_mask.host.shape == (8 * 32,)


class TestDecodeSlot:
    @pytest.mark.gpu
    def test_create_slot(self, device):
        config = PhotonConfig()
        stream = torch.cuda.Stream(device=device)
        slot = create_decode_slot(
            slot_id=0,
            device=device,
            dtype=config.dtype,
            max_batch=config.max_batch_size,
            kv_cache_pages=config.kv_cache_pages,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            compute_stream=stream,
        )
        assert slot.slot_id == 0
        assert slot.logits.shape == (config.max_batch_size, config.vocab_size)
        assert slot.hidden_last.shape == (config.max_batch_size, config.hidden_dim)
        assert slot.sampled_ids.shape == (config.max_batch_size,)
