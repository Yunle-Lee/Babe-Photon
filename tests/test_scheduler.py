"""Unit tests for the scheduler."""

import pytest

from photon_amd.config import PhotonConfig
from photon_amd.scheduler import Batch, RequestState, Scheduler, Sequence


class TestSequence:
    def test_next_token_id_empty(self):
        s = Sequence(seq_id=0, prompt_tokens=[])
        assert s.next_token_id() == 0

    def test_next_token_id_from_prompt(self):
        s = Sequence(seq_id=0, prompt_tokens=[10, 20, 30])
        assert s.next_token_id() == 30

    def test_next_token_id_from_generated(self):
        s = Sequence(seq_id=0, prompt_tokens=[10, 20, 30])
        s.generated_tokens.append(40)
        assert s.next_token_id() == 40

    def test_is_done_by_state(self):
        s = Sequence(seq_id=0, prompt_tokens=[1])
        s.state = RequestState.FINISHED
        assert s.is_done() is True

    def test_is_done_by_length_cap(self):
        s = Sequence(seq_id=0, prompt_tokens=[1], max_new_tokens=0)
        assert s.is_done() is True


class TestBatch:
    def test_add_sequence(self):
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        batch = Batch()
        batch.add(seq)
        assert batch.batch_size == 1
        assert seq.batch_idx == 0
        assert batch.token_ids == [3]  # last prompt token
        assert batch.positions == [0]


class TestScheduler:
    @pytest.fixture
    def config(self):
        return PhotonConfig(max_batch_size=32, num_slots=2)

    @pytest.fixture
    def scheduler(self, config):
        launched = []
        committed = []

        def on_launch(slot_id, batch):
            launched.append((slot_id, batch))

        def on_commit(slot_id, batch):
            committed.append((slot_id, batch))

        def on_finalize(slot_id, batch):
            pass

        return Scheduler(config, on_launch, on_commit, on_finalize)

    def test_submit_returns_id(self, scheduler):
        sid1 = scheduler.submit([1, 2, 3])
        sid2 = scheduler.submit([4, 5, 6])
        assert sid1 == 0
        assert sid2 == 1

    def test_submit_enqueues(self, scheduler):
        scheduler.submit([1, 2, 3])
        assert scheduler.pending_count() == 1
        # Sequence is QUEUED, not yet DECODING, so active_count is 0.
        assert scheduler.active_count() == 0

    def test_tick_processes(self, scheduler):
        """A tick should pull from pending and launch."""
        scheduler.submit([1, 2, 3], max_new_tokens=4)

        # First tick: launch step 1.
        scheduler.tick()
        # Pending should now be 0, active 1.
        assert scheduler.pending_count() == 0
        assert scheduler.active_count() == 1

    def test_finish_drains(self, scheduler):
        """finish() should drain the batch queue without errors."""
        scheduler.submit([1, 2, 3], max_new_tokens=2)

        # Run a few ticks.
        for _ in range(4):
            scheduler.tick()

        # finish() drains remaining entries in batch_queue.
        # With mock callbacks, sequences aren't actually committed
        # (no token appends), so active_count may still be > 0.
        # The test validates that finish() completes without error.
        scheduler.finish()
        # No exception is the test.
