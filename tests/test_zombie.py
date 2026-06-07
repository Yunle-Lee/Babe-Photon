"""Unit tests for zombie lifecycle (Mechanism 3)."""

import pytest

from photon_amd.zombie import ZombieState, ZombieTracker


class TestZombieState:
    """Tests for ZombieState dataclass defaults."""

    def test_default_state(self):
        z = ZombieState()
        assert z.finalized is False
        assert z.inflight_refs == 0
        assert z.kv_page_count == 0


class TestZombieTracker:
    """Tests for ZombieTracker state machine."""

    @pytest.fixture
    def tracker(self):
        t = ZombieTracker()
        s0 = ZombieState()
        s1 = ZombieState()
        t.register(s0)
        t.register(s1)
        return t

    def test_launch_ref_increments(self, tracker):
        tracker.launch_ref(0)
        assert tracker.states[0].inflight_refs == 1
        assert tracker.states[1].inflight_refs == 0

    def test_launch_ref_double(self, tracker):
        """Two launches before commit → inflight_refs = 2 (pipelined)."""
        tracker.launch_ref(0)
        tracker.launch_ref(0)
        assert tracker.states[0].inflight_refs == 2

    def test_commit_normal(self, tracker):
        """Normal commit without EOS returns None."""
        tracker.launch_ref(0)
        result = tracker.commit(0, eos_detected=False)
        assert result is None
        assert tracker.states[0].finalized is False

    def test_commit_eos(self, tracker):
        """Commit with EOS returns 'finalized'."""
        tracker.launch_ref(0)
        result = tracker.commit(0, eos_detected=True)
        assert result == "finalized"
        assert tracker.states[0].finalized is True

    def test_commit_skip_zombie(self, tracker):
        """Commit on already-finalized sequence returns 'skip'."""
        tracker.launch_ref(0)
        tracker.commit(0, eos_detected=True)   # finalized
        tracker.launch_ref(0)                  # zombie launched
        result = tracker.commit(0, eos_detected=False)  # commit zombie
        assert result == "skip"

    def test_release_full_cycle(self, tracker):
        """Full zombie lifecycle: launch → EOS → zombie launch → release."""
        # Step 1: sequence starts.
        tracker.launch_ref(0)
        assert tracker.states[0].inflight_refs == 1

        # Step 1 commit: EOS detected.
        result = tracker.commit(0, eos_detected=True)
        assert result == "finalized"
        assert tracker.states[0].finalized is True

        # Step 2: zombie was already launched (inflight_refs = 1 from step 1).
        tracker.launch_ref(0)  # step 2 launch (zombie rides along)
        assert tracker.states[0].inflight_refs == 2

        # Step 2 commit: should skip zombie.
        result = tracker.commit(0, eos_detected=False)
        assert result == "skip"

        # Release step 1's reference.
        tracker.release(0)
        assert tracker.states[0].inflight_refs == 1
        assert tracker.states[0].finalized is True  # still finalized

        # Release step 2's reference → should recycle.
        tracker.release(0)
        assert tracker.states[0].inflight_refs == 0
        assert tracker.states[0].finalized is False  # reset after recycle

    def test_should_skip(self, tracker):
        """should_skip returns True only for finalized zombies."""
        assert tracker.should_skip(0) is False

        tracker.launch_ref(0)
        tracker.commit(0, eos_detected=True)
        assert tracker.should_skip(0) is True

    def test_out_of_bounds_row(self, tracker):
        """Out-of-bounds row index should not crash."""
        tracker.launch_ref(99)
        # No crash is the test.
