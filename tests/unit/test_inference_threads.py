"""Tests for the single-state inference thread cap.

MCTS evaluates one position at a time, so self-play is batch-of-1 inference.
Torch defaults its intra-op pool to the core count, which on an oversubscribed
VM makes each forward pass dramatically slower (measured 1.54 ms at 2 threads
vs 347 ms at 4 on a 4-vCPU box). These tests pin the resolution rules and the
restore-on-exit contract, not the timing, which is machine-dependent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from alpha_zero.utils.device import (  # noqa: E402
    DEFAULT_MAX_INFERENCE_THREADS,
    INFERENCE_THREADS_ENV_VAR,
    inference_thread_limit,
    resolve_inference_threads,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an ambient CIV6AZ_INFERENCE_THREADS out of these assertions."""
    monkeypatch.delenv(INFERENCE_THREADS_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _restore_threads():
    """Fail loudly rather than leaking a thread setting into later tests."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


class TestResolveInferenceThreads:
    def test_default_caps_at_the_documented_maximum(self) -> None:
        resolved = resolve_inference_threads()
        assert resolved is not None
        assert resolved <= DEFAULT_MAX_INFERENCE_THREADS

    def test_default_never_exceeds_the_current_torch_setting(self) -> None:
        # A single-core box must not be asked for two threads.
        assert resolve_inference_threads() <= torch.get_num_threads()

    def test_explicit_argument_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, "1")
        assert resolve_inference_threads(3) == 3

    @pytest.mark.parametrize("value", ["auto", "0", "", "  AUTO  "])
    def test_opt_out_values_resolve_to_none(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, value)
        assert resolve_inference_threads() is None

    def test_explicit_zero_opts_out(self) -> None:
        assert resolve_inference_threads(0) is None

    def test_env_integer_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, "3")
        assert resolve_inference_threads() == 3

    def test_unparseable_env_degrades_to_no_change(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A typo in an env var must not take down a training run.
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, "two")
        with caplog.at_level("WARNING"):
            assert resolve_inference_threads() is None
        assert INFERENCE_THREADS_ENV_VAR in caplog.text


class TestInferenceThreadLimit:
    def test_applies_and_restores(self) -> None:
        before = torch.get_num_threads()
        with inference_thread_limit(1) as applied:
            assert applied == 1
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == before

    def test_restores_on_exception(self) -> None:
        before = torch.get_num_threads()
        with pytest.raises(RuntimeError), inference_thread_limit(1):
            raise RuntimeError("boom")
        assert torch.get_num_threads() == before

    def test_opt_out_leaves_threads_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, "auto")
        before = torch.get_num_threads()
        with inference_thread_limit() as applied:
            assert applied is None
            assert torch.get_num_threads() == before

    def test_nested_limits_restore_in_order(self) -> None:
        before = torch.get_num_threads()
        with inference_thread_limit(2):
            outer = torch.get_num_threads()
            with inference_thread_limit(1):
                assert torch.get_num_threads() == 1
            assert torch.get_num_threads() == outer
        assert torch.get_num_threads() == before

    def test_noop_when_already_at_target(self) -> None:
        torch.set_num_threads(2)
        with inference_thread_limit(2) as applied:
            assert applied == 2
            assert torch.get_num_threads() == 2
        assert torch.get_num_threads() == 2
