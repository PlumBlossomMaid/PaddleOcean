"""S1: SIGTERM triggers graceful shutdown; handlers are restored on teardown."""

import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ocean

# Windows has no true SIGTERM soft-interrupt semantics: `os.kill(pid, SIGTERM)`
# terminates the process (exit code 15) instead of dispatching to an in-process
# handler, so the live-signal test below can't run there. The register/teardown
# tests above it deliver no real signal and run on all platforms.
_SKIP_WINDOWS_SIGTERM = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM) on Windows terminates the process rather than "
    "dispatching to an in-process handler; live-signal behavior is "
    "exercised under POSIX only.",
)


def _trainer():
    return ocean.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)


def test_register_saves_original_and_installs_handler():
    original = signal.getsignal(signal.SIGTERM)
    trainer = _trainer()
    sc = trainer._signal_connector
    try:
        sc.register_signal_handlers()
        # original handler recorded for later restoration
        assert signal.SIGTERM in sc._original_handlers
        assert sc._original_handlers[signal.SIGTERM] == original
        # a handler is now installed (not the previous one)
        current = signal.getsignal(signal.SIGTERM)
        assert current != original
    finally:
        sc.teardown()


def test_sigterm_requests_graceful_stop():
    trainer = _trainer()
    sc = trainer._signal_connector
    assert sc.received_sigterm is False
    assert trainer.should_stop is False

    sc._sigterm_handler(signal.SIGTERM, None)

    assert sc.received_sigterm is True
    assert trainer.should_stop is True  # fit loop honors this at the next boundary


def test_teardown_restores_original_handler():
    original = signal.getsignal(signal.SIGTERM)
    trainer = _trainer()
    sc = trainer._signal_connector
    sc.register_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) != original
    sc.teardown()
    assert signal.getsignal(signal.SIGTERM) == original
    assert sc._original_handlers == {}


@_SKIP_WINDOWS_SIGTERM
def test_real_sigterm_signal_sets_flag():
    """Deliver an actual SIGTERM and confirm the installed handler runs."""
    original = signal.getsignal(signal.SIGTERM)
    trainer = _trainer()
    sc = trainer._signal_connector
    try:
        sc.register_signal_handlers()
        os.kill(os.getpid(), signal.SIGTERM)  # our handler swallows it (no exit)
        assert sc.received_sigterm is True
        assert trainer.should_stop is True
    finally:
        sc.teardown()
        assert signal.getsignal(signal.SIGTERM) == original
