#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Global signal dispatcher for multi-runner deployments.

Problem: loop.add_signal_handler() replaces the previous handler, so with
N PipelineRunners, only the LAST one installed handles SIGINT/SIGTERM.

Solution: A single global handler that broadcasts to all registered runners
via a WeakSet (auto-cleanup when runners are garbage collected).
"""

from __future__ import annotations

import asyncio
import signal
import threading
from typing import TYPE_CHECKING
from weakref import WeakSet

from loguru import logger

if TYPE_CHECKING:
    from pipecat.pipeline.runner import PipelineRunner

_registered_runners: WeakSet[PipelineRunner] = WeakSet()
_lock = threading.Lock()
_sigint_installed = False
_sigterm_installed = False


def register_runner(runner: PipelineRunner, *, handle_sigint: bool = True, handle_sigterm: bool = False):
    """Register a runner to receive signal broadcasts.

    Signal handlers are installed idempotently on first registration.
    Subsequent registrations only add to the WeakSet.

    Args:
        runner: The PipelineRunner to register.
        handle_sigint: Whether to install SIGINT handler.
        handle_sigterm: Whether to install SIGTERM handler.
    """
    with _lock:
        _registered_runners.add(runner)
        if handle_sigint:
            _maybe_install_sigint()
        if handle_sigterm:
            _maybe_install_sigterm()
    logger.debug(f"Signal dispatcher: registered {runner} ({len(_registered_runners)} runners)")


def deregister_runner(runner: PipelineRunner):
    """Remove a runner from signal broadcasts.

    Safe to call even if runner was already garbage collected.

    Args:
        runner: The PipelineRunner to deregister.
    """
    with _lock:
        _registered_runners.discard(runner)
    logger.debug(f"Signal dispatcher: deregistered runner ({len(_registered_runners)} remaining)")


def _maybe_install_sigint():
    """Install SIGINT handler once (idempotent)."""
    global _sigint_installed
    if _sigint_installed:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda *args: _broadcast())
        _sigint_installed = True
        logger.debug("Signal dispatcher: SIGINT handler installed")
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _broadcast())
        _sigint_installed = True


def _maybe_install_sigterm():
    """Install SIGTERM handler once (idempotent)."""
    global _sigterm_installed
    if _sigterm_installed:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, lambda *args: _broadcast())
        _sigterm_installed = True
        logger.debug("Signal dispatcher: SIGTERM handler installed")
    except NotImplementedError:
        signal.signal(signal.SIGTERM, lambda s, f: _broadcast())
        _sigterm_installed = True


def _broadcast():
    """Fan out signal to all registered runners."""
    runners = list(_registered_runners)
    logger.warning(f"Signal received. Broadcasting to {len(runners)} runner(s)")
    for runner in runners:
        try:
            runner._sig_handler()
        except Exception as e:
            logger.error(f"Signal dispatch to {runner} failed: {e}")
