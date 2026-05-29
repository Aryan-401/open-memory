"""Helper backing the synchronous wrappers exposed on async-first classes.

The core of open-memory is async. For scripts and notebooks we expose thin sync
wrappers (``add``/``get_context``/...) that delegate to their ``a*`` counterparts via
:func:`run_sync`. Calling a sync wrapper from inside a running event loop is an error
(it would deadlock); we raise a clear message steering callers to the async method.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[object, object, T]) -> T:
    """Run *coro* to completion from synchronous code.

    Raises ``RuntimeError`` if invoked while an event loop is already running, since
    the correct call in that context is the ``a*`` (async) method directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to drive one ourselves.
        return asyncio.run(coro)

    coro.close()
    raise RuntimeError(
        "Synchronous method called from within a running event loop. "
        "Use the async variant (the 'a'-prefixed method, e.g. `aadd`/`aget_context`) instead."
    )
