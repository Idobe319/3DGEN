import os
import time
import json
from typing import Any, Callable

LOG_PATH = os.path.join('workspace', 'neoforge.log')

os.makedirs('workspace', exist_ok=True)

# Progress callbacks are not used across subprocess boundaries, but they
# are useful when the logger is imported in the same process (tests / UI).
_progress_callbacks: list[Callable[[dict[str, Any]], None]] = []


def log(msg: str, level: str = 'INFO'):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {level}: {msg}"
    try:
        with open(LOG_PATH, 'a', encoding='utf8') as fh:
            fh.write(line + '\n')
    except Exception:
        pass
    print(line)


def register_progress_callback(cb: Callable[[dict[str, Any]], None]) -> None:
    """Register a callback that receives progress dicts when progress is emitted.

    Callbacks are executed synchronously in the logging thread/process.
    """
    try:
        _progress_callbacks.append(cb)
    except Exception:
        pass


def _emit_progress_to_callbacks(payload: dict[str, Any]) -> None:
    for cb in list(_progress_callbacks):
        try:
            cb(payload)
        except Exception:
            pass


def progress(stage: str, percent: float | None = None, details: dict | None = None) -> None:
    """Emit a structured progress update.

    This writes a human-readable log line and also prints a machine-parseable
    JSON prefix (PROGRESS_JSON:) so external UIs (running this as a subprocess)
    can parse and render progress without separate IPC.
    """
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    payload: dict[str, Any] = {"ts": ts, "stage": stage}
    if percent is not None:
        try:
            payload["percent"] = float(percent)
        except Exception:
            payload["percent"] = percent
    if details is not None:
        payload["details"] = details

    # Log as normal
    try:
        with open(LOG_PATH, 'a', encoding='utf8') as fh:
            fh.write(f"[{ts}] PROGRESS: {json.dumps(payload, ensure_ascii=False)}\n")
    except Exception:
        pass

    # Also print a machine-parsable JSON line so UIs can capture it from stdout
    try:
        print("PROGRESS_JSON: " + json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

    # Notify in-process callbacks (useful for tests that import the logger)
    try:
        _emit_progress_to_callbacks(payload)
    except Exception:
        pass


# Backwards-compatibility: some modules import the logger function directly
# (from neoforge_logger import log as nlog) and expect to call `nlog.progress(...)`.
# Attach helper functions as attributes on the `log` function to preserve that
# calling style without changing every import site.
try:
    # `log` is a function object; attach attributes if possible
    setattr(log, 'progress', progress)
    setattr(log, 'register_progress_callback', register_progress_callback)
except Exception:
    # If attaching fails, ignore; callers should still be able to import helpers
    pass
