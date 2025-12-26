"""Minimal stub for `tsr` package used by neoforge_core.py for local testing.

This stub provides a `TSR.from_pretrained` factory that returns a
lightweight DummyModel producing a simple mesh. Replace with the real
TripoSR/TRELLIS implementation for production.
"""

from .system import TSR  # re-export

__all__ = ["TSR"]
