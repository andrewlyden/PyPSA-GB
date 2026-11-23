"""Old name of :mod:`scripts.market.cfd_strikes` (year-generic stage 5, 27 Oct 2026).

Kept so modules outside ``scripts/market`` that still import ``cfd_strikes_2050`` keep
working: this name is the same module object, so a constant patched through either
name is seen by both. New code imports ``scripts.market.cfd_strikes``.
"""

import sys

from scripts.market import cfd_strikes as _module

sys.modules[__name__] = _module
