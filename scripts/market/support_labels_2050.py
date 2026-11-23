"""Old name of :mod:`scripts.market.support_labels` (year-generic stage 5, 27 Oct 2026).

Kept so tests and modules outside ``scripts/market`` that import ``support_labels_2050``
keep working: this name is the same module object. New code imports
``scripts.market.support_labels``; the rule runs ``scripts/market/support_labels.py``.
"""

import sys

from scripts.market import support_labels as _module

sys.modules[__name__] = _module
