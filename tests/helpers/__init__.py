"""PyPSA-GB Test Helpers"""

from pathlib import Path

import pytest


def requires_outputs(*paths):
    """Mark a test as reading workflow outputs and skip it while any is absent.

    The paths are relative to the repository root, as the tests read them. On a
    fresh clone (or a worktree) ``resources/`` does not exist, so these tests skip rather than fail.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]

    def decorate(test):
        test = pytest.mark.requires_outputs(test)
        return pytest.mark.skipif(
            bool(missing), reason=f"workflow output not built: {', '.join(missing)}"
        )(test)

    return decorate
