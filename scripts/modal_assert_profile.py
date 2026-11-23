"""Exit non-zero unless the active Modal profile is faheemgurkani (faheemgurkani@gmail.com)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modal_app.settings import assert_expected_modal_profile  # noqa: E402

try:
    assert_expected_modal_profile()
except RuntimeError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
