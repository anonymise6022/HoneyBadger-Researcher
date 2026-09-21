"""Put the repository root on sys.path so tests can import `quant_pipeline`.

The project is not pip-installed, and pytest's default import mode prepends
the test file's own directory rather than the root. Without this, every
test module would need its own path manipulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
