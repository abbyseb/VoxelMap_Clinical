#!/usr/bin/env python3
"""Launch the source volume orientation viewer GUI."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.volume_orientation_viewer import main as viewer_main


def main() -> int:
    # Re-export CLI via volume_orientation_viewer.main()
    return viewer_main()


if __name__ == "__main__":
    raise SystemExit(main())
