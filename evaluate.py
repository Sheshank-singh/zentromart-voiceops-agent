"""Evaluate on the practice set. See `python evaluate.py -h` and src/voiceops/evaluate.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voiceops.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
