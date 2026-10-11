"""CI entry point.  All numbers come from evaluation/run_eval.py; this file only
chooses CI-friendly options (lexical stage, no model downloads) and forwards the
exit code, so a failed gate fails the build."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.run_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--mode", "bm25", "--generate"] + sys.argv[1:]))
