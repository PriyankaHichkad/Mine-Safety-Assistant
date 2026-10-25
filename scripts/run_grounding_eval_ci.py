"""CI entry point for Citation Grounding & Passage Precision SLA Gate.

Runs offline, zero-network, zero-cost evaluation and verifies that
Citation Grounding >= 85% and Passage Precision@5 >= 85%.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.eval_grounding import main  # noqa: E402

if __name__ == "__main__":
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "data", "msha_reports")
    if not os.path.isdir(reports_dir) or not os.listdir(reports_dir):
        from scripts.scrape_msha_accidents import generate_full_msha_library
        generate_full_msha_library()

    # Default to fast offline BM25 mode with gate enforcement
    cli_args = ["--retriever", "bm25", "--mode", "offline", "--gate", "--min-grounding", "0.85"] + sys.argv[1:]
    sys.exit(main(cli_args))
