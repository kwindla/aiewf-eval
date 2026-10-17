"""Build a lossless judge view; hold ambiguous ordering for review."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["live", "gemini"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.provider == "live":
        from multi_turn_eval.auditing.live_reconstruction import reconstruct
    else:
        from multi_turn_eval.auditing.gemini_reconstruction import reconstruct
    if not reconstruct(args.run):
        raise SystemExit("Reconstruction needs review; do not judge automatically")


if __name__ == "__main__":
    main()
