"""Audit the five-second Live lifecycle/playback completion gate."""

import argparse
from multi_turn_eval.auditing.live_gate import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    if not audit(args.run)["approved"]:
        raise SystemExit("Completion gate audit requires review")


if __name__ == "__main__":
    main()
