"""HNEP command-line interface.

Phase 1 stub — currently supports ``hnep --version`` and an informational
banner. ``hnep evaluate`` and ``hnep replay`` are wired up in Phase 5 once
the probes and reports are real.
"""

from __future__ import annotations

import argparse
import sys

from hnep import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnep",
        description="Hybrid Network Evaluation Protocol — CLI",
    )
    parser.add_argument("--version", action="version", version=f"hnep {__version__}")
    sub = parser.add_subparsers(dest="command")

    # evaluate (Phase 5 will fill in the implementation)
    eval_p = sub.add_parser("evaluate", help="(stub) run HNEP on a saved model")
    eval_p.add_argument("model", help="Path to model checkpoint")
    eval_p.add_argument("dataset", help="Path to dataset")
    eval_p.add_argument("--output", default="report.html", help="Output report path")

    # replay (Phase 5)
    replay_p = sub.add_parser("replay", help="(stub) replay an evaluation from a manifest")
    replay_p.add_argument("manifest", help="Path to hnep_manifest.yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print(f"hnep {__version__}")
        print("Run `hnep --help` to see available commands.")
        return 0

    if args.command in {"evaluate", "replay"}:
        print(
            f"`hnep {args.command}` is a stub in v0.1.0.dev0. "
            "Wire-up lands once Phase 5 is complete."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
