"""HNEP command-line interface.

Subcommands
-----------

* ``hnep card <result.json>``      — single-model summary card
* ``hnep compare <a.json> <b.json> [...]`` — side-by-side comparison
* ``hnep evaluate`` / ``hnep replay`` — stubs (Phase 5 wire-up)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hnep import __version__
from hnep.card import (
    HNEPCard,
    compare_cards_html,
    compare_cards_markdown,
    compare_cards_text,
    load_result_from_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnep",
        description="Hybrid Network Evaluation Protocol — CLI",
    )
    parser.add_argument(
        "--version", action="version", version=f"hnep {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # card
    card_p = sub.add_parser(
        "card", help="Render a single HNEP card from a result JSON file",
    )
    card_p.add_argument("result", help="Path to HNEP result JSON")
    card_p.add_argument(
        "--format", choices=("text", "markdown", "html"), default="text",
        help="Output format (default: text)",
    )
    card_p.add_argument(
        "--output", "-o",
        help="Optional path to write to (otherwise prints to stdout)",
    )

    # compare
    cmp_p = sub.add_parser(
        "compare", help="Side-by-side comparison of two or more HNEP results",
    )
    cmp_p.add_argument(
        "results", nargs="+",
        help="Paths to two or more HNEP result JSON files",
    )
    cmp_p.add_argument(
        "--format", choices=("text", "markdown", "html"), default="text",
    )
    cmp_p.add_argument(
        "--output", "-o",
        help="Optional output path (otherwise prints to stdout)",
    )

    # evaluate (stub)
    eval_p = sub.add_parser(
        "evaluate", help="(stub) run HNEP on a saved model",
    )
    eval_p.add_argument("model", help="Path to model checkpoint")
    eval_p.add_argument("dataset", help="Path to dataset")
    eval_p.add_argument(
        "--output", default="report.html", help="Output report path",
    )

    # replay (stub)
    replay_p = sub.add_parser(
        "replay", help="(stub) replay an evaluation from a manifest",
    )
    replay_p.add_argument("manifest", help="Path to hnep_manifest.yaml")

    return parser


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)


def _run_card(args) -> int:
    try:
        result = load_result_from_json(args.result)
    except FileNotFoundError:
        print(f"error: file not found: {args.result}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to load {args.result}: {exc}", file=sys.stderr)
        return 2

    card = HNEPCard(result)
    if args.format == "text":
        out = card.to_text()
    elif args.format == "markdown":
        out = card.to_markdown()
    else:
        out = card.to_html()
    _emit(out, args.output)
    return 0


def _run_compare(args) -> int:
    if len(args.results) < 2:
        print("error: `hnep compare` needs at least two result files",
              file=sys.stderr)
        return 2
    results = []
    for path in args.results:
        try:
            results.append(load_result_from_json(path))
        except FileNotFoundError:
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: failed to load {path}: {exc}", file=sys.stderr)
            return 2

    if args.format == "text":
        out = compare_cards_text(results)
    elif args.format == "markdown":
        out = compare_cards_markdown(results)
    else:
        out = compare_cards_html(results)
    _emit(out, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print(f"hnep {__version__}  —  Hybrid Network Evaluation Protocol")
        print("")
        print("Common commands:")
        print("  hnep card <result.json>           single-model summary card")
        print("  hnep compare a.json b.json [...]  side-by-side comparison")
        print("  hnep evaluate <model> <dataset>   run the full protocol (stub)")
        print("")
        print("Run `hnep --help` or `hnep <command> --help` for details.")
        return 0

    if args.command == "card":
        return _run_card(args)
    if args.command == "compare":
        return _run_compare(args)

    if args.command in {"evaluate", "replay"}:
        print(
            f"`hnep {args.command}` is a stub in v0.1.0. "
            "Wire-up lands once the trainer integration is complete."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
