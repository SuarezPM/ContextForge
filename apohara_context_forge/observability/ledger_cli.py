"""CLI: verify a FORGE-LEDGER hash-chained ledger.
Usage: python -m apohara_context_forge.observability.ledger_cli verify <path>
Exit codes: 0 = chain intact, 2 = tampered/broken, 64 = usage error."""
from __future__ import annotations
import json
import sys

from apohara_context_forge.observability.ledger import Ledger


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "verify":
        print(
            "usage: python -m apohara_context_forge.observability.ledger_cli verify <path>",
            file=sys.stderr,
        )
        return 64
    result = Ledger(argv[1]).verify()
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
