"""`python -m recoup.compare` -- score every strategy against the same holdout.

This is the headline result of the project, so it gets a real command rather than a
one-liner. A long quoted one-liner is fine to type once and a liability everywhere
else: it wraps when pasted into a document, breaks when copied out of a PDF, and
quotes differently in bash, PowerShell and cmd. The command you have to run in front
of an audience should be short enough that it cannot break.
"""

from __future__ import annotations

import sys

from recoup.generator.synthetic import ScenarioGenerator
from recoup.measure.harness import compare, run
from recoup.policy.strategies import STRATEGIES


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    n_events = 5000
    for i, arg in enumerate(args):
        if arg in ("-n", "--events") and i + 1 < len(args):
            n_events = int(args[i + 1])

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    scenario = ScenarioGenerator().generate(n_events=n_events)
    print()
    print(compare([run(scenario, fn, name) for name, fn in STRATEGIES.items()]))
    print()
    print("  GROSS       every rupee that arrived on an event we touched.")
    print("  INCREMENTAL what arrived BECAUSE we touched it, against a random holdout.")
    print("  Gross is what every recovery tool reports. Incremental is what it caused.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
