"""Do the documents still agree with the results they quote?

    python scripts/check_docs.py --results results

Fails if any generated table in the README or `docs/results.md` differs from
what the current result files would produce, and lists the lines that differ.

This exists because the alternative does not work. A README with hand-typed
numbers is correct on the day it is written and wrong a month later, and
nothing announces it: the prose still reads perfectly well. Running this in
`make check` means a stale number is a build failure rather than a thing a
reader has to take on trust.
"""

from __future__ import annotations

import argparse
import difflib
import sys

import _bootstrap  # noqa: F401
import _tables


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(_bootstrap.default_results_root()))
    p.add_argument("--documents", nargs="*", default=None)
    p.add_argument("--allow-missing", action="store_true",
                   help="pass when a table has no results yet, instead of failing")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    documents = args.documents or _tables.DOCUMENTS

    problems = []
    for name in documents:
        path = _bootstrap.ROOT / name
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        expected, tables = _tables.render(current, args.results)
        if not tables:
            continue

        if _tables.MISSING in expected and not args.allow_missing:
            missing = [t for t in tables
                       if _tables.MISSING in _tables.BUILDERS[t](args.results)]
            problems.append(
                "%s quotes %s but the result files for them are absent; run "
                "`make experiments`" % (name, ", ".join(missing))
            )
            continue

        if expected != current:
            diff = list(difflib.unified_diff(
                current.splitlines(), expected.splitlines(),
                fromfile="%s (committed)" % name, tofile="%s (from results)" % name,
                lineterm="", n=1,
            ))
            problems.append(
                "%s is out of date; run `make tables`.\n    %s"
                % (name, "\n    ".join(diff[:40]))
            )
        else:
            say("  %s: up to date (%s)" % (name, ", ".join(tables)))

    if problems:
        say("")
        for p in problems:
            say("FAIL  " + p)
        return 1
    say("documents agree with the results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
