"""Rewrite every generated table in the documents from the result files.

    python scripts/make_tables.py --results results

Every number quoted in a table in the README or in `docs/results.md` is written
by this script from the JSON the pipeline produced. Nothing is typed in by
hand, which is the only way a document and a result stay in agreement over
more than about a week.

`scripts/check_docs.py` is the other half: it runs the same builders and fails
if a document has drifted.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import _tables


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(_bootstrap.default_results_root()))
    p.add_argument("--documents", nargs="*", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    documents = args.documents or _tables.DOCUMENTS

    changed = 0
    for name in documents:
        path = _bootstrap.ROOT / name
        if not path.exists():
            say("  %s does not exist, skipping" % name)
            continue
        before = path.read_text(encoding="utf-8")
        after, tables = _tables.render(before, args.results)
        if not tables:
            say("  %s has no generated tables" % name)
            continue
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
            say("  %s: rewrote %s" % (name, ", ".join(tables)))
        else:
            say("  %s: already up to date (%s)" % (name, ", ".join(tables)))

    say("%d document(s) changed" % changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
