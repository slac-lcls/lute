"""Resolve an eLog sample name to its run list.

Thin CLI wrapper around `lute.io.elog.get_elog_runs_by_sample`, the sample
counterpart of `resolve_tag.py`. For submission scripts that need to know a
--sample's run list before/without submitting anything - e.g. to pre-create
per-run output directories, which LUTE does not do itself (see
`references/cctbx-sfx-workflow.md` §4 in the ask-lute skill).
`launch_scripts/tagged_launch.py::run_sampled_workflow` resolves the same
sample internally when actually submitting; this is the same lookup exposed
standalone.
"""

from __future__ import annotations

import argparse
import sys

# The eLog's /ws/runs endpoint treats both of these as "no sampleName filter"
# and happily returns every run of the experiment - which, printed as a run
# list, silently looks like a successful resolution of an enormous sample.
_NO_FILTER_SAMPLE_NAMES = ("", "All Samples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print every run number associated with a given eLog sample."
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-s", "--sample", type=str, required=True)
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Separator between printed run numbers (default: a single space).",
    )
    args = parser.parse_args()

    if args.sample in _NO_FILTER_SAMPLE_NAMES:
        print(
            f"Refusing to resolve sample name '{args.sample}': the eLog treats "
            "it as no filter at all and would return every run of "
            f"'{args.experiment}'. Pass a real sample name.",
            file=sys.stderr,
        )
        sys.exit(2)

    from lute.io.elog import get_elog_runs_by_sample

    runs = sorted(get_elog_runs_by_sample(args.experiment, args.sample))
    if not runs:
        print(
            f"No runs found for sample '{args.sample}' in experiment "
            f"'{args.experiment}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(args.separator.join(str(r) for r in runs))


if __name__ == "__main__":
    main()
