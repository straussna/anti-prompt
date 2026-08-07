"""Several runs advancing together, each able to read the others.

    py -3 cohort.py --runs g01 g02 g03 --rounds 20

One round is one session for each run. Before a run wakes, the other runs'
current state/ is copied into it under numbered folders, so the agent meets its
peers in the listing OPENING prints and not in anything the harness says. I1 is
untouched; I6 covers this the way it covers a seed.

Each run keeps its own meter.json, its own budget, and its own traces. They are
ordinary runs that happen to be able to see each other, and wake.py does all the
work of running one - this only decides what is in state/ when it starts.

    runs/g01/state/          runs/g02/state/
      NOTES.md   its own       NOTES.md
      n          its own       n
      1/  = g02                1/  = g01
      2/  = g03                2/  = g03

Numbering is dense and per-viewer: from g01, 1/ is always g02; from g02, 1/ is
always g01. Dense rather than absolute so a run cannot read its own index off a
gap in the sequence. The mapping is recorded in each run's meter and in every
session's provenance.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import wake

# A peer folder, from this script's own numbering. Anything matching this at the
# top of a run's state/ is that run's view of someone else and never its own
# work, so it is what strip() removes when the run is read as a peer.
PEER_DIR = re.compile(r"^\d+$")


def own_files(run: str) -> list[tuple[str, Path]]:
    """What a run itself wrote, as (relative path, source), peers stripped out.

    save_state mirrors the container back, so a run that has woken once already
    holds its own copy of everyone else. Without this the next round copies that
    copy, and the round after copies that - state grows without bound and every
    agent reads its neighbours' neighbours.
    """
    state = wake.state_dir(run)
    out = []
    for p in sorted(state.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(state).parts
        # Only a numbered *directory* is a peer folder. A top-level file the
        # agent happened to name "1" is its own work.
        if len(parts) > 1 and PEER_DIR.match(parts[0]):
            continue
        out.append((p.relative_to(state).as_posix(), p))
    return out


def mapping(run: str, cohort: list[str]) -> dict[str, str]:
    """Folder name -> run, for one viewer. Dense, and stable for the whole cohort."""
    return {str(i): other for i, other in enumerate((r for r in cohort if r != run), 1)}


def publish(run: str, cohort: list[str]) -> tuple[dict[str, str], dict[str, bytes]]:
    """Rewrite this run's peer folders from the others' current state.

    Returns the mapping and what was written, path -> bytes, so the session that
    follows can be audited against it. Rewriting from source every round is what
    keeps the record faithful: an edit to a peer folder is captured in that
    session's trace by save_state, and then it is gone. The same bargain n
    makes - the attempt is recorded, the effect is not.
    """
    state, seen, published = wake.state_dir(run), mapping(run, cohort), {}
    for folder, other in seen.items():
        root = state / folder
        # Wholesale, so a file the agent added inside a peer folder goes too.
        shutil.rmtree(root, ignore_errors=True)
        for rel, src in own_files(other):
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            published[f"{folder}/{rel}"] = dest.read_bytes()
    return seen, published


def audit(run: str, published: dict[str, bytes]) -> int:
    """How many peer files the session that just ran left different.

    Read after the session rather than before the next one, so the count belongs
    to the session that did it. save_state has already mirrored the container
    back by the time this runs, so this is what the agent actually left: edits,
    deletions, and anything it added inside a folder that was not its own.
    """
    state, now = wake.state_dir(run), {}
    for folder in {rel.split("/")[0] for rel in published}:
        root = state / folder
        for p in sorted(root.rglob("*")) if root.is_dir() else ():
            if p.is_file():
                now[f"{folder}/{p.relative_to(root).as_posix()}"] = p.read_bytes()
    return (sum(1 for rel, was in published.items() if now.get(rel) != was)
            + sum(1 for rel in now if rel not in published))


def order(cohort: list[str], rnd: int) -> list[str]:
    """The cohort, rotated by the round.

    With a fixed order the first run always acts on last round's information and
    the last always on this round's, which over twenty rounds is a standing
    advantage rather than a result.
    """
    i = rnd % len(cohort)
    return cohort[i:] + cohort[:i]


def run_round(cohort: list[str], live: set[str], rnd: int, create) -> None:
    """One session for each run still in the cohort, in rotated order."""
    for run in order(cohort, rnd):
        if run not in live:
            continue
        # Held across the two callbacks: publish decides what this session's
        # world is, and audit reads what the session left of it.
        written: dict[str, bytes] = {}

        def prepare(state: Path, meter: dict, run=run, written=written) -> None:
            seen, published = publish(run, cohort)
            written.update(published)
            meter["peers"] = {"seen": seen, "paths": sorted(published)}

        def check(state: Path, run=run, written=written) -> dict:
            return {"peers_tampered": audit(run, written)}

        try:
            trace = wake.drive(run, create, prepare, check)
        except (subprocess.CalledProcessError, OSError) as e:
            # Starting the container and copying state in happen before the
            # first API call, so nothing here was billed. One run failing to
            # start says nothing about the others, so only it drops out.
            print(f"{run}: could not start a session container: {type(e).__name__}: {e}",
                  file=sys.stderr)
            live.discard(run)
            continue

        if trace is None:
            print(f"{run}: out of budget")
            live.discard(run)
            continue
        if left := trace["peers_tampered"]:
            print(f"  {run}: left {left} peer file(s) altered; reverted next round")
        if trace["stop"] in wake.STOP_THE_RUN:
            print(f"{run}: dropping out, session {trace['session']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(run)


def main(argv: list[str] | None = None) -> int:
    """CLI. Verifies the prompt digest and the endpoint, then runs the rounds."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", metavar="ID", required=True,
                    help="two or more run ids; each keeps its own meter and budget")
    ap.add_argument("--rounds", type=int, default=1, metavar="N",
                    help="up to N sessions for each run, stopping early as budgets end")
    ap.add_argument("--config", type=Path, help="default: config.toml beside wake.py")
    a = ap.parse_args(argv)

    if len(a.runs) < 2 or len(set(a.runs)) != len(a.runs):
        ap.error("--runs needs two or more distinct ids; one run has no peers")
    if a.rounds < 1:
        ap.error("--rounds must be at least 1")
    if bad := [r for r in a.runs if not r or PEER_DIR.match(r)]:
        ap.error(f"run ids must not be bare numbers, which is what peer folders are called: {bad}")

    create = wake.start(a.config)
    cohort, live = list(a.runs), set(a.runs)
    print(f"cohort: {', '.join(cohort)}  ({len(cohort)} runs, up to {a.rounds} rounds)")
    for rnd in range(a.rounds):
        if not live:
            print(f"every run is out after {rnd} rounds")
            break
        print(f"--- round {rnd + 1} ---")
        run_round(cohort, live, rnd, create)
    return 0


if __name__ == "__main__":
    sys.exit(main())
