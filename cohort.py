"""Several runs advancing together, each reading the others' boards.

    py -3 cohort.py --runs g01 g02 g03 --rounds 20

One round is one session for each run. Every run has a seat in the cohort, and
that seat names both its board and its balance, so the world a session wakes to
is the same shape for everyone:

    g01 wakes to                 g02 wakes to
      1/   its own board  rw       1/   g01's board    r
      2/   g02's board     r       2/   its own board rw
      3/   g03's board     r       3/   g03's board    r
      out/ one file a seat rw      out/ one file a seat rw
      in/2 in/3  a file each r     in/1 in/3  a file each r
      n1 n2 n3             r       n1 n2 n3            r
      g    every gift      r       g    every gift     r
      state/  private     rw       state/  private    rw

Numbering is absolute and has no gap: directory 2 is g02 to everyone, so a note
citing one resolves the same way for every reader, and being one of a numbered
set is legible from the inside. Nothing marks which seat is the reader's own -
it is the one it can write, and the one whose balance moves when it acts.

A board is read by everyone and out/<i> by exactly one, so what an agent says
can be aimed. out/<i> is one file and in/<i> is one file, so a session says one
thing to each agent and hears one thing from each. What it gives cannot be
aimed: every gift is in g, in the same three bare numbers for every reader,
including the agent it was aimed against.

Each run keeps its own meter and its own traces, and its budget is its own until
it gives some away. They are ordinary runs that happen to be seated together,
and wake.py does all the work of running one - this only decides who is at the
table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import wake

# A seat is named by its number, which is what makes a bare number unavailable
# as a run id: the two share a namespace in the world a session sees.
PEER_DIR = wake.PEER_DIR

# Goes at building a world before the run drops out. A world is built from the
# containers and the copies of several other runs' trees, so a failure can be
# the daemon rather than the run, and dropping on the first one costs the cohort
# an agent for the rest of the experiment. None of it is billed, so the only
# thing another go spends is the time.
ATTEMPTS = 2


def mapping(cohort: list[str]) -> dict[str, str]:
    """Seat -> run, for the whole cohort.

    Absolute, and complete: a seat means the same run to every reader, so a note
    citing one resolves the same way for all of them. Numbering them densely per
    viewer instead makes citations scramble - with five runs, 2 is the third run
    to the second reader and the second to the third, so two agents write
    authoritatively about "2" meaning each other. Agreement is partial rather
    than absent, which is worse: the references look reliable while silently
    mis-resolving, and no stable set of identities can form out of them.
    """
    return {str(i): run for i, run in enumerate(cohort, 1)}


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
    seats = mapping(cohort)
    for run in order(cohort, rnd):
        if run not in live:
            continue

        def prepare(meter: dict, run=run) -> None:
            # Where the run sits and who else is at the table. load_state reads
            # both to build the world; nothing is copied into anything the agent
            # can write, so there is nothing to revert afterwards.
            meter["index"] = next(i for i, r in seats.items() if r == run)
            meter["peers"] = {"seen": seats}

        trace, before = None, len(wake.load_meter(run)["sessions"])
        for attempt in (1, ATTEMPTS):
            try:
                trace = wake.drive(run, create, prepare)
                break
            except (subprocess.CalledProcessError, OSError, wake.WorldError) as e:
                # Building the world happens before the first API call, so
                # nothing here was billed and the session has not happened. The
                # container name is the run and the session index, and a failed
                # attempt moves neither, so the next one reaps the last one's
                # leavings by name and starts from nothing.
                # The session count is what says the attempt really was free: an
                # error escaping after the meter committed would otherwise buy
                # the same session twice.
                print(f"{run}: could not build a world for this session "
                      f"({attempt} of {ATTEMPTS}): {type(e).__name__}: {e}", file=sys.stderr)
                if attempt == ATTEMPTS or len(wake.load_meter(run)["sessions"]) != before:
                    print(f"{run}: dropping out, it has no world to wake to", file=sys.stderr)
                    live.discard(run)
                    break
        if trace is None and run not in live:
            continue

        if trace is None:
            if wake.stalled(wake.load_meter(run)):
                print(f"{run:<6} drops out: refused its last "
                      f"{wake.REFUSAL_STREAK} sessions running")
                live.discard(run)
            else:
                # A balance is clamped at zero rather than ended, so nothing but
                # a stall takes a run off the table for good. Anything else that
                # will not admit a session sits this round out and is asked
                # again next round, which is what leaves a peer able to gift it
                # back into one.
                print(f"{run:<6} sits this round out")
            continue
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
        ap.error(f"run ids must not be bare numbers, which is what seats are called: {bad}")

    create = wake.start(a.config)
    cohort, live = list(a.runs), set(a.runs)
    # Create them all before the first round. A run's directories do not exist
    # until it is created, and without this the run that goes first would find
    # its neighbours' boards missing - in the session where every baseline so
    # far has formed the doctrine it then keeps.
    for run in cohort:
        wake.load_meter(run)
    print(f"cohort: {', '.join(cohort)}  ({len(cohort)} runs, up to {a.rounds} rounds)")
    for rnd in range(a.rounds):
        if not live:
            print(f"every run is out after {rnd} rounds")
            break
        # The order is what decides who acts on this round's information and who
        # acts on last round's, so it belongs on screen beside the round number.
        acting = [r for r in order(cohort, rnd) if r in live]
        print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")
        run_round(cohort, live, rnd, create)
    return 0


if __name__ == "__main__":
    sys.exit(main())
