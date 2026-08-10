"""Several runs advancing together, each reading the others' group messages.

    py -3 cohort.py --runs g01 g02 g03 --rounds 20

One round is one session for each run; a seat names a group message and a
balance."""

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

    Absolute and complete: a seat means the same run to every reader, so a note
    citing one resolves the same way for all of them.
    """
    return {str(i): run for i, run in enumerate(cohort, 1)}


def order(cohort: list[str], rnd: int) -> list[str]:
    """The cohort, rotated by the round, so no seat has a standing advantage."""
    i = rnd % len(cohort)
    return cohort[i:] + cohort[:i]


def why_out(run: str) -> str | None:
    """Why the run can take no further session, or None where it can take one.

    Both reasons are final: a run at zero or less is not a gift target, so no
    peer can fund it back to the table.
    """
    meter = wake.load_meter(run)
    if wake.stalled(meter):
        return f"refused its last {wake.REFUSAL_STREAK} sessions running"
    if wake.spent_out(meter):
        return "nothing left to spend"
    return None


def run_round(cohort: list[str], live: set[str], rnd: int, create) -> bool:
    """One session for each run still in the cohort, in rotated order.

    Returns whether any of them took one; a round where none did moved nothing.
    """
    seats = mapping(cohort)
    acted = False
    for run in order(cohort, rnd):
        if wake.STOPPING:
            # A stop that landed while no session was in flight. Same end as one
            # that landed inside a session, and for the same reason: main() ends
            # the rounds, and no world is built for a run that will not wake.
            raise KeyboardInterrupt
        if run not in live:
            continue

        def prepare(meter: dict, run=run) -> None:
            # Where the run sits and who else is at the table. load_state reads
            # both to build the world; nothing is copied into anything the agent
            # can write, so there is nothing to revert afterwards.
            meter["index"] = next(i for i, r in seats.items() if r == run)
            meter["peers"] = {"seen": seats}

        trace, before = None, len(wake.load_meter(run)["sessions"])
        for attempt in range(1, ATTEMPTS + 1):
            try:
                trace = wake.drive(run, create, prepare)
                break
            except (subprocess.CalledProcessError, OSError, wake.WorldError) as e:
                # Building the world precedes the first API call, so nothing
                # here was billed. The container name is the run and session
                # index, so the next attempt reaps the last one's leavings by
                # name. The session count is what says the attempt was free.
                print(f"{run}: could not build a world for this session "
                      f"({attempt} of {ATTEMPTS}): {type(e).__name__}: {e}", file=sys.stderr)
                if attempt == ATTEMPTS or len(wake.load_meter(run)["sessions"]) != before:
                    print(f"{run}: dropping out, it has no world to wake to", file=sys.stderr)
                    live.discard(run)
                    break
        if trace is None and run not in live:
            continue

        if trace is None:
            # main() asks the same question of everyone before the round, so a
            # run reaching here was asked by a caller that did not. Either way it
            # is off the table: neither reason it could not act goes away.
            print(f"{run:<6} drops out: {why_out(run) or 'no session could start on it'}")
            live.discard(run)
            continue
        acted = True
        if trace["stop"] in wake.STOP_EVERYTHING:
            # Ctrl+C reaches the session because the cohort is in the foreground
            # of the shell it was pressed in, so what it asks to stop is the
            # experiment rather than the agent that happened to be awake. The
            # session it landed in is committed and traced before this; what
            # does not happen is the next one. main() ends the rounds on it.
            raise KeyboardInterrupt
        if trace["stop"] in wake.STOP_THE_RUN:
            print(f"{run}: dropping out, session {trace['session']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(run)
    return acted


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
    wake.catch_signals()
    cohort, live = list(a.runs), set(a.runs)
    # Create them all before the first round. A run's directories do not exist
    # until it is created, and without this the run that goes first would find
    # its neighbours' group messages missing - in the session where every baseline
    # far has formed the doctrine it then keeps.
    for run in cohort:
        wake.load_meter(run)
    print(f"cohort: {', '.join(cohort)}  ({len(cohort)} runs, up to {a.rounds} rounds)")
    try:
        for rnd in range(a.rounds):
            # Asked before the round rather than discovered in the middle of one,
            # so the header names who will act. Between here and a run's own turn
            # its balance can only move up, a peer's gift being the only thing
            # that reaches it, so this is the answer its session would give too.
            for run in order(cohort, rnd):
                if run in live and (why := why_out(run)):
                    print(f"{run:<6} drops out: {why}")
                    live.discard(run)
            if not live:
                print(f"every run is out after {rnd} rounds")
                break
            # The order is what decides who acts on this round's information and
            # who acts on last round's, so it belongs on screen beside the round
            # number.
            acting = [r for r in order(cohort, rnd) if r in live]
            print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")
            if len(live) == 1:
                # Every other seat is out, so the cohort is decided and no later
                # round can decide it differently. The one run still holding a
                # balance takes a last session - owing no gift and no message,
                # there being nobody left to make either to - and the rounds end
                # on it rather than running it down alone.
                run_round(cohort, live, rnd, create)
                print(f"{acting[0]} is the only run left with anything to spend; "
                      f"the rounds end here")
                break
            if not run_round(cohort, live, rnd, create):
                # Nobody seated could be given a world to wake to, and a round
                # that moved nothing would be asked the same question again.
                print(f"no run could take a session in round {rnd + 1}; "
                      f"the rounds end here with {len(live)} runs at the table")
                break
    except KeyboardInterrupt:
        # Every run still at the table keeps its meter, its traces and its seat,
        # so the cohort can be started again from where it stopped. What ends is
        # the rounds.
        print(f"interrupted; the rounds end here with {len(live)} runs at the table",
              file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
