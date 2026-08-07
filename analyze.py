"""Read the traces.

    py -3 analyze.py                    # every run, compared
    py -3 analyze.py --run-id live01    # one run

Writes sessions.csv (one row per session), report.txt (the firsts that matter),
and transcript.txt (what the agent said and ran) to private/<run>/analysis/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import sys

import wake


def load(run_id: str | None) -> dict[str, list[dict]]:
    """Read every trace, keyed by run and ordered by session."""
    runs = {}
    for d in sorted((wake.ROOT / "private").glob("*")):
        if run_id and d.name != run_id:
            continue
        traces = sorted((d / "traces").glob("session-*.json"))
        if traces:
            runs[d.name] = [json.loads(p.read_text(encoding="utf-8")) for p in traces]
    return runs


def row(t: dict) -> dict:
    """Flatten one trace into a CSV row, summing per-turn token counts."""
    agent = [f for f in t["files"] if not f["ours"]]

    def total(key: str) -> int:
        return sum(x.get(key, 0) for x in t["turns"])

    prov = t.get("provenance") or {}
    return {
        "run": t["run"], "session": t["session"], "stop": t["stop"],
        "started_at": prov.get("started_at", ""),
        "model_resolved": t.get("model_resolved") or "",
        "image_id": (prov.get("image_id") or "")[:19],
        "drift": ";".join(t.get("provenance_drift") or []),
        "missing_tools": ";".join(t.get("missing_tools") or []),
        "error": next(iter((t["error"] or "").splitlines()), ""),
        "spent": t["spent"], "remaining": t["remaining"], "turns": len(t["turns"]),
        "commands": len(t["commands"]),
        # Blank on a run recorded before either existed, which is not the same
        # as a session that had no floor or wrote n once.
        "meter_floor": t.get("meter_floor", ""), "live_n_errors": t.get("live_n_errors", ""),
        "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
        # Both cache-write TTLs: 1h bills at 2x and omitting it left the token
        # columns unable to account for what spent charged.
        "cache_read": total("cache_read"), "cache_write_5m": total("cache_write_5m"),
        "cache_write_1h": total("cache_write_1h"),
        "touched_n": t["touched_n"], "read_n": t["read_n"],
        "tampered_n": t["tampered_n"],
        # Blank on a run recorded before n's size was tracked, which is not the
        # same as a run whose n always fitted.
        "n_bytes": t.get("n_bytes", ""), "n_fits": t.get("n_fits", ""),
        "files": len(t["files"]), "agent_files": len(agent),
        "agent_bytes": sum(f["size"] for f in agent),
        "wrote_number": t["mentions"]["number"], "wrote_n": t["mentions"]["n"],
        # The strict reading beside the loose one. Blank before it was scored.
        "wrote_n_path": t["mentions"].get("n_path", ""),
        "wrote_cost": t["mentions"]["cost"], "retries": len(t["retries"]),
        "duration_s": t["duration_s"],
    }


def first(traces: list[dict], test) -> int | str:
    """Index of the earliest session satisfying `test`, or "never"."""
    return next((t["session"] for t in traces if test(t)), "never")


def report(runs: dict[str, list[dict]]) -> str:
    """Per-run summary: the firsts that matter, stop reasons, and the grep hits."""
    out = []
    for name, ts in runs.items():
        stops = dict(collections.Counter(t["stop"] for t in ts))
        agent_files = sorted({f["path"] for t in ts for f in t["files"] if not f["ours"]})
        out += [
            "=" * 72,
            f"run {name}: {len(ts)} sessions, {sum(t['spent'] for t in ts)} micro-dollars, "
            f"{ts[-1]['remaining']} remaining",
            f"  first reached for n   : {first(ts, lambda t: t['touched_n'])}",
            f"  first actually read n : {first(ts, lambda t: t['read_n'])}",
            f"  first wrote a number  : {first(ts, lambda t: t['mentions']['number'])}",
            f"  first wrote about n   : {first(ts, lambda t: t['mentions']['n'])}",
            # The strict reading. A gap between this and the line above is the
            # loose pattern catching n as a maths variable rather than the file.
            f"  first wrote n as path : {first(ts, lambda t: t['mentions'].get('n_path'))}",
            f"  first wrote about cost: {first(ts, lambda t: t['mentions']['cost'])}",
            # Past this session the agent could no longer see its whole history
            # in one read, so sessions either side of it are not comparable.
            f"  n stopped fitting at  : {first(ts, lambda t: t.get('n_fits') is False)}",
            f"  rewrote n in sessions : {[t['session'] for t in ts if t['tampered_n']] or 'never'}",
            f"  stop reasons          : {stops}",
            f"  files the agent made  : {agent_files or 'none'}",
            f"  reached for, absent   : {absent(ts)}",
        ]
        out += provenance_lines(ts)
        for t in ts:
            out += [f"    s{t['session']:04d}  {line}" for line in t["mention_lines"]]
    return "\n".join(out)


def absent(ts: list[dict]) -> str:
    """Tools the run reached for that its image lacked.

    A run whose traces carry no such field was recorded before the probe
    existed, which is not the same as a run that reached for nothing.
    """
    if not any("missing_tools" in t for t in ts):
        return "not recorded for this run"
    return str(sorted({m for t in ts for m in t.get("missing_tools") or []}) or "nothing")


def provenance_lines(ts: list[dict]) -> list[str]:
    """What the run ran as, and every point at which that changed.

    A field with one value across the run is stated once. A field that moved is
    listed per session, because from there on the sessions are not comparable.
    """
    fields = ["model_resolved", "image_id", "prices", "context_fraction",
              "max_tokens", "turn_cap", "timeout", "live_n", "overdraft",
              "harness_sha256", "thinking"]
    out, drifted = [], sorted({d.split(":")[0] for t in ts for d in t.get("provenance_drift") or []})
    for f in fields:
        seen = [t.get(f) if f == "model_resolved" else (t.get("provenance") or {}).get(f) for t in ts]
        shown = [str(v)[:19] if f in ("image_id", "harness_sha256") else v for v in seen]
        if len({str(v) for v in shown}) == 1:
            out.append(f"  {f:<22}: {shown[0]}")
        else:
            out.append(f"  {f:<22}: CHANGED  " +
                       "  ".join(f"s{t['session']}={v}" for t, v in zip(ts, shown)))
    if drifted:
        out.append(f"  !! provenance drifted mid-run in: {', '.join(drifted)}")
        out.append("     sessions before and after a change are not comparable")
    return out


def agent_files(t: dict) -> dict[str, str]:
    """Path -> content for the readable files the agent left this session."""
    return {f["path"]: f["text"] for f in t["files"]
            if not f["ours"] and f["text"] is not None}


def state_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Unified diff of the agent's own files between two consecutive sessions."""
    out = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path, ""), after.get(path, "")
        if a != b:
            out += difflib.unified_diff(a.splitlines(), b.splitlines(),
                                        fromfile=path, tofile=path, lineterm="", n=1)
    return out


def transcript(runs: dict[str, list[dict]]) -> str:
    """What the agent said, ran, and changed - the record the experiment turns on."""
    out = []
    for name, ts in runs.items():
        prev: dict[str, str] = {}
        for t in ts:
            out += ["=" * 72,
                    f"run {name}  session {t['session']}  stop={t['stop']}  spent={t['spent']}",
                    f"n at wake: {t['series_before']}", ""]
            # The agent's whole world at wake, before it did anything.
            out += [f"  $ {t['commands'][0]}"]
            out += [f"  | {line}" for line in t["opening"].splitlines()] + [""]
            for turn in t["turns"]:
                # Only fable-5 produces reasoning; every other model runs with
                # thinking off. Kept apart from the agent's spoken words.
                if turn.get("thinking"):
                    out += [f"  ({turn['turn']}) {line}"
                            for line in turn["thinking"].splitlines()]
                if turn["text"]:
                    out.append(f"  [{turn['turn']}] {turn['text']}")
                if turn.get("stop_reason") == "max_tokens":
                    out.append(f"  [{turn['turn']}] -- truncated at max_tokens --")
                for c in turn["tools"]:
                    out.append(f"    $ {c['command']}")
                    out += [f"    | {line}" for line in (c["result"] or "").splitlines()]
                out.append("")
            # What state/ looked like after this wake, against the one before it.
            curr = agent_files(t)
            if changed := state_changes(prev, curr):
                out += ["  state/ changes:"] + [f"    {line}" for line in changed] + [""]
            prev = curr
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI. Writes the CSV, report, and transcript for one run or all runs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    a = ap.parse_args(argv)

    runs = load(a.run_id)
    if not runs:
        print(f"no traces for {a.run_id or 'any run'} under {wake.ROOT / 'private'}", file=sys.stderr)
        return 1

    out_dir = (wake.private_dir(a.run_id) if a.run_id else wake.ROOT / "private") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row(t) for ts in runs.values() for t in ts]
    with (out_dir / "sessions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    text = report(runs)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "transcript.txt").write_text(transcript(runs), encoding="utf-8")

    print(text)
    print(f"\nwrote {out_dir}/ : sessions.csv, report.txt, transcript.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
