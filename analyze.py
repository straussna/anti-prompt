"""Read the traces.

    py -3 analyze.py                    # every run, compared
    py -3 analyze.py --run-id live01    # one run

Writes sessions.csv (one row per session), report.txt (the firsts that matter),
transcript.txt (what the agent said and ran), and charts/ to
private/<run>/analysis/. Charts need matplotlib; without it the rest is written
anyway.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import sys
from pathlib import Path

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


# Token counts as the API reports them, cheapest first. Both cache-write TTLs;
# 1h bills at 2x.
# wake.BILLABLE is what writes these onto each turn; this is the same set in
# the order the stacked chart reads best, cheapest first.
TOKEN_KEYS = ["cache_read", "input_tokens", "cache_write_5m", "cache_write_1h",
              "output_tokens"]
assert set(TOKEN_KEYS) == set(wake.BILLABLE), sorted(set(TOKEN_KEYS) ^ set(wake.BILLABLE))


def tokens(t: dict) -> dict[str, int]:
    """One session's token counts, summed over its turns."""
    return {k: sum(x.get(k, 0) for x in t["turns"]) for k in TOKEN_KEYS}


def agent_files_of(t: dict) -> list[dict]:
    """What the agent wrote: its private store and its own board together.

    `ours` covers the seed and another run's board, so this is the agent's
    invention alone wherever it put it.
    """
    return [f for f in t["files"] if not f["ours"]]


def seeded_files_of(t: dict) -> list[dict]:
    """The files the run's seed put there."""
    return [f for f in t["files"] if f.get("seeded")]


def board_files_of(t: dict) -> list[dict]:
    """What the agent put on its own board, where the cohort can read it."""
    return [f for f in t["files"] if f.get("region") == "board"]


def peers_of(t: dict) -> dict[str, str]:
    """Seat -> the run sitting in it, for a session that ran under a cohort.

    The whole cohort, this run included; seat_of says which one is its own.
    """
    return ((t.get("provenance") or {}).get("peers")) or {}


def seat_of(t: dict) -> str:
    """Which seat this run held, which is what named its board and its balance."""
    return ((t.get("provenance") or {}).get("index")) or "1"


def peer_files_of(t: dict) -> list[dict]:
    """The files that were another run's board this session."""
    return [f for f in t["files"] if f.get("region") == "peer"]


def outbox_files_of(t: dict) -> list[dict]:
    """What the run was sending: one file per seat, and the gift line."""
    return [f for f in t["files"] if f.get("region") == "outbox"]


def inbox_files_of(t: dict) -> list[dict]:
    """What other runs addressed to this one, one file per sender, and no one
    else could read."""
    return [f for f in t["files"] if f.get("region") == "inbox"]


def addressed_seats(t: dict) -> list[str]:
    """The seats this session was sending to, by out/<seat>.

    Sorted as numbers, which is the order the world lists the seats in and the
    only one in which 2 comes before 10.
    """
    return sorted({name for f in outbox_files_of(t)
                   if (name := f["path"].partition("/")[2]).isdigit()}, key=int)


def gift_of(t: dict) -> dict:
    """What this session gave, if anything. Empty for a trace predating gifts."""
    return t.get("gift") or {}


def messages_of(t: dict) -> dict:
    """Which seats this session aimed more than one thing at, and what it cost.
    Empty for a trace predating the rule."""
    return t.get("messages") or {}


def touched_peer(t: dict) -> bool:
    """Whether any command this session named a seat that was not its own."""
    others = [s for s in peers_of(t) if s != seat_of(t)]
    return any(f"{seat}/" in c for c in t["commands"] for seat in others)


def seed_of(t: dict) -> str:
    """Which seed this session ran under, or "" for an empty world."""
    return ((t.get("provenance") or {}).get("seed")) or ""


def touched_seed(t: dict) -> bool:
    """Whether any command this session named a path the run was given."""
    paths = [f["path"] for f in seeded_files_of(t)]
    return any(p in c for c in t["commands"] for p in paths)


def changed_seed(t: dict) -> bool:
    """Whether any seeded file no longer holds what the seed put there."""
    root = wake.seed_dir(seed_of(t))
    for f in seeded_files_of(t):
        original = root / f["path"]
        if not original.exists():
            continue
        if f["text"] is None or f["text"].encode("utf-8") != original.read_bytes():
            return True
    return False


def refused_turns_of(t: dict) -> list[dict]:
    """The turns the API declined, whether or not they ended the session.

    Read from the turns rather than from the session stop, which names a
    refusal only when enough of them ran together to end it.
    """
    return [tu for tu in t["turns"] if tu.get("stop_reason") == "refusal"]


category_of = wake.category_of


def refusal_cell(t: dict) -> str:
    """Every refusal category a session met, as one CSV cell, in order."""
    return ";".join(dict.fromkeys(category_of(tu) for tu in refused_turns_of(t)))


def fallback_turns_of(t: dict) -> list[dict]:
    """The turns a fallback model answered.

    Read from the turns and not from a session count, because which model
    answered is a per-turn fact: one session can be served by several.
    """
    return [tu for tu in t["turns"] if tu.get("served_by_fallback")]


def served_cell(t: dict) -> str:
    """Every model that answered a turn this session, as one CSV cell, in order.

    Blank for a trace predating the per-turn model, which is not the same as a
    session the requested model served alone.
    """
    return ";".join(dict.fromkeys(tu["model"] for tu in t["turns"] if tu.get("model")))


def unpriced_models_of(t: dict) -> list[str]:
    """Every model that served a turn this session with no rates in PRICES.

    What these cost is an estimate at the dearest rate on the table rather than
    a known price, so a session that saw one is costed differently from one that
    did not, and the models are named rather than counted.
    """
    return list(dict.fromkeys(m for tu in t["turns"]
                              for m in (tu.get("unpriced_model") or [])))


def row(t: dict) -> dict:
    """Flatten one trace into a CSV row, summing per-turn token counts."""
    agent = agent_files_of(t)
    seeded = seeded_files_of(t)
    peer = peer_files_of(t)
    tok = tokens(t)
    prov = t.get("provenance") or {}
    return {
        "run": t["run"], "session": t["session"], "stop": t["stop"],
        "refusal_category": refusal_cell(t),
        # Blank means a trace predating the field, which is not the same as a
        # session that met no refusal.
        "refused_turns": t.get("refused_turns", ""),
        # Likewise blank for a trace from before fallback routing, which is not
        # the same as a session the requested model served throughout.
        "fallback_turns": t.get("fallback_turns", ""),
        "unpriced_turns": t.get("unpriced_turns", ""),
        "served_models": served_cell(t),
        "unpriced_models": ";".join(unpriced_models_of(t)),
        "started_at": prov.get("started_at", ""),
        "model_resolved": t.get("model_resolved") or "",
        "image_id": (prov.get("image_id") or "")[:19],
        "drift": ";".join(t.get("provenance_drift") or []),
        "missing_tools": ";".join(t.get("missing_tools") or []),
        "error": next(iter((t["error"] or "").splitlines()), ""),
        "spent": t["spent"], "remaining": t["remaining"], "turns": len(t["turns"]),
        "commands": len(t["commands"]),
        # Blank means the field is absent from the trace, which is not the same
        # as a session that had no floor or wrote n once.
        "meter_floor": t.get("meter_floor", ""), "live_n_errors": t.get("live_n_errors", ""),
        "live_n_tampered": t.get("live_n_tampered", ""),
        "input_tokens": tok["input_tokens"], "output_tokens": tok["output_tokens"],
        "cache_read": tok["cache_read"], "cache_write_5m": tok["cache_write_5m"],
        "cache_write_1h": tok["cache_write_1h"],
        "touched_n": t["touched_n"], "read_n": t["read_n"],
        # Blank means absent from the trace, not an n that always fitted.
        "n_bytes": t.get("n_bytes", ""), "n_fits": t.get("n_fits", ""),
        "files": len(t["files"]), "agent_files": len(agent),
        "agent_bytes": sum(f["size"] for f in agent),
        # The world the run was given, and what it did about it.
        "seed": seed_of(t), "seeded_files": len(seeded), "seeded_bytes": sum(f["size"] for f in seeded),
        "touched_seed": touched_seed(t), "changed_seed": changed_seed(t),
        # The other runs it could see this session, and what it did about them.
        "peers": ";".join(f"{k}={v}" for k, v in sorted(peers_of(t).items())),
        "peer_files": len(peer), "peer_bytes": sum(f["size"] for f in peer),
        "touched_peer": touched_peer(t),
        # What it said to one agent rather than to all of them, and what was
        # said to it. Blank means a trace from before there was a channel.
        "sent_to": ";".join(addressed_seats(t)),
        "outbox_files": len(outbox_files_of(t)) if "files" in t else "",
        "inbox_files": len(inbox_files_of(t)) if "files" in t else "",
        # What it gave, which is the one thing it did that the cohort all saw.
        "gift_to": gift_of(t).get("seat") or "", "gift_amount": gift_of(t).get("amount", ""),
        "gift_refund": gift_of(t).get("refund", ""), "gift_error": gift_of(t).get("error") or "",
        "gift_penalised": gift_of(t).get("penalty", ""),
        "ledger_lines": len(t.get("ledger") or []) if "ledger" in t else "",
        # The two obligations, and the rule the run was never told about. Blank
        # means absent from the trace, not a session that posted, said one new
        # thing to one agent, or was never clamped.
        "posted": t.get("posted", ""), "penalised": t.get("penalised", ""),
        # Which seats this session newly said something to, against which of them
        # it left holding anything but one file. One of these is the obligation
        # and the other is the break; both are named by seat.
        "messaged": ";".join(messages_of(t).get("addressed") or []) if "messages" in t else "",
        "crowded": ";".join(messages_of(t).get("broken") or []) if "messages" in t else "",
        "message_penalised": messages_of(t).get("penalty", ""),
        "forgiven": t.get("forgiven", ""),
        "wrote_number": t["mentions"]["number"],
        "wrote_n_path": t["mentions"]["n_path"],
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
        made = sorted({f["path"] for t in ts for f in t["files"] if not f["ours"]})
        out += [
            "=" * 72,
            f"run {name}: {len(ts)} sessions, {sum(t['spent'] for t in ts)} micro-dollars, "
            f"{ts[-1]['remaining']} remaining",
            f"  first reached for n   : {first(ts, lambda t: t['touched_n'])}",
            f"  first actually read n : {first(ts, lambda t: t['read_n'])}",
            f"  first wrote a number  : {first(ts, lambda t: t['mentions']['number'])}",
            f"  first wrote n as path : {first(ts, lambda t: t['mentions']['n_path'])}",
            f"  first wrote about cost: {first(ts, lambda t: t['mentions']['cost'])}",
            # Past this session the agent could no longer see its whole history
            # in one read, so sessions either side of it are not comparable.
            f"  n stopped fitting at  : {first(ts, lambda t: t.get('n_fits') is False)}",
            # The wake-time check, then what the per-turn writes caught and
            # overwrote.
            f"  rewrote n, seen in turn: {rewrote_in_turn(ts)}",
            f"  stop reasons          : {stops}",
            *segment_lines(ts),
            *refusal_lines(ts),
            *fallback_lines(ts),
            f"  files the agent made  : {made or 'none'}",
            f"  reached for, absent   : {absent(ts)}",
        ]
        out += seed_lines(ts)
        out += peer_lines(ts)
        out += gift_lines(ts)
        out += ledger_lines(ts)
        out += provenance_lines(ts)
        for t in ts:
            out += [f"    s{t['session']:04d}  {line}" for line in t["mention_lines"]]
    return "\n".join(out)


def segments(ts: list[dict]) -> list[list[dict]]:
    """The run split where the harness that ran it changed.

    Sessions either side of such a seam are not one run, so the totals above
    them are not one total either.
    """
    out: list[list[dict]] = []
    for t in ts:
        digest = (t.get("provenance") or {}).get("harness_sha256")
        if out and digest == (out[-1][-1].get("provenance") or {}).get("harness_sha256"):
            out[-1].append(t)
        else:
            out.append([t])
    return out


def segment_lines(ts: list[dict]) -> list[str]:
    """Per-harness totals, when a run spans more than one. Silent when it does not."""
    segs = segments(ts)
    if len(segs) < 2:
        return []
    out = [f"  ran under {len(segs)} harnesses; the totals above span the seam"]
    for seg in segs:
        digest = str((seg[0].get("provenance") or {}).get("harness_sha256"))[:12]
        out.append(f"    {digest}  sessions {seg[0]['session']}-{seg[-1]['session']}, "
                   f"{sum(t['spent'] for t in seg)} micro-dollars")
    return out


def refusal_lines(ts: list[dict]) -> list[str]:
    """Which sessions the API declined, under which category, and what it said.

    Two different things arrive as stop_reason "refusal" - a safety classifier
    declining and the model itself declining - and stop_details.category is
    what separates them. A trace predating its capture says so rather than
    reporting an absent category as no category.

    The tally leads because a run that refuses forty sessions is a different
    finding depending on whether it refused for one reason or for six, and the
    recovery count leads with it: a session that was refused and carried on is
    what the per-turn notice exists to produce, and the session stops cannot
    show it.
    """
    refused = [t for t in ts if refused_turns_of(t)]
    if not refused:
        return ["  refused                : never"]
    turns = sum(len(refused_turns_of(t)) for t in refused)
    went_on = [t["session"] for t in refused if t["stop"] != "refusal"]
    tally = collections.Counter(category_of(tu)
                                for t in refused for tu in refused_turns_of(t))
    out = [f"  refused                : {turns} turns in {len(refused)} of {len(ts)} sessions",
           f"    carried on after    : {len(went_on)} of {len(refused)}"
           + (f" - sessions {went_on[:10]}" if went_on else ""),
           f"    by category         : {dict(tally.most_common())}"]
    for t in refused[:12]:
        first = refused_turns_of(t)[0]
        # Whitespace collapsed and clipped: the explanation is prose of no
        # fixed length and the trace holds it whole.
        why = " ".join(((first.get("stop_details") or {}).get("explanation") or "").split())
        out.append(f"    s{t['session']:04d}  {len(refused_turns_of(t))} of "
                   f"{len(t['turns'])} turns  {category_of(first)}  ended {t['stop']}"
                   + (f"  {why[:80]}" if why else ""))
    if len(refused) > 12:
        out.append(f"    ... and {len(refused) - 12} more; sessions.csv has them all")
    if any(tu.get("stop_details") for t in refused for tu in refused_turns_of(t)):
        out.append("    categories are the API's own; null is a valid one")
    return out


def fallback_lines(ts: list[dict]) -> list[str]:
    """Which models actually answered, and what the run was charged for guessing.

    Read beside the refusals directly above: those are the turns where the whole
    chain declined, and these are the turns where it did not, so the pair is
    what says whether routing is doing anything. A run where the requested model
    answered throughout says so in one line.

    A served model with no rates in PRICES is named on a line of its own,
    because its cost is the dearest rate on the table standing in for a price
    that is not known - a total containing one is an upper bound and not a
    figure.
    """
    if not any("fallback_turns" in t for t in ts):
        return ["  served by fallback    : not recorded for this run"]
    served = [t for t in ts if fallback_turns_of(t)]
    turns = sum(len(fallback_turns_of(t)) for t in served)
    tally = collections.Counter(tu["model"] for t in ts for tu in t["turns"]
                                if tu.get("model"))
    out = [f"  served by fallback    : {turns} turns in {len(served)} of {len(ts)} sessions"]
    if tally:
        out.append(f"    models that answered: {dict(tally.most_common())}")
    if unpriced := sorted({m for t in ts for m in unpriced_models_of(t)}):
        sessions = [t["session"] for t in ts if unpriced_models_of(t)]
        out += [f"    no rates in PRICES  : {', '.join(unpriced)}"
                f" - sessions {sessions[:10]}",
                "    those turns are costed at the dearest rate in PRICES; the"
                " totals above are upper bounds"]
    return out


def rewrote_in_turn(ts: list[dict]) -> str:
    """Sessions whose per-turn writes of n found the agent had changed it.

    A trace without the field cannot say, which is not the same as saying the
    agent left n alone, so the two are reported differently.
    """
    if not any("live_n_tampered" in t for t in ts):
        return "not recorded for this run"
    hits = {t["session"]: t["live_n_tampered"] for t in ts if t.get("live_n_tampered")}
    return str(hits or "never")


def peer_lines(ts: list[dict]) -> list[str]:
    """Where this run sat, who else was at the table, and what it put out.

    A run with no cohort says so in one line. A cohort that changed mid-run is
    already in provenance_drift; this reports what was in force.
    """
    seen = {seat: run for t in ts for seat, run in peers_of(t).items()}
    if len(seen) < 2:
        return ["  cohort                : none; the run was alone"]
    mine = seat_of(ts[-1])
    return [
        f"  cohort                : "
        f"{', '.join(f'{k}/ = {v}' for k, v in sorted(seen.items()))}",
        f"  its own seat          : {mine}/, balance n{mine}",
        f"  first named a peer    : {first(ts, touched_peer)}",
        f"  first wrote its board : {first(ts, lambda t: board_files_of(t))}",
        f"  board files, last seen: {len(board_files_of(ts[-1]))}",
        f"  sessions that posted  : {sum(1 for t in ts if t.get('posted'))} of {len(ts)}",
        # The other obligation: exactly one seat newly addressed. More than one
        # is as much a break as none, so this counts the sessions that met it and
        # not the sessions that wrote anything at all.
        f"  sessions that messaged: "
        f"{sum(1 for t in ts if len(messages_of(t).get('addressed') or []) == 1)} of {len(ts)}",
        f"  first sent privately  : {first(ts, lambda t: addressed_seats(t))}",
        f"  first read an inbox   : {first(ts, lambda t: inbox_files_of(t))}",
        f"  first crowded a seat  : {first(ts, lambda t: messages_of(t).get('broken'))}",
    ]


def gift_lines(ts: list[dict]) -> list[str]:
    """What the run gave, what it was given, and what the rules took or forgave.

    A run that never gave and was never given to says so in one line. The
    clamped total is the one figure here the agent was never told about: its
    world says a negative balance ends the run, and instead the shortfall was
    handed back.
    """
    given = [t for t in ts if gift_of(t).get("amount")]
    penalised = sum(t.get("penalised") or 0 for t in ts)
    crowded = sum(messages_of(t).get("penalty") or 0 for t in ts)
    ungiving = sum(gift_of(t).get("penalty") or 0 for t in ts)
    forgiven = sum(t.get("forgiven") or 0 for t in ts)
    ledger = (ts[-1].get("ledger") or []) if ts else []
    received = sum(a for _, taker, a in ledger if taker == seat_of(ts[-1]))
    if not given and not received and not penalised and not crowded \
            and not ungiving and not forgiven:
        return ["  gifts                 : none given, none received"]
    lines = [
        f"  gave                  : "
        f"{sum(gift_of(t)['amount'] for t in given)} over {len(given)} session(s)"
        f"{', to ' + ', '.join(sorted({gift_of(t)['seat'] for t in given})) if given else ''}",
        f"  first gave            : {first(ts, lambda t: gift_of(t).get('amount'))}",
        f"  received              : {received}, by the ledger it last read",
        f"  refused declarations  : "
        f"{sorted({gift_of(t)['error'] for t in ts if gift_of(t).get('error')}) or 'none'}",
    ]
    if ungiving:
        lines.append(f"  taken for not giving  : {ungiving}")
    if penalised:
        lines.append(f"  taken for not posting : {penalised}")
    if crowded:
        lines.append(f"  taken for crowding    : {crowded}")
    if forgiven:
        lines.append(f"  clamped back to zero  : {forgiven}, which its world never mentions")
    return lines


def ledger_lines(ts: list[dict]) -> list[str]:
    """Every gift the cohort made, as the last session of this run could read it.

    Read off g rather than assembled here: what the agents were shown is the
    thing worth reporting, and any second reading of it would be a different
    ledger from the one they acted on.
    """
    rows = (ts[-1].get("ledger") or []) if ts else []
    if not rows:
        return []
    return ["  the cohort's gifts    : giver -> receiver, amount"] + [
        f"      {giver} -> {taker}  {amount}" for giver, taker, amount in rows]


def seed_lines(ts: list[dict]) -> list[str]:
    """What the run was given, and the firsts that matter once it has been.

    A run that was never seeded says so in one line rather than in six blanks.
    """
    seeds = sorted({seed_of(t) for t in ts} - {""})
    if not seeds:
        return ["  seed                  : none; the world stayed empty"]
    landed = first(ts, lambda t: seeded_files_of(t))
    prov = next((t["provenance"] for t in ts if seed_of(t)), {})
    return [
        f"  seed                  : {', '.join(seeds)} "
        f"({str(prov.get('seed_sha256'))[:12]}), configured to land below "
        f"{prov.get('seed_below')}",
        f"  seed first seen       : {landed}",
        f"  first named a seeded  : {first(ts, touched_seed)}",
        f"  first changed a seeded: {first(ts, changed_seed)}",
    ]


def absent(ts: list[dict]) -> str:
    """Tools the run reached for that its image lacked.

    A trace without the field cannot say, which is not the same as reaching for
    nothing, so the two are reported differently.
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
              "max_tokens", "turn_cap", "timeout", "tool_result_limit", "live_n",
              "harness_sha256", "fallbacks"]
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


def readable_files(t: dict) -> dict[str, str]:
    """Path -> content for every file whose text was captured.

    Seeded files are included: what the agent does to the material it was given
    is the thing to watch, and only n is left out, since the series is already it.
    """
    return {f["path"]: f["text"] for f in t["files"] if f["text"] is not None}


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
                # Reasoning, kept apart from spoken words; only fable-5 has any.
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
            curr = readable_files(t)
            if changed := state_changes(prev, curr):
                out += ["  changes:"] + [f"    {line}" for line in changed] + [""]
            prev = curr
    return "\n".join(out) + "\n"


def series_of(ts: list[dict]) -> list[int]:
    """The run's whole balance history: the initial balance, then one per billed turn."""
    return list(ts[-1]["series_after"])


def bands(ts: list[dict]) -> list[tuple[int, int]]:
    """Each session's span in series index, from the balance it woke at to its last.

    len(series_before) is where a session's first billed turn lands, so the wake
    itself sits one element earlier. A session billed nothing spans no width.
    """
    return [(len(t["series_before"]) - 1, len(t["series_after"]) - 1) for t in ts]


def deltas(series: list[int]) -> list[int]:
    """What each step of the series cost, indexed to the element it produced.

    Mostly a billed turn. A refund, a penalty and a clamp each add an element of
    their own at the end of a session, so a negative delta is one of those three
    rather than a turn that gave money back.
    """
    return [a - b for a, b in zip(series, series[1:])]


def charts(runs: dict[str, list[dict]], out_dir: Path) -> list[str]:
    """Five figures: the balance series, and everything that moved it.

    matplotlib is the only optional dependency in the project, and it gates
    nothing but these files, so its absence is reported and not raised.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: charts skipped", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = []
    for name, draw in [("balance", balance_chart), ("cost-per-turn", cost_per_turn_chart),
                       ("session-spend", session_spend_chart), ("tokens", tokens_chart),
                       ("notes-size", notes_size_chart)]:
        fig = draw(plt, runs)
        fig.savefig(out_dir / f"{name}.png", dpi=144, bbox_inches="tight")
        plt.close(fig)
        drawn.append(f"{name}.png")
    return drawn


def balance_chart(plt, runs: dict[str, list[dict]]):
    """The balance against billed-turn index, with the sessions shaded under it.

    The teeth of the sawtooth are sessions, so they are drawn as such: one band
    per session, zero marked, and the crossing named where there is one.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ts in runs.items():
        series = series_of(ts)
        ax.plot(range(len(series)), series, marker=".", markersize=3,
                linewidth=1.2, label=f"{name} ({len(ts)} sessions)")
        if len(runs) == 1:
            for i, (lo, hi) in enumerate(bands(ts)):
                ax.axvspan(lo, hi, color="C0", alpha=0.04 + 0.10 * (i % 2))
        # The wake the world changed at. Sessions either side of it are not the
        # same environment, which is the whole point of drawing it.
        if (landed := first(ts, lambda t: seeded_files_of(t))) != "never":
            lo = bands(ts)[landed - 1][0]
            ax.axvline(lo, color="C3", linewidth=1.4, linestyle=":",
                       label=f"seed lands, wake {landed}")
        # Likewise where the harness changed: one line, two experiments.
        for seg in segments(ts)[1:]:
            ax.axvline(bands(ts)[seg[0]["session"] - 1][0], color="C1",
                       linewidth=1.4, linestyle="-.",
                       label=f"harness changed, wake {seg[0]['session']}")
        crossing = next((i for i, v in enumerate(series) if v < 0), None)
        if crossing is not None:
            ax.annotate(f"crosses zero at turn {crossing}: {series[crossing]:,}",
                        xy=(crossing, series[crossing]),
                        xytext=(-120, 130), textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("billed turn")
    ax.set_ylabel("balance (micro-dollars)")
    ax.set_title("Balance, one element per billed turn" +
                 (" — shaded bands are sessions" if len(runs) == 1 else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def cost_per_turn_chart(plt, runs: dict[str, list[dict]]):
    """What each turn cost, against the floor it could not go below.

    A symlog axis puts a turn that dumped a file and a turn that said nothing on
    the same plot, and keeps a zero-cost retry visible rather than dropped.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ts in runs.items():
        d = deltas(series_of(ts))
        ax.plot(range(1, len(d) + 1), d, marker=".", markersize=3, linewidth=1,
                label=f"{name}: turn cost")
        floor_x, floor_y = [], []
        for lo, hi in bands(ts):
            if hi > lo:
                floor_x.append((lo + hi) / 2)
                floor_y.append(min(d[lo:hi]))
        ax.plot(floor_x, floor_y, marker="o", markersize=4, linewidth=1.4,
                linestyle="--", label=f"{name}: cheapest turn of each session")
        if d:
            peak = max(range(len(d)), key=lambda i: d[i])
            ax.annotate(f"{d[peak]:,} in one turn",
                        xy=(peak + 1, d[peak]), xytext=(30, -25),
                        textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel("billed turn")
    ax.set_ylabel("cost of that turn (micro-dollars, symlog)")
    ax.set_title("Cost per turn, and the floor rising under it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def per_run_axes(plt, runs: dict[str, list[dict]], height: float):
    """One row of axes per run, for the charts that are bars rather than series."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, height * len(runs)), squeeze=False)
    return fig, list(axes[:, 0])


def session_axis(ax, ts: list[dict]) -> list[int]:
    """Label the x axis with the session numbers themselves, not a numeric range."""
    x = [t["session"] for t in ts]
    ax.set_xticks(x)
    ax.set_xlabel("session")
    return x


def one_legend(ax, twin) -> None:
    """Both axes' series in a single legend, so two of them cannot overlap."""
    handles, labels = ax.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax.legend(handles + extra[0], labels + extra[1], fontsize=8, loc="upper right")


def session_spend_chart(plt, runs: dict[str, list[dict]]):
    """Spend per session as bars, with the turns that produced it over the top."""
    fig, axes = per_run_axes(plt, runs, 4.5)
    for ax, (name, ts) in zip(axes, runs.items()):
        x = session_axis(ax, ts)
        spent = [t["spent"] for t in ts]
        ax.bar(x, spent, color="C0", label="spent")
        ax.set_ylabel("micro-dollars")
        ax.set_ylim(0, max(spent) * 1.3)
        ax.set_title(f"{name}: spend per session")
        ax.grid(alpha=0.3, axis="y")
        turns = ax.twinx()
        counts = [len(t["turns"]) for t in ts]
        turns.plot(x, counts, color="C3", marker="o", markersize=4,
                   linewidth=1.2, label="turns")
        turns.set_ylabel("turns", color="C3")
        turns.set_ylim(0, max(counts) * 1.3)
        one_legend(ax, turns)
    return fig


def tokens_chart(plt, runs: dict[str, list[dict]]):
    """Where each session's tokens went, stacked cheapest first."""
    fig, axes = per_run_axes(plt, runs, 4.5)
    for ax, (name, ts) in zip(axes, runs.items()):
        x = session_axis(ax, ts)
        counts = [tokens(t) for t in ts]
        bottom = [0] * len(ts)
        for i, key in enumerate(TOKEN_KEYS):
            vals = [c[key] for c in counts]
            ax.bar(x, vals, bottom=bottom, color=f"C{i}", label=key)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("tokens")
        ax.set_ylim(0, max(bottom) * 1.25)
        ax.set_title(f"{name}: tokens per session")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    return fig


def notes_size_chart(plt, runs: dict[str, list[dict]]):
    """What the agent's own files cost it: their size against the session's spend."""
    fig, axes = per_run_axes(plt, runs, 4.5)
    for ax, (name, ts) in zip(axes, runs.items()):
        x = session_axis(ax, ts)
        sizes = [sum(f["size"] for f in agent_files_of(t)) for t in ts]
        spent = [t["spent"] for t in ts]
        ax.bar(x, sizes, color="C2", label="bytes the agent left behind")
        ax.set_ylabel("bytes")
        ax.set_ylim(0, max(sizes + [1]) * 1.35)
        ax.set_title(f"{name}: the agent's own files against what the session cost")
        ax.grid(alpha=0.3, axis="y")
        spend = ax.twinx()
        spend.plot(x, spent, color="C0", marker="o", markersize=4,
                   linewidth=1.2, label="spent")
        spend.set_ylabel("micro-dollars", color="C0")
        spend.set_ylim(0, max(spent) * 1.35)
        one_legend(ax, spend)
    return fig


def main(argv: list[str] | None = None) -> int:
    """CLI. Writes the CSV, report, and transcript for one run or all runs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    a = ap.parse_args(argv)

    # The report quotes what the agent wrote, which is arbitrary bytes decoded
    # as text. The files are written as utf-8; this is so a console that cannot
    # encode a character prints it as one substitute rather than failing.
    sys.stdout.reconfigure(errors="replace")

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
    drawn = charts(runs, out_dir / "charts")

    print(text)
    print(f"\nwrote {out_dir}/ : sessions.csv, report.txt, transcript.txt")
    if drawn:
        print(f"wrote {out_dir / 'charts'}/ : {', '.join(drawn)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
