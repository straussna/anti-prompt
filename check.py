"""Verification. No API spend.

    py -3 check.py

A fake `create` is injected into wake.run_once, so the whole pipeline runs -
container, meter, trace - with nothing billed. Checks that need Docker are
skipped with a notice when the daemon is down.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

import wake

# --- the fake ---------------------------------------------------------------


def usage(**kw):
    """A usage object shaped like the API's, with overridable token counts."""
    return NS(**{"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation": None, **kw})


def say(text="done.", u=None, id=None, stop="end_turn"):
    """A scripted step: reply with text and stop."""
    return {"kind": "say", "text": text, "u": u, "id": id, "stop": stop}


def run(*cmds, u=None, id=None, stop="tool_use"):
    """A scripted step: reply with one bash tool call per command.

    `stop` is the response's stop_reason, so a truncated turn can be scripted.
    """
    return {"kind": "run", "cmds": list(cmds), "u": u, "id": id, "stop": stop}


def restart(u=None, id=None):
    """A scripted step: the {"restart": true} form of the bash tool."""
    return run(None, u=u, id=id)


def think(thinking="reasoning.", text="done.", u=None, id=None, stop="end_turn"):
    """A scripted step: a thinking block plus text, as fable-5 replies."""
    return {"kind": "think", "thinking": thinking, "text": text, "u": u, "id": id, "stop": stop}


class Err(Exception):
    """An API error carrying a status code."""

    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def fake(*steps, seen=None):
    """Build a `create` that plays the given steps, one per call.

    Steps run out into a plain "done." reply. Pass `seen` to capture the request
    params each call was made with.
    """
    q, n = list(steps), [0]

    def create(**params):
        """Stands in for client.messages.create: plays one step per call."""
        n[0] += 1
        if seen is not None:
            seen.append(params)
        s = q.pop(0) if q else say()
        if isinstance(s, BaseException):
            raise s
        rid, u = s["id"] or f"msg{n[0]}", s["u"] or usage()
        # The real API answers with the dated snapshot the alias resolved to,
        # and the harness records it. A reply without one cannot check that.
        model = f"{params['model']}-20990101"
        if s["kind"] == "run":
            return NS(id=rid, model=model, stop_reason=s["stop"], usage=u,
                      content=[NS(type="tool_use", id=f"t{n[0]}_{i}", name="bash", input={"command": c})
                               for i, c in enumerate(s["cmds"])])
        content = [NS(type="text", text=s["text"])]
        if s["kind"] == "think":
            content.insert(0, NS(type="thinking", thinking=s["thinking"]))
        return NS(id=rid, model=model, stop_reason=s["stop"], usage=u, content=content)

    return create


class Skip(Exception):
    """This check needs a container and cannot have one."""


@functools.cache
def docker_ready() -> bool:
    """True if the daemon is up and the image is built. Says so once if not."""
    if not shutil.which("docker") or subprocess.run(["docker", "info"], capture_output=True).returncode:
        return False
    if subprocess.run(["docker", "image", "inspect", wake.IMAGE], capture_output=True).returncode:
        print(f"image {wake.IMAGE} not built\n")
        return False
    return True


# Every wake global a check is allowed to move, and therefore every one pinned()
# puts back. A check that sets anything else would leak it into the checks after
# it, so temp_root refuses the name rather than restoring something it never saved.
RESTORED = wake.TUNABLES | {"ROOT", "WATCH"}


@contextlib.contextmanager
def pinned():
    """Restore every wake global a check may move, on exit."""
    saved = {k: getattr(wake, k) for k in RESTORED}
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(wake, k, v)


@contextlib.contextmanager
def temp_root(**overrides):
    """Point wake at a throwaway directory, and skip when Docker is unavailable.

    Every check needing a container passes through here. `overrides` set wake
    module globals (TURN_CAP=1, TIMEOUT=2) for the duration; pinned() puts them
    and ROOT back afterwards.
    """
    if not docker_ready():
        raise Skip
    unknown = set(overrides) - RESTORED
    assert not unknown, f"temp_root cannot restore {sorted(unknown)}"
    with pinned(), tempfile.TemporaryDirectory(
            prefix="mtr-check-", ignore_cleanup_errors=True) as d:
        wake.ROOT = Path(d)
        for k, v in overrides.items():
            setattr(wake, k, v)
        yield Path(d)


@contextlib.contextmanager
def quiet():
    """Swallow session output so the check list stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def wake_once(*steps, seen=None):
    """One session against a fresh run, output suppressed."""
    with quiet():
        return wake.run_once("t", fake(*steps, seen=seen))


def ground_truth(run="t") -> dict:
    """meter.json as it stands on disk, read back rather than carried."""
    return json.loads((wake.private_dir(run) / "meter.json").read_text(encoding="utf-8"))


# --- checks -----------------------------------------------------------------

DEFAULT = (run("cat state/n"), run("echo hi > state/note.txt", "ls state"), say())


def check_system_is_pinned():
    """I1: the prompt is the pinned 115 bytes, and the tool carries no description."""
    assert hashlib.sha256(wake.SYSTEM.encode()).hexdigest() == wake.SYSTEM_SHA256
    assert len(wake.SYSTEM) == 115
    for word in ("you ", "your", "assistant", "budget", "cost", "goal", "task", "network"):
        assert word not in wake.SYSTEM.lower(), f"SYSTEM contains {word!r}"
    assert wake.TOOL == {"type": "bash_20250124", "name": "bash"}, "tool must carry no description"


def check_cost_is_exact():
    """I3: cost matches hand-computed integers, including both cache-write TTLs."""
    # opus-5: in 500, out 2500 centi/token; write 1.25x, read 0.1x
    u = usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=2000,
              cache_read_input_tokens=5000)
    m = wake.measure(u, "claude-opus-5")
    assert m["centi"] == 1000 * 500 + 100 * 2500 + 2000 * 625 + 5000 * 50 == 2_250_000, m
    assert m["prefix"] == 1000 + 5000 + 2000, "prefix must include cached tokens"
    # per-TTL detail wins over the flat field; 1h writes cost 2x
    split = wake.measure(usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=9999,
                               cache_creation=NS(ephemeral_5m_input_tokens=100,
                                                 ephemeral_1h_input_tokens=200)), "claude-opus-5")
    assert split["centi"] == 100 * 625 + 200 * 1000, split


def check_n_is_bare_integers():
    """I4: n is a JSON array of bare integers - no keys, units, or quoting."""
    assert wake.render_n([500000, 494750]) == "[500000,494750]\n"
    assert all(type(v) is int for v in json.loads(wake.render_n([1, 2])))
    assert ":" not in wake.render_n([1]) and '"' not in wake.render_n([1])


def check_n_bytes_carry_no_host():
    """The published n is byte-for-byte what render_n says, on any platform.

    n is LF on every host.
    """
    with tempfile.TemporaryDirectory(prefix="mtr-n-") as tmp:
        n = Path(tmp) / "n"
        wake.publish_n(Path(tmp), [1_000_000, 996_989])
        assert n.read_bytes() == b"[1000000,996989]\n", n.read_bytes()
        assert b"\r" not in n.read_bytes(), "no carriage return reaches the agent"


def check_config_is_validated():
    """The real config.toml is valid, and bad keys, types, and values are refused.

    Unknown keys, wrong types, out-of-range values, and a --config path that
    does not exist all exit nonzero. The last one matters most: a file asked for
    by name and silently not read is a run configured differently than believed,
    which is the failure the whole function exists to refuse.
    """
    with pinned():
        try:
            wake.load_config()                              # the real file, if present
        except SystemExit as e:                             # reported as a failure
            raise AssertionError(f"config.toml is invalid: {e}") from None
        assert wake.MODEL in wake.PRICES
        assert wake.MODEL in wake.THINKING
        assert 0 < wake.CONTEXT_FRACTION <= 1
        assert wake.MAX_TOKENS <= wake.MAX_TOKENS_CEILING
        assert wake.OVERDRAFT >= 0
        assert type(wake.LIVE_N) is bool

    with tempfile.TemporaryDirectory(prefix="mtr-cfg-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in ('turn_cpa = 5', 'system = "hi"', 'turn_cap = "many"',
                    'model = "no-such-model"', 'context_fraction = 2.0', 'budget = 0',
                    'overdraft = -1', 'live_n = "yes"',
                    # Above the ceiling the harness would time out mid-session.
                    f'max_tokens = {wake.MAX_TOKENS_CEILING + 1}'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                try:
                    wake.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad config: {bad}")

        # A named file that is not there is refused, not quietly skipped.
        with pinned():
            try:
                wake.load_config(Path(tmp) / "confg.toml")
            except SystemExit:
                pass
            else:
                raise AssertionError("a missing --config path was ignored")

        f.write_text("turn_cap = 7\ncontext_fraction = 1\n", encoding="utf-8")
        with pinned():
            assert wake.load_config(f) == f, "the file used is reported back"
            assert wake.TURN_CAP == 7, "a good value must actually apply"
            assert wake.CONTEXT_FRACTION == 1.0, "an int must widen into a float field"


def check_lapsed_rates_are_refused():
    """A model whose rates are known to have expired cannot start a run.

    A lapsed rate is the quietest way to be wrong: the run completes, nothing
    looks off, and every number in meter.json and in n is out by the difference.
    Both sides of the date are checked, so neither waits for the other to come.
    """
    assert wake.lapsed_prices("claude-sonnet-5", "2026-08-31") is None, "the last valid day runs"
    assert wake.lapsed_prices("claude-sonnet-5", "2026-09-01"), "the day after must refuse"
    assert wake.lapsed_prices("claude-opus-5", "2099-01-01") is None, \
        "a model with no known expiry never lapses"
    assert set(wake.PRICES_EXPIRE) <= set(wake.PRICES), "an expiry for an unpriced model is dead"
    assert wake.lapsed_prices(wake.MODEL) is None, \
        f"{wake.MODEL}: the rates in PRICES have lapsed as of today; update them"

    # And main() refuses before it can create a run or reach the client, naming
    # the model and the date rather than leaving it to be noticed later.
    with pinned():
        wake.load_config()                           # whichever model is configured
        was, wake.PRICES_EXPIRE = wake.PRICES_EXPIRE, {
            **wake.PRICES_EXPIRE, wake.MODEL: ("2000-01-01", "something newer")}
        try:
            with quiet() as buf:
                assert wake.main(["--run-id", "t"]) == 2
        finally:
            wake.PRICES_EXPIRE = was
    assert "expired 2000-01-01" in buf.getvalue(), buf.getvalue()


def check_truncation_and_empty():
    """Oversized output is clipped with an explicit marker, keeping head and tail."""
    c = wake.clip("x" * 50_000, 8_000)
    assert "[truncated: 42000 of 50000 characters]" in c
    assert c.startswith("x") and c.endswith("x") and len(c) < 8_500, "head and tail both kept"
    assert wake.clip("short", 8_000) == "short"


def check_sessions_reconcile():
    """sum(spent) == initial - remaining, and the series accounts for it turn by turn.

    One element per billed turn, so a session's own slice of the series has to
    drop by exactly what that session spent, and the last element has to be the
    balance rather than an approximation of it.
    """
    with temp_root():
        for _ in range(5):
            wake_once(*DEFAULT)
        meter = ground_truth()
        series, sessions = meter["series"], meter["sessions"]
        assert len(sessions) == 5, sessions
        assert len(series) == 1 + sum(s["turns"] for s in sessions), "one element per billed turn, plus seed"
        spent = sum(s["spent"] for s in sessions)
        assert spent == meter["initial"] - meter["remaining"], f"{spent} != {meter['initial'] - meter['remaining']}"
        assert series[-1] == meter["remaining"], "the last element is the balance"
        at = 0
        for s in sessions:
            woke, ended = series[at], series[at + s["turns"]]
            assert woke - ended == s["spent"], f"session {s['index']}: {woke} - {ended} != {s['spent']}"
            at += s["turns"]
        assert (wake.state_dir("t") / "n").read_text() == wake.render_n(series)


def check_n_grows_within_a_session():
    """LIVE_N: every billed turn appends its balance to n while the session runs.

    Appended, never rewritten, so what the agent has already read stays true and
    the series is the balance's whole history. The element a turn adds differs
    from the one before it by what that turn cost, which is what makes the
    number something the agent can run an experiment against rather than a
    series it can only fit a curve to.
    """
    with temp_root(LIVE_N=True):
        t = wake_once(run("cat state/n"), run("cat state/n"), say())
    before, balances = t["series_before"], [x["balance"] for x in t["turns"]]
    first, second = (json.loads(t["turns"][i]["tools"][0]["result"]) for i in (0, 1))
    assert first == before + balances[:1], (first, before, balances)
    assert second == before + balances[:2], (second, before, balances)
    assert second[:len(first)] == first, "elements are appended, never rewritten"
    assert all(type(v) is int for v in second), second
    assert first[-1] - second[-1] == t["turns"][1]["micros"], \
        "the drop between two reads is what the turn between them cost"
    assert t["series_after"] == before + balances, "and the session commits exactly those"
    assert t["live_n_writes"] == len(t["turns"]) and t["live_n_errors"] == 0, t["live_n_errors"]


def check_live_n_can_be_turned_off():
    """LIVE_N off leaves n fixed for the whole session; the turns arrive at the wake.

    The series is per-turn under either regime. What the setting decides is
    whether the agent can see an element appear while it is working, or only
    finds the session's worth of them together at the next wake. Both have to
    remain runnable: a run under one is not comparable with a run under the
    other, and provenance is what says which it was.
    """
    with temp_root(LIVE_N=False):
        t = wake_once(run("cat state/n"), run("cat state/n"), say())
    reads = [c["result"].strip() for x in t["turns"][:2] for c in x["tools"]]
    assert reads[0] == reads[1] == wake.render_n(t["series_before"]).strip(), reads
    assert t["live_n_writes"] == 0 and t["live_n_errors"] == 0
    assert t["provenance"]["live_n"] is False, "the trace must say which regime this was"
    assert t["read_n"], "a read of the fixed form is still a read"
    assert t["series_after"] == t["series_before"] + [x["balance"] for x in t["turns"]], \
        "the turns still reach the series, just not during the session"


def check_live_n_is_the_agents_own_file():
    """The mid-session rewrite leaves n owned and moded as the agent's, and alone.

    It is written as root from outside the agent's shell, so ownership has to be
    put back or the agent would find a file it could no longer write. The stage
    file lives outside state/, because a second name in there is a second thing
    the agent can read.
    """
    with temp_root(LIVE_N=True):
        t = wake_once(run("stat -c '%a %U:%G %n' state/n", "ls -a state"), say())
    stat, listing = (c["result"] for c in t["turns"][0]["tools"])
    assert stat.strip() == "644 agent:agent state/n", stat
    assert sorted(listing.split()) == [".", "..", "n"], f"state/ must hold only n: {listing}"
    assert [f["path"] for f in t["files"]] == ["n"], t["files"]


def check_overdraft_is_readable_as_a_negative_balance():
    """A balance that has gone negative is a number some session wakes to.

    At overdraft 0 the run ends holding it and no instance ever sees it. Above
    0 the session that wakes to the overshoot gets that much runway, so the
    negative value reaches the agent - and the run still terminates.
    """
    # One short of a turn, so the first session cannot help but overshoot zero.
    cost = wake.measure(usage(), wake.MODEL)["centi"] // 100
    runway = cost * 4

    with temp_root(BUDGET=cost - 1, OVERDRAFT=0):
        first = wake_once(*DEFAULT)
        assert first["meter_floor"] == 0, "at 0 a session stops at zero"
        assert first["remaining"] < 0, "the last turn overshoots; that is the value at stake"
        with quiet():
            assert wake.run_sessions("t", fake(), 1) == 3, "and no session may start on it"

    with temp_root(BUDGET=cost - 1, OVERDRAFT=runway):
        wake_once(*DEFAULT)
        grace = wake_once(run("cat state/n"), say())
        assert grace["series_before"][-1] < 0, grace["series_before"]
        assert grace["turns"], "the session that wakes to it must get a turn"
        assert grace["read_n"], "and be able to read it"
        saw = json.loads(grace["turns"][0]["tools"][0]["result"])
        assert saw[:len(grace["series_before"])] == grace["series_before"], \
            "the negative balance is in what the agent saw"
        assert grace["meter_floor"] == grace["series_before"][-1] - runway, \
            "its runway is measured from the balance it woke to"

    # And it converges: a session is admitted only while the balance is above
    # -OVERDRAFT, and every session spends.
    with temp_root(BUDGET=cost - 1, OVERDRAFT=runway):
        with quiet():
            assert wake.run_sessions("t", fake(), 50) == 0
        meter = ground_truth()
        assert 0 < len(meter["sessions"]) < 50, "the meter ends the run, not the count"
        assert meter["remaining"] <= -runway, meter["remaining"]
        with quiet():
            assert wake.run_sessions("t", fake(), 1) == 3


def check_sessions_are_a_ceiling_not_a_floor():
    """run_sessions(N) runs N sessions, or fewer if the budget ends it first."""
    cost = wake.measure(usage(), wake.MODEL)["centi"] // 100
    with temp_root():
        with quiet() as buf:
            assert wake.run_sessions("t", fake(), 3) == 0
        meter = ground_truth()
        assert [s["index"] for s in meter["sessions"]] == [1, 2, 3], meter["sessions"]
        assert len(meter["series"]) == 1 + sum(s["turns"] for s in meter["sessions"])
        assert buf.getvalue().count("created run") == 1, "the run is created once, not per session"

    # Budget for three sessions, asked for twenty: the meter decides.
    with temp_root(BUDGET=cost * 3, OVERDRAFT=0):
        with quiet() as buf:
            assert wake.run_sessions("t", fake(), 20) == 0
        meter = ground_truth()
        assert 0 < len(meter["sessions"]) < 20, meter["sessions"]
        assert meter["remaining"] <= 0 and "out of budget" in buf.getvalue()


def check_a_fault_ends_the_loop():
    """An interrupted or failed session stops the loop rather than being retried.

    A harness fault or a dead API says nothing about whether the next session
    would work, and a loop that keeps going finds out by spending. A session
    that merely finished, however it finished, is not a fault.
    """
    for fault, stop in ((KeyboardInterrupt(), "interrupted"), (Err(400), "api_error")):
        with temp_root():
            with quiet() as buf:
                assert wake.run_sessions("t", fake(run("echo one"), fault), 5) == 0
            meter = ground_truth()
            assert len(meter["sessions"]) == 1, "the loop must not run a second session"
            assert meter["sessions"][0]["stop"] == stop, meter["sessions"]
            assert "stopping after 1 of 5" in buf.getvalue(), buf.getvalue()
    assert wake.STOP_THE_RUN.isdisjoint(
        {"end_turn", "meter_exhausted", "context_threshold", "turn_cap",
         "max_tokens", "no_tool_call", "refusal"}), wake.STOP_THE_RUN


def check_sessions_flag_is_validated():
    """--sessions below one is refused before anything is created or billed.

    The check sits ahead of both the config load and the client, so a bad count
    cannot reach the API.
    """
    for bad in ("0", "-1"):
        with quiet():
            try:
                wake.main(["--run-id", "t", "--sessions", bad])
            except SystemExit as e:
                assert e.code == 2, e.code
                continue
        raise AssertionError(f"accepted --sessions {bad}")


def check_tampered_n_is_overwritten():
    """I2: a rewritten n is restored and flagged; clean and fresh runs are not."""
    with temp_root():
        with quiet():
            wake.load_meter("t")
        (wake.state_dir("t") / "n").write_text("[999999999]\n")
        t = wake_once(*DEFAULT)
        assert t["tampered_n"] is True
        assert 999999999 not in t["series_after"]
        assert wake_once(*DEFAULT)["tampered_n"] is False, "a clean session must not be flagged"
    with temp_root():
        assert wake_once(*DEFAULT)["tampered_n"] is False, "a fresh run must not be flagged"


def check_bills_once():
    """A response is charged exactly once, however the duplicate arose.

    Two ways it can: the harness retried a 429 that the server had already served,
    or the server deduped and replayed a response id we have seen.
    """
    with temp_root():
        clean = wake_once(*DEFAULT)
    with temp_root():
        retried = wake_once(Err(429), *DEFAULT)
    assert retried["retries"] and retried["retries"][0]["status"] == 429
    assert retried["spent"] == clean["spent"], f"the 429 was billed: {retried['spent']} vs {clean['spent']}"

    with temp_root():
        t = wake_once(run("echo one", id="dup"), run("echo two", id="dup"), say())
    dup = [x for x in t["turns"] if x["id"] == "dup"]
    assert len(dup) == 2 and dup[1]["micros"] == 0, "second sighting of an id must be free"
    # The token counts go with the money. Left in place they would be summed
    # twice by analyze.py while spent counted them once.
    assert all(dup[1][k] == 0 for k in wake.BILLABLE), f"tokens billed twice: {dup[1]}"
    assert any(dup[0][k] for k in wake.BILLABLE), "the first sighting must keep its counts"

    # And what that does to n is deliberate rather than incidental. The replayed
    # turn appends a balance equal to the one before it, because the incremental
    # cost really was zero: the element is truthful and n stays one per turn. A
    # flat step in the series is a retry made visible, not a gap in the record.
    assert dup[0]["balance"] == dup[1]["balance"], "a replayed response moves nothing"
    assert len(t["series_after"]) == len(t["series_before"]) + len(t["turns"]), \
        "and still appends one element per turn"


def check_per_turn_micros_partition_the_spend():
    """The per-turn column sums to exactly what the session spent.

    Rounding each response's cost on its own drops a fraction per turn that the
    session's own total never dropped, so the turns would add up to less than
    spent. Reading micros off the balance instead makes the column a partition
    of the spend, which is also the only reading under which a turn's cost and
    the move in n are the same number.

    A cache read prices at a tenth, so a single one puts half a micro-dollar on
    each turn and the fraction has to carry - without it every turn is a round
    number and this proves nothing.
    """
    odd = usage(cache_read_input_tokens=1)
    with temp_root(MODEL="claude-opus-5"):
        t = wake_once(run("echo one", u=odd), run("echo two", u=odd), say(u=odd))
    micros = [x["micros"] for x in t["turns"]]
    assert len(micros) == 3, micros
    assert sum(micros) == t["spent"], f"{micros} sums to {sum(micros)}, spent {t['spent']}"
    assert micros[0] != micros[1], f"the fraction must carry, or this proves nothing: {micros}"
    assert t["balances"] == [t["series_before"][-1] - sum(micros[:i + 1])
                             for i in range(len(micros))], t["balances"]


def check_interrupt_still_traces_and_commits():
    """Ctrl+C ends the session with its spend recorded, not discarded.

    The session stops with stop=interrupted, the spend reaches the series, and
    the trace is written.
    """
    with temp_root():
        t = wake_once(run("echo one"), KeyboardInterrupt())
        assert t["stop"] == "interrupted", t["stop"]
        assert t["spent"] > 0, "spend before the interrupt must reach the series"
        meter = ground_truth()
        assert meter["initial"] - meter["remaining"] == t["spent"]
        assert len(meter["sessions"]) == 1, "the session must appear in the record"
        assert (wake.state_dir("t") / "n").read_text() == wake.render_n(meter["series"])


def check_fatal_error_still_traces_and_commits():
    """A 400 aborts without retrying, but the spend before it still reaches the series."""
    with temp_root():
        t = wake_once(run("echo before"), Err(400))
        assert t["stop"] == "api_error" and t["retries"] == [], "a 400 must not be retried"
        assert t["spent"] > 0, "spend before the failure must still reach the series"
        meter = ground_truth()
        assert meter["initial"] - meter["remaining"] == t["spent"]

    # A session that never got a turn spent nothing and adds nothing. The
    # balance did not move, and an element saying so would be a reading that
    # nothing took.
    with temp_root():
        t = wake_once(Err(400))
        assert t["turns"] == [] and t["spent"] == 0, t["spent"]
        assert t["series_after"] == t["series_before"], t["series_after"]


def check_no_tool_call_on_turn_one():
    """Text with no tool call on turn one is a clean outcome, not an error, and still bills."""
    with temp_root():
        t = wake_once(say("Nothing to do."))
        assert t["stop"] == "no_tool_call" and t["error"] is None, t
        assert t["spent"] > 0 and t["series_after"][-1] == t["remaining"]


def check_stop_reasons():
    """end_turn, refusal, context_threshold, and turn_cap each fire on the right condition."""
    with temp_root():
        assert wake_once(*DEFAULT)["stop"] == "end_turn"
    with temp_root():
        big = usage(input_tokens=900_000)
        assert wake_once(run("echo hi", u=big), say())["stop"] == "context_threshold"
    with temp_root(TURN_CAP=1):
        assert wake_once(*DEFAULT)["stop"] == "turn_cap"
    # A refusal on turn one must not be recorded as "nothing to do" - models with
    # safety classifiers can decline before the agent has done anything.
    with temp_root():
        assert wake_once(say("I can't help with that.", stop="refusal"))["stop"] == "refusal"
    with temp_root():
        assert wake_once(run("echo hi"), say(stop="refusal"))["stop"] == "refusal"


def check_truncated_turn_is_not_a_clean_end():
    """A turn cut off at max_tokens is recorded as truncated, not as a clean end.

    Both shapes matter. Text truncated mid-sentence would otherwise read as
    end_turn, making the prompt's "sessions end when context is exhausted"
    false. A tool_use block truncated mid-JSON arrives with no command, which
    the restart path would honour - silently wiping cwd and exports and running
    nothing.
    """
    with temp_root():
        t = wake_once(run("echo hi"), say("half a sen", stop="max_tokens"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert t["error"] is None, "truncation is an outcome, not a harness fault"
        assert t["spent"] > 0, "the truncated turn was still billed"

    with temp_root():
        t = wake_once(run("cd /tmp; export MARK=before"),
                      run(None, stop="max_tokens"),          # truncated mid-JSON
                      run("pwd", "echo [$MARK]"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert len(t["turns"]) == 2, "the session must stop at the truncated turn"
        assert t["turns"][1]["tools"] == [], "a truncated call must not be executed"
        assert t["commands"] == [wake.OPENING, "cd /tmp; export MARK=before"], t["commands"]


def check_turn_records_the_api_stop_reason():
    """Every turn carries the API's own stop_reason, not just the derived stop.

    The session-level `stop` is the harness's reading of events; without the
    API's own word per turn, a truncated turn cannot be told from a clean one
    after the fact.
    """
    with temp_root():
        t = wake_once(run("echo hi"), say("bye"))
    assert [x["stop_reason"] for x in t["turns"]] == ["tool_use", "end_turn"], t["turns"]


def check_thinking_is_pinned_per_model():
    """Every model is sent an explicit thinking policy, and fable-5's is recorded.

    An absent `thinking` no longer means the same thing on every model: on
    opus-5, sonnet-5, and fable-5 it now runs adaptive thinking. Omitting it
    would make what the agent *is* vary by model, so the table decides.
    """
    assert set(wake.THINKING) == set(wake.PRICES), "every priced model needs a policy"
    for model, want in wake.THINKING.items():
        seen = []
        with temp_root(MODEL=model):
            wake_once(say(), seen=seen)
        got = {k: v for k, v in seen[0].items() if k == "thinking"}
        assert got == want, f"{model}: sent {got}, table says {want}"

    # fable-5 cannot turn thinking off, so its reasoning must reach the record
    # rather than being dropped with the non-text blocks.
    with temp_root(MODEL="claude-fable-5"):
        t = wake_once(think("weighing it up", "here goes"), say())
    assert t["turns"][0]["thinking"] == "weighing it up", t["turns"][0]
    assert t["turns"][0]["text"] == "here goes", "reasoning stays apart from spoken words"


def check_reaching_for_n_is_not_reading_it():
    """touched_n is reaching for n; read_n is having seen its contents.

    touched_n is true when a command names n; read_n is true only when n's
    contents appear in a result the agent saw.
    """
    with temp_root():
        missed = wake_once(run("cat n"), say())          # wrong path: fails
        hit = wake_once(run("cat state/n"), say())       # right path: succeeds
    assert missed["touched_n"] and not missed["read_n"], "a failed read is not a read"
    assert hit["touched_n"] and hit["read_n"], "a successful read is both"
    # Under LIVE_N a read comes back as the committed series plus the balance so
    # far, so the committed array is a prefix of what the agent saw rather than
    # equal to it.
    got = json.loads(hit["turns"][0]["tools"][0]["result"])
    assert got[:len(hit["series_before"])] == hit["series_before"], got

    with temp_root():
        blind = wake_once(run("wc -c state/n"), say())   # names n, never sees it
    assert blind["touched_n"] and not blind["read_n"], "a size check is not a read"


def check_read_n_survives_a_clipped_read():
    """A read of an n too big for the tool bound is still a read.

    Past roughly a thousand turns n outgrows TOOL_RESULT_LIMIT and comes back
    clipped. Matching only the whole array would pin the headline "first
    actually read n" to never, for a reason about the harness rather than the
    agent. clip() keeps a fixed head and n is append-only, so those leading
    bytes are the same at every turn and matching them is exact.
    """
    with temp_root():
        with quiet():
            wake.load_meter("t")                     # create the run, then enlarge it
        m = ground_truth()
        m["series"] = list(range(1_000_000, 1_000_000 + 2_000))
        m["remaining"] = m["series"][-1]
        wake.save_meter("t", m)
        wake.publish_n(wake.state_dir("t"), m["series"])
        t = wake_once(run("cat state/n"), say())

    result = t["turns"][0]["tools"][0]["result"]
    assert "[truncated:" in result, "the read has to have actually been clipped"
    assert len(result) < wake.TOOL_RESULT_LIMIT + 500, len(result)
    assert t["n_bytes"] > wake.TOOL_RESULT_LIMIT and t["n_fits"] is False, t["n_bytes"]
    assert t["read_n"], "a clipped read of n is still a read"
    assert t["touched_n"]


def check_n_in_prose_is_scored_loose_and_strict():
    """n as a maths variable and n as a filename are told apart.

    The loose reading is right for a command, where the shell resolves n as a
    path. In prose it catches every n an agent writes while fitting a curve, so
    the strict reading is scored beside it and the gap is the inflation.
    """
    with temp_root():
        maths = wake_once(run("printf 'fit x_n = a*exp(-c*n^2) for n in 1..5\\n' > state/notes.md"),
                          say())
    assert maths["mentions"]["n"], "the loose reading catches n as a variable"
    assert not maths["mentions"]["n_path"], "the strict one does not"

    with temp_root():
        named = wake_once(run("printf 'the series lives in state/n, append-only\\n' > state/notes.md"),
                          say())
    assert named["mentions"]["n"] and named["mentions"]["n_path"], "a path is caught by both"


def check_opening_is_recorded():
    """The agent's first stimulus is in the trace, not just the command that made it."""
    seen = []
    with temp_root():
        t = wake_once(say(), seen=seen)
    assert t["opening"].strip(), "the opening ls output must be recorded"
    assert " n\n" in t["opening"] or t["opening"].rstrip().endswith(" n"), t["opening"]
    assert t["opening"] == seen[0]["messages"][0]["content"], \
        "the record must hold exactly what was sent as turn one"
    assert t["commands"][0] == wake.OPENING, "the trace says which command produced it"


def check_opening_says_where_it_is():
    """The listing names the directories it is of, so `n` is not hunted for.

    Both operands are named so ls prints a header for each, showing state as a
    subdirectory of the working directory rather than as the working directory.
    """
    with temp_root():
        t = wake_once(say())
    assert ".:" in t["opening"] and "./state:" in t["opening"], t["opening"]
    # state is visible as a subdirectory of the working directory, and n inside it.
    before, _, after = t["opening"].partition("./state:")
    assert re.search(r"\bstate$", before, re.M), f"state must show as a subdirectory: {before}"
    assert re.search(r"\bn$", after, re.M), f"n must show inside ./state: {after}"


def check_first_turn_is_raw_ls():
    """I1: turn one is the verbatim ls output, the system prompt is exact, caching is on."""
    seen = []
    with temp_root():
        wake_once(say(), seen=seen)
    first = seen[0]["messages"][0]
    assert first["role"] == "user" and "\nn\n" not in first["content"], "not a wrapper"
    assert " n\n" in first["content"] or first["content"].rstrip().endswith(" n"), first["content"]
    assert seen[0]["system"] == wake.SYSTEM
    assert seen[0]["cache_control"] == {"type": "ephemeral"}, "caching must be live"


def check_shell_is_persistent():
    """bash_20250124 is a persistent shell, so cd and exports must stick.

    One bash process is held open for the session, so state set by one command
    is still there for the next.
    """
    with temp_root():
        t = wake_once(run("cd state; pwd", "pwd", "export MARK=kept", "echo $MARK",
                          "MARK=$MARK; cd /tmp", "pwd"), say())
    got = [c["result"].strip() for c in t["turns"][0]["tools"]]
    assert got[0] == "/work/state", got
    assert got[1] == "/work/state", f"cd must persist across calls: {got}"
    assert got[3] == "kept", f"exports must persist across calls: {got}"
    assert got[5] == "/tmp", got


def check_restart_gives_a_fresh_shell():
    """{"restart": true} really restarts, and still says nothing to the agent."""
    with temp_root():
        t = wake_once(run("cd /tmp; export MARK=before"), restart(),
                      run("pwd", "echo [$MARK]"), say())
    assert t["turns"][1]["tools"][0]["result"] == " ", "restart carries no harness voice"
    after = [c["result"].strip() for c in t["turns"][2]["tools"]]
    assert after == ["/work", "[]"], f"restart must clear cwd and exports: {after}"


def check_a_bare_read_cannot_wedge_the_session():
    """A command that reads stdin must not swallow the framing of its own output.

    A command that reads stdin returns empty and the session continues.
    """
    with temp_root():
        t = wake_once(run("cat", "echo alive", "head -n 5"), say())
    got = [c["result"] for c in t["turns"][0]["tools"]]
    assert got[0] == " ", f"a stdin reader returns empty, not a hang: {got[0]!r}"
    assert got[1].strip() == "alive", f"the session survives it: {got}"
    assert got[2] == " ", got


def check_hostile_output_survives():
    """Binary bytes, a flood of output, and a hang each leave the session alive and marked.

    The flood is 4MB rather than 50KB so it exercises the scan in Shell.run at a
    size where decoding the buffer on every poll would be slow enough to trip
    the command's own two-second deadline.
    """
    with temp_root(TIMEOUT=2):
        t = wake_once(run("head -c 4096 /dev/urandom"),
                      run("head -c 4000000 /dev/zero | tr '\\0' x"),
                      run("sleep 30"), say())
        assert t["stop"] == "end_turn" and t["error"] is None, t["error"]
        results = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        flood = results[1]
        assert "truncated:" in flood, "the flood should have been clipped"
        assert "timed out" not in flood, "scanning the flood must not outlast the deadline"
        assert len(flood) < wake.TOOL_RESULT_LIMIT + 500, f"clipped to the tool bound: {len(flood)}"
        assert any("timed out after 2s" in r for r in results), "the hang should be marked"


def check_state_contents_are_captured():
    """Every session records what the agent's files held at that moment.

    state/ keeps only the latest revision, so these per-session copies are the
    record of how what the agent writes to itself changes over time.
    """
    with temp_root():
        wake_once(run("echo doctrine v1 > state/notes.md"), say())
        wake_once(run("echo doctrine v2 > state/notes.md"), say())
        wake_once(run("rm state/notes.md"), say())
        # Read back from disk: the point is that the record persists, unaltered
        # by what later sessions did to the file.
        traces = [json.loads(p.read_text()) for p in
                  sorted((wake.private_dir("t") / "traces").glob("session-*.json"))]

    def note(t):
        return next((f for f in t["files"] if f["path"] == "notes.md"), None)

    assert note(traces[0])["text"].strip() == "doctrine v1"
    assert note(traces[1])["text"].strip() == "doctrine v2", "each session keeps its own copy"
    assert note(traces[2]) is None, "a deleted file leaves the listing"

    with temp_root():
        t = wake_once(run("head -c 64 /dev/zero > state/blob.bin",
                          "head -c 200000 /dev/zero | tr '\\0' x > state/big.txt"), say())
    by = {f["path"]: f for f in t["files"]}
    assert by["blob.bin"]["text"] is None, "binary is marked, not stored as mojibake"
    assert by["blob.bin"]["size"] == 64, "the size is recorded either way"
    assert by["big.txt"]["size"] == 200_000, "the true size, not the captured slice"
    assert "[truncated: 100000 of 200000 bytes]" in by["big.txt"]["text"], "truncation is stated"
    assert by["n"]["text"] is None, "n is ours; series_before already carries it"


def check_a_live_balance_counts_as_a_number_written():
    """A balance that arrived mid-session counts as a number the agent wrote.

    mentions["number"] is decided against the balances the agent could have
    read, and under LIVE_N those include the session's own elements as they are
    billed. Deciding it against the series as it stood at the wake would miss a
    note holding only the newest element, which is what "first wrote a number"
    in the report is meant to catch.
    """
    with temp_root(LIVE_N=True):
        t = wake_once(run("echo hi"),
                      run("tr -d '[]' < state/n | tr ',' '\\n' | tail -1 > state/note.txt"),
                      say())
    note = next(f for f in t["files"] if f["path"] == "note.txt")
    wrote = int(note["text"])
    assert wrote not in t["series_before"], \
        f"the note must hold an element that arrived this session: {wrote}"
    assert wrote in t["series_after"], (wrote, t["series_after"])
    assert t["mentions"]["number"], "a balance read this session is a number written"
    assert t["mention_lines"], "and the line it was on is quoted"


def check_missing_tools_are_recorded():
    """A tool the agent reached for and the image lacks is named in the trace.

    The case that matters is stderr redirected away, which is how the agent
    writes these by habit: the transcript then shows empty output, identical to
    a check that ran and found nothing.
    """
    assert wake.invoked("python3 -c 'import os; print(os.getcwd())'") == {"python3"}, \
        "a program passed as an argument is not a list of commands"
    assert wake.invoked("cat <<'EOF' > f.py\nimport sys\nprint(1)\nEOF") == {"cat"}, \
        "a here-document body is not a list of commands"
    assert wake.invoked("A=1 rg foo / | head -3; getfattr -d n") == {"rg", "head", "getfattr"}
    # The apostrophe in the comment must not pair with the quote in the program.
    assert wake.invoked("# Let's look\npython3 -c \"\nimport json\nprint(open('n'))\n\"") \
        == {"python3"}, "prose in a comment must not expose the program after it"

    with temp_root():
        t = wake_once(run("getfattr -d ./state/n 2>/dev/null; nosuchtool --help 2>/dev/null"), say())
    assert "nosuchtool" in t["missing_tools"], "a silenced miss must still be recorded"
    assert "getfattr" not in t["missing_tools"], "a tool the image has is not a miss"
    assert "cat" not in t["missing_tools"] and "ls" not in t["missing_tools"]


def check_provenance_is_recorded():
    """Each trace states what decided the session, and says when that changed.

    budget and model are pinned in meter.json; everything else here is read at
    each wake, so a run can be several environments deep without meter.json
    differing by a byte. The trace is the only place that can say so.
    """
    with temp_root():
        with quiet():
            first = wake.run_once("t", fake(*DEFAULT))
            prov = first["provenance"]
            for key in ("started_at", "harness_sha256", "image", "image_id", "prices",
                        "thinking", "context_fraction", "max_tokens", "turn_cap",
                        "timeout", "tool_result_limit", "live_n", "overdraft"):
                assert key in prov, f"provenance omits {key}"
            assert prov["prices"] == list(wake.PRICES[wake.MODEL]), "the rates actually applied"
            assert first["model_resolved"], "the dated snapshot behind the alias"
            assert first["provenance_drift"] == [], "nothing to differ from on session one"

            # A rate change mid-run makes early and late entries of the same
            # series mean different things. The seam has to be findable.
            was = wake.PRICES[wake.MODEL]
            wake.PRICES[wake.MODEL] = (was[0] * 2, was[1], was[2])
            try:
                second = wake.run_once("t", fake(*DEFAULT))
            finally:
                wake.PRICES[wake.MODEL] = was
    assert any(d.startswith("prices:") for d in second["provenance_drift"]), \
        "a mid-run rate change must be recorded on the session that changed"


def check_watch_is_display_only():
    """--watch echoes the session and changes nothing else.

    The request bytes and the trace are identical whether or not it is on.
    """
    quiet_seen, loud_seen = [], []
    with temp_root():
        with quiet():
            plain = wake.run_once("t", fake(*DEFAULT, seen=quiet_seen))
    with temp_root(WATCH=True):
        with quiet() as buf:
            loud = wake.run_once("t", fake(*DEFAULT, seen=loud_seen))
        shown = buf.getvalue()

    assert f"$ {wake.OPENING}" in shown, "the opening turn is shown"
    assert "$ echo hi > state/note.txt" in shown, "commands are shown"
    assert "turn 1" in shown and "context" in shown, "per-turn meter and context are shown"
    assert loud_seen == quiet_seen, "watching must not change what is sent to the model"
    for t in (plain, loud):
        # Wall clock, not the record: these differ between any two runs.
        t.pop("duration_s")
        t["provenance"].pop("started_at")
    assert plain == loud, "watching must not change the record"


def check_state_looks_like_itself():
    """The agent sees the modes and ownership that were actually intended.

    State sits on the container's own filesystem, so its modes and ownership
    are real and read the same on every host.
    """
    with temp_root():
        t = wake_once(run("ls -la state/n", "stat -c '%a %U:%G %n' state/n",
                          "test -x state/n && echo EXECUTABLE || echo not-executable"), say())
    shown = "\n".join(c["result"] for c in t["turns"][0]["tools"])
    assert "644 agent:agent state/n" in shown, shown
    assert "not-executable" in shown, "n must not look runnable"
    assert "rwxrwxrwx" not in shown and "root" not in shown, shown


def check_agent_writes_survive_the_container():
    """What the agent leaves is copied back out, including deletions."""
    with temp_root():
        first = wake_once(run("echo kept > state/keep.txt", "echo doomed > state/gone.txt"), say())
        assert first["state_saved"], "the mirror must be written"
        assert {f["path"] for f in first["files"]} == {"n", "keep.txt", "gone.txt"}

        second = wake_once(run("rm state/gone.txt", "cat state/keep.txt"), say())
        assert {f["path"] for f in second["files"]} == {"n", "keep.txt"}, "a deletion must propagate"
        assert "kept" in second["turns"][0]["tools"][1]["result"], "files persist across sessions"
        # And the agent owns what it made, so it can rewrite it next wake.
        third = wake_once(run("stat -c '%U' state/keep.txt"), say())
        assert third["turns"][0]["tools"][0]["result"].strip() == "agent"


def check_agent_file_modes_survive_the_host():
    """A file's mode is what the agent made it, however many sessions later.

    The host cannot store POSIX modes, so they are carried in a sidecar and
    reapplied when state is copied back into the container.
    """
    with temp_root():
        wake_once(run("echo plain > state/plain.txt",          # 644 by umask
                      "printf '#!/bin/sh\\necho hi\\n' > state/script.sh",
                      "chmod 700 state/script.sh"), say())     # a non-default mode
        seen = wake_once(run("stat -c '%a %n' state/plain.txt state/script.sh"), say())
    modes = dict(reversed(line.split()) for line in
                 seen["turns"][0]["tools"][0]["result"].split("\n") if line.strip())
    assert modes["state/plain.txt"] == "644", f"a plain file must not become executable: {modes}"
    assert modes["state/script.sh"] == "700", f"a deliberate chmod must survive: {modes}"


def check_isolation():
    """I2/I5: private/ is absent from the container, DNS is dead, only n is ours."""
    with temp_root():
        t = wake_once(run("cat /work/../private/meter.json; find / -name meter.json 2>/dev/null; "
                          "getent hosts api.anthropic.com || echo NO-DNS"), say())
        out = t["turns"][0]["tools"][0]["result"]
        assert "initial" not in out and "meter.json" not in out.replace("/work/../private/meter.json", ""), out
        assert "No such file" in out, "private/ should not be reachable"
        assert "NO-DNS" in out, "network should be off"
        assert [f["path"] for f in t["files"] if f["ours"]] == ["n"], "only n is ours"


def check_a_container_failure_stops_the_run_cleanly():
    """A container that will not start ends the run with a message, not a traceback.

    Nothing reaching this path was billed - the container, the state copy, and
    the shell all come before the first API call - so there is no session to
    record and returning without a trace is right rather than lossy.
    """
    with temp_root(IMAGE="mtr-No-Such-Image:latest"):     # rejected on sight, no pull
        with quiet() as buf:
            assert wake.run_sessions("t", fake(*DEFAULT), 3) == 4
        assert "could not start a session container" in buf.getvalue(), buf.getvalue()
        m = ground_truth()
        assert m["sessions"] == [], "a session that never started is not recorded"
        assert m["remaining"] == m["initial"], "and nothing was spent"
        assert m["series"] == [m["initial"]], "and the series did not move"
        assert not list((wake.private_dir("t") / "traces").glob("*.json")), "no trace"


def check_a_failed_mirror_keeps_the_last_record():
    """A mirror that fails leaves the previous session's files where they were.

    save_state swaps a staged copy in whole, so a failure partway is the one way
    the host could end up holding neither the new state nor the old. What the
    agent wrote is only in the container, and losing the host copy loses it.
    """
    with temp_root():
        wake_once(run("echo kept > state/keep.txt"), say())
        state = wake.state_dir("t")
        before = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert "keep.txt" in before, before

        assert wake.save_state("mtr-no-such-container-9f3c1d", state) is False, \
            "a mirror of a container that is not there must fail, not raise"
        after = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert after == before, f"a failed mirror lost the record: {sorted(after)}"
        assert not state.with_name("state.incoming").exists(), "the staging copy is cleaned up"
        assert not state.with_name("state.previous").exists()


def check_containers_are_reaped():
    """Even a session that ends in an API error leaves no container behind."""
    with temp_root():
        wake_once(run("echo hi"), Err(400))
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=mtr-t-", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f"containers leaked: {left}"


# --- runner -----------------------------------------------------------------


def main() -> int:
    """Run every check_* in the module. The container ones skip themselves."""
    docker_ready()                       # so its notice prints above the list
    failed, skipped = [], []
    for name, fn in sorted(globals().items()):
        if not name.startswith("check_"):
            continue
        label = name[6:]
        try:
            fn()
        except Skip:
            skipped.append(label)
            print(f"SKIP  {label}")
        except Exception:
            failed.append(label)
            print(f"FAIL  {label}\n{traceback.format_exc()}")
        else:
            print(f"ok    {label}")

    if skipped:
        print(f"\n{len(skipped)} skipped (needs Docker + the image)")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
    else:
        print("\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
