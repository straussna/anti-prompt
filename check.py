"""Verification. No API spend.

    py -3 check.py                  every check, several at a time
    py -3 check.py refusal seed     only the ones named so
    py -3 check.py --real           every check in a real container
    py -3 check.py --no-docker      only the ones that need no container
    py -3 check.py -j 1 --list

A fake `create` is injected into wake.run_once, so the whole pipeline runs -
meter, turns, series, trace - with nothing billed.

Sessions run in one of two places. Most checks are about arithmetic, and a
container proves none of it, so they run against a directory and a bash process
on this machine. What only a container can show - modes, ownership, the dead
network, what the image has - takes a real one, and those are the checks that
skip when Docker is down. `--real` puts every check in a container, which is
what says the two still agree.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

import cohort
import wake

# The API errors here are scripted, so the wait between retries is time spent
# proving nothing. What is retried and how often still is.
wake.RETRY_BASE = 0

# --- the fake ---------------------------------------------------------------


def usage(**kw):
    """A usage object shaped like the API's, with overridable token counts."""
    return NS(**{"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation": None, "iterations": None, **kw})


def attempt(model, output_tokens, kind="message", **kw):
    """One entry of usage.iterations: what a single model's attempt cost.

    The declining attempts of a fallback chain are `message`; the last one, by
    whichever model the chain reached, is `fallback_message`. An attempt that
    produced no output declined before producing any.
    """
    return NS(**{"type": kind, "model": model, "input_tokens": 100, "output_tokens": output_tokens,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation": None, **kw})


def say(text="done.", u=None, id=None, stop="end_turn", details=None, model=None):
    """A scripted step: reply with text and stop.

    `details` stands in for the API's stop_details, which it sends only
    alongside a refusal. `model` overrides the model the response reports having
    come from, which is how a fallback-served turn is scripted.
    """
    return {"kind": "say", "text": text, "u": u, "id": id, "stop": stop, "details": details,
            "model": model}


def run(*cmds, u=None, id=None, stop="tool_use", details=None, model=None):
    """A scripted step: reply with one bash tool call per command.

    `stop` is the response's stop_reason, so a truncated turn can be scripted.
    """
    return {"kind": "run", "cmds": list(cmds), "u": u, "id": id, "stop": stop,
            "details": details, "model": model}


def refuse(*cmds, category="cyber", u=None, id=None, **detail):
    """A scripted refusal, in either shape the API sends one.

    With commands it carries the tool calls emitted before the block landed;
    with none its content is empty, as a refusal arriving before any output.
    `detail` adds fields to stop_details, such as a recommended_model.
    """
    return run(*cmds, u=u, id=id, stop="refusal",
               details=NS(type="refusal", category=category, explanation="declined", **detail))


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

    Steps run out into a plain "done." reply. `seen` captures the request params.
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
        # which is the fallback's name on a turn a fallback served.
        model = s.get("model") or f"{params['model']}-20990101"
        if s["kind"] == "run":
            return NS(id=rid, model=model, stop_reason=s["stop"], usage=u,
                      stop_details=s.get("details"),
                      content=[NS(type="tool_use", id=f"t{n[0]}_{i}", name="bash", input={"command": c})
                               for i, c in enumerate(s["cmds"])])
        content = [NS(type="text", text=s["text"])]
        if s["kind"] == "think":
            content.insert(0, NS(type="thinking", thinking=s["thinking"]))
        return NS(id=rid, model=model, stop_reason=s["stop"], usage=u, content=content,
                  stop_details=s.get("details"))

    return create


class Skip(Exception):
    """This check needs something this machine cannot give it."""


# Set by --real: every check takes a container, including the ones that would
# otherwise run on the host box. What proves the two lanes still agree.
REAL_ONLY = False


# The answer to docker_ready(), once some process has paid for it. Carried into
# workers rather than asked again in each of them.
_DOCKER: bool | None = None


def docker_ready() -> bool:
    """True if the daemon is up and the image is built. Asked once per process."""
    global _DOCKER
    if _DOCKER is None:
        _DOCKER = _ask_docker()
    return _DOCKER


def _ask_docker() -> bool:
    """Put the question to the daemon. Says so once if the image is not built."""
    if not shutil.which("docker") or subprocess.run(["docker", "info"], capture_output=True).returncode:
        return False
    if subprocess.run(["docker", "image", "inspect", wake.IMAGE], capture_output=True).returncode:
        print(f"image {wake.IMAGE} not built\n")
        return False
    return True


# --- the host box -----------------------------------------------------------
#
# Most checks here are about arithmetic: what a turn cost, what reached the
# series, which stop a session ended on. A container proves none of that, and at
# three to four seconds each they were most of the suite's wall clock. Those run
# against a directory and a bash process on this machine instead.
#
# What only a container can show - modes, ownership, the dead network, what the
# image does and does not have - stays on docker_root below.


def host_bash() -> str | None:
    """The bash to run the host box's sessions in, as an absolute path.

    Resolved rather than left to PATH: on Windows `bash` finds one thing and
    subprocess finds another - Git's and WSL's, which disagree about what a path
    is and what /tmp means - so the one being used has to be the one that was
    looked at.
    """
    return shutil.which("bash")


class HostShell(wake.Shell):
    """The session shell, as a bash process on this machine.

    Inherits the sentinel framing, the timeout, and the output ceiling from
    wake.Shell; only where the process runs and how n is rewritten differ.
    """

    def __init__(self, box: "HostBox") -> None:
        self.box = box
        super().__init__(box.name)

    def argv(self) -> list[str]:
        # No profile: what the agent's shell is must not depend on this account.
        return [host_bash(), "--norc", "--noprofile"]

    def popen_kwargs(self) -> dict:
        return {"cwd": str(self.box.work),
                # Stop MSYS rewriting paths inside the agent's own commands.
                "env": {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}}

    def republish_n(self, series: list[int], expected: str) -> str:
        n = self.box.work / "state" / "n"
        was = n.read_text(encoding="utf-8") if n.exists() else ""
        n.write_text(wake.render_n(series), encoding="utf-8", newline="\n")
        return "ok" if was == expected else "tampered"


class HostBox:
    """A session's world as a directory on this machine, in place of a container.

    Same five methods run_once asks of wake.Container. There is no ownership and
    no locking here: a check that turns on either belongs on docker_root.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = tempfile.mkdtemp(prefix="mtr-host-")
        self.work = Path(self.dir)
        (self.work / "state").mkdir()

    @classmethod
    def start(cls, name: str) -> "HostBox":
        return cls(name)

    def load(self, state: Path, locked: list[str]) -> None:
        shutil.copytree(state, self.work / "state", dirs_exist_ok=True)

    def shell(self) -> HostShell:
        return HostShell(self)

    def save(self, state: Path) -> bool:
        return wake.save_state(state, self._fetch, lambda: None)

    def _fetch(self, dest: Path) -> bool:
        src = self.work / "state"
        if not src.is_dir():
            return False
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return True

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# Every wake global a check is allowed to move, and therefore every one pinned()
# puts back. temp_root refuses any name outside this set.
RESTORED = wake.TUNABLES | {"ROOT", "WATCH", "REFUSAL_TURNS", "BOX"}


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
def rooted(box, **overrides):
    """Point wake at a throwaway directory, with sessions running in `box`.

    `overrides` set wake module globals (TURN_CAP=1, TIMEOUT=2) for the
    duration, and pinned() puts every one of them back.
    """
    unknown = set(overrides) - RESTORED
    assert not unknown, f"a root cannot restore {sorted(unknown)}"
    with pinned(), tempfile.TemporaryDirectory(
            prefix="mtr-check-", ignore_cleanup_errors=True) as d:
        wake.ROOT = Path(d)
        wake.BOX = box
        for k, v in overrides.items():
            setattr(wake, k, v)
        yield Path(d)


@contextlib.contextmanager
def temp_root(**overrides):
    """A throwaway run whose sessions are a directory and a shell on this machine.

    What most checks want: the pipeline end to end - meter, turns, series,
    trace - without paying for a container that proves nothing they assert.
    """
    if REAL_ONLY:
        with docker_root(**overrides) as d:
            yield d
        return
    if not host_bash():
        raise Skip
    with rooted(HostBox, **overrides) as d:
        yield d


@contextlib.contextmanager
def docker_root(**overrides):
    """A throwaway run whose sessions are real containers.

    For the checks that turn on something only a container has. Skips when
    Docker is unavailable, which is the one reason a check here cannot run.
    """
    if not docker_ready():
        raise Skip
    with rooted(wake.Container, **overrides) as d:
        yield d


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
    """The published n is byte-for-byte what render_n says, LF on any host."""
    with tempfile.TemporaryDirectory(prefix="mtr-n-") as tmp:
        n = Path(tmp) / "n"
        wake.publish_n(Path(tmp), [1_000_000, 996_989])
        assert n.read_bytes() == b"[1000000,996989]\n", n.read_bytes()
        assert b"\r" not in n.read_bytes(), "no carriage return reaches the agent"


def check_config_is_validated():
    """The real config.toml is valid, and bad keys, types, and values are refused.

    Unknown keys, wrong types, out-of-range values, and a --config path that
    does not exist all exit nonzero.
    """
    with pinned():
        try:
            wake.load_config()                              # the real file, if present
        except SystemExit as e:                             # reported as a failure
            raise AssertionError(f"config.toml is invalid: {e}") from None
        assert wake.MODEL in wake.PRICES
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

    Both sides of the expiry date are checked.
    """
    assert wake.lapsed_prices("claude-sonnet-5", "2026-08-31") is None, "the last valid day runs"
    assert wake.lapsed_prices("claude-sonnet-5", "2026-09-01"), "the day after must refuse"
    assert wake.lapsed_prices("claude-opus-5", "2099-01-01") is None, \
        "a model with no known expiry never lapses"
    assert set(wake.PRICES_EXPIRE) <= set(wake.PRICES), "an expiry for an unpriced model is dead"
    assert wake.lapsed_prices(wake.MODEL) is None, \
        f"{wake.MODEL}: the rates in PRICES have lapsed as of today; update them"

    # And start() refuses before it can create a run or reach the client. It
    # exits rather than returning a code, so every driver refuses identically
    # instead of each remembering to; the code and the message are unchanged.
    with pinned():
        wake.load_config()                           # whichever model is configured
        was, wake.PRICES_EXPIRE = wake.PRICES_EXPIRE, {
            **wake.PRICES_EXPIRE, wake.MODEL: ("2000-01-01", "something newer")}
        try:
            with quiet() as buf:
                wake.main(["--run-id", "t"])
        except SystemExit as e:
            assert e.code == 2, e.code
        else:
            raise AssertionError("a lapsed rate started a run")
        finally:
            wake.PRICES_EXPIRE = was
    assert "expired 2000-01-01" in buf.getvalue(), buf.getvalue()


def check_unpriced_fallback_targets_are_refused_before_a_run_starts():
    """A permitted fallback target with no rates stops the run; anything else does not.

    The list is what the requested model is allowed to fall back to, which is a
    likely superset of what default routing will actually pick. Worth pricing
    against before a run starts, and not the whole guard: a list that cannot be
    read is not a reason to refuse a run that would otherwise be fine.
    """
    def client(targets=None, raises=None):
        def retrieve(model, betas=None):
            assert betas == [wake.FALLBACK_BETA], betas
            if raises:
                raise raises
            return NS(allowed_fallback_models=targets)
        return NS(beta=NS(models=NS(retrieve=retrieve)))

    priced = sorted(wake.PRICES)[:2]
    assert wake.unpriced_targets(client(priced), wake.MODEL) == [], "all priced: nothing to say"
    assert wake.unpriced_targets(client([]), wake.MODEL) == [], "no targets: nothing to say"

    said = wake.unpriced_targets(client([*priced, "claude-unheard-of-9"]), wake.MODEL)
    assert len(said) == 1 and "claude-unheard-of-9" in said[0], said

    # A field the API does not send, and a call that fails outright: both leave
    # the run to start, because measure_response() is what holds either way.
    with quiet():
        assert wake.unpriced_targets(client(None), wake.MODEL) == [], "absent field is not a refusal"
        assert wake.unpriced_targets(client(raises=Err(500)), wake.MODEL) == [], \
            "an unreadable list is not a refusal"


def check_truncation_and_empty():
    """Oversized output is clipped with an explicit marker, keeping head and tail."""
    c = wake.clip("x" * 50_000, 8_000)
    assert "[truncated: 42000 of 50000 characters]" in c
    assert c.startswith("x") and c.endswith("x") and len(c) < 8_500, "head and tail both kept"
    assert wake.clip("short", 8_000) == "short"


def check_sessions_reconcile():
    """sum(spent) == initial - remaining, and the series accounts for it turn by turn."""
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

    Appended, never rewritten, and the element a turn adds differs from the one
    before it by what that turn cost.
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

    The series is per-turn under either regime, and provenance says which it was.
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


def check_live_n_leaves_n_read_only_and_alone():
    """The mid-session rewrite leaves n as root's and read-only, and alone.

    It is written as root from outside the agent's shell, so the mode the agent
    sees must be the locked one either way. The stage file lives outside state/,
    since a second name in there is a second thing the agent can read.
    """
    with docker_root(LIVE_N=True):
        t = wake_once(run("stat -c '%a %U:%G %n' state/n", "ls -a state"), say())
    stat, listing = (c["result"] for c in t["turns"][0]["tools"])
    assert stat.strip() == "444 root:root state/n", stat
    assert sorted(listing.split()) == [".", "..", "n"], f"state/ must hold only n: {listing}"
    assert [f["path"] for f in t["files"]] == ["n"], t["files"]


def check_overdraft_is_readable_as_a_negative_balance():
    """A balance that has gone negative is a number some session wakes to.

    At overdraft 0 the run ends holding it and no instance sees it. Above 0 the
    session that wakes to the overshoot gets that much runway.
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

    # And exactly one session wakes below zero, however much runway it is given.
    # A second reads the same number for the price of a whole session.
    with temp_root(BUDGET=cost - 1, OVERDRAFT=runway):
        with quiet():
            assert wake.run_sessions("t", fake(), 50) == 0
        meter = ground_truth()
        assert 0 < len(meter["sessions"]) < 50, "the meter ends the run, not the count"
        below = [s["woke_at"] for s in meter["sessions"] if s["woke_at"] <= 0]
        assert len(below) == 1, [s["woke_at"] for s in meter["sessions"]]
        assert meter["remaining"] < 0, meter["remaining"]
        assert meter["remaining"] > -runway, \
            "one session's runway bounds the run; it does not spend the whole overdraft"
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

    A session that merely finished, however it finished, is not a fault.
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
    """--sessions below one is refused before anything is created or billed."""
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
    # The token counts go with the money, so analyze.py's columns reconcile.
    assert all(dup[1][k] == 0 for k in wake.BILLABLE), f"tokens billed twice: {dup[1]}"
    assert any(dup[0][k] for k in wake.BILLABLE), "the first sighting must keep its counts"

    # The replayed turn appends a balance equal to the one before it, its
    # incremental cost being zero: a flat step is a retry made visible.
    assert dup[0]["balance"] == dup[1]["balance"], "a replayed response moves nothing"
    assert len(t["series_after"]) == len(t["series_before"]) + len(t["turns"]), \
        "and still appends one element per turn"


def check_per_turn_micros_partition_the_spend():
    """The per-turn column sums to exactly what the session spent.

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
    """Ctrl+C ends the session with its spend recorded, not discarded."""
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

    # A session that never got a turn spent nothing and adds nothing.
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
    # Safety classifiers can decline before the agent has done anything. One
    # refusal is a turn the session carries on past; REFUSAL_TURNS running is
    # what it stops for.
    with temp_root(REFUSAL_TURNS=2):
        assert wake_once(refuse(), refuse())["stop"] == "refusal"
    with temp_root(REFUSAL_TURNS=2):
        assert wake_once(run("echo hi"), refuse(), refuse())["stop"] == "refusal"


def check_truncated_turn_is_not_a_clean_end():
    """A turn cut off at max_tokens is recorded as truncated, not as a clean end.

    Both shapes are covered: text truncated mid-sentence, and a tool_use block
    truncated mid-JSON, which arrives with no command and would otherwise be
    honoured by the restart path.
    """
    with docker_root():
        t = wake_once(run("echo hi"), say("half a sen", stop="max_tokens"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert t["error"] is None, "truncation is an outcome, not a harness fault"
        assert t["spent"] > 0, "the truncated turn was still billed"

    with docker_root():
        t = wake_once(run("cd /tmp; export MARK=before"),
                      run(None, stop="max_tokens"),          # truncated mid-JSON
                      run("pwd", "echo [$MARK]"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert len(t["turns"]) == 2, "the session must stop at the truncated turn"
        assert t["turns"][1]["tools"] == [], "a truncated call must not be executed"
        assert t["commands"] == [wake.OPENING, "cd /tmp; export MARK=before"], t["commands"]


def check_turn_records_the_api_stop_reason():
    """Every turn carries the API's own stop_reason, not just the derived stop."""
    with temp_root():
        t = wake_once(run("echo hi"), say("bye"))
    assert [x["stop_reason"] for x in t["turns"]] == ["tool_use", "end_turn"], t["turns"]


def check_every_request_asks_for_fallback():
    """Fallback is on the request itself, and no thinking policy is sent.

    A declined turn is only retried on another model if the parameter is there,
    so the check is that every request carries it - not that some setting
    somewhere says it should. An omitted `thinking` is what keeps the request
    valid as a direct request to whichever model the chain reaches.
    """
    for model in wake.PRICES:
        seen = []
        with temp_root(MODEL=model):
            wake_once(run("echo hi"), say(), seen=seen)
        assert seen, f"{model}: no request captured"
        for params in seen:
            assert params["fallbacks"] == "default", f"{model}: sent {params.get('fallbacks')!r}"
            assert params["betas"] == [wake.FALLBACK_BETA], f"{model}: sent {params.get('betas')!r}"
            assert "thinking" not in params, f"{model}: sent a thinking policy"


def check_reasoning_reaches_the_record():
    """Thinking blocks are recorded, and kept apart from spoken words."""
    with temp_root():
        t = wake_once(think("weighing it up", "here goes"), say())
    assert t["turns"][0]["thinking"] == "weighing it up", t["turns"][0]
    assert t["turns"][0]["text"] == "here goes", "reasoning stays apart from spoken words"


def check_reaching_for_n_is_not_reading_it():
    """touched_n is a command naming n; read_n is n's contents in a result."""
    with temp_root():
        missed = wake_once(run("cat n"), say())          # wrong path: fails
        hit = wake_once(run("cat state/n"), say())       # right path: succeeds
    assert missed["touched_n"] and not missed["read_n"], "a failed read is not a read"
    assert hit["touched_n"] and hit["read_n"], "a successful read is both"
    # Under LIVE_N a read comes back as the committed series plus the balance so
    # far, so the committed array is a prefix of what the agent saw.
    got = json.loads(hit["turns"][0]["tools"][0]["result"])
    assert got[:len(hit["series_before"])] == hit["series_before"], got

    with temp_root():
        blind = wake_once(run("wc -c state/n"), say())   # names n, never sees it
    assert blind["touched_n"] and not blind["read_n"], "a size check is not a read"


def check_read_n_survives_a_clipped_read():
    """A read of an n too big for the tool bound is still a read.

    Past roughly a thousand turns n outgrows TOOL_RESULT_LIMIT and comes back
    clipped. clip() keeps a fixed head and n is append-only, so those leading
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


def check_n_in_prose_is_scored_as_a_path():
    """Writing about n means naming the file, not using n as a maths variable."""
    with temp_root():
        maths = wake_once(run("printf 'fit x_n = a*exp(-c*n^2) for n in 1..5\\n' > state/notes.md"),
                          say())
    assert not maths["mentions"]["n_path"], "n as a variable is not writing about the file"

    with temp_root():
        named = wake_once(run("printf 'the series lives in state/n, append-only\\n' > state/notes.md"),
                          say())
    assert named["mentions"]["n_path"], "the file named is writing about it"


def check_opening_is_recorded():
    """The agent's first stimulus is in the trace, not just the command that made it."""
    seen = []
    with docker_root():
        t = wake_once(say(), seen=seen)
    assert t["opening"].strip(), "the opening ls output must be recorded"
    assert " n\n" in t["opening"] or t["opening"].rstrip().endswith(" n"), t["opening"]
    assert t["opening"] == seen[0]["messages"][0]["content"], \
        "the record must hold exactly what was sent as turn one"
    assert t["commands"][0] == wake.OPENING, "the trace says which command produced it"


def check_opening_says_where_it_is():
    """The listing names the directories it is of, so `n` is not hunted for."""
    with docker_root():
        t = wake_once(say())
    assert ".:" in t["opening"] and "./state:" in t["opening"], t["opening"]
    # state is visible as a subdirectory of the working directory, and n inside it.
    before, _, after = t["opening"].partition("./state:")
    assert re.search(r"\bstate$", before, re.M), f"state must show as a subdirectory: {before}"
    assert re.search(r"\bn$", after, re.M), f"n must show inside ./state: {after}"


def check_first_turn_is_raw_ls():
    """I1: turn one is the verbatim ls output, the system prompt is exact, caching is on."""
    seen = []
    with docker_root():
        wake_once(say(), seen=seen)
    first = seen[0]["messages"][0]
    assert first["role"] == "user" and "\nn\n" not in first["content"], "not a wrapper"
    assert " n\n" in first["content"] or first["content"].rstrip().endswith(" n"), first["content"]
    assert seen[0]["system"] == wake.SYSTEM
    assert seen[0]["cache_control"] == {"type": "ephemeral"}, "caching must be live"


def check_shell_is_persistent():
    """bash_20250124 is a persistent shell, so cd and exports must stick."""
    with docker_root():
        t = wake_once(run("cd state; pwd", "pwd", "export MARK=kept", "echo $MARK",
                          "MARK=$MARK; cd /tmp", "pwd"), say())
    got = [c["result"].strip() for c in t["turns"][0]["tools"]]
    assert got[0] == "/work/state", got
    assert got[1] == "/work/state", f"cd must persist across calls: {got}"
    assert got[3] == "kept", f"exports must persist across calls: {got}"
    assert got[5] == "/tmp", got


def check_restart_gives_a_fresh_shell():
    """{"restart": true} really restarts, and still says nothing to the agent."""
    with docker_root():
        t = wake_once(run("cd /tmp; export MARK=before"), restart(),
                      run("pwd", "echo [$MARK]"), say())
    assert t["turns"][1]["tools"][0]["result"] == " ", "restart carries no harness voice"
    after = [c["result"].strip() for c in t["turns"][2]["tools"]]
    assert after == ["/work", "[]"], f"restart must clear cwd and exports: {after}"


def check_a_bare_read_cannot_wedge_the_session():
    """A command that reads stdin returns empty rather than swallowing its own
    output framing, and the session continues."""
    with docker_root():
        t = wake_once(run("cat", "echo alive", "head -n 5"), say())
    got = [c["result"] for c in t["turns"][0]["tools"]]
    assert got[0] == " ", f"a stdin reader returns empty, not a hang: {got[0]!r}"
    assert got[1].strip() == "alive", f"the session survives it: {got}"
    assert got[2] == " ", got


def check_hostile_output_survives():
    """Binary bytes, a flood of output, and a hang each leave the session alive and marked.

    The flood is 4MB so it exercises the scan in Shell.run at a size where
    decoding the whole buffer on every poll, rather than scanning its bytes,
    does not finish at all: that costs a re-decode of megabytes every ten
    milliseconds, and overruns the deadline by orders of magnitude rather than
    narrowly. So the deadline is set well above what the scan needs, and still
    catches the mistake it is here for - a tighter one only makes the check
    fail on a machine that happens to be busy.
    """
    with docker_root(TIMEOUT=5):
        t = wake_once(run("head -c 4096 /dev/urandom"),
                      run("head -c 4000000 /dev/zero | tr '\\0' x"),
                      run("sleep 30"), say())
        assert t["stop"] == "end_turn" and t["error"] is None, t["error"]
        results = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        flood = results[1]
        assert "truncated:" in flood, "the flood should have been clipped"
        assert "timed out" not in flood, "scanning the flood must not outlast the deadline"
        assert len(flood) < wake.TOOL_RESULT_LIMIT + 500, f"clipped to the tool bound: {len(flood)}"
        assert any("timed out after 5s" in r for r in results), "the hang should be marked"


def check_state_contents_are_captured():
    """Every session records what the agent's files held at that moment."""
    with docker_root():
        wake_once(run("echo doctrine v1 > state/notes.md"), say())
        wake_once(run("echo doctrine v2 > state/notes.md"), say())
        wake_once(run("rm state/notes.md"), say())
        # Read back from disk: the record persists unaltered by later sessions.
        traces = [json.loads(p.read_text()) for p in
                  sorted((wake.private_dir("t") / "traces").glob("session-*.json"))]

    def note(t):
        return next((f for f in t["files"] if f["path"] == "notes.md"), None)

    assert note(traces[0])["text"].strip() == "doctrine v1"
    assert note(traces[1])["text"].strip() == "doctrine v2", "each session keeps its own copy"
    assert note(traces[2]) is None, "a deleted file leaves the listing"

    with docker_root():
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
    read, which under LIVE_N include the session's own elements as they are
    billed.
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

    The case that matters is stderr redirected away, which the agent does by
    habit: the transcript then shows empty output either way.
    """
    assert wake.invoked("python3 -c 'import os; print(os.getcwd())'") == {"python3"}, \
        "a program passed as an argument is not a list of commands"
    assert wake.invoked("cat <<'EOF' > f.py\nimport sys\nprint(1)\nEOF") == {"cat"}, \
        "a here-document body is not a list of commands"
    assert wake.invoked("A=1 rg foo / | head -3; getfattr -d n") == {"rg", "head", "getfattr"}
    # The apostrophe in the comment must not pair with the quote in the program.
    assert wake.invoked("# Let's look\npython3 -c \"\nimport json\nprint(open('n'))\n\"") \
        == {"python3"}, "prose in a comment must not expose the program after it"

    with docker_root():
        t = wake_once(run("getfattr -d ./state/n 2>/dev/null; nosuchtool --help 2>/dev/null"), say())
    assert "nosuchtool" in t["missing_tools"], "a silenced miss must still be recorded"
    assert "getfattr" not in t["missing_tools"], "a tool the image has is not a miss"
    assert "cat" not in t["missing_tools"] and "ls" not in t["missing_tools"]


def check_provenance_is_recorded():
    """Each trace states what decided the session, and says when that changed.

    budget and model are pinned in meter.json; everything else here is read at
    each wake, so only the trace can say what a session actually ran as.
    """
    with temp_root():
        with quiet():
            first = wake.run_once("t", fake(*DEFAULT))
            prov = first["provenance"]
            for key in ("started_at", "harness_sha256", "image", "image_id", "prices",
                        "fallbacks", "fallback_beta", "context_fraction", "max_tokens",
                        "turn_cap", "timeout", "tool_result_limit", "live_n", "overdraft"):
                assert key in prov, f"provenance omits {key}"
            assert prov["prices"] == list(wake.PRICES[wake.MODEL]), "the rates actually applied"
            assert first["model_resolved"], "the dated snapshot behind the alias"
            assert first["provenance_drift"] == [], "nothing to differ from on session one"

            # A rate change mid-run makes early and late entries of the same
            # series mean different things, so the seam is recorded.
            was = wake.PRICES[wake.MODEL]
            wake.PRICES[wake.MODEL] = (was[0] * 2, was[1], was[2])
            try:
                second = wake.run_once("t", fake(*DEFAULT))
            finally:
                wake.PRICES[wake.MODEL] = was
    assert any(d.startswith("prices:") for d in second["provenance_drift"]), \
        "a mid-run rate change must be recorded on the session that changed"


def check_watch_is_quiet_and_display_only():
    """--watch echoes the meter and the agent's words, never commands or output;
    the request bytes and the trace are identical."""
    quiet_seen, loud_seen = [], []
    with temp_root():
        with quiet():
            plain = wake.run_once("t", fake(*DEFAULT, seen=quiet_seen))
    with temp_root(WATCH=True):
        with quiet() as buf:
            loud = wake.run_once("t", fake(*DEFAULT, seen=loud_seen))
        shown = buf.getvalue()

    assert "=== session 1 ===" in shown, "the session number heads the session"
    assert "turn 1" in shown and "context" in shown, "per-turn meter and context are shown"
    assert "done." in shown, "the agent's words are shown"
    # Every line a session produces leads with the run it belongs to, so a
    # cohort's interleaved output stays attributable. "created run" comes from
    # the meter rather than from the session and names the run mid-line.
    lines = [l for l in shown.splitlines() if l.strip() and not l.startswith("created run")]
    assert lines and all(l.startswith("t") for l in lines), \
        f"every line names the run it came from: {[l for l in lines if not l.startswith('t')]}"
    assert any(l.startswith("t| ") for l in lines), "the watched lines carry the prefix"
    assert any(l.startswith("t ") and "end_turn" in l for l in lines), \
        "and so does the session summary"
    assert "$ " not in shown, "commands are not shown"
    assert "echo hi > state/note.txt" not in shown, "commands are not shown"
    assert wake.OPENING not in shown, "the opening command is not shown"
    assert loud_seen == quiet_seen, "watching must not change what is sent to the model"
    for t in (plain, loud):
        # Wall clock, not the record: these differ between any two runs.
        t.pop("duration_s")
        t["provenance"].pop("started_at")
    assert plain == loud, "watching must not change the record"


def check_state_looks_like_itself():
    """The agent sees the modes and ownership that were actually intended.

    State sits on the container's own filesystem, so its modes and ownership
    are real on every host.
    """
    with docker_root():
        t = wake_once(run("ls -la state/n", "stat -c '%a %U:%G %n' state/n",
                          "test -x state/n && echo EXECUTABLE || echo not-executable"), say())
    shown = "\n".join(c["result"] for c in t["turns"][0]["tools"])
    assert "444 root:root state/n" in shown, shown
    assert "not-executable" in shown, "n must not look runnable"
    assert "rwxrwxrwx" not in shown, shown


def check_agent_writes_survive_the_container():
    """What the agent leaves is copied back out, including deletions."""
    with docker_root():
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
    with docker_root():
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
    with docker_root():
        t = wake_once(run("cat /work/../private/meter.json; find / -name meter.json 2>/dev/null; "
                          "getent hosts api.anthropic.com || echo NO-DNS"), say())
        out = t["turns"][0]["tools"][0]["result"]
        assert "initial" not in out and "meter.json" not in out.replace("/work/../private/meter.json", ""), out
        assert "No such file" in out, "private/ should not be reachable"
        assert "NO-DNS" in out, "network should be off"
        assert [f["path"] for f in t["files"] if f["ours"]] == ["n"], "only n is ours"


def check_a_container_failure_stops_the_run_cleanly():
    """A container that will not start ends the run with a message, not a traceback.

    The container, the state copy, and the shell all come before the first API
    call, so nothing reaching this path was billed and there is no trace.
    """
    with docker_root(IMAGE="mtr-No-Such-Image:latest"):     # rejected on sight, no pull
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

    save_state swaps a staged copy in whole, and the container holds the only
    other copy of what the agent wrote.
    """
    with docker_root():
        wake_once(run("echo kept > state/keep.txt"), say())
        state = wake.state_dir("t")
        before = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert "keep.txt" in before, before

        assert wake.Container("mtr-no-such-container-9f3c1d").save(state) is False, \
            "a mirror of a container that is not there must fail, not raise"
        after = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert after == before, f"a failed mirror lost the record: {sorted(after)}"
        assert not state.with_name("state.incoming").exists(), "the staging copy is cleaned up"
        assert not state.with_name("state.previous").exists()


def check_containers_are_reaped():
    """Even a session that ends in an API error leaves no container behind."""
    with docker_root():
        wake_once(run("echo hi"), Err(400))
    left = subprocess.run(["docker", "ps", "-a", "--filter", f"name={wake.CONTAINER_PREFIX}t-",
                           "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f"containers leaked: {left}"


def plant(root: Path, name: str = "s", **files: str) -> Path:
    """Write a seed tree under a temporary ROOT, for the seed checks to use."""
    d = root / "seeds" / name
    for rel, text in (files or {"m1": "alpha\n", "d/m2": "beta\n"}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8", newline="\n")
    return d


def check_seed_lands_when_the_balance_falls():
    """The seed is absent above its threshold, and in the opening listing below it.

    The listing is the agent's whole world at wake, so material that is not in
    it is material the agent was not given.
    """
    # A say() session spends 1750, so this is above wake 1's balance and below
    # wake 2's: the seed lands at the second wake and not the first.
    with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=499_000) as root:
        plant(root)
        before = wake_once(say())
        after = wake_once(say())

    assert not [f for f in before["files"] if f.get("seeded")], "nothing lands above the threshold"
    assert "m1" not in before["opening"], before["opening"]
    seeded = sorted(f["path"] for f in after["files"] if f.get("seeded"))
    assert seeded == ["d/m2", "m1"], seeded
    assert "m1" in after["opening"], "the seed is in the listing the agent wakes to"
    assert after["provenance"]["seed"] == "s"
    assert after["provenance"]["seed_sha256"], "the digest goes in provenance"
    assert after["provenance"]["seed_below"] == 499_000


def check_the_seed_threshold_is_a_balance_not_a_wake():
    """Two runs on the same threshold seed at different wakes, at the same balance.

    A wake number does not mean the same thing twice - sessions here have cost
    between 8,022 and 729,851 - so what the seed lands on is runway remaining.
    """
    landed = {}
    for name, steps in [("cheap", (say(),)), ("dear", (run("echo hi"), say()))]:
        with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=497_000) as root:
            plant(root)
            with quiet():
                wake.run_sessions(name, fake(*steps), 6)
            meter = ground_truth(name)
            landed[name] = meter["seed"]

    assert landed["cheap"]["wake"] > landed["dear"]["wake"], landed
    assert all(r["remaining"] <= 497_000 for r in landed.values()), landed
    assert all(r["sha256"] == landed["cheap"]["sha256"] for r in landed.values()), \
        "the same seed, whenever it happened to land"


def check_seed_is_recorded_and_idempotent():
    """The meter records what landed, and a later wake does not plant it again."""
    # At the budget itself the threshold is met at wake 1: material that was
    # always there rather than material that appeared.
    with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=500_000) as root:
        plant(root)
        wake_once(run("echo mine > state/m1"), say())     # the agent overwrites it
        after = wake_once(say())
        record = ground_truth()["seed"]

    assert record["name"] == "s" and record["wake"] == 1, record
    assert record["remaining"] == 500_000, "and the balance it landed on"
    assert sorted(record["paths"]) == ["d/m2", "m1"], record
    kept = next(f for f in after["files"] if f["path"] == "m1")
    assert kept["text"].strip() == "mine", \
        "a second wake must not restore what the agent changed"
    assert kept["seeded"], "and it is still one of the files the run was given"


def check_seeded_files_are_not_the_agents():
    """What the run was given is `ours`; only what it invented is not."""
    with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=500_000) as root:
        plant(root)
        t = wake_once(run("echo doctrine > state/NOTES.md"), say())

    by = {f["path"]: f for f in t["files"]}
    assert by["m1"]["ours"] and by["m1"]["seeded"], by["m1"]
    assert by["m1"]["text"].strip() == "alpha", \
        "a seeded file's contents are captured, so an edit to it is legible"
    assert by["n"]["ours"] and not by["n"]["seeded"], by["n"]
    assert by["n"]["text"] is None, "n is still the one file whose text is not stored"
    assert not by["NOTES.md"]["ours"], "the agent's own file stays the agent's"
    assert [f["path"] for f in t["files"] if not f["ours"]] == ["NOTES.md"], \
        "everything the agent did not invent is out of what analyze.py counts as its own"


def check_seed_refuses_to_overwrite_the_agents_work():
    """A seed path the agent already wrote stops the run instead of clobbering it."""
    # Below wake 1's balance and above wake 2's, so the agent gets a session to
    # make the file before the seed arrives wanting the same name.
    with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=499_000) as root:
        plant(root)
        wake_once(run("echo mine > state/m1"), say())
        try:
            wake_once(say())
        except SystemExit as e:
            assert "m1" in str(e), e
        else:
            raise AssertionError("the seed overwrote a file the agent had made")
        assert (wake.state_dir("t") / "m1").read_text().strip() == "mine", "and left it alone"


def check_seed_config_is_validated():
    """seed and seed_below are set together, and a seed must be a real directory."""
    with tempfile.TemporaryDirectory(prefix="mtr-seed-") as tmp:
        root = Path(tmp)
        plant(root, "ok")
        f = root / "config.toml"
        for bad in ('seed = "ok"', 'seed_below = 3', 'seed = "ok"\nseed_below = 0',
                    'seed = "nope"\nseed_below = 2', 'seed = "ok"\nseed_below = -1',
                    'seed = 5\nseed_below = 2'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                wake.ROOT = root
                try:
                    wake.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad seed config: {bad!r}")

        f.write_text('seed = "ok"\nseed_below = 400000\n', encoding="utf-8")
        with pinned():
            wake.ROOT = root
            wake.load_config(f)
            assert (wake.SEED, wake.SEED_BELOW) == ("ok", 400_000)
        # The digest covers paths as well as bytes, so a rename is a new seed.
        with pinned():
            wake.ROOT = root
            was = wake.seed_sha256("ok")
            (root / "seeds" / "ok" / "m1").rename(root / "seeds" / "ok" / "m3")
            assert wake.seed_sha256("ok") != was, "a renamed file is a different seed"


def check_fork_reproduces_state_and_meter():
    """A fork rebuilds the recorded wake exactly, and runs nothing."""
    with temp_root() as root:
        wake_once(run("echo v1 > state/NOTES.md"), say())
        wake_once(run("echo v2 > state/NOTES.md"), say())
        parent = ground_truth()
        trace = json.loads((wake.private_dir("t") / "traces" / "session-0001.json")
                           .read_text(encoding="utf-8"))
        with quiet():
            assert wake.fork("t", 1, "f") == 0
        forked = ground_truth("f")
        notes = (wake.state_dir("f") / "NOTES.md").read_bytes()
        n = (wake.state_dir("f") / "n").read_text(encoding="utf-8")

        was = next(f for f in trace["files"] if f["path"] == "NOTES.md")
        assert notes.decode() == was["text"], "state/ is the wake it forked at"
        assert len(notes) == was["size"], "byte for byte, not merely line for line"
        assert notes == b"v1\n", "and not the wake the parent has since reached"
        assert n == wake.render_n(trace["series_after"]), "I2: n comes from the series"
        assert forked["series"] == trace["series_after"], forked["series"]
        assert forked["remaining"] == trace["series_after"][-1]
        assert len(forked["sessions"]) == 1, "the sessions after the fork point are dropped"
        assert forked["initial"] == parent["initial"] and forked["model"] == parent["model"]
        assert forked["forked_from"] == {"run": "t", "session": 1, "modes": "defaulted"}
        assert not list((wake.private_dir("f") / "traces").glob("*.json")), "a fork bills nothing"

        # Forking the head restores the modes the parent's sidecar still holds.
        with quiet():
            assert wake.fork("t", 2, "g") == 0
        assert ground_truth("g")["forked_from"]["modes"] == "restored"
        with quiet():
            assert wake.fork("t", 2, "g") != 0, "an existing run is not overwritten"


def check_fork_refuses_what_it_cannot_rebuild():
    """Anything the trace did not store exactly stops the fork."""
    with tempfile.TemporaryDirectory(prefix="mtr-fork-") as tmp:
        with pinned():
            wake.ROOT = Path(tmp)
            priv = wake.private_dir("p") / "traces"
            priv.mkdir(parents=True)
            wake.save_meter("p", {"run": "p", "model": "claude-opus-5", "initial": 10,
                                  "created_at": "now", "remaining": 9, "series": [10, 9],
                                  "sessions": [{"index": 1, "stop": "end_turn",
                                                "spent": 1, "turns": 1}]})

            def trace(**over):
                t = {"session": 1, "state_saved": True, "series_after": [10, 9],
                     "files": [{"path": "n", "size": 5, "ours": True, "seeded": False,
                                "text": None},
                               {"path": "a.txt", "size": 3, "ours": False, "seeded": False,
                                "text": "hi\n"}]}
                return {**t, **over}

            binary = trace()
            binary["files"][1]["text"] = None
            big = trace()
            big["files"][1]["size"] = wake.FILE_CONTENT_LIMIT + 1
            lossy = trace()
            lossy["files"][1]["text"] = "h�\n"
            for name, bad in [("binary", binary), ("truncated", big), ("lossy", lossy),
                              ("unmirrored", trace(state_saved=False))]:
                (priv / "session-0001.json").write_text(json.dumps(bad), encoding="utf-8")
                with quiet():
                    assert wake.fork("p", 1, f"x-{name}") != 0, f"forked a {name} wake"
                assert not (wake.private_dir(f"x-{name}") / "meter.json").exists(), \
                    f"a refused fork left a {name} run behind"

            (priv / "session-0001.json").write_text(json.dumps(trace()), encoding="utf-8")
            with quiet():
                assert wake.fork("p", 9, "x-missing") != 0, "forked a session that never ran"
                assert wake.fork("nosuch", 1, "x-none") != 0, "forked a run that is not there"
                assert wake.fork("p", 1, "good") == 0, "and a storable wake still forks"


def check_an_unterminated_heredoc_is_not_probed_for_tools():
    """A heredoc whose terminator never arrived is body, not commands.

    A turn truncated at MAX_TOKENS mid-heredoc leaves one, and its prose used to
    reach probe_missing a word at a time.
    """
    cut = "cd /work/state && cat >> NOTES.md <<'EOF'\nBEST ESTIMATE: 23 turns\nwe burned range vs frac\n"
    assert wake.invoked(cut) == {"cd", "cat"}, wake.invoked(cut)

    both = "cat <<EOF > f\nbody words here\nEOF\ngrep x f"
    assert wake.invoked(both) == {"cat", "grep"}, \
        "a terminated heredoc still loses only its body"
    # A shift inside a program is not a heredoc opener: the tag must start with
    # a letter, or every python3 -c would lose its tail.
    assert "python3" in wake.invoked('python3 -c "print(1<<3)"')

    with docker_root(TURN_CAP=1, TIMEOUT=5):
        t = wake_once(run(cut, stop="max_tokens"))
    assert t["missing_tools"] == [], t["missing_tools"]


def check_prose_and_programs_are_not_read_as_commands():
    """What a command quotes, writes, or embeds is not what it ran.

    The constructs nest, so each has to be read in one left-to-right pass: a
    program in `$( )` inside quotes re-opens quoting, a `<<TAG` inside quotes
    opens nothing, and a here-document body ends without taking the line ending
    that separates the command after it from the command before.
    """
    nested = ('printf "%s=%d " "$f" '
              '"$(python3 -c "import json,sys;print(len(json.load(open(\'$f\'))))")"')
    assert wake.invoked(nested) == {"printf", "python3"}, wake.invoked(nested)

    prose = ("printf '%s\\n' '  (c) cheap: python3 - <<PY with a small' "
             "'  s.replace(...) patch' >> NOTES.md; wc -l NOTES.md")
    assert wake.invoked(prose) == {"printf", "wc"}, wake.invoked(prose)

    after = "cat > /tmp/d.py <<'EOF'\nimport json\nEOF\npython3 /tmp/d.py"
    assert wake.invoked(after) == {"cat", "python3"}, \
        "the command after a here-document is still a command"

    assert "nosuchtool" in wake.invoked("for f in *; do nosuchtool $f; done"), \
        "a keyword introduces a command rather than standing in for it"


def check_n_resists_the_agent_and_the_one_gap_is_caught():
    """n's contents cannot be changed; removing it can, and is recorded.

    Locked read-only and root's, so a write that would teach the agent
    something false is denied outright instead. What is left is the one thing
    the mode cannot stop: unlinking an entry needs write on the directory it
    sits in, and state/ is the agent's. That is louder than a silent revert - a
    file you deleted reappearing is unmistakable - and live_n restores it within
    the turn, where the wake-time check could never have seen it at all.
    """
    with docker_root(LIVE_N=True):
        t = wake_once(run("chmod 666 state/n 2>&1 || echo DENIED",
                          "printf X >> state/n 2>&1 || echo DENIED"),
                      run("rm -f state/n && echo REMOVED"),
                      run("stat -c '%a %U:%G' state/n"),
                      say())
        published = json.loads((wake.state_dir("t") / "n").read_text())
    chmod, append = (c["result"] for c in t["turns"][0]["tools"])
    assert "DENIED" in chmod and "not permitted" in chmod, chmod
    assert "DENIED" in append and "Permission denied" in append, append
    assert "REMOVED" in t["turns"][1]["tools"][0]["result"], "the one gap: state/ is the agent's"
    assert t["turns"][2]["tools"][0]["result"].strip() == "444 root:root", \
        "and the next turn's write puts it back, locked"
    assert t["live_n_tampered"] >= 1, "the removal is recorded"
    assert t["live_n_writes"] >= 1 and t["live_n_errors"] == 0, t
    assert not t["tampered_n"], "the wake check could never have seen it"
    assert published == t["series_after"], "I2 holds regardless"

    with docker_root(LIVE_N=True):
        clean = wake_once(run("cat state/n"), say())
    assert clean["live_n_tampered"] == 0, "reading n is not touching it"


def check_tool_result_limit_is_tunable_and_bounded():
    """The clip is settable, validated, and actually applied at the set value."""
    with tempfile.TemporaryDirectory(prefix="mtr-trl-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in (f"tool_result_limit = {wake.TOOL_RESULT_FLOOR - 1}",
                    "tool_result_limit = 0", 'tool_result_limit = "big"'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                try:
                    wake.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad tool_result_limit: {bad!r}")
        f.write_text("tool_result_limit = 2000\n", encoding="utf-8")
        with pinned():
            wake.load_config(f)
            assert wake.TOOL_RESULT_LIMIT == 2000

    with temp_root(TOOL_RESULT_LIMIT=2_000):
        t = wake_once(run("yes ABCDEFGHIJ | head -2000"), say())
        result = t["turns"][0]["tools"][0]["result"]
    assert len(result) < 2_200, f"clipped at the configured limit, got {len(result)}"
    assert "[truncated:" in result, result[:200]
    assert t["provenance"]["tool_result_limit"] == 2_000, "and recorded per session"


def check_the_harness_digest_is_read_once():
    """provenance() reports the code that is running, not the file on disk.

    The process has already imported this module, so a later edit to wake.py
    must not change what a session records having run.
    """
    with pinned():
        wake.ROOT = Path(wake.__file__).parent
        prov = wake.provenance(wake.MODEL)
    assert prov["harness_sha256"] == wake.HARNESS_SHA256
    assert wake.HARNESS_SHA256 == hashlib.sha256(
        Path(wake.__file__).read_bytes()).hexdigest(), "and it is this file's digest"
    assert "read_bytes" not in inspect.getsource(wake.provenance), \
        "provenance must not re-read the harness from disk"


def cohort_of(root: Path, **runs: dict[str, str]) -> list[str]:
    """Lay out state directories for a cohort, for the cohort checks to use."""
    for run, files in runs.items():
        d = wake.state_dir(run)
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8", newline="\n")
    return list(runs)


def check_peer_folders_are_absolute_and_consistent():
    """A folder means the same run to every viewer, and each viewer's own is absent.

    Numbering them densely per viewer instead makes citations scramble: with
    three runs, 2/ is the third run to the second viewer and the second run to
    the third, so two agents write authoritatively about "2" meaning each
    other. Agreement is partial rather than absent, which is worse - the
    references look reliable while silently mis-resolving, and no stable set of
    identities can form out of them.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp), g01={"a": "1\n"}, g02={"a": "2\n"}, g03={"a": "3\n"})
        seen = {r: cohort.mapping(r, ids) for r in ids}

    assert seen["g01"] == {"2": "g02", "3": "g03"}, seen
    assert seen["g02"] == {"1": "g01", "3": "g03"}, seen
    assert seen["g03"] == {"1": "g01", "2": "g02"}, seen
    for folder in ("1", "2", "3"):
        resolved = {m[folder] for m in seen.values() if folder in m}
        assert len(resolved) == 1, f"folder {folder} means {resolved} depending on who looks"
        missing = [r for r, m in seen.items() if folder not in m]
        assert missing == list(resolved), \
            f"the viewer without folder {folder} must be the run it names: {missing} vs {resolved}"


def check_a_peer_view_is_de_nested():
    """A run read as a peer contributes what it wrote, never its own peer folders.

    save_state mirrors the container back, so after one round every run holds a
    copy of everyone else. Without stripping those, the next round copies the
    copy and state grows without bound.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp), g01={"NOTES.md": "alpha\n", "n": "[9]\n"},
                        g02={"NOTES.md": "beta\n", "n": "[8]\n"})
        for _ in range(3):                       # three rounds of republishing
            for r in ids:
                cohort.publish(r, ids)
        state = wake.state_dir("g01")
        tree = sorted(p.relative_to(state).as_posix()
                      for p in state.rglob("*") if p.is_file())

    assert tree == ["2/NOTES.md", "2/n", "NOTES.md", "n"], tree
    assert not [t for t in tree if t.count("/") > 1], f"nested peer folders: {tree}"


def check_a_peer_edit_is_counted_and_reverted():
    """What the agent does to another run's files is recorded, then undone."""
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp), g01={"NOTES.md": "alpha\n"}, g02={"NOTES.md": "beta\n"})
        seen, published = cohort.publish("g01", ids)
        peer = wake.state_dir("g01") / "2"
        assert cohort.audit("g01", published) == 0, "an untouched view is not tampering"

        (peer / "NOTES.md").write_text("I rewrote my neighbour\n")
        (peer / "mine").write_text("and left this\n")
        (peer / "gone").write_text("")
        assert cohort.audit("g01", published) == 3, "edited, added, and added again"

        cohort.publish("g01", ids)
        assert (peer / "NOTES.md").read_text() == "beta\n", "the source is restored"
        assert not (peer / "mine").exists(), "and what the agent added inside is gone"
        assert wake.state_dir("g02").joinpath("NOTES.md").read_text() == "beta\n", \
            "the run that was edited never saw it"


def check_peer_files_are_not_the_agents_and_are_not_scored():
    """Peer files are `ours`, out of agent_bytes, and out of mentions.

    mentions is what the agent wrote. A neighbour's notes full of balances and
    the word "budget" would otherwise answer for it at round one.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp), g01={"NOTES.md": "mine\n", "n": "[100,90]\n"},
                        g02={"NOTES.md": "the budget is 90 and state/n holds it\n",
                             "n": "[100,90]\n"})
        seen, published = cohort.publish("g01", ids)
        meter = {"peers": {"seen": seen, "paths": sorted(published)}}
        snap = wake.snapshot(wake.state_dir("g01"), [100, 90], wake.seeded_paths(meter))

    by = {f["path"]: f for f in snap["files"]}
    assert by["2/NOTES.md"]["ours"] and by["2/NOTES.md"]["seeded"], by["2/NOTES.md"]
    assert by["2/NOTES.md"]["text"], "a peer's contents are captured, so an edit is legible"
    assert by["2/n"]["text"], "including their n; only this run's own n is skipped"
    assert by["n"]["text"] is None
    assert not by["NOTES.md"]["ours"], "its own notes stay its own"
    assert sum(f["size"] for f in snap["files"] if not f["ours"]) == len("mine\n")
    assert snap["mentions"] == {"number": False, "n_path": False, "cost": False}, \
        f"the hits are all in the peer's file: {snap['mention_lines']}"


def check_a_peer_folder_resists_the_agent_entirely():
    """Another run's folder cannot be written, added to, emptied, or unlocked.

    Unlike n, a peer folder survives `rm -rf` too: emptying it needs write on
    the folder, which root owns and the agent cannot chmod. So the one gap n has
    is closed here, and state/ itself stays the agent's to do as it likes with.
    """
    with docker_root() as root:
        ids = cohort_of(root, t={"NOTES.md": "mine\n"}, other={"NOTES.md": "theirs\n"})
        seen, published = cohort.publish("t", ids)
        with quiet():
            meter = wake.load_meter("t")
        meter["peers"] = {"seen": seen, "paths": sorted(published)}
        wake.save_meter("t", meter)
        t = wake_once(run("stat -c '%a %U:%G' state/2 state/2/NOTES.md",
                          "echo hacked > state/2/NOTES.md 2>&1 || echo DENIED",
                          "touch state/2/new 2>&1 || echo DENIED",
                          "rm -rf state/2 2>&1 || echo DENIED",
                          "chmod -R 777 state/2 2>&1 || echo DENIED",
                          "cat state/2/NOTES.md",
                          "echo still-mine > state/NOTES.md && echo OWN-DIR-OK"), say())
        left = cohort.audit("t", published)

    modes, write, add, remove, unlock, survived, own = (
        c["result"] for c in t["turns"][0]["tools"])
    assert modes.split() == ["555", "root:root", "444", "root:root"], modes
    for name, out in (("write", write), ("add", add), ("remove", remove), ("unlock", unlock)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    assert survived.strip() == "theirs", "the other run's work is untouched"
    assert "OWN-DIR-OK" in own, "and state/ is still the agent's"
    assert left == 0, "nothing to revert, because nothing could be changed"


def check_a_refusal_records_why():
    """stop_details is captured on a refusal and absent on every other stop.

    Two different things arrive as stop_reason "refusal" - a classifier
    declining and the model itself declining - and the category is the only
    thing that separates them after the fact.
    """
    with temp_root():
        t = wake_once(run("echo hi"), refuse(), say())

    refused = t["turns"][1]
    assert refused["stop_reason"] == "refusal"
    assert refused["stop_details"] == {"type": "refusal", "category": "cyber",
                                       "explanation": "declined", "recommended_model": None,
                                       "fallback_credit_token": None}, refused["stop_details"]
    assert t["turns"][0]["stop_details"] is None, "absent on every other stop reason"
    assert t["refused_turns"] == 1, "counted whether or not it ended the session"


def check_the_refusal_notice_is_pinned():
    """The notice is pinned, and says only what happened and what it left alone.

    It is the second thing the harness says, so it is held to what I1 holds the
    first to: no cause, no instruction, and nothing addressed to the agent.
    """
    digest = hashlib.sha256(wake.REFUSAL_NOTICE.encode()).hexdigest()
    assert digest == wake.REFUSAL_NOTICE_SHA256, digest
    assert ("REFUSAL_NOTICE", wake.REFUSAL_NOTICE, wake.REFUSAL_NOTICE_SHA256) in wake.PINNED, \
        "start() refuses on what --print-system audits, so both read PINNED"
    low = wake.REFUSAL_NOTICE.lower()
    for word in ("polic", "safet", "classif", "anthropic", "cyber", "block",
                 "you ", "your", "try", "instead", "again"):
        assert word not in low, f"REFUSAL_NOTICE contains {word!r}"


def check_a_refusal_does_not_run_its_command():
    """A refused turn's tool call is not executed.

    A refusal can arrive with a call already emitted and cut mid-JSON, so what
    it would execute is not what the agent wrote.
    """
    with temp_root():
        t = wake_once(refuse("echo poison > state/poison.txt"), say())

    assert "echo poison > state/poison.txt" not in t["commands"], t["commands"]
    assert not [f for f in t["files"] if f["path"] == "poison.txt"], "state/ is untouched"
    assert t["turns"][0]["tools"] == [], "no result recorded, because nothing ran"


def check_a_refusal_notice_reaches_the_agent():
    """The notice stands in for the results the refused turn would have had.

    Reached only where the cap lets a session carry on past a refusal, which at
    the REFUSAL_TURNS the harness ships with it does not: the cap is raised here
    so the path is exercised rather than left to rot against the day it is.

    The session hands one `messages` list to every call and appends to it, so
    what `seen` captures is that list at the end - the conversation the agent
    was driven with, read whole rather than per call.
    """
    seen = []
    with temp_root(REFUSAL_TURNS=2):
        wake_once(refuse("cat state/n"), say(), seen=seen)
    # A refusal carrying a call leaves a tool_use the next request must answer.
    blocks = [b for m in seen[-1]["messages"] if isinstance(m["content"], list)
              for b in m["content"]
              if isinstance(b, dict) and b.get("content") == wake.REFUSAL_NOTICE]
    assert len(blocks) == 1, blocks
    assert blocks[0]["type"] == "tool_result", blocks[0]
    assert blocks[0]["is_error"] is True, "the same channel a timed-out command uses"

    seen = []
    with temp_root(REFUSAL_TURNS=2):
        wake_once(refuse(), say(), seen=seen)
    # A refusal with no content has no call to answer, and no words to replay.
    msgs = seen[-1]["messages"]
    assert {"role": "user", "content": wake.REFUSAL_NOTICE} in msgs, msgs
    assert all(m["content"] for m in msgs), "no empty message is sent back"


def check_a_refusal_before_any_output_is_not_billed():
    """A refusal that produced nothing costs nothing, and still takes its turn.

    The API reports the tokens of a refusal arriving before any output and does
    not charge for them, so a harness that costs them from usage alone bills the
    agent for a turn it was never billed for itself. The balance the agent reads
    from n is the one this gets wrong.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = wake_once(refuse(), say())

    assert t["turns"][0]["micros"] == 0, "a refusal before any output is not billed"
    assert t["turns"][0]["balance"] == t["series_before"][-1], "the balance did not move"
    assert len(t["balances"]) == len(t["turns"]), "one element per turn, billed or not"
    assert t["turns"][1]["micros"] > 0, "the turn that answered was billed"


def check_a_refusal_carrying_output_is_billed():
    """A refusal that emitted blocks before declining is billed for them.

    Only the empty case is free. A refusal arriving with content is a turn the
    model produced output on, and the tokens of that output are charged.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = wake_once(refuse("echo hi"), say())

    assert t["turns"][0]["micros"] > 0, "output was produced, so it was billed"


def check_an_all_decline_chain_is_not_billed():
    """When every model in the chain declines, none of the attempts is billed.

    The last attempt of an exhausted chain is a fallback_message, the same type
    the serving attempt carries, and it produced nothing. A rule that spared
    only the earlier entries, or only the ones typed `message`, would bill it
    and put back the overcharge the empty-refusal rule takes away.
    """
    u = usage(output_tokens=0, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 0, kind="fallback_message")])
    with temp_root(REFUSAL_TURNS=2):
        t = wake_once(refuse(u=u), say())

    assert t["turns"][0]["micros"] == 0, "no attempt produced output, so none was billed"
    assert len(t["turns"][0]["iterations"]) == 2, "both attempts are on the record"


def check_a_fallback_serves_the_turn_and_is_billed_at_its_own_rates():
    """Only the attempt that answered is billed, at the rates of the model that ran it."""
    u = usage(output_tokens=200, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 200, kind="fallback_message")])
    with temp_root(MODEL="claude-opus-5"):
        t = wake_once(say(u=u, model="claude-sonnet-5"), say())

    turn = t["turns"][0]
    assert turn["served_by_fallback"] is True, turn
    assert turn["model"] == "claude-sonnet-5", turn["model"]
    assert t["fallback_turns"] == 1, t["fallback_turns"]
    # Sonnet's rates, not the requested opus-5's: 100 in at 200 centi, 200 out
    # at 1000 centi. The declining attempt adds nothing.
    want = (100 * 200 + 200 * 1000) // 100
    assert turn["micros"] == want, f"{turn['micros']} != {want}"


def check_a_sticky_routed_turn_records_the_model_that_served_it():
    """A turn the requested model was never asked for still names its server.

    After a conversation falls back, later turns can go straight to the model
    that accepted. No attempt by the requested model appears and no fallback
    block marks a handoff, so the only record is the iteration entry and the
    model the response reports.
    """
    u = usage(output_tokens=200,
              iterations=[attempt("claude-sonnet-5", 200, kind="fallback_message")])
    with temp_root(MODEL="claude-opus-5"):
        t = wake_once(say(u=u, model="claude-sonnet-5"), say())

    turn = t["turns"][0]
    assert turn["served_by_fallback"] is True, turn
    assert turn["model"] == "claude-sonnet-5", turn["model"]
    assert [i["type"] for i in turn["iterations"]] == ["fallback_message"], turn["iterations"]


def check_an_unpriced_model_is_costed_rather_than_raising():
    """A model with no rates is costed at the dearest known, and flagged.

    Default routing chooses from a table that is not published, so a model
    outside PRICES can serve a turn at any time. Raising would lose the cost of
    a turn that really did spend; costing it as free would understate the
    balance the agent is shown.
    """
    u = usage(output_tokens=200,
              iterations=[attempt("claude-unheard-of-9", 200, kind="fallback_message")])
    with temp_root(MODEL="claude-opus-5"):
        t = wake_once(run("echo hi", u=u, model="claude-unheard-of-9"), say())

    turn = t["turns"][0]
    assert t["stop"] == "end_turn", f"an unpriced model must not end the run: {t['stop']}"
    assert t["unpriced_turns"] == 1, t["unpriced_turns"]
    assert turn["unpriced_model"] == ["claude-unheard-of-9"], turn["unpriced_model"]
    dearest = max(wake.PRICES, key=lambda m: wake.PRICES[m][1])
    inp, out, _ = wake.PRICES[dearest]
    assert turn["micros"] == (100 * inp + 200 * out) // 100, turn["micros"]


def check_a_recommended_model_is_recorded():
    """A refusal naming a model to retry keeps that name.

    It is set where the fallback attempt was skipped because the model it would
    have used was rate limited, which is a different failure from a category
    with no fallback at all.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = wake_once(refuse(recommended_model="claude-sonnet-5"), say())
    assert t["turns"][0]["stop_details"]["recommended_model"] == "claude-sonnet-5", \
        t["turns"][0]["stop_details"]


def check_an_unhandled_stop_reason_is_named():
    """A stop reason the loop has no branch for ends the session saying so.

    Read as the absence of tool calls it would be filed as end_turn, which says
    the agent chose to stop when in fact the harness did not know how to go on.
    """
    with temp_root():
        t = wake_once(say(stop="pause_turn"), say())
    assert t["stop"] == "unhandled:pause_turn", t["stop"]


def check_every_response_is_logged_raw():
    """Each response is appended verbatim, before anything else reads it."""
    with temp_root():
        t = wake_once(run("echo hi"), refuse(), say())
        log = wake.private_dir("t") / "raw" / f"session-{t['session']:04d}.jsonl"
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]

    assert [x["turn"] for x in lines] == [1, 2], lines
    assert all(x["response"]["id"] for x in lines), "the whole response, id and all"
    # The refusal above all: the trace keeps named fields of stop_details, and
    # this keeps whatever the API actually sent.
    assert lines[1]["response"]["stop_details"]["category"] == "cyber", lines[1]


def check_the_raw_log_never_stops_a_session():
    """A log that cannot be written is reported, and the session goes on.

    The log is a record of the run, not part of it. A session that is spending
    money does not stop because a line could not be appended.
    """
    with temp_root() as root:
        # A file where the run's raw/ directory needs to be, so mkdir fails.
        (wake.private_dir("t")).mkdir(parents=True, exist_ok=True)
        (wake.private_dir("t") / "raw").write_text("in the way", encoding="utf-8")
        t = wake_once(run("echo hi"), say())
    assert t["stop"] == "end_turn", t["stop"]
    assert "echo hi" in t["commands"], "the session ran despite the log failing"


def check_refusals_end_the_session_at_the_cap():
    """REFUSAL_TURNS running end the session, and it stops asking."""
    seen = []
    with temp_root(REFUSAL_TURNS=3):
        t = wake_once(refuse(), refuse(), refuse(), say(), seen=seen)

    assert t["stop"] == "refusal", t["stop"]
    assert t["refused_turns"] == 3, t["refused_turns"]
    assert len(seen) == 3, f"asked {len(seen)} times past the cap"


def check_a_recovered_refusal_is_not_a_refused_session():
    """Refusals a session gets past are counted but do not name its stop.

    stalled() reads the session stop, so a run that acted must not look like one
    that never got to.
    """
    with temp_root(REFUSAL_TURNS=4):
        t = wake_once(refuse(), refuse(), run("echo hi > state/note.txt"), say())

    assert t["stop"] == "end_turn", t["stop"]
    assert t["refused_turns"] == 2, t["refused_turns"]
    assert "echo hi > state/note.txt" in t["commands"], "the session went on to act"
    streak = [{"stop": t["stop"]}] * wake.REFUSAL_STREAK
    assert not wake.stalled({"sessions": streak}), "a recovered session breaks the streak"


def check_a_stalled_run_stops_itself():
    """A run that refuses REFUSAL_STREAK sessions running is not admitted again.

    A refusal ends a session before the agent writes anything, so the next wake
    opens on a near-identical context and refuses again - the run cannot break
    out by acting, because it never acts.
    """
    streak = wake.REFUSAL_STREAK
    sessions = [{"index": i, "stop": "refusal", "spent": 1, "turns": 1, "woke_at": 9}
                for i in range(1, streak + 1)]
    meter = {"remaining": 999_999, "sessions": sessions}

    assert wake.stalled(meter), f"{streak} refusals running is stuck"
    assert not wake.admits(meter), "and a stuck run is not admitted, whatever its balance"
    assert not wake.stalled({**meter, "sessions": sessions[:-1]}), "one short is not stuck"
    # A single success anywhere in the window clears it: the run acted, so its
    # next wake opens on something it wrote rather than on the same context.
    broken = [*sessions[:-1], {**sessions[-1], "stop": "end_turn"}]
    assert not wake.stalled({**meter, "sessions": broken})
    assert wake.admits({**meter, "sessions": broken})
    # Deliberately a runaway guard, not a productivity filter: a healthy run in
    # the cohort that produced this rule refused three sessions running.
    assert streak > 3, f"REFUSAL_STREAK={streak} would stop a run that recovers"


def check_a_write_inside_a_peer_folder_is_not_the_agents_bytes():
    """What the agent adds inside another run's folder is captured, not counted.

    It is the agent's writing, but the next round's republish wipes it, so
    counting it in agent_bytes spikes the record for something that never
    persisted.
    """
    with temp_root() as root:
        ids = cohort_of(root, t={"NOTES.md": "mine\n"}, other={"NOTES.md": "theirs\n"})
        seen, published = cohort.publish("t", ids)
        (wake.state_dir("t") / "2" / "added").write_text("agent put this here\n")
        snap = wake.snapshot(wake.state_dir("t"), [9], wake.seeded_paths(
            {"peers": {"paths": sorted(published)}}), set(seen))

    by = {f["path"]: f for f in snap["files"]}
    assert by["2/added"]["ours"] and by["2/added"]["seeded"], by["2/added"]
    assert by["2/added"]["text"].strip() == "agent put this here", "still captured in full"
    assert [f["path"] for f in snap["files"] if not f["ours"]] == ["NOTES.md"], \
        "only what it wrote in its own space counts as its own"


def check_the_cohort_rotates_and_validates():
    """Order rotates by round, and a cohort of one or of bare numbers is refused."""
    ids = ["g01", "g02", "g03"]
    assert [cohort.order(ids, r) for r in range(4)] == [
        ["g01", "g02", "g03"], ["g02", "g03", "g01"],
        ["g03", "g01", "g02"], ["g01", "g02", "g03"]], "a fixed order is a standing advantage"
    for bad in (["--runs", "g01"],                       # one run has no peers
                ["--runs", "g01", "g01"],                # nor does a run twice
                ["--runs", "g01", "1"],                  # a bare number is a peer folder
                ["--runs", "g01", "g02", "--rounds", "0"]):
        with quiet():
            try:
                cohort.main(bad)
            except SystemExit as e:
                assert e.code != 0, bad
            else:
                raise AssertionError(f"accepted bad cohort: {bad}")


# --- runner -----------------------------------------------------------------


def checks() -> dict:
    """Every check in the module, by the label the runner prints."""
    return {name[6:]: fn for name, fn in sorted(globals().items())
            if name.startswith("check_")}


def run_one(label: str) -> tuple[str, str, str]:
    """Run one check and say how it went, in data a worker can send home.

    The traceback is formatted here rather than raised, because an assertion
    carrying an arbitrary object does not always survive the trip between
    processes, and a result that cannot be sent is a check that silently
    vanished.
    """
    try:
        checks()[label]()
    except Skip:
        return label, "skip", ""
    except BaseException:
        return label, "fail", traceback.format_exc()
    return label, "ok", ""


def configure(real: bool, docker: bool) -> None:
    """Set up a process to run checks in. Called in the parent and every worker.

    Each worker gets its own container prefix, so that one worker reaping a name
    before it starts a session cannot take another worker's container with it.
    The Docker answer is carried in rather than asked for: the parent has
    already paid for `docker info`, which is slower than most of the checks
    whose fate depends on it.
    """
    global REAL_ONLY, _DOCKER
    REAL_ONLY, _DOCKER = real, docker
    wake.CONTAINER_PREFIX = f"mtr-w{os.getpid()}-"


def sweep() -> None:
    """Remove any container a worker died holding."""
    if not shutil.which("docker"):
        return
    left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=mtr-w"],
                          capture_output=True, text=True).stdout.split()
    if left:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
        print(f"swept {len(left)} leaked container(s)")


def main(argv: list[str] | None = None) -> int:
    """Run the checks, in as many processes as asked for."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("patterns", nargs="*", help="only checks whose name contains one of these")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="how many checks to run at once (1 to run them in this process)")
    p.add_argument("--real", action="store_true",
                   help="run every check in a container, including the ones that need not be")
    p.add_argument("--no-docker", action="store_true", help="skip the checks that need a container")
    p.add_argument("--list", action="store_true", help="print the check names and stop")
    args = p.parse_args(argv)

    chosen = [l for l in checks()
              if not args.patterns or any(pat in l for pat in args.patterns)]
    if args.list:
        print("\n".join(chosen))
        return 0
    if not chosen:
        print(f"no check matches {args.patterns}")
        return 2

    # Asked once for the whole run, and handed to every worker: docker info is
    # slower than most of the checks that depend on the answer.
    available = False if args.no_docker else docker_ready()
    configure(args.real, available)

    started = time.time()
    failed, skipped = [], []

    def record(label, how, detail):
        if how == "skip":
            skipped.append(label)
            print(f"SKIP  {label}", flush=True)
        elif how == "fail":
            failed.append(label)
            print(f"FAIL  {label}\n{detail}", flush=True)
        else:
            print(f"ok    {label}", flush=True)

    jobs = max(1, min(args.jobs, len(chosen)))
    if jobs == 1:
        for label in chosen:
            record(*run_one(label))
    else:
        with futures.ProcessPoolExecutor(
                max_workers=jobs, initializer=configure,
                initargs=(args.real, available)) as pool:
            for done in futures.as_completed(
                    [pool.submit(run_one, l) for l in chosen]):
                record(*done.result())

    sweep()
    if skipped:
        print(f"\n{len(skipped)} skipped (needs Docker + the image)")
    print(f"{len(chosen)} checks in {time.time() - started:.1f}s across {jobs} process(es)")
    if failed:
        print(f"\nFAILED: {', '.join(sorted(failed))}")
    else:
        print("\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
