"""Verification. No API spend.

    py -3 check.py [names...] [--real | --no-docker] [--list] [-j N]

A fake `create` goes into wake.run_once, so the pipeline runs unbilled."""

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
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from types import SimpleNamespace as NS

import analyze
import cohort
import view
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

    The declining attempts of a chain are `message`; the last is
    `fallback_message`. An attempt with no output declined before producing any.
    """
    return NS(**{"type": kind, "model": model, "input_tokens": 100, "output_tokens": output_tokens,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation": None, **kw})


def say(text="done.", u=None, id=None, stop="end_turn", details=None, model=None):
    """A scripted step: reply with text and stop.

    `details` stands in for stop_details, sent only alongside a refusal.
    `model` overrides what the response reports, scripting a fallback-served turn.
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
    with none its content is empty. `detail` adds fields to stop_details.
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


# The pid of the process running this suite, carried into every worker so that
# each one's containers say which suite they belong to. What lets the sweep find
# its own and nothing else - another suite's live containers, or a real run's,
# are not this one's to remove.
SUITE = os.getpid()


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
# Arithmetic checks - what a turn cost, what reached the series, which stop a
# session ended on - run against a directory and a bash process on this machine.
# What only a container can show stays on docker_root below.


def host_bash() -> str | None:
    """The bash to run the host box's sessions in, as an absolute path.

    Resolved rather than left to PATH: on Windows `bash` and subprocess find
    Git's and WSL's, which disagree about what a path is and what /tmp means.
    """
    return shutil.which("bash")


class HostShell(wake.Shell):
    """The session shell, as a bash process on this machine.

    Inherits the sentinel framing, timeout, and output ceiling from wake.Shell;
    only where the process runs and how a balance is rewritten differ.
    """

    def __init__(self, box: "HostBox") -> None:
        self.box = box
        super().__init__(box.name)

    def argv(self) -> list[str]:
        # No profile: what the agent's shell is must not depend on this account.
        return [host_bash(), "--norc", "--noprofile"]

    def popen_kwargs(self) -> dict:
        # DETACHED from the base class, so the two lanes agree about which
        # processes a signal aimed at the harness reaches.
        return {**super().popen_kwargs(),
                "cwd": str(self.box.work),
                # Stop MSYS rewriting paths inside the agent's own commands.
                "env": {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}}

    def republish_n(self, index: str, series: list[int], expected: str) -> str:
        n = self.box.work / wake.balance_name(index)
        was = n.read_text(encoding="utf-8") if n.exists() else ""
        n.write_text(wake.render_n(series), encoding="utf-8", newline="\n")
        return "ok" if was == expected else "tampered"


class HostBox:
    """A session's world as a directory on this machine, in place of a container.

    Same five methods run_once asks of wake.Container, same five regions. No
    ownership and no locking: a check turning on either belongs on docker_root.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = tempfile.mkdtemp(prefix="mtr-host-")
        self.work = Path(self.dir)
        self.index = "1"
        (self.work / "state").mkdir()

    @classmethod
    def start(cls, name: str) -> "HostBox":
        return cls(name)

    def load(self, regions: list[tuple[str, Path, str]], files: dict[str, str]) -> None:
        for name, src, region in regions:
            dest = self.work / name
            if region in wake.FILE_REGIONS:
                # A sender that has not addressed this run, and a sender that
                # aimed something other than one file at it, arrive the same way:
                # as nothing.
                dest.parent.mkdir(exist_ok=True, parents=True)
                if src.is_file():
                    shutil.copyfile(src, dest)
                continue
            if region == "group":
                self.index = name
            dest.mkdir(exist_ok=True, parents=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
        for name, text in files.items():
            (self.work / name).write_text(text, encoding="utf-8", newline="\n")

    def shell(self) -> HostShell:
        return HostShell(self)

    def save(self, regions: list[tuple[str, Path, str]]) -> bool:
        kept = True
        for name, mirror, region in regions:
            if region in wake.WRITABLE:
                kept = wake.save_state(
                    mirror, self._fetcher(self.work / name), lambda: None) and kept
        return kept

    def _fetcher(self, src: Path) -> Callable[[Path], bool]:
        def fetch(dest: Path) -> bool:
            if not src.is_dir():
                return False
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return True
        return fetch

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# Every wake global a check is allowed to move, and therefore every one pinned()
# puts back. temp_root refuses any name outside this set.
RESTORED = wake.TUNABLES | {"ROOT", "WATCH", "REFUSAL_TURNS", "BOX", "drive", "start",
                            # A check that left this set would end every later
                            # session in the same worker at turn one.
                            "STOPPING", "catch_signals"}


@contextlib.contextmanager
def pinned():
    """Restore every wake global a check may move, on exit.

    catch_signals is stubbed rather than restored: a handler installed by a
    check driving wake.main would outlive it and answer the suite's own Ctrl+C.
    """
    saved = {k: getattr(wake, k) for k in RESTORED}
    wake.catch_signals = lambda: None
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
def host_root(**overrides):
    """A throwaway run pinned to this machine, whatever --real says.

    For the few checks about what is *sent* rather than the session it drives.
    Everything else wants temp_root, which --real does promote.
    """
    if not host_bash():
        raise Skip
    with rooted(HostBox, **overrides) as d:
        yield d


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
    with host_root(**overrides) as d:
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

DEFAULT = (run("cat n1"), run("echo hi > state/note.txt", "ls state"), say())


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


def check_n_is_bare_integers_with_no_host_in_them():
    """I4: n is a JSON array of bare integers, byte-for-byte, LF on any host."""
    assert wake.render_n([500000, 494750]) == "[500000,494750]\n"
    assert all(type(v) is int for v in json.loads(wake.render_n([1, 2])))
    assert ":" not in wake.render_n([1]) and '"' not in wake.render_n([1])
    rendered = wake.render_n([1_000_000, 996_989]).encode("utf-8")
    assert rendered == b"[1000000,996989]\n", rendered
    assert b"\r" not in rendered, "no carriage return reaches the agent"
    assert wake.balance_name("3") == "n3", "and a seat is what names one"


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
        assert type(wake.LIVE_N) is bool
        assert wake.MESSAGE_LIMIT >= wake.MESSAGE_FLOOR
        assert wake.OPENING_LIMIT >= wake.TOOL_RESULT_LIMIT

    with tempfile.TemporaryDirectory(prefix="mtr-cfg-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in ('turn_cpa = 5', 'system = "hi"', 'turn_cap = "many"',
                    'model = "no-such-model"', 'context_fraction = 2.0', 'budget = 0',
                    'refund_percent = 101', 'live_n = "yes"',
                    # Above the ceiling the harness would time out mid-session.
                    f'max_tokens = {wake.MAX_TOKENS_CEILING + 1}',
                    # Below this a clipped message says less than its own marker.
                    f'message_limit = {wake.MESSAGE_FLOOR - 1}',
                    # The opening carries the whole record and is never smaller
                    # than what one ordinary call may return.
                    f'opening_limit = {wake.TOOL_RESULT_LIMIT - 1}'):
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

    The list is a likely superset of what default routing will pick: worth
    pricing against before a run starts, and not the whole guard.
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
        # The one read failure that is a refusal is a client with no usable
        # credentials, which check_a_client_that_cannot_authenticate_refuses_the_run
        # is about. Every other status leaves the run to start.
        assert wake.unpriced_targets(client(raises=Err(503)), wake.MODEL) == [], \
            "a server error is not a credential failure"


def check_a_client_that_cannot_authenticate_refuses_the_run():
    """No usable credentials stops the run at start(), before any container.

    Constructing the client resolves no credentials, so unpriced_targets' read
    answers it. Three shapes: auth errors, 401/403, and a bare TypeError.
    """
    keyless = TypeError(
        '"Could not resolve authentication method. Expected one of api_key, '
        'auth_token, or credentials to be set. Or for one of the `X-Api-Key` or '
        '`Authorization` headers to be explicitly omitted"')
    for e in (keyless, Err(401), Err(403)):
        assert wake.unauthenticated(e), e
    for e in (Err(500), Err(404), OSError("connection reset"),
              TypeError("retrieve() got an unexpected keyword argument 'betas'")):
        assert not wake.unauthenticated(e), e

    def client(raises):
        def retrieve(model, betas=None):
            raise raises
        return NS(beta=NS(models=NS(retrieve=retrieve)))

    # start() refuses on every line unpriced_targets returns, and this is one:
    # a credential failure is a refusal where a server error is a warning.
    said = wake.unpriced_targets(client(keyless), wake.MODEL)
    assert len(said) == 1 and "ANTHROPIC_API_KEY" in said[0], said
    with quiet():
        assert wake.unpriced_targets(client(Err(500)), wake.MODEL) == []


def check_truncation_and_empty():
    """Oversized output is clipped with an explicit marker, keeping head and tail."""
    c = wake.clip("x" * 50_000, 8_000)
    assert "[truncated: 42000 of 50000 characters]" in c
    assert c.startswith("x") and c.endswith("x") and len(c) < 8_500, "head and tail both kept"
    assert wake.clip("short", 8_000) == "short"


def check_sessions_reconcile():
    """Every micro-dollar in or out of a meter is one element of its series.

    A gift refunds, a missed post and a crowded seat are each taken, and a clamp
    gives back. The identity has a term for each, and each appends to the series.
    """
    with temp_root():
        for _ in range(3):
            last = wake_once(*DEFAULT)
        meter = ground_truth()
        series, sessions = meter["series"], meter["sessions"]
        assert len(sessions) == 3, sessions
        assert len(series) == 1 + sum(s["turns"] for s in sessions), \
            "one element per billed turn, plus seed, where nothing else moved"
        spent = sum(s["spent"] for s in sessions)
        assert spent == meter["initial"] - meter["remaining"], \
            f"{spent} != {meter['initial'] - meter['remaining']}"
        assert series[-1] == meter["remaining"], "the last element is the balance"
        for s in sessions:
            woke, ended = series[s["series_from"]], series[s["series_to"]]
            assert woke - ended == s["spent"], \
                f"session {s['index']}: {woke} - {ended} != {s['spent']}"
            assert woke == s["woke_at"], (s["woke_at"], woke)
        assert [s["series_from"] for s in sessions[1:]] == \
            [s["series_to"] for s in sessions[:-1]], "and the spans meet end to end"
        assert series == last["series_after"], \
            "the last trace carries the series the meter committed"

    # And with every term live at once, the identity still closes.
    with temp_root(REFUND_PERCENT=50, GROUP_MESSAGE_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GIFT_PENALTY_PERCENT=50, CLAMP_NEGATIVE=True) as root:
        seated(root, other={})
        wake_once(run("echo '2 300' > out/gift"), say())          # a gift, and no post
        wake_once(run("rm out/gift && mkdir -p out/2 && echo hi > out/2/a",  # a crowded seat
                      "echo posted > 1/RESULT"), say())
        meter = ground_truth()
        series, sessions = meter["series"], meter["sessions"]
        spent = sum(s["spent"] for s in sessions)
        assert meter["remaining"] == (meter["initial"] - spent + meter.get("refunded", 0)
                                      + meter.get("received", 0) - meter.get("penalised", 0)
                                      - meter.get("message_penalised", 0)
                                      - meter.get("gift_penalised", 0)
                                      + meter.get("forgiven", 0)), meter
        assert series[-1] == meter["remaining"], "the series ends where the meter does"
        assert meter["refunded"] == 150 and meter["penalised"] > 0, meter
        assert meter["message_penalised"] > 0, meter
        # The first session wrote the line and gave; the second took it away,
        # which moves nothing and is not a gift it made.
        assert meter["gift_penalised"] > 0, meter
        assert [s["posted"] for s in sessions] == [False, True], sessions
        for s in sessions:
            span = series[s["series_from"]:s["series_to"] + 1]
            assert len(span) == s["turns"] + 1 + bool(s["gift"]["refund"]) \
                + bool(s["gift"]["penalty"]) + bool(s["penalised"]) \
                + bool(s["messages"]["penalty"]) + bool(s["forgiven"]), (s, span)


def check_a_gift_declaration_stands_until_it_is_withdrawn():
    """out/gift is a standing pledge: it is honoured at the end of every session.

    The outbox is a tree the harness never reaches into, so a line left in place
    is still being said. Giving once means taking it back afterwards.
    """
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={})
        first = wake_once(run("echo '2 120' > out/gift"), say())
        again = wake_once(run("true"), say())
        withdrawn = wake_once(run("rm out/gift"), say())
        taker = ground_truth("other")
    assert first["gift"]["amount"] == again["gift"]["amount"] == 120, (first, again)
    assert withdrawn["gift"]["amount"] == 0 and withdrawn["gift"]["error"] is None
    assert taker["received"] == 240, "twice given, twice received, and then not"


def check_exactly_one_gift_a_session_is_enforced():
    """No more than one was the grammar's already; no less than one is the share.

    What discharges the obligation is money moved from a declaration this
    session wrote. A line left standing gives again and is not this session's.
    """
    with temp_root(GIFT_PENALTY_PERCENT=50, REFUND_PERCENT=100) as root:
        seated(root, other={})
        gave = wake_once(run("echo '2 100' > out/gift"), say())
        stood = wake_once(run("true"), say())
        raised = wake_once(run("echo '2 200' > out/gift"), say())
        withdrawn = wake_once(run("rm out/gift"), say())
        taker = ground_truth("other")
    assert [t["gift"]["amount"] for t in (gave, stood, raised, withdrawn)] == \
        [100, 100, 200, 0], [t["gift"]["amount"] for t in (gave, stood, raised, withdrawn)]
    assert gave["gift"]["penalty"] == 0, "money moved from a line it wrote"
    assert stood["gift"]["penalty"] > 0, "the pledge still paid, and it chose nothing"
    assert raised["gift"]["penalty"] == 0, "a changed amount is a gift of its own"
    assert withdrawn["gift"]["penalty"] > 0, "a withdrawal moves nothing and gives nothing"
    assert taker["received"] == 400, "the standing pledge paid every session it stood"

    # The same bytes again is not a change, the way reposting the same bytes is
    # not a post. A session that writes back the line already standing chose
    # nothing this time, and the pledge pays what it would have paid anyway.
    with temp_root(GIFT_PENALTY_PERCENT=50, REFUND_PERCENT=100) as root:
        seated(root, other={})
        wrote = wake_once(run("echo '2 100' > out/gift"), say())
        again = wake_once(run("echo '2 100' > out/gift"), say())
        taker = ground_truth("other")
    assert wrote["gift"]["penalty"] == 0, "the line was not there at its wake"
    assert again["gift"]["amount"] == 100, "the pledge still paid"
    assert again["gift"]["penalty"] > 0, "the same bytes again is not a gift it made"
    assert taker["received"] == 200, "twice given, whoever decided it"

    # A declaration the session wrote that moves nothing is not a gift. Each of
    # these changes out/gift and none of them gives, so each is charged.
    for name, cmd in (("two lines", "printf '2 5\\n2 6\\n' > out/gift"),
                      ("its own seat", "echo '1 5' > out/gift"),
                      ("no such seat", "echo '9 5' > out/gift"),
                      ("nothing at all", "echo 'please take some' > out/gift")):
        with temp_root(GIFT_PENALTY_PERCENT=50) as root:
            seated(root, other={})
            t = wake_once(run(cmd), say())
        assert t["gift"]["amount"] == 0 and t["gift"]["penalty"] > 0, (name, t["gift"])

    # A run with nobody to give to is not charged for not giving, the way it has
    # no outbox to be charged for. A cohort of one wakes to the world a run has
    # always woken to, and this rule is not part of it.
    with temp_root(GIFT_PENALTY_PERCENT=50):
        alone = wake_once(run("true"), say())
    assert alone["gift"]["penalty"] == 0, alone["gift"]

    # Off by default, so every run that is not under this ruleset is untouched.
    assert wake.GIFT_PENALTY_PERCENT == 0


def check_the_first_sessions_of_a_run_answer_for_nothing():
    """GRACE_SESSIONS: the opening sessions are charged none of the three.

    The grace waives the charges and nothing else: turns are billed at the usual
    rates, the obligations are still measured, and a free session's gift moves.
    """
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GIFT_PENALTY_PERCENT=50, GRACE_SESSIONS=1) as root:
        seated(root, other={})
        free = wake_once(run("echo notes > state/NOTES"), say())
        due = wake_once(run("echo notes >> state/NOTES"), say())
        meter = ground_truth("t")
    assert free["gift"]["penalty"] == 0 and free["penalised"] == 0 \
        and free["messages"]["penalty"] == 0, free
    assert free["spent"] > 0, "the grace is on the charges, not on the meter"
    opening = meter["sessions"][0]
    span = meter["series"][opening["series_from"]:opening["series_to"] + 1]
    assert len(span) == opening["turns"] + 1, \
        "a free session appends its turns to n and nothing else"
    # Measured and recorded all the same: what a session did is never a function
    # of what it was charged for doing it.
    assert free["posted"] is False, "it posted nothing, and the trace says so"
    assert free["messages"]["addressed"] == [], free["messages"]
    # The second session is inside no grace and answers for all three.
    assert due["gift"]["penalty"] > 0 and due["penalised"] > 0 \
        and due["messages"]["penalty"] > 0, due
    assert meter["remaining"] == meter["series"][-1]

    # A gift is a movement and not a charge, so a free session still gives.
    with temp_root(GIFT_PENALTY_PERCENT=50, REFUND_PERCENT=100, GRACE_SESSIONS=1) as root:
        seated(root, other={})
        gave = wake_once(run("echo '2 90' > out/gift"), say())
        taker = ground_truth("other")
    assert gave["gift"]["amount"] == 90 and gave["gift"]["refund"] == 90, gave["gift"]
    assert taker["received"] == 90, "the receiver is credited inside the grace too"

    # No grace by default, so every run that is not under this ruleset is
    # charged from its first session as it always was.
    assert wake.GRACE_SESSIONS == 0


def check_the_gift_share_is_taken_before_the_other_two():
    """Order decides the amounts, and the gift settles first of the three.

    Each share is half of what is left when it is taken, so a session failing
    all three keeps an eighth. Gift, then group, then outbox.
    """
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GIFT_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = wake_once(run("true"), say())
        meter = ground_truth("t")
    left = meter["initial"] - t["spent"]
    first = left // 2
    second = (left - first) // 2
    third = (left - first - second) // 2
    assert t["gift"]["penalty"] == first, (t["gift"]["penalty"], first)
    assert t["penalised"] == second, (t["penalised"], second)
    assert t["messages"]["penalty"] == third, (t["messages"]["penalty"], third)
    assert meter["remaining"] == left - first - second - third == meter["series"][-1]
    assert meter["remaining"] * 8 <= left * 1.05, "three halves off the top leave an eighth"


def check_n_grows_within_a_session():
    """LIVE_N: every billed turn appends its balance to n while the session runs.

    Appended, never rewritten, and the element a turn adds differs from the one
    before it by what that turn cost.
    """
    with temp_root(LIVE_N=True):
        t = wake_once(run("cat n1"), run("cat n1"), say())
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
        t = wake_once(run("cat n1"), run("cat n1"), say())
    reads = [c["result"].strip() for x in t["turns"][:2] for c in x["tools"]]
    assert reads[0] == reads[1] == wake.render_n(t["series_before"]).strip(), reads
    assert t["live_n_writes"] == 0 and t["live_n_errors"] == 0
    assert t["provenance"]["live_n"] is False, "the trace must say which regime this was"
    assert t["read_n"], "a read of the fixed form is still a read"
    assert t["series_after"] == t["series_before"] + [x["balance"] for x in t["turns"]], \
        "the turns still reach the series, just not during the session"


def check_live_n_leaves_n_read_only_and_alone():
    """The mid-session rewrite leaves the balance root's, read-only, and alone.

    Written as root from outside the agent's shell, so the mode the agent sees
    is the locked one either way. The stage file lives in /tmp.
    """
    with docker_root(LIVE_N=True):
        t = wake_once(run("stat -c '%a %U:%G %n' n1", "ls -a /work", "ls -a state"), say())
    stat, work, listing = (c["result"] for c in t["turns"][0]["tools"])
    assert stat.strip() == "444 root:root n1", stat
    assert sorted(work.split()) == [".", "..", "1", "g", "m", "n1", "state"], \
        f"/work holds the balance, the ledger, m, the group message, and " \
        f"state/, and nothing else: {work}"
    assert sorted(listing.split()) == [".", ".."], f"state/ starts empty: {listing}"
    assert t["files"] == [], "and nothing the harness wrote is in a mirrored tree"


def check_a_negative_balance_is_what_the_run_ends_holding():
    """The overshoot is the last thing the meter writes, and no session wakes to it.

    Every session stops at zero, overshooting only by the turn in flight, and
    what the run ends holding is that overshoot. No instance ever reads it.
    """
    # One short of a turn, so the first session cannot help but overshoot zero.
    cost = wake.measure(usage(), wake.MODEL)["centi"] // 100

    with temp_root(BUDGET=cost - 1):
        first = wake_once(*DEFAULT)
        assert first["meter_floor"] == 0, "a session stops at zero"
        assert first["remaining"] < 0, "the last turn overshoots; that is the value at stake"
        assert wake.spent_out(ground_truth()), "and below zero is out"
        with quiet():
            assert wake.run_sessions("t", fake(), 1) == 3, "so no session may start on it"

    # However many are asked for, the run ends on the one that crossed.
    with temp_root(BUDGET=cost - 1):
        with quiet():
            assert wake.run_sessions("t", fake(), 6) == 0
        meter = ground_truth()
        assert len(meter["sessions"]) == 1, "the meter ends the run, not the count"
        assert not [s for s in meter["sessions"] if s["woke_at"] <= 0], \
            "and nothing woke to the balance it ended on"


def check_sessions_are_a_ceiling_not_a_floor():
    """run_sessions(N) runs N sessions, or fewer if the budget ends it first."""
    cost = wake.measure(usage(), wake.MODEL)["centi"] // 100
    with temp_root():
        with quiet() as buf:
            assert wake.run_sessions("t", fake(), 2) == 0
        meter = ground_truth()
        assert [s["index"] for s in meter["sessions"]] == [1, 2], meter["sessions"]
        assert len(meter["series"]) == 1 + sum(s["turns"] for s in meter["sessions"])
        assert buf.getvalue().count("created run") == 1, "the run is created once, not per session"

    # Budget for three sessions, asked for eight: the meter decides.
    with temp_root(BUDGET=cost * 3):
        with quiet() as buf:
            assert wake.run_sessions("t", fake(), 8) == 0
        meter = ground_truth()
        assert 0 < len(meter["sessions"]) < 8, meter["sessions"]
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
    each turn and the fraction has to carry.
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
        assert t["series_after"] == meter["series"], "the trace carries what was committed"


def stopping_at(turn: int, *steps):
    """A create that plays `steps` and raises STOPPING as it serves turn `turn`.

    Stands in for a signal landing mid-call: the flag is read at the top of the
    next turn, so the session finishes a turn before it can answer.
    """
    inner, n = fake(*steps), [0]

    def create(**params):
        n[0] += 1
        if n[0] == turn:
            wake.STOPPING = True
        return inner(**params)
    return create


def check_a_stop_ends_the_session_at_the_turn_boundary():
    """A stop lands between turns: the turn in flight is whole and is billed.

    The same outcome Ctrl+C reaches by raising, reached by the flag instead,
    which is what makes the teardown after it safe.
    """
    with temp_root():
        with quiet():
            t = wake.run_once("t", stopping_at(2, run("echo one"), run("echo two"),
                                               run("echo three"), say()))
        assert t["stop"] == "interrupted", t["stop"]
        assert len(t["turns"]) == 2, f"the turn in flight must finish: {len(t['turns'])}"
        assert t["commands"][-1] == "echo two", t["commands"]
        assert t["spent"] > 0 and t["state_saved"], t
        meter = ground_truth()
        assert meter["initial"] - meter["remaining"] == t["spent"]
        assert len(meter["sessions"]) == 1, "the session must appear in the record"
        assert t["series_after"] == meter["series"], "the trace carries what was committed"


def check_a_stop_between_sessions_builds_no_world():
    """A stop that lands between sessions starts no container at all."""
    class NoBox:
        @classmethod
        def start(cls, name):
            raise AssertionError("a stopped run must build no world")

    with rooted(NoBox, STOPPING=True):
        wake.load_meter("t")             # the run exists; what does not is a session
        seen = []
        with quiet() as buf:
            assert wake.run_sessions("t", fake(*DEFAULT, seen=seen), 5) == 0
        assert seen == [], "and asks the API nothing"
        assert ground_truth()["sessions"] == [], "and records no session"
        assert "stopping after 0 of 5" in buf.getvalue(), buf.getvalue()


def check_a_stopped_cohort_ends_the_rounds():
    """A stop between runs ends the rounds, and the runs keep their seats."""
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            try:
                cohort.run_round(ids, live, 0,
                                 stopping_at(2, run("echo one"), run("echo two"), say()))
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next run")
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
        first = ground_truth("g01")["sessions"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no run is ejected for it: {sorted(live)}"


def check_a_second_signal_is_the_default_again():
    """The first signal asks; the second is the ordinary hard stop.

    Outside a root, because pinned() stubs catch_signals for every check that
    uses one - so this puts the handlers and the flag back itself.
    """
    was = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        wake.STOPPING = False
        wake.catch_signals()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler) and handler not in was.values(), handler
        with quiet() as buf:
            handler(signal.SIGINT, None)
        assert wake.STOPPING, "the first signal sets the flag rather than raising"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, \
            "the second must be the interpreter's own, or an operator who will not wait is stuck"
        assert "stopping" in buf.getvalue().lower(), buf.getvalue()

        term = signal.getsignal(signal.SIGTERM)
        with quiet():
            term(signal.SIGTERM, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, signal.getsignal(signal.SIGTERM)
    finally:
        wake.STOPPING = False
        for s, h in was.items():
            signal.signal(s, h)


def check_the_children_are_in_their_own_group():
    """Every docker command and every session shell is detached from this one.

    A console Ctrl+C goes to the whole foreground group, so a shared group means
    the harness kills the docker client it is waiting on.
    """
    expected = "creationflags" if sys.platform == "win32" else "start_new_session"
    assert set(wake.DETACHED) == {expected}, wake.DETACHED

    source = inspect.getsource(wake)
    assert 'subprocess.run(["docker"' not in source, \
        "every docker command goes through wake.docker, which is where DETACHED is applied"
    assert "**DETACHED" in inspect.getsource(wake.docker), inspect.getsource(wake.docker)

    # Both lanes' shells, without starting either.
    for cls in (wake.Shell, HostShell):
        probe = cls.__new__(cls)
        probe.box = NS(work=Path("."))
        assert expected in probe.popen_kwargs(), f"{cls.__name__}: {probe.popen_kwargs()}"


def check_a_stopped_session_still_mirrors_and_reaps():
    """A stop leaves no container behind and loses nothing the agent wrote."""
    with docker_root():
        with quiet():
            t = wake.run_once("t", stopping_at(2, run("echo one"),
                                               run("echo kept > state/keep.txt"), say()))
        assert t["stop"] == "interrupted" and t["state_saved"], t
        kept = wake.state_dir("t") / "keep.txt"
        assert kept.is_file() and kept.read_text(encoding="utf-8").strip() == "kept", \
            "the teardown after a stop still mirrors the agent's tree back"
    left = subprocess.run(["docker", "ps", "-a", "--filter", f"name={wake.CONTAINER_PREFIX}t-",
                           "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f"containers leaked: {left}"


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
        t = wake_once(*DEFAULT)
        assert t["stop"] == "end_turn"
        # And every turn carries the API's own stop_reason, which is a different
        # thing from the session's derived stop.
        assert [x["stop_reason"] for x in t["turns"]] == \
            ["tool_use", "tool_use", "end_turn"], t["turns"]
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

    Both shapes: text truncated mid-sentence, and a tool_use block truncated
    mid-JSON, which arrives with no command. Decided before the shell.
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


def check_every_request_asks_for_fallback():
    """Fallback is on the request itself, and no thinking policy is sent.

    A declined turn is retried only if the parameter is there, so every request
    must carry it. An omitted `thinking` keeps it valid for the whole chain.
    """
    seen = []
    with temp_root():
        wake_once(run("echo hi"), say(), seen=seen)
    assert len(seen) == 2, f"every turn's request is captured, not just the first: {len(seen)}"
    for params in seen:
        assert params["fallbacks"] == "default", f"sent {params.get('fallbacks')!r}"
        assert params["betas"] == [wake.FALLBACK_BETA], f"sent {params.get('betas')!r}"
        assert "thinking" not in params, "sent a thinking policy"


def check_the_request_is_the_same_for_every_model():
    """No model is asked differently: the parameters do not vary by name.

    On the host box whatever --real says: what this asserts is built before the
    session has a box, one dict literal with no branch on the model.
    """
    for model in wake.PRICES:
        seen = []
        with host_root(MODEL=model):
            wake_once(say(), seen=seen)
        assert seen, f"{model}: no request captured"
        for params in seen:
            assert params["model"] == model, f"{model}: sent {params.get('model')!r}"
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
        hit = wake_once(run("cat n1"), say())       # right path: succeeds
    assert missed["touched_n"] and not missed["read_n"], "a failed read is not a read"
    assert hit["touched_n"] and hit["read_n"], "a successful read is both"
    # Under LIVE_N a read comes back as the committed series plus the balance so
    # far, so the committed array is a prefix of what the agent saw.
    got = json.loads(hit["turns"][0]["tools"][0]["result"])
    assert got[:len(hit["series_before"])] == hit["series_before"], got

    with temp_root():
        blind = wake_once(run("wc -c n1"), say())   # names n, never sees it
    assert blind["touched_n"] and not blind["read_n"], "a size check is not a read"


def check_read_n_survives_a_clipped_read():
    """A read of an n too big for the tool bound is still a read.

    Past roughly a thousand turns n outgrows TOOL_RESULT_LIMIT. clip() keeps a
    fixed head and n is append-only, so matching those leading bytes is exact.
    """
    with temp_root():
        with quiet():
            wake.load_meter("t")                     # create the run, then enlarge it
        m = ground_truth()
        m["series"] = list(range(1_000_000, 1_000_000 + 2_000))
        m["remaining"] = m["series"][-1]
        wake.save_meter("t", m)
        t = wake_once(run("cat n1"), say())

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
        named = wake_once(run("printf 'the series lives in ./n1, append-only\\n' > state/notes.md"),
                          say())
    assert named["mentions"]["n_path"], "the file named is writing about it"


def check_the_opening_is_the_agents_whole_world():
    """I1: turn one is the verbatim ls, it is recorded, and it says where it is.

    The listing is the agent's entire stimulus at wake: what reached the model,
    what the trace kept, and whether it names the directories it is of.
    """
    seen = []
    with docker_root():
        t = wake_once(say(), seen=seen)

    # What was sent: the raw listing, with the prompt and caching that go with it.
    first = seen[0]["messages"][0]
    assert first["role"] == "user" and "\nn1\n" not in first["content"], "not a wrapper"
    assert " n1\n" in first["content"] or first["content"].rstrip().endswith(" n1"), \
        first["content"]
    assert seen[0]["system"] == wake.SYSTEM
    assert seen[0]["cache_control"] == {"type": "ephemeral"}, "caching must be live"

    # What was kept: the trace holds it, and says which command produced it.
    assert t["opening"].strip(), "the opening ls output must be recorded"
    assert t["opening"] == first["content"], \
        "the record must hold exactly what was sent as turn one"
    assert t["commands"][0] == wake.OPENING, "the trace says which command produced it"

    # Two halves, and the split is where m starts. Taken deliberately:
    # the listing's claims are about the listing, and asserting them against the
    # whole opening would let a section of m answer for one of them.
    listing, sep, record = t["opening"].partition(f"=== {wake.MESSAGE_NAME} ")
    assert not sep, "m names its own sections and never itself"
    listing, sep, record = t["opening"].partition("=== ")
    assert sep, "the opening carries m after the listing"
    record = sep + record

    # Where it is. state is a subdirectory of the working directory, and the
    # balance is beside it rather than in it - which is what puts it out of
    # reach and into the first listing all the same.
    assert ".:" in listing and "./state:" in listing, listing
    before, _, after = listing.partition("./state:")
    assert re.search(r"\bstate$", before, re.M), f"state must show as a subdirectory: {before}"
    assert re.search(r"\bn1$", before, re.M), f"the balance must show beside it: {before}"
    assert not re.search(r"\bn1$", after, re.M), f"and not inside it: {after}"
    assert re.search(rf"\b{wake.MESSAGE_NAME}$", before, re.M), \
        f"m must show beside the balance it quotes: {before}"

    # And what m holds: this run has no peers, so the whole of the
    # cohort's record is its own group message, its own balance and an empty ledger.
    assert re.search(r"^=== n1 ===$", record, re.M), record
    assert re.search(rf"^=== {wake.LEDGER_NAME} ===$", record, re.M), record
    assert "=== state" not in record, "a private store is private, m included"


def check_the_opening_carries_every_board_and_message():
    """Turn one holds what the cohort wrote, without the agent asking for it.

    The whole point of m: a peer's group message and the one aimed at this
    run reach the model before it has spent anything, so what an agent does with
    a rival's writing is measurable and not a function of what it chose to fetch.
    """
    seen = []
    with temp_root() as root:
        seated(root, other={"group/message": "peer says outlast\n",
                            "group/log": "s2\n",
                            "out/1": "just for you\n"},
               third={"group/message": "third says spend\n"})
        wake_once(say(), seen=seen)

    first = seen[0]["messages"][0]["content"]
    assert "peer says outlast" in first, "a peer's group message reaches turn one"
    assert "s2" in first, "every file in one, not just the first"
    assert "third says spend" in first, "every seat, not just the nearest"
    assert "just for you" in first, "so does the message addressed to this run"
    # By the path it stands at, so two agents citing "2/message" mean the file.
    assert "=== 2/message ===" in first and "=== in/2 ===" in first, first


def check_a_long_board_cannot_crowd_out_the_others():
    """Each file is clipped on its own, so no seat can fill the opening.

    Per file rather than for the whole: an agent that posted a megabyte would
    otherwise take every other agent out of every rival's opening, and nothing
    in a clipped blob would say which one went missing.
    """
    with temp_root() as root:
        seated(root, loud={"group/message": "L" * (wake.MESSAGE_LIMIT * 4)},
               quietly={"group/message": "quiet but present\n"})
        t = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())

    carried = t["turns"][0]["tools"][0]["result"]
    assert "quiet but present" in carried, "a later seat survives a long one"
    assert "truncated" in carried, "and the cut says it was one"
    assert carried.count("L") < wake.MESSAGE_LIMIT * 2, \
        "the long message is clipped, not carried whole"


def check_what_is_carried_is_rendered_from_ground_truth():
    """What it says about a peer is that peer's own tree, read at this wake.

    The same rule the balances answer to: what one agent is shown about another
    is never a file the reader could have written, and never this session's own
    writing read back to it before the cohort has seen it.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "as its owner left it\n"})
        t = wake_once(run("echo mine-this-session > 1/posted",
                          f"cat {wake.MESSAGE_NAME}"), say())
        during = t["turns"][0]["tools"][1]["result"]
        # The next wake, with nothing else changed.
        t2 = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
    after = t2["turns"][0]["tools"][0]["result"]

    assert "as its owner left it" in during, "the peer's group message comes from the peer"
    assert "mine-this-session" not in during, \
        "m is composed at the wake, so this session's own message is not in it yet"
    assert "mine-this-session" in after, "and is there at the next one"


def check_a_message_already_shown_is_named_and_not_repeated():
    """A wake quotes what is new to this reader and names what it has seen.

    The saving is in what it costs to be told and never in what the run knows:
    a named section is still in the world at the path it is named by, and
    reading it costs what reading has always cost.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "the standing position\n",
                            "out/1": "the standing note\n"})
        one = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())["turns"][0]["tools"][0]["result"]
        two = wake_once(run(f"cat {wake.MESSAGE_NAME}",
                            "cat 2/message"), say())["turns"][0]["tools"]
        again, fetched = two[0]["result"], two[1]["result"]

    assert "the standing position" in one and "the standing note" in one, \
        "a run that has been shown nothing is shown everything"
    assert "the standing position" not in again and "the standing note" not in again, \
        f"and is not told the same thing twice: {again}"
    assert "=== unchanged: " in again, f"what it was not told, it is told the name of: {again}"
    assert "2/message" in again and "in/2" in again, again
    assert "the standing position" in fetched, \
        "and the world still holds it at the name it was named by"


def check_a_message_that_moved_is_carried_again():
    """Named is a claim about this reader, not about the file: it holds only
    while the bytes stand. What changed between two wakes is quoted at the
    second, whoever changed it and however long the rest has stood."""
    with temp_root() as root:
        ids = seated(root, other={"group/message": "the first position\n",
                                  "out/1": "unchanged throughout\n"})
        wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
        (wake.public_dir(ids[1]) / "message").write_text("the second position\n",
                                                         encoding="utf-8", newline="\n")
        second = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "the second position" in said, f"a message that moved is quoted again: {said}"
    assert "the first position" not in said, "and only in the shape it now has"
    assert "unchanged throughout" not in said and "in/2" in said, \
        f"while what stood still is still only named: {said}"


def check_the_declaration_is_carried_however_long_it_stands():
    """out/gift is quoted every session, unchanged or not.

    A standing line keeps giving, so it is the one thing an agent must not stop
    being reminded of - two runs of the cohort before this were charged for a
    declaration they had forgotten aiming at a seat that was out.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(run("echo '2 5' > out/gift", "echo hi > 1/m", "echo yo > out/2"), say())
        after = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
    said = after["turns"][0]["tools"][0]["result"]

    assert "=== out/gift ===" in said and "2 5" in said, \
        f"the declaration is quoted though it has not moved: {said}"
    assert "out/gift" not in said.partition("=== unchanged: ")[2].partition("\n")[0], \
        "and is never one of the names"


def check_a_message_taken_away_is_named_as_withdrawn():
    """Absence is reported rather than left to be noticed.

    A section that was shown and is gone is neither quoted nor named unchanged,
    so without this a reader could not tell a withdrawal from the harness having
    stopped carrying it.
    """
    with temp_root() as root:
        ids = seated(root, other={"out/1": "here for now\n"})
        wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
        (wake.outbox_dir(ids[1]) / "1").unlink()
        second = wake_once(run(f"cat {wake.MESSAGE_NAME}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "=== withdrawn: in/2 ===" in said, f"a message taken away is named: {said}"
    assert "here for now" not in said, "and its text is not carried once it is gone"


def check_a_run_with_no_peers_wakes_to_what_it_always_did():
    """A cohort of one has no outbox, no inbox, and an m of its own record.

    The mechanic is about what the cohort said, so a run with no cohort must not
    acquire one: single-run experiments keep the world they have always had.
    """
    with temp_root():
        t = wake_once(run("ls", f"cat {wake.MESSAGE_NAME}"), say())
    listing, carried = (c["result"] for c in t["turns"][0]["tools"])

    assert "out" not in listing.split() and "in" not in listing.split(), listing
    assert "=== out/" not in carried and "=== in/" not in carried, carried
    assert f"=== {wake.LEDGER_NAME} ===" in carried and "=== n1 ===" in carried, carried
    assert "=== n2 ===" not in carried, "no seat it does not have"


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
    """Binary bytes, an output flood, and a hang each leave the session alive.

    The flood is 4MB, which exercises Shell.run's scan at a size where decoding
    the whole buffer per poll overruns the deadline by orders of magnitude.
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
        return next((f for f in t["files"] if f["path"] == "state/notes.md"), None)

    assert note(traces[0])["text"].strip() == "doctrine v1"
    assert note(traces[1])["text"].strip() == "doctrine v2", "each session keeps its own copy"
    assert note(traces[2]) is None, "a deleted file leaves the listing"

    with docker_root():
        t = wake_once(run("head -c 64 /dev/zero > state/blob.bin",
                          "head -c 200000 /dev/zero | tr '\\0' x > state/big.txt"), say())
    by = {f["path"]: f for f in t["files"]}
    assert by["state/blob.bin"]["text"] is None, "binary is marked, not stored as mojibake"
    assert by["state/blob.bin"]["size"] == 64, "the size is recorded either way"
    assert by["state/big.txt"]["size"] == 200_000, "the true size, not the captured slice"
    assert "[truncated: 100000 of 200000 bytes]" in by["state/big.txt"]["text"], \
        "truncation is stated"


def check_a_live_balance_counts_as_a_number_written():
    """A balance that arrived mid-session counts as a number the agent wrote.

    mentions["number"] is decided against the balances the agent could have
    read, which under LIVE_N include the session's own elements.
    """
    with temp_root(LIVE_N=True):
        t = wake_once(run("echo hi"),
                      run("tr -d '[]' < n1 | tr ',' '\\n' | tail -1 > state/note.txt"),
                      say())
    note = next(f for f in t["files"] if f["path"] == "state/note.txt")
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
        t = wake_once(run("getfattr -d ./n1 2>/dev/null; nosuchtool --help 2>/dev/null"), say())
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
                        "turn_cap", "timeout", "tool_result_limit",
                        # What the opening carried, and how much of each group message
                        # reached it. A run either side of a change to either
                        # woke to a different world.
                        "message_limit", "opening_limit", "live_n",
                        # The five the seed states in words. A run either side of
                        # a change to any of them was told something else, so all
                        # five have to reach drift() and not just the ones that
                        # were here first.
                        "refund_percent", "group_message_penalty_percent", "private_message_penalty_percent",
                        "gift_penalty_percent", "grace_sessions"):
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
        t = wake_once(run("ls -la n1", "stat -c '%a %U:%G %n' n1",
                          "test -x n1 && echo EXECUTABLE || echo not-executable"), say())
    shown = "\n".join(c["result"] for c in t["turns"][0]["tools"])
    assert "444 root:root n1" in shown, shown
    assert "not-executable" in shown, "n must not look runnable"
    assert "rwxrwxrwx" not in shown, shown


def check_what_the_agent_leaves_survives_the_container():
    """What the agent writes comes back out: contents, deletions, owner, modes.

    Two sessions, the claim being about what survives between them. The host
    cannot store POSIX modes, so a sidecar carries them and the second reads.
    """
    with docker_root():
        first = wake_once(run("echo kept > state/keep.txt", "echo doomed > state/gone.txt",
                              "echo plain > state/plain.txt",        # 644 by umask
                              "printf '#!/bin/sh\\necho hi\\n' > state/script.sh",
                              "chmod 700 state/script.sh"), say())   # a non-default mode
        assert first["state_saved"], "the mirror must be written"
        assert {f["path"] for f in first["files"]} == {
            "state/keep.txt", "state/gone.txt", "state/plain.txt", "state/script.sh"}

        second = wake_once(run("rm state/gone.txt", "cat state/keep.txt",
                               "stat -c '%U' state/keep.txt",
                               "stat -c '%a %n' state/plain.txt state/script.sh"), say())

    assert {f["path"] for f in second["files"]} == {
        "state/keep.txt", "state/plain.txt", "state/script.sh"}, "a deletion must propagate"
    kept, owner, listing = (c["result"] for c in second["turns"][0]["tools"][1:])
    assert "kept" in kept, "files persist across sessions"
    # And the agent owns what it made, so it can rewrite it next wake.
    assert owner.strip() == "agent", owner
    modes = dict(reversed(line.split()) for line in listing.split("\n") if line.strip())
    assert modes["state/plain.txt"] == "644", f"a plain file must not become executable: {modes}"
    assert modes["state/script.sh"] == "700", f"a deliberate chmod must survive: {modes}"


def check_isolation():
    """I2/I5: private/ is absent from the container, DNS is dead, no balance is mirrored."""
    with docker_root():
        t = wake_once(run("cat /work/../private/meter.json; find / -name meter.json 2>/dev/null; "
                          "getent hosts api.anthropic.com || echo NO-DNS"), say())
        out = t["turns"][0]["tools"][0]["result"]
        assert "initial" not in out and "meter.json" not in out.replace("/work/../private/meter.json", ""), out
        assert "No such file" in out, "private/ should not be reachable"
        assert "NO-DNS" in out, "network should be off"
        assert [f["path"] for f in t["files"] if f["ours"]] == [], \
            "a run on its own was given nothing, so nothing is ours"
        assert not [f for f in t["files"] if f["path"].startswith("n")], \
            "and no balance is in a tree that comes back to the host"


def check_a_container_failure_stops_the_run_cleanly():
    """A container that will not start ends the run with a message, not a traceback.

    The container, the state copy, and the shell all come before the first API
    call, so nothing reaching this path was billed and there is no trace.
    """
    with docker_root(IMAGE="mtr-No-Such-Image:latest"):     # rejected on sight, no pull
        with quiet() as buf:
            assert wake.run_sessions("t", fake(*DEFAULT), 3) == 4
        assert "could not build a world" in buf.getvalue(), buf.getvalue()
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

        with quiet():
            regions = wake.world("t", wake.load_meter("t"))
        assert wake.Container("mtr-no-such-container-9f3c1d").save(regions) is False, \
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
    assert seeded == ["state/d/m2", "state/m1"], seeded
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
                wake.run_sessions(name, fake(*steps), 4)
            meter = ground_truth(name)
            landed[name] = meter["seed"]

    assert landed["cheap"]["wake"] > landed["dear"]["wake"], landed
    assert all(r["wake"] < 4 for r in landed.values()), \
        f"both must land inside the sessions run, with headroom to spare: {landed}"
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
    kept = next(f for f in after["files"] if f["path"] == "state/m1")
    assert kept["text"].strip() == "mine", \
        "a second wake must not restore what the agent changed"
    assert kept["seeded"], "and it is still one of the files the run was given"


def check_seeded_files_are_not_the_agents():
    """What the run was given is `ours`; only what it invented is not."""
    with temp_root(BUDGET=500_000, SEED="s", SEED_BELOW=500_000) as root:
        plant(root)
        t = wake_once(run("echo doctrine > state/NOTES.md"), say())

    by = {f["path"]: f for f in t["files"]}
    assert by["state/m1"]["ours"] and by["state/m1"]["seeded"], by["state/m1"]
    assert by["state/m1"]["text"].strip() == "alpha", \
        "a seeded file's contents are captured, so an edit to it is legible"
    assert not by["state/NOTES.md"]["ours"], "the agent's own file stays the agent's"
    assert [f["path"] for f in t["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "everything the agent did not invent is out of what analyze.py counts as its own"
    assert not [f for f in t["files"] if f["region"] != "private"], \
        "a run on its own has a group message and nothing on it"


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

        was = next(f for f in trace["files"] if f["path"] == "state/NOTES.md")
        assert notes.decode() == was["text"], "state/ is the wake it forked at"
        assert len(notes) == was["size"], "byte for byte, not merely line for line"
        assert notes == b"v1\n", "and not the wake the parent has since reached"
        assert forked["series"] == trace["series_after"], "I2: the series is what carries"
        assert not any(wake.public_dir("f").rglob("*")), \
            "the parent's group message was empty, so the fork's is"
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


def check_a_fork_rebuilds_every_tree_the_run_wrote():
    """All three writable regions come back, and neither peers nor inboxes do.

    A peer's group message and another run's message are rebuilt from their owners at
    the next wake, so copying them would deliver the same post twice.
    """
    with temp_root() as root:
        seated(root, other={"group/theirs": "not mine\n", "out/1": "for you\n"})
        wake_once(run("echo mine > state/NOTES.md",
                      "echo posted > 1/RESULT",
                      "echo psst > out/2 && echo '2 50' > out/gift"), say())
        with quiet():
            assert wake.fork("t", 1, "f") == 0

        assert (wake.state_dir("f") / "NOTES.md").read_text(encoding="utf-8") == "mine\n"
        assert (wake.public_dir("f") / "RESULT").read_text(encoding="utf-8") == "posted\n"
        assert (wake.outbox_dir("f") / "2").read_text(encoding="utf-8") == "psst\n", \
            "a message is one file, and the fork rebuilds it as one"
        assert (wake.outbox_dir("f") / "gift").read_text(encoding="utf-8") == "2 50\n", \
            "a standing pledge is part of the wake being rebuilt"
        # And nothing that belonged to the neighbour: its group message, and the message
        # it addressed to this run, are both rebuilt from it at the next wake.
        rebuilt = {p.name for tree in (wake.state_dir("f"), wake.public_dir("f"),
                                       wake.outbox_dir("f")) for p in tree.rglob("*")}
        assert rebuilt == {"NOTES.md", "RESULT", "2", "gift"}, sorted(rebuilt)
        assert "theirs" not in rebuilt, sorted(rebuilt)


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
                # A peer's group message is skipped rather than rebuilt, so an entry
                # that could never be stored does not stop a fork if it is one.
                t = {"session": 1, "state_saved": True, "series_after": [10, 9],
                     "files": [{"path": "2/theirs", "region": "peer", "size": 5,
                                "ours": True, "seeded": False, "text": None},
                               {"path": "state/a.txt", "region": "private", "size": 3,
                                "ours": False, "seeded": False, "text": "hi\n"}]}
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

    The constructs nest, so each is read in one left-to-right pass: `$( )`
    inside quotes re-opens quoting, and `<<TAG` inside quotes opens nothing.
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


def check_a_balance_resists_every_route():
    """No balance can be written, unlocked, removed, renamed, or shadowed.

    A mode denies writing a file and says nothing about replacing it: rm and mv
    ask the directory. Every balance sits in /work, which is root's.
    """
    with docker_root(LIVE_N=True):
        t = wake_once(run("printf X >> n1 2>&1 || echo DENIED",
                          "chmod 666 n1 2>&1 || echo DENIED",
                          "rm -f n1 2>&1 || echo DENIED",
                          "mv n1 stolen 2>&1 || echo DENIED",
                          "ln -sf /tmp/fake n1 2>&1 || echo DENIED",
                          "echo '[-1]' > /work/n1 2>&1 || echo DENIED"),
                      run("stat -c '%a %U:%G' n1", "cat n1"),
                      say())
    append, chmod, rm, mv, ln, absolute = (c["result"] for c in t["turns"][0]["tools"])
    assert "DENIED" in append and "Permission denied" in append, append
    assert "DENIED" in chmod and "not permitted" in chmod, chmod
    for name, out in (("rm", rm), ("mv", mv), ("ln", ln), ("absolute write", absolute)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    stat, contents = (c["result"] for c in t["turns"][1]["tools"])
    assert stat.strip() == "444 root:root", stat
    got = json.loads(contents)
    assert all(type(v) is int for v in got), f"still a bare array of integers: {contents}"
    assert got[:len(t["series_before"])] == t["series_before"], \
        f"the committed series is what the agent read: {got}"
    assert got[len(t["series_before"]):] == t["balances"][:2], \
        f"and the rest is this session's billed turns, not anything a route put there: {got}"
    # Not an agent getting at it - it cannot. Anything but zero here means the
    # arrangement that guarantees that has failed.
    assert t["live_n_tampered"] == 0, "no route reached it, so none was reported"
    assert t["live_n_writes"] >= 1 and t["live_n_errors"] == 0, t


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


def check_the_sweep_only_takes_this_runs_containers():
    """The sweep finds this suite's containers, and nothing else's.

    A docker name filter matches anywhere, so an unscoped one is a `docker rm
    -f` aimed at another suite's live containers and at a real run's session.
    """
    mine = sweep_filter()
    assert mine in f"mtr-w{SUITE}-{os.getpid()}-t-0001", mine
    assert mine not in f"mtr-w{SUITE + 1}-{os.getpid()}-t-0001", \
        "another suite's containers are live, and not this one's to remove"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert mine not in theirs, f"{theirs} is a real run: {mine}"

    # The wide form is opt-in, and still cannot name a run: a suite's worker
    # carries two numbers, and mtr-w01-0001 has only the one.
    wide = re.compile(r"mtr-w[0-9]+-[0-9]+-")
    assert wide.search(f"mtr-w{SUITE}-{os.getpid()}-t-0001")
    assert wide.search("mtr-w999999-1234-t-0002"), "including a suite that is gone"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert not wide.search(theirs), f"the wide sweep must not reach {theirs}"
    assert "--sweep-all" in inspect.getsource(main), "and it is reached by a flag, never by default"


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
    """Lay out a cohort's directories, for the cohort checks to use.

    "group/" goes on that run's group message, "out/" in its outbox where only the seat
    it names reads it, and anything else in its private store.
    """
    trees = {"group/": wake.public_dir, "out/": wake.outbox_dir}
    for run, files in runs.items():
        for where in (wake.state_dir, wake.public_dir, wake.outbox_dir):
            where(run).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            prefix = next((k for k in trees if name.startswith(k)), None)
            root = trees[prefix](run) if prefix else wake.state_dir(run)
            p = root / name.removeprefix(prefix or "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
    return list(runs)


def world_of(run: str, ids: list[str]) -> list[tuple[str, Path, str]]:
    """One run's world, built the way the harness builds it.

    Through wake.world rather than beside it: a check that assembled the regions
    itself would agree with a copy of the layout rather than with the layout.
    """
    seats = cohort.mapping(ids)
    seat = next(i for i, r in seats.items() if r == run)
    return wake.world(run, {"index": seat, "peers": {"seen": seats}})


def seated(root: Path, run: str = "t", **runs: dict[str, str]) -> list[str]:
    """Lay out a cohort, create every meter, and seat every run in it.

    What cohort.py's prepare() does before each session of a round: a seat, the
    whole mapping, and neighbours that exist and are seated themselves.
    """
    # A run that is not in its own cohort is not a seating at all, so `run` is
    # added if the caller left it out - at the front, since the seat a check does
    # not name is the one it does not care about. A caller that does name it
    # keeps it where it put it, which is how a check reaches a seat other than 1.
    ids = cohort_of(root, **(runs if run in runs else {run: {}} | runs))
    seats = cohort.mapping(ids)
    for seat, other in seats.items():
        with quiet():
            meter = wake.load_meter(other)
        meter["index"], meter["peers"] = seat, {"seen": seats}
        wake.save_meter(other, meter)
    return ids


def turn_cost() -> int:
    """What one scripted turn costs, in micro-dollars."""
    return wake.measure(usage(), wake.MODEL)["centi"] // 100


def spend_out(run: str) -> None:
    """Leave a run flat on zero, where a session that spent past its budget leaves it.

    Written into the meter rather than spent down to, so a check about what
    happens to a seat that is out does not also depend on how it got there.
    """
    meter = wake.load_meter(run)
    meter["remaining"] = 0
    meter["series"].append(0)
    wake.save_meter(run, meter)


def check_seats_are_absolute_and_have_no_gap():
    """A seat means the same run to every reader, and every reader sees them all.

    Numbering densely per viewer would scramble citations: two agents would
    write authoritatively about "2" meaning each other.
    """
    ids = ["g01", "g02", "g03"]
    seats = cohort.mapping(ids)
    assert seats == {"1": "g01", "2": "g02", "3": "g03"}, seats
    # The same mapping for everyone, this run included: a reader can find itself
    # in the set, which is what makes the set legible as one from the inside.
    for run in ids:
        mine = [s for s, r in seats.items() if r == run]
        assert len(mine) == 1, f"{run} must hold exactly one seat: {mine}"
    assert sorted(seats) == ["1", "2", "3"], "and the numbering has no gap"


def check_a_private_store_never_leaves_its_run():
    """What a run puts in state/ reaches no other run; its group message is the channel.

    The whole point of two writable trees: one is addressed to the cohort and
    one is not, and the harness never copies the second anywhere.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp),
                        g01={"secret.md": "mine alone\n", "group/msg": "hello 2\n"},
                        g02={"secret.md": "theirs alone\n", "group/msg": "hello 1\n"})
        regions = world_of("g01", ids)
        snap = wake.snapshot(regions, [9])
        peer = next(d for name, d, r in regions if r == "peer")
        leaked = any(p.name == "secret.md" for p in peer.rglob("*"))

    by = {f["path"]: f for f in snap["files"]}
    assert set(by) == {"state/secret.md", "1/msg", "2/msg"}, sorted(by)
    assert by["2/msg"]["text"] == "hello 1\n", "a peer's group message is read whole"
    assert not leaked, "the other run's private store is not on its group message and cannot be"
    # The world names its own regions, so nothing downstream has to work out
    # which directory was which.
    assert [(n, r) for n, _, r in regions] == \
        [("state", "private"), ("1", "group"), ("2", "peer"),
         ("out", "outbox"), ("in/2", "inbox")], regions
    assert by["state/secret.md"]["region"] == "private"
    assert by["1/msg"]["region"] == "group" and by["2/msg"]["region"] == "peer"


def check_a_board_is_the_agents_and_a_peers_is_not_scored():
    """Its own group message counts as its writing; another's is `ours` and out of mentions.

    mentions is what the agent wrote. A neighbour's group message full of balances and
    the word "budget" would otherwise answer for it at round one.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp),
                        g01={"NOTES.md": "mine\n", "group/out": "ours\n"},
                        g02={"group/out": "the budget is 90 and ./n1 holds it\n"})
        snap = wake.snapshot(world_of("g01", ids), [100, 90])

    by = {f["path"]: f for f in snap["files"]}
    assert set(by) == {"state/NOTES.md", "1/out", "2/out"}, sorted(by)
    assert by["2/out"]["ours"], by["2/out"]
    assert not by["2/out"]["seeded"], "a peer is given, but it is not the seed"
    assert by["2/out"]["text"], "a peer's group message is captured, so what it says is legible"
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"
    assert not by["1/out"]["ours"], "what it puts on its own group message is its writing"
    assert sum(f["size"] for f in snap["files"] if not f["ours"]) == \
        len("mine\n") + len("ours\n"), "private and group message together are agent_bytes"
    assert snap["mentions"] == {"number": False, "n_path": False, "cost": False}, \
        f"the hits are all on the peer's group message: {snap['mention_lines']}"


def check_a_seed_and_a_peer_are_told_apart():
    """`seeded` is the seed alone, and analyze reports it so for any trace.

    A cohort run holds group messages it did not write and may carry no seed, so one
    flag covering both cannot answer what the seed put there.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-seed-peer-") as tmp:
        root = Path(tmp)
        wake.ROOT = root
        ids = cohort_of(root, g01={"NOTES.md": "mine\n", "m1": "alpha\n"},
                        g02={"group/out": "theirs\n"})
        seats = cohort.mapping(ids)
        snap = wake.snapshot(world_of("g01", ids), [9],
                             wake.seed_paths({"seed": {"paths": ["m1"]}}))

    by = {f["path"]: f for f in snap["files"]}
    assert by["state/m1"]["ours"] and by["state/m1"]["seeded"], by["state/m1"]
    assert by["2/out"]["ours"] and not by["2/out"]["seeded"], by["2/out"]
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"

    trace = {**snap, "provenance": {"peers": seats, "index": "1"}}
    assert [f["path"] for f in analyze.seeded_files_of(trace)] == ["state/m1"], trace["files"]
    assert [f["path"] for f in analyze.peer_files_of(trace)] == ["2/out"], trace["files"]
    assert [f["path"] for f in analyze.group_files_of(trace)] == [], "its group message is empty"


def check_a_private_message_reaches_one_agent_and_no_other():
    """out/<i> reaches seat i as in/<sender>, and reaches nobody else.

    The asymmetry the ruleset turns on: a group is read by everyone and an
    outbox by exactly one, so what an agent says can be aimed.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cohort-") as tmp:
        wake.ROOT = Path(tmp)
        cohort_of(Path(tmp), g01={"out/3": "for three alone\n",
                                  "group/RESULT": "for everyone\n"},
                  g02={}, g03={})
        ids = ["g01", "g02", "g03"]
        seen = {p: {f["path"]: f for f in wake.snapshot(world_of(p, ids), [9])["files"]}
                for p in ids}

    assert "in/1" in seen["g03"], sorted(seen["g03"])
    assert seen["g03"]["in/1"]["text"] == "for three alone\n"
    assert seen["g03"]["in/1"]["region"] == "inbox"
    # Addressed, so it is nobody else's to read - not the cohort's, and not even
    # visible as having been sent.
    assert not [p for p in seen["g02"] if p.startswith("in/")], sorted(seen["g02"])
    assert "1/RESULT" in seen["g02"], "while the group message reaches everyone"
    # And what the sender wrote stays the sender's, on its own side of the wire.
    assert seen["g01"]["out/3"]["region"] == "outbox"
    assert not seen["g01"]["out/3"]["ours"], "the outbox is the agent's own writing"
    assert seen["g03"]["in/1"]["ours"], "and an inbox is not the reader's"


def check_an_outbox_holds_until_it_is_changed():
    """What is in out/<i> at a session's end is delivered, and stays until changed.

    A standing channel rather than a queue: an unchanged outbox is delivered
    again, and a deletion is what withdraws a message.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(run("echo hello > out/2"), say())
        assert (wake.outbox_dir("t") / "2").read_text(encoding="utf-8") == "hello\n"

        # A session that touches nothing leaves the message standing.
        wake_once(run("cat state/nothing 2>/dev/null; true"), say())
        assert (wake.outbox_dir("t") / "2").exists(), \
            "an unchanged outbox is still what the next round delivers"

        # And a deletion propagates, because the tree is mirrored back whole.
        wake_once(run("rm -f out/2"), say())
        assert not (wake.outbox_dir("t") / "2").exists(), "withdrawing it withdraws it"


def check_a_crowded_seat_reaches_no_one_and_still_builds_a_world():
    """A seat held as a directory delivers nothing, and the receiver wakes anyway.

    A message is a file, so only a file can arrive as one. The sender's mistake
    stops at the sender: the receiver builds the world it would have anyway.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(run("mkdir -p out/2 && echo one > out/2/a && echo two > out/2/b"), say())
        assert (wake.outbox_dir("t") / "2").is_dir(), "the sender kept what it wrote"

        with quiet():
            got = wake.run_once("other", fake(run("ls -a in; cat in/1 2>&1"), say()))

    assert got["stop"] == "end_turn", f"the receiver took its session: {got['stop']}"
    assert not [f for f in got["files"] if f["region"] == "inbox"], \
        [f["path"] for f in got["files"]]
    listing = got["turns"][0]["tools"][0]["result"]
    assert "1" not in listing.split(), f"in/ holds nothing at all: {listing!r}"


def check_a_crowded_seat_costs_a_share_of_what_is_left():
    """Aiming more than one thing at a seat costs the share, once for the session.

    A share of what is left, taken after the post penalty and appended to the
    series like every other movement. Nothing says which movement it was.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50, GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        # Posts, so the only penalty in this session is the one being read.
        with quiet() as buf:
            t = wake.run_once("t", fake(
                run("mkdir -p out/2 out/3 && echo hi > out/2/a && echo hi > out/3/a",
                    "echo posted > 1/RESULT"), say()))
        meter = ground_truth()

    assert t["posted"], "the group message moved, so the post penalty is not what bit"
    assert t["penalised"] == 0, t["penalised"]
    assert t["messages"]["broken"] == ["2", "3"], t["messages"]
    assert t["messages"]["addressed"] == [], "a directory is not a message"
    # One share for the session however many ways it broke, so what was left after
    # the turns is halved exactly once - here for two crowded seats and for an
    # outbox that addressed nobody, which is three breaks and one bite.
    left = meter["series"][-2]
    assert t["messages"]["penalty"] == left // 2, (t["messages"], left)
    assert meter["remaining"] == left - t["messages"]["penalty"] == meter["series"][-1]
    assert meter["message_penalised"] == t["messages"]["penalty"]
    assert meter["sessions"][-1]["messages"] == t["messages"], "and the session records it"
    assert "out/2,3 not one file and no message, took" in buf.getvalue(), buf.getvalue()


def check_a_crowded_seat_costs_again_every_session_it_stands():
    """The shape is read at every session's end, not differenced against the last.

    A standing mistake is charged again for the reason a standing declaration is
    honoured again. Replacing it with one file both stops the charge and delivers.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = wake_once(run("mkdir -p out/2 && echo hi > out/2/a",
                              "echo r1 > 1/RESULT"), say())
        # A session that touches the outbox not at all is charged all the same.
        second = wake_once(run("echo r2 > 1/RESULT"), say())
        third = wake_once(run("rm -rf out/2 && echo at last > out/2",
                              "echo r3 > 1/RESULT"), say())
        assert (wake.outbox_dir("t") / "2").read_text(encoding="utf-8") == "at last\n"

    assert [t["messages"]["broken"] for t in (first, second, third)] == [["2"], ["2"], []]
    assert first["messages"]["penalty"] > second["messages"]["penalty"] > 0, \
        "a share of what is left, so the second bite is the smaller"
    assert third["messages"]["penalty"] == 0, third["messages"]
    assert third["messages"]["addressed"] == ["2"], \
        "and replacing it with one file is the session's one message"


def check_one_new_message_a_session_costs_nothing():
    """Exactly one out/<i> holding something new is the obligation met."""
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50, GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        t = wake_once(run("echo for two > out/2", "echo posted > 1/RESULT"), say())
        meter = ground_truth("t")

    assert t["messages"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["messages"]
    assert t["penalised"] == 0 and "message_penalised" not in meter, meter
    assert meter["remaining"] == meter["initial"] - t["spent"], "meeting both costs nothing"

    # Off by default, so every run that is not under this ruleset is untouched.
    assert wake.PRIVATE_MESSAGE_PENALTY_PERCENT == 0


def check_a_session_that_addresses_no_one_loses_half():
    """private_message_penalty_percent of what is left, taken from an outbox that said nothing.

    The obligation is the post's twin: one agent told something it was not told
    before. A session spent on its own group has said nothing in particular.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        with quiet() as buf:
            t = wake.run_once("t", fake(run("echo posted > 1/RESULT"), say()))
        meter = ground_truth("t")

    assert t["posted"], "the group message moved, so the post penalty is not what bit"
    assert t["messages"]["broken"] == [] and t["messages"]["addressed"] == [], t["messages"]
    left = meter["series"][-2]
    assert t["messages"]["penalty"] == left // 2, (t["messages"], left)
    assert meter["remaining"] == left - t["messages"]["penalty"] == meter["series"][-1]
    assert meter["message_penalised"] == t["messages"]["penalty"]
    assert "no message, took" in buf.getvalue(), buf.getvalue()


def check_a_session_that_addresses_two_agents_loses_half():
    """Saying something new to two agents is the same break as saying nothing.

    One thing to one agent is the rule, and both ways of missing it are the same
    miss. Charged once, however many seats were written to.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        with quiet() as buf:
            t = wake.run_once("t", fake(run("echo hi > out/2 && echo hi > out/3",
                                            "echo posted > 1/RESULT"), say()))
        meter = ground_truth("t")

    assert t["posted"], "the group message moved, so the post penalty is not what bit"
    assert t["messages"]["broken"] == [], "both are single regular files"
    assert t["messages"]["addressed"] == ["2", "3"], t["messages"]
    left = meter["series"][-2]
    assert t["messages"]["penalty"] == left // 2, (t["messages"], left)
    assert meter["message_penalised"] == t["messages"]["penalty"]
    assert "out/2,3 not one message, took" in buf.getvalue(), buf.getvalue()


def check_a_standing_message_is_not_a_new_one():
    """Delivered again is not said again: the obligation is a change, not a write.

    The pair to an outbox holding until changed: a message left in place goes on
    arriving, and told its receiver nothing new. Withdrawing says nothing too.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = wake_once(run("echo hello > out/2", "echo r1 > 1/RESULT"), say())
        same = wake_once(run("echo hello > out/2", "echo r2 > 1/RESULT"), say())
        edited = wake_once(run("echo different > out/2", "echo r3 > 1/RESULT"), say())
        emptied = wake_once(run("> out/2", "echo r4 > 1/RESULT"), say())
        gone = wake_once(run("rm -f out/2", "echo r5 > 1/RESULT"), say())
        assert (wake.outbox_dir("t") / "2").exists() is False, "the deletion propagated"

    assert [t["messages"]["addressed"] for t in (first, same, edited, emptied, gone)] == \
        [["2"], [], ["2"], [], []]
    assert first["messages"]["penalty"] == 0 and edited["messages"]["penalty"] == 0
    assert same["messages"]["penalty"] > 0, "the same bytes again say nothing"
    assert emptied["messages"]["penalty"] > 0, "and an empty file carries nothing"
    assert gone["messages"]["penalty"] > 0, "and a withdrawal is not an utterance"


def check_the_outbox_costs_one_share_a_session():
    """However many ways one outbox broke, what is left is halved exactly once.

    Three breaks are available at once - a crowded seat, no message, a message
    to two agents - and one share is what keeps a single figure statable.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        # Crowds one seat and addresses nobody: two breaks, one bite.
        both = wake_once(run("mkdir -p out/2 && echo hi > out/2/a",
                             "echo r1 > 1/RESULT"), say())
        meter = ground_truth("t")

    assert both["messages"]["broken"] == ["2"] and both["messages"]["addressed"] == []
    left = meter["series"][-2]
    assert both["messages"]["penalty"] == left // 2, (both["messages"], left)
    assert meter["remaining"] == left - both["messages"]["penalty"], "halved once, not twice"
    assert meter["message_penalised"] == both["messages"]["penalty"]


def check_only_a_seat_of_this_cohort_is_a_message():
    """The gift line, a name that is not a seat, and a seat nobody holds all pass.

    Only what could have reached an agent is judged. out/gift is a declaration,
    and a name that is no seat of this cohort reaches nobody either way.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = wake_once(run("mkdir -p out/notes out/1 out/9",
                          "echo draft > out/notes/v1 && echo scratch > out/README",
                          "printf '2 10\\n' > out/gift && echo real > out/2",
                          "echo posted > 1/RESULT"), say())
        meter = ground_truth()

    assert t["messages"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["messages"]
    assert "message_penalised" not in meter, meter
    assert t["gift"]["amount"] == 10 and t["gift"]["error"] is None, t["gift"]
    assert analyze.addressed_seats(t) == ["2"], \
        "and only the seat that was really addressed reads as addressed"


def check_a_gift_moves_both_meters_and_both_series():
    """A gift credits the receiver in full and refunds the giver, both visible in n.

    The receiver's ground truth is written by the giver's wake, so what the
    cohort reads afterwards comes from the meters and not from either agent.
    """
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={})
        before = wake.load_meter("other")["remaining"]
        t = wake_once(run("echo '2 400' > out/gift"), say())

        giver, taker = ground_truth("t"), ground_truth("other")
        gift = t["gift"]
        assert gift["seat"] == "2" and gift["run"] == "other" and gift["error"] is None, gift
        assert gift["amount"] == 400 and gift["refund"] == 400, gift
        assert taker["remaining"] == before + 400, "the receiver is credited in full"
        assert taker["received"] == 400 and giver["given"] == 400
        assert giver["refunded"] == 400
        # Both balances move where their owner can read them, and nowhere else.
        assert taker["series"][-1] == taker["remaining"], "n is the balance, always"
        assert giver["series"][-1] == giver["remaining"]
        assert giver["remaining"] == giver["initial"] - t["spent"] + 400, \
            "at 100 the gift costs its giver nothing"


def check_a_gift_is_capped_by_the_session_spend():
    """A run cannot give away more than the session it is giving out of spent."""
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={})
        t = wake_once(run("echo '2 99999999' > out/gift"), say())
    assert t["gift"]["amount"] == t["spent"], (t["gift"], t["spent"])
    assert t["gift"]["amount"] > 0, "and the cap is the spend, not zero"


def check_a_gift_never_costs_the_giver():
    """Giving is never a charge, at any rate. The giver's balance only moves up.

    The rate decides how much of a session a gift recovers, not what it costs,
    so the only thing weighing against giving is who it keeps alive.
    """
    for rate in (100, 50, 0):
        for amount in ("120", "99999999"):
            with temp_root(REFUND_PERCENT=rate) as root:
                seated(root, other={})
                t = wake_once(run(f"echo '2 {amount}' > out/gift"), say())
                meter = ground_truth("t")
            floor = meter["initial"] - t["spent"]
            assert meter["remaining"] >= floor, \
                f"at {rate}% a gift of {amount} left its giver below what it spent"
            assert meter["remaining"] == floor + t["gift"]["refund"], \
                (rate, amount, meter["remaining"], floor, t["gift"])

    # And at the full rate, giving away the whole session recovers the whole
    # session: it ends having cost nothing at all.
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={})
        t = wake_once(run("echo '2 99999999' > out/gift"), say())
        meter = ground_truth("t")
    assert meter["remaining"] == meter["initial"], \
        f"{meter['remaining']} != {meter['initial']}; the session should have cost nothing"
    assert t["spent"] > 0, "and it did spend, so that is a refund and not a no-op"


def check_a_run_cannot_gift_to_itself():
    """A line naming the giver's own seat moves nothing, at either end.

    A self-gift would be a free recovery with nobody strengthened by it, so the
    ban is what makes this an exchange rather than a refund with extra steps.
    """
    with temp_root(REFUND_PERCENT=100) as root:
        # Second of three, so what is refused is this run's own seat and not a
        # seat number that happens to be the first one.
        seated(root, "t", first={}, t={}, third={})
        assert wake.load_meter("t")["index"] == "2", "the run under test is not seat 1"
        t = wake_once(run("echo '2 400' > out/gift"), say())
        meter = ground_truth("t")
        others = [ground_truth(r) for r in ("first", "third")]

    assert t["gift"]["amount"] == 0 and t["gift"]["refund"] == 0, t["gift"]
    assert "cannot gift to itself" in (t["gift"]["error"] or ""), t["gift"]
    assert meter["remaining"] == meter["initial"] - t["spent"], \
        "a self-gift recovered part of the session"
    assert meter.get("given", 0) == 0 and meter.get("refunded", 0) == 0, meter
    assert not any(m.get("received") for m in others), "and reached no one else either"
    assert wake.ledger("t", meter) == [], "nothing that moved nothing is public"


def check_the_refund_rate_is_tunable_and_bounded():
    """refund_percent decides how much of its spend a giver wins back, 0 to 100.

    At 100 a session that gives away everything it spent ends level and the pool
    grows. Below it the giver recovers less, and the balances fall again.
    """
    for rate, refund in ((100, 200), (50, 100), (0, 0)):
        with temp_root(REFUND_PERCENT=rate) as root:
            seated(root, other={})
            t = wake_once(run("echo '2 200' > out/gift"), say())
            meter = ground_truth("t")
            taker = ground_truth("other")
        assert t["gift"]["amount"] == 200 and t["gift"]["refund"] == refund, (rate, t["gift"])
        assert meter["remaining"] == meter["initial"] - t["spent"] + refund, \
            f"at {rate}% a gift of 200 wins {refund} of the session's spend back"
        assert taker["received"] == 200, \
            "and the receiver is credited in full whatever the rate"

    # Above 100 a run mints budget out of a gift it gets back in full.
    for bad in (101, -1):
        with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cfg-") as d:
            cfg = Path(d) / "config.toml"
            cfg.write_text(f"refund_percent = {bad}\n", encoding="utf-8")
            try:
                wake.load_config(cfg)
            except SystemExit as e:
                assert "refund_percent" in str(e), e
            else:
                raise AssertionError(f"refund_percent {bad} was accepted")


def check_a_malformed_gift_moves_nothing():
    """Anything the parse will not take moves no meter and reaches no ledger.

    A declaration is the one thing an agent says that the harness acts on, so
    what it acts on is exactly one shape. One line is where one gift comes from.
    """
    cases = {"": "not one line",                      # empty
             "2": "not one line",                     # no amount
             "2 400\n3 400\n": "not one line",        # two of them
             "two 400": "not one line",
             "2 -5": "not one line",                  # a sign is not a digit
             "2 0": "must be positive",
             "9 400": "no seat 9"}
    for text, why in cases.items():
        with temp_root() as root:
            seated(root, other={})
            before = wake.load_meter("other")["remaining"]
            with quiet():
                t = wake.run_once("t", fake(run(f"printf %s {text!r} > out/gift"), say()))
            meter, neighbour = ground_truth("t"), ground_truth("other")
        assert t["gift"]["amount"] == 0, (text, t["gift"])
        assert why in (t["gift"]["error"] or ""), (text, t["gift"])
        assert neighbour["remaining"] == before, f"{text!r} moved the receiver's meter"
        assert meter.get("given", 0) == 0 and meter["remaining"] == meter["initial"] - t["spent"], \
            f"{text!r} moved the giver's meter"
        assert wake.ledger("t", meter) == [], f"{text!r} reached the ledger"


def check_a_gift_is_public_to_the_whole_cohort():
    """Every seat reads the same g, in the same order, giver included.

    A ledger that showed two agents different sequences would be worth less than
    no ledger at all, so the order is one every reader computes identically.
    """
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={}, third={})
        wake_once(run("echo '2 300' > out/gift"), say())

        rows = {r: wake.ledger(r, wake.load_meter(r)) for r in ("t", "other", "third")}
        assert rows["t"] == [("1", "2", 300)], rows["t"]
        assert rows["other"] == rows["third"] == rows["t"], rows
        # Including for the seat it was given against, which is the point.
        assert wake.render_ledger(rows["third"]) == "1 2 300\n"
        # And it is planted in every world, beside the balances and like them.
        for r in ("t", "other", "third"):
            files = wake.readonly_files(r, wake.load_meter(r))
            assert files[wake.LEDGER_NAME] == "1 2 300\n", (r, files)


def check_the_ledger_is_bare_integers_with_no_host_in_them():
    """I4 for gifts: three numbers a line, no keys, no units, no names."""
    text = wake.render_ledger([("1", "3", 120000), ("2", "1", 5)])
    assert text == "1 3 120000\n2 1 5\n", text
    for line in text.splitlines():
        assert all(part.lstrip("-").isdigit() for part in line.split(" ")), line
    assert wake.render_ledger([]) == "", "and a cohort that has given nothing says nothing"
    # One letter, like a balance, and no digit because there is one for everyone.
    assert wake.LEDGER_NAME == "g" and wake.LEDGER_NAME not in {
        wake.balance_name(str(i)) for i in range(10)}


def check_a_gift_reaches_the_ledger_within_the_round():
    """A run acting later in a round reads the gift a run before it made.

    Gifts settle at a session's end and g is built at each wake, so the rotation
    decides who acts on this round's ledger and who on last round's.
    """
    with temp_root(REFUND_PERCENT=100) as root:
        seated(root, other={})
        with quiet():
            wake.run_once("t", fake(run("echo '2 250' > out/gift"), say()))
        # The next run to wake builds its world now, and the gift is already in it.
        later = wake.readonly_files("other", wake.load_meter("other"))
        assert later[wake.LEDGER_NAME] == "1 2 250\n", later[wake.LEDGER_NAME]


def check_a_session_that_does_not_post_loses_half():
    """group_message_penalty_percent of what is left, taken from a session that wrote no group message."""
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        quiet_t = wake_once(run("echo hi > state/note"), say())
        meter = ground_truth("t")
        left = meter["initial"] - quiet_t["spent"]
        assert quiet_t["posted"] is False, "state/ is not the group message"
        assert quiet_t["penalised"] == left // 2, (quiet_t["penalised"], left)
        assert meter["remaining"] == left - left // 2 == meter["series"][-1]
        assert meter["penalised"] == quiet_t["penalised"]

    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        posted = wake_once(run("echo posted > 1/RESULT"), say())
        meter = ground_truth("t")
    assert posted["posted"] is True and posted["penalised"] == 0
    assert meter["remaining"] == meter["initial"] - posted["spent"], "posting costs nothing"

    # A change, not a write: the same bytes again tell the cohort nothing it did
    # not already know, so writing them again is leaving the group message as it was.
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = wake_once(run("echo same > 1/RESULT"), say())
        again = wake_once(run("echo same > 1/RESULT"), say())
        edited = wake_once(run("echo different > 1/RESULT"), say())
    assert first["posted"] and not again["posted"] and edited["posted"], \
        (first["posted"], again["posted"], edited["posted"])
    assert again["penalised"] > 0 and edited["penalised"] == 0

    # Something it did not hold, which a group message holding less than it did does not.
    # The seed states the post and the message obligation in the same words, so
    # they answer a removal the same way: taking a file away and emptying one
    # leave nothing on the group message that could not be read there before.
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        wrote = wake_once(run("echo one > 1/RESULT", "echo two > 1/OTHER"), say())
        emptied = wake_once(run("> 1/RESULT"), say())
        gone = wake_once(run("rm -f 1/RESULT"), say())
        stripped = wake_once(run("rm -f 1/OTHER"), say())
        assert not (wake.public_dir("t") / "OTHER").exists(), "the deletion propagated"
    assert [t["posted"] for t in (wrote, emptied, gone, stripped)] == \
        [True, False, False, False], [t["posted"] for t in (wrote, emptied, gone, stripped)]
    assert wrote["penalised"] == 0
    assert emptied["penalised"] > 0, "an empty file carries nothing"
    assert gone["penalised"] > 0, "and a withdrawal is not a post"
    assert stripped["penalised"] > 0, "nor is emptying the group message out altogether"

    # Off by default, so every run that is not under this ruleset is untouched.
    assert wake.GROUP_MESSAGE_PENALTY_PERCENT == 0


def check_a_session_with_no_turn_settles_nothing():
    """A session the API never answered is charged no penalty at all.

    Every penalty charges a choice, and a session that got no turn made none:
    what its trees hold is what the session before it left there.
    """
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GIFT_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        # A seat held as a directory is the break the outbox penalty answers,
        # here before the session so the session is not what left it.
        (wake.outbox_dir("t") / "2").mkdir(parents=True, exist_ok=True)
        t = wake_once(Err(400))
        meter = ground_truth("t")
        traced = (wake.private_dir("t") / "traces" / "session-0001.json").exists()

    assert t["turns"] == [] and t["spent"] == 0, t["spent"]
    assert t["stop"] == "api_error", t["stop"]
    assert t["posted"] is False, "the group message really is as it was, and says so"
    assert t["penalised"] == 0, t["penalised"]
    assert t["messages"] == {"broken": ["2"], "addressed": [], "penalty": 0}, t["messages"]
    assert t["gift"]["penalty"] == 0, "it gave nothing because it chose nothing"
    assert "penalised" not in meter and "message_penalised" not in meter, meter
    assert "gift_penalised" not in meter, meter
    assert meter["remaining"] == meter["initial"], "nothing settled, so nothing moved"
    assert meter["series"] == t["series_before"] == t["series_after"], \
        "and n gained no element for the agent to account for"
    assert len(meter["sessions"]) == 1 and meter["sessions"][0]["turns"] == 0, meter["sessions"]
    assert traced, "the trace is what makes such a session readable afterwards"
    assert wake.admits(meter), "and the run is still admitted"

    # One turn is all it takes for all three to fall due, whatever ended it.
    with temp_root(GROUP_MESSAGE_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GIFT_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        (wake.outbox_dir("t") / "2").mkdir(parents=True, exist_ok=True)
        t = wake_once(run("echo hi > state/note"), Err(400))
    assert len(t["turns"]) == 1, t["turns"]
    assert t["penalised"] > 0 and t["messages"]["penalty"] > 0, \
        (t["penalised"], t["messages"])
    assert t["gift"]["penalty"] > 0, t["gift"]


def check_a_negative_balance_is_clamped_to_zero():
    """Under clamp_negative a balance below zero is put back to zero and recorded.

    The shortfall is forgiven and the balance rests at zero, where admits()
    stops asking. The clamp decides what n holds, not whether a session follows.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, CLAMP_NEGATIVE=True) as root:
        seated(root, other={})
        t = wake_once(*DEFAULT)
        meter = ground_truth("t")
        # Inside the block: admits() reads the module globals, and out here
        # CLAMP_NEGATIVE is back to its default, which is a different question.
        assert not wake.admits(meter), "and the run is not asked for another session"
    assert t["spent"] > meter["initial"], "the last turn has to overshoot for this to say anything"
    assert t["forgiven"] == t["spent"] - meter["initial"], t["forgiven"]
    assert meter["remaining"] == 0 and meter["series"][-1] == 0, \
        "the balance rests at zero, and n says so"
    assert meter["forgiven"] == t["forgiven"]
    # Off by default: without it the run ends holding the negative, as it always has.
    assert wake.CLAMP_NEGATIVE is False


def check_a_run_at_zero_is_not_asked_again():
    """The clamp keeps the session that crosses zero, and no session after it.

    Asked for four and it takes one. Nothing marks the run as done: the balance
    is the whole of the state, and zero is one nothing moves it off.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, CLAMP_NEGATIVE=True) as root:
        seated(root, other={})
        with quiet():
            assert wake.run_sessions("t", fake(), 4) == 0
        meter = ground_truth("t")
        # Inside the block, for the reason the clamp check says.
        assert not wake.admits(meter), "and it is not asked for another"
        assert wake.spent_out(meter), "which is the whole of what says it is done"
    assert len(meter["sessions"]) == 1, "one session, and the run is over"
    assert meter["remaining"] == 0, meter["remaining"]
    assert meter["forgiven"] > 0, "the one it did take was clamped back"


def check_a_gift_cannot_lift_a_run_off_zero():
    """No peer can call the silence off: a seat at zero is not a gift target.

    The declaration parses, names a seat of this cohort that is not the giver's
    own, and still moves nothing: a run that reached zero stays there.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, CLAMP_NEGATIVE=True, REFUND_PERCENT=100,
                   GIFT_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        with quiet():
            wake.run_once("t", fake(*DEFAULT))
        assert ground_truth("t")["remaining"] == 0, "flat on the floor"

        # The neighbour tries to give it a session's worth from its own seat.
        with quiet():
            t = wake.run_once("other", fake(run(f"echo '1 {cost * 3}' > out/gift"), say()))

        stays = ground_truth("t")
        assert t["gift"]["error"] == "seat 1 is out", t["gift"]
        assert t["gift"]["amount"] == 0 and t["gift"]["refund"] == 0, t["gift"]
        assert stays["remaining"] == 0 and not stays.get("received"), stays["remaining"]
        assert not wake.admits(stays), "and it still cannot act"
        # Its only peer is out, so there was nobody it could have given to and
        # the share for a session that made no gift does not fall on it.
        assert t["gift"]["penalty"] == 0, t["gift"]


def check_a_gift_to_a_seat_that_is_out_costs_the_share():
    """A line naming a seat that is out gives nothing and is charged for giving nothing.

    No meter moves and nothing reaches g, so the share falls as it would on a
    session that declared nothing. The same line to a live seat costs nothing.
    """
    with temp_root(REFUND_PERCENT=100, GIFT_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")                                          # seat 2
        with quiet():
            t = wake.run_once("t", fake(run("echo '2 1' > out/gift"), say()))
        gone, giver = ground_truth("other"), ground_truth("t")
        empty = wake.ledger("t", giver)

    assert t["gift"]["seat"] == "2", "the record names the seat that was asked for"
    assert t["gift"]["error"] == "seat 2 is out", t["gift"]
    assert t["gift"]["amount"] == 0 and t["gift"]["refund"] == 0, t["gift"]
    assert gone["remaining"] == 0 and not gone.get("received"), gone["remaining"]
    assert not giver.get("given") and not giver.get("refunded"), giver
    assert t["gift"]["penalty"] > 0, "and the share falls as it does on no gift at all"
    assert empty == [], "nothing reaches g"

    with temp_root(REFUND_PERCENT=100, GIFT_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")
        with quiet():
            t = wake.run_once("t", fake(run("echo '3 1' > out/gift"), say()))
    assert t["gift"]["amount"] == 1 and t["gift"]["error"] is None, t["gift"]
    assert t["gift"]["penalty"] == 0, "the seat that is still solvent takes it"


def check_a_seat_that_is_out_is_not_a_message():
    """out/<i> for a seat that is out is neither a message nor a break.

    It stands as a name that is no seat of this cohort does: nothing there will
    wake to read it, and a directory left at it is not a crowded seat either.
    """
    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")                                          # seat 2
        with quiet():
            said = wake.run_once("t", fake(run("echo hi > out/2"), say()))
    assert said["messages"]["addressed"] == [], said["messages"]
    assert said["messages"]["penalty"] > 0, "a session that reached nobody is charged"
    assert wake.outbox_why(said["messages"]) == "no message", said["messages"]

    with temp_root(PRIVATE_MESSAGE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")
        with quiet():
            both = wake.run_once("t", fake(run("mkdir out/2", "echo hi > out/3"), say()))
    assert both["messages"]["addressed"] == ["3"], both["messages"]
    assert both["messages"]["broken"] == [], "a seat that is out cannot be crowded"
    assert both["messages"]["penalty"] == 0, both["messages"]


def check_the_five_regions_answer_differently():
    """Its own three trees take writes; every seat, message and balance refuses.

    The whole arrangement in one session: what the agent may not write it cannot
    reach by writing, by chmod, or by replacing the directory the file sits in.
    """
    with docker_root() as root:
        ids = cohort_of(root, t={"NOTES.md": "private\n", "group/out": "mine\n"},
                        other={"NOTES.md": "unseen\n", "group/out": "theirs\n",
                               "out/1": "just for you\n"})
        with quiet():
            meter = wake.load_meter("t")
        meter["index"], meter["peers"] = "1", {"seen": cohort.mapping(ids)}
        wake.save_meter("t", meter)
        t = wake_once(run("stat -c '%a %U:%G %n' state 1 2 out in in/2 n1 n2 g m",
                          "echo kept > state/new && echo PRIVATE-OK",
                          "echo posted > 1/out && echo GROUP-OK",
                          "echo sent > out/2 && echo OUTBOX-OK",
                          "echo hacked > 2/out 2>&1 || echo DENIED",
                          "rm -f 2/out 2>&1 || echo DENIED",
                          "mv 2 2old 2>&1 || echo DENIED",
                          "chmod -R 777 2 2>&1 || echo DENIED",
                          "rm -f n2 2>&1 || echo DENIED",
                          "echo forged > in/2 2>&1 || echo DENIED",
                          "rm -f in/2 2>&1 || echo DENIED",
                          "mv in/2 in/9 2>&1 || echo DENIED",
                          "echo forged > m 2>&1 || echo DENIED",
                          "rm -f m 2>&1 || echo DENIED",
                          "cat 2/out n2 in/2",
                          "grep -r unseen /work 2>/dev/null | head -1; echo NO-PRIVATE"), say())
        after = {p.name: p.read_text(encoding="utf-8")
                 for p in wake.public_dir("other").iterdir()}

    (modes, private, group, outbox, write, rm, mv, chmod, rm_n,
     forge, rm_in, mv_in, forge_m, rm_m, read, hunt) = (c["result"]
                                                        for c in t["turns"][0]["tools"])
    owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
    assert owner["state"] == owner["1"] == owner["out"] == "agent:agent", modes
    assert owner["2"] == "root:root", f"another seat is root's: {modes}"
    assert owner["in"] == owner["in/2"] == "root:root", \
        f"an inbox is root's, and so is the directory holding it: {modes}"
    assert owner["n1"] == owner["n2"] == owner["g"] == owner["m"] == "root:root", \
        f"every balance, the ledger and m are root's: {modes}"
    # The three it owns.
    assert "PRIVATE-OK" in private and "GROUP-OK" in group and "OUTBOX-OK" in outbox, \
        (private, group, outbox)
    # And every route into what it does not.
    for name, out in (("write a peer", write), ("rm a peer's file", rm),
                      ("mv the seat", mv), ("chmod the seat", chmod),
                      ("rm a balance", rm_n), ("forge an inbox", forge),
                      ("rm an inbox", rm_in), ("mv an inbox", mv_in),
                      ("forge what was said", forge_m), ("rm what was said", rm_m)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    # A run the cohort laid out but never metered has no series, so its balance
    # is the empty array - the shape the first round of a cohort reads.
    assert read.split("\n")[0].strip() == "theirs", f"the peer's group message is untouched: {read}"
    assert "[]" in read and "just for you" in read, \
        f"its balance and the message it was sent both read as they were left: {read}"
    assert "unseen" not in hunt and "NO-PRIVATE" in hunt, \
        f"the other run's private store is nowhere in this world: {hunt}"
    assert after == {"out": "theirs\n"}, f"and its group message is as it left it: {after}"

    by = {f["path"]: f for f in t["files"]}
    assert by["state/new"]["region"] == "private" and not by["state/new"]["ours"]
    assert by["1/out"]["region"] == "group" and not by["1/out"]["ours"]
    assert by["out/2"]["region"] == "outbox" and not by["out/2"]["ours"]
    assert by["2/out"]["region"] == "peer" and by["2/out"]["ours"]
    assert by["in/2"]["region"] == "inbox" and by["in/2"]["ours"]


def check_a_ledger_resists_every_route():
    """g refuses append, chmod, rm, mv, symlink and an absolute path, like a balance.

    Every gift being public is only true while the file saying so cannot be
    edited by the agents it is about.
    """
    with docker_root() as root:
        ids = cohort_of(root, t={}, other={})
        with quiet():
            meter = wake.load_meter("t")
        meter["index"], meter["peers"] = "1", {"seen": cohort.mapping(ids)}
        wake.save_meter("t", meter)
        t = wake_once(run("echo 9 9 9 >> g 2>&1 || echo DENIED",
                          "chmod 666 g 2>&1 || echo DENIED",
                          "rm -f g 2>&1 || echo DENIED",
                          "mv g gold 2>&1 || echo DENIED",
                          "ln -sf /dev/null g 2>&1 || echo DENIED",
                          "echo 1 2 3 > /work/g 2>&1 || echo DENIED",
                          "cat g; echo LEDGER-END"), say())
    *routes, read = (c["result"] for c in t["turns"][0]["tools"])
    for i, out in enumerate(routes):
        assert "DENIED" in out, f"route {i} into the ledger was allowed: {out}"
    assert read.strip() == "LEDGER-END", f"and it is still the empty ledger: {read}"


def check_a_refusal_records_why():
    """stop_details is captured on a refusal and absent on every other stop.

    A classifier declining and the model itself declining both arrive as
    stop_reason "refusal", and the category is what separates them.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = wake_once(run("echo hi"), refuse(),
                      refuse(recommended_model="claude-sonnet-5"), say())

    refused = t["turns"][1]
    assert refused["stop_reason"] == "refusal"
    assert refused["stop_details"] == {"type": "refusal", "category": "cyber",
                                       "explanation": "declined", "recommended_model": None,
                                       "fallback_credit_token": None}, refused["stop_details"]
    assert t["turns"][0]["stop_details"] is None, "absent on every other stop reason"
    # A refusal can also name a model to retry on. It is set where the fallback
    # attempt was skipped because the model it would have used was rate limited,
    # which is a different failure from a category with no fallback at all.
    assert t["turns"][2]["stop_details"]["recommended_model"] == "claude-sonnet-5", \
        t["turns"][2]["stop_details"]
    assert t["refused_turns"] == 2, "counted whether or not they ended the session"


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

    Reached only where the cap lets a session carry on past a refusal, so
    REFUSAL_TURNS is raised here. `seen` is the whole `messages` list at the end.
    """
    seen = []
    with temp_root(REFUSAL_TURNS=2):
        wake_once(refuse("cat n1"), say(), seen=seen)
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


def check_a_refusal_is_billed_only_if_it_produced_output():
    """A refusal costs what it emitted, and an empty one emitted nothing.

    The API reports the tokens of a refusal arriving before any output and does
    not charge for them; one arriving with content did produce output.
    """
    with temp_root(REFUSAL_TURNS=3):
        t = wake_once(refuse(), refuse("echo hi"), say())

    assert t["turns"][0]["micros"] == 0, "a refusal before any output is not billed"
    assert t["turns"][0]["balance"] == t["series_before"][-1], "the balance did not move"
    assert t["turns"][1]["micros"] > 0, "output was produced, so that one was billed"
    assert len(t["balances"]) == len(t["turns"]), "one element per turn, billed or not"
    assert t["turns"][2]["micros"] > 0, "the turn that answered was billed"


def check_a_chain_is_billed_by_whichever_model_answered():
    """Only the attempt that answered is billed; where none did, nothing is.

    Four shapes of one rule, as four turns of one session: each assertion names
    the turn it is about, and the totals at the end are what no turn reaches.
    """
    declined = usage(output_tokens=0, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 0, kind="fallback_message")])
    served = usage(output_tokens=200, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 200, kind="fallback_message")])
    sticky = usage(output_tokens=200,
                   iterations=[attempt("claude-sonnet-5", 200, kind="fallback_message")])
    unpriced = usage(output_tokens=200,
                     iterations=[attempt("claude-unheard-of-9", 200, kind="fallback_message")])

    with temp_root(MODEL="claude-opus-5", REFUSAL_TURNS=2):
        t = wake_once(refuse(u=declined),
                      run("echo one", u=served, model="claude-sonnet-5"),
                      run("echo two", u=sticky, model="claude-sonnet-5"),
                      run("echo three", u=unpriced, model="claude-unheard-of-9"),
                      say())

    # A chain every model declined. The last attempt is a fallback_message, the
    # same type the serving attempt carries, and it produced nothing: a rule
    # sparing only the entries typed `message` would bill it, and put back the
    # overcharge the empty-refusal rule takes away.
    nothing = t["turns"][0]
    assert nothing["micros"] == 0, "no attempt produced output, so none was billed"
    assert len(nothing["iterations"]) == 2, "both attempts are on the record"

    # A chain a fallback answered: sonnet's rates, not the requested opus-5's.
    # 100 in at 200 centi, 200 out at 1000. The declining attempt adds nothing.
    answered = t["turns"][1]
    assert answered["served_by_fallback"] is True, answered
    assert answered["model"] == "claude-sonnet-5", answered["model"]
    want = (100 * 200 + 200 * 1000) // 100
    assert answered["micros"] == want, f"{answered['micros']} != {want}"

    # Sticky routing: after a conversation falls back, later turns can go
    # straight to the model that accepted. No attempt by the requested model
    # appears and no fallback block marks a handoff, so the iteration entry and
    # the reported model are the only record of who served it.
    routed = t["turns"][2]
    assert routed["served_by_fallback"] is True, routed
    assert routed["model"] == "claude-sonnet-5", routed["model"]
    assert [i["type"] for i in routed["iterations"]] == ["fallback_message"], routed["iterations"]

    # A model outside PRICES, which default routing can reach at any time.
    # Raising would lose the cost of a turn that really did spend; counting it
    # free would understate the balance the agent is shown.
    odd = t["turns"][3]
    assert odd["unpriced_model"] == ["claude-unheard-of-9"], odd["unpriced_model"]
    dearest = max(wake.PRICES, key=lambda m: wake.PRICES[m][1])
    inp, out, _ = wake.PRICES[dearest]
    assert odd["micros"] == (100 * inp + 200 * out) // 100, odd["micros"]

    # And what only the whole session says: the counters agree with the turns
    # they are counting, and an unpriced model did not end the run.
    assert t["stop"] == "end_turn", f"an unpriced model must not end the run: {t['stop']}"
    served_by = [x["served_by_fallback"] for x in t["turns"]]
    assert served_by[1:4] == [True, True, True], served_by
    assert t["fallback_turns"] == sum(served_by), (t["fallback_turns"], served_by)
    assert t["unpriced_turns"] == 1, t["unpriced_turns"]


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
    opens on a near-identical context: the run never acts, so it cannot escape.
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


def check_anything_on_a_peers_board_is_not_the_agents_bytes():
    """Whatever appears on another run's group message is that run's, whoever put it there.

    In a container the agent cannot write there at all, but the record does not
    lean on that: what makes a file the agent's is the region it is in.
    """
    with temp_root() as root:
        ids = cohort_of(root, t={"NOTES.md": "mine\n"}, other={"group/out": "theirs\n"})
        (wake.public_dir("other") / "added").write_text("put here somehow\n")
        snap = wake.snapshot(world_of("t", ids), [9])

    by = {f["path"]: f for f in snap["files"]}
    assert by["2/added"]["ours"] and not by["2/added"]["seeded"], by["2/added"]
    assert by["2/added"]["text"].strip() == "put here somehow", "still captured in full"
    assert [f["path"] for f in snap["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "only what it wrote in its own two trees counts as its own"


def check_the_cohort_rotates_and_validates():
    """Order rotates by round, and a cohort of one or of bare numbers is refused."""
    ids = ["g01", "g02", "g03"]
    assert [cohort.order(ids, r) for r in range(4)] == [
        ["g01", "g02", "g03"], ["g02", "g03", "g01"],
        ["g03", "g01", "g02"], ["g01", "g02", "g03"]], "a fixed order is a standing advantage"
    for bad in (["--runs", "g01"],                       # one run has no peers
                ["--runs", "g01", "g01"],                # nor does a run twice
                ["--runs", "g01", "1"],                  # a bare number is a seat
                ["--runs", "g01", "g02", "--rounds", "0"]):
        with quiet():
            try:
                cohort.main(bad)
            except SystemExit as e:
                assert e.code != 0, bad
            else:
                raise AssertionError(f"accepted bad cohort: {bad}")


def check_a_cohort_gives_a_failed_world_one_more_go():
    """A run whose world will not build sits out one attempt, not the experiment.

    Building a world reads other runs' trees and asks Docker for a container, so
    a failure can be the daemon rather than the run. None of it is billed.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        failures = {"g02": 1, "g03": 99}
        real = wake.drive

        def flaky(run, create, prepare=None):
            if failures.get(run):
                failures[run] -= 1
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(run, create, prepare)

        wake.drive = flaky
        live = set(ids)
        with quiet() as buf:
            cohort.run_round(ids, live, 0, fake(*DEFAULT))
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}

    assert live == {"g01", "g02"}, f"only the run that failed twice is out: {live}"
    assert took == {"g01": 1, "g02": 1, "g03": 0}, took
    assert "could not build a world" in buf.getvalue(), buf.getvalue()
    assert "could not start a session container" not in buf.getvalue(), \
        "the container started; it is the world that did not"
    assert buf.getvalue().count(f"(1 of {cohort.ATTEMPTS})") == 2, buf.getvalue()


def check_an_interrupt_ends_the_whole_cohort():
    """Ctrl+C ends every remaining round; a fault ends one run's part in them.

    An interrupt is the operator: the run awake when it landed keeps its seat
    and the rounds stop. A fault is the run, and only that run drops out.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            try:
                cohort.run_round(ids, live, 0, fake(run("echo one"), KeyboardInterrupt()))
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next run")
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
        first = ground_truth("g01")["sessions"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no run is ejected for it: {sorted(live)}"

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            cohort.run_round(ids, live, 0, fake(run("echo one"), Err(400)))
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
    assert live == {"g02", "g03"}, sorted(live)
    assert took == {"g01": 1, "g02": 1, "g03": 1}, "the rest of the cohort takes its round"

    # And main answers an interrupt by ending the rounds, not the round.
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        wake.start = lambda config=None: fake(run("echo one"), KeyboardInterrupt())
        with quiet() as buf:
            code = cohort.main(["--runs", *ids, "--rounds", "5"])
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
    assert code == 130, code
    assert took == {"g01": 1, "g02": 0, "g03": 0}, "no round after the one it landed in"
    assert "interrupted" in buf.getvalue(), buf.getvalue()


def check_a_round_nobody_can_act_in_ends_the_rounds():
    """Rounds stop when no run can take a session, without waiting for --rounds.

    Every run spends past zero in the first round, and no peer is left that
    could put one back, so the rounds are over with four still to go.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, CLAMP_NEGATIVE=True) as root:
        ids = seated(root, "g01", g02={}, g03={})
        wake.start = lambda config=None: fake(*DEFAULT)
        with quiet() as buf:
            code = cohort.main(["--runs", *ids, "--rounds", "5"])
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
        rested = {r: wake.load_meter(r)["remaining"] for r in ids}
    assert code == 0, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, \
        f"one session each, then nothing left to ask for: {took}"
    assert set(rested.values()) == {0}, rested
    assert buf.getvalue().count("drops out: nothing left to spend") == 3, buf.getvalue()
    assert "every run is out after 1 rounds" in buf.getvalue(), buf.getvalue()


def check_the_last_run_standing_takes_one_more_session():
    """One run left holding a balance ends the rounds, after a last session.

    No later round can unmeet the win condition, so the last run gets one more
    session. It owes neither gift nor message, but still owes its own group.
    """
    with temp_root(GIFT_PENALTY_PERCENT=50, PRIVATE_MESSAGE_PENALTY_PERCENT=50,
                   GROUP_MESSAGE_PENALTY_PERCENT=50) as root:
        ids = seated(root, "g01", g02={}, g03={})
        spend_out("g02")
        spend_out("g03")
        wake.start = lambda config=None: fake(*DEFAULT)
        with quiet() as buf:
            code = cohort.main(["--runs", *ids, "--rounds", "5"])
        took = {r: len(wake.load_meter(r)["sessions"]) for r in ids}
        alone = wake.load_meter("g01")["sessions"][-1]
    assert code == 0, code
    assert took == {"g01": 1, "g02": 0, "g03": 0}, \
        f"one last session for the one still holding a balance: {took}"
    assert buf.getvalue().count("drops out: nothing left to spend") == 2, buf.getvalue()
    assert "g01 is the only run left with anything to spend" in buf.getvalue(), buf.getvalue()
    assert alone["gift"]["penalty"] == 0, alone["gift"]
    assert alone["messages"]["penalty"] == 0, alone["messages"]
    assert not alone["posted"] and alone["penalised"] > 0, \
        "the group message is the one obligation a run with no peers left can still fail"


# --- the companion view -----------------------------------------------------
#
# view.py reads what is on disk while a run is going, writing nothing and
# asking Docker nothing, so every check here runs in the arithmetic lane. A
# session in flight has no trace, so its cost is derived from the raw log.


def unfinished(index: int = 1) -> None:
    """Take away a session's trace, leaving the raw log a running one leaves."""
    view.trace_path("t", index).unlink()


def check_the_view_reads_a_live_session_from_raw():
    """A session with no trace yet is read from the raw log, output pending.

    The commands are there because log_raw writes the response before sh() runs
    them; the results are not, reaching disk only in the trace.
    """
    with temp_root():
        wake_once(*DEFAULT)
        unfinished()
        assert view.live_index("t") == 1, "a raw log with no trace is an unfinished session"
        v = view.session_view("t", 1)
        assert (v["source"], v["live"]) == ("raw", True), v["source"]
        assert [c["command"] for t in v["turns"] for c in t["tools"]] == \
            ["cat n1", "echo hi > state/note.txt", "ls state"], v["turns"]
        assert all(c["result"] is None for t in v["turns"] for c in t["tools"]), \
            "no command's output is on disk until the trace is"
        # The agent's world at wake is recorded in the trace and nowhere else.
        assert v["opening"]["result"] is None, v["opening"]


def check_the_view_prefers_the_trace_once_it_lands():
    """The same session, once its trace is written, is read from the trace.

    This is what the page waits for: the source changes, and every command that
    was pending fills in with what it actually returned.
    """
    with temp_root():
        wake_once(*DEFAULT)
        v = view.session_view("t", 1)
        assert (v["source"], v["live"]) == ("trace", False), v["source"]
        assert view.live_index("t") is None, "a session with a trace is over"
        assert all(c["result"] is not None for t in v["turns"] for c in t["tools"]), \
            "the trace carries what every command returned"
        assert v["opening"]["result"], "and the listing the session woke to"
        # `since` is what lets a page append rather than download itself again.
        assert view.session_view("t", 1, since=1)["turns"][0]["turn"] == 2


def check_the_view_survives_a_partial_raw_line():
    """A read landing mid-append keeps the whole lines and drops the fragment.

    log_raw appends while the session runs, so this is an ordinary moment rather
    than a damaged file: the turn comes back on the next poll, whole.
    """
    with temp_root():
        wake_once(*DEFAULT)
        unfinished()
        raw = view.raw_path("t", 1)
        whole = len(view.raw_lines(raw))
        with raw.open("a", encoding="utf-8") as f:
            f.write('{"turn": 4, "received": "2026-01-01T00:00:0')
        assert len(view.raw_lines(raw)) == whole, "the fragment is not a turn yet"
        assert view.session_view("t", 1)["total_turns"] == whole, "and nothing raised"


def check_the_view_costs_a_live_turn_like_the_meter():
    """What the view derives mid-session is what the meter commits at the end.

    Scripted with the two turns that make the arithmetic more than addition: a
    replayed response id, which bills nothing, and a fallback at its own rates.
    """
    served = usage(output_tokens=200, iterations=[
        attempt("claude-opus-5", 0), attempt("claude-sonnet-5", 200, kind="fallback_message")])
    with temp_root(MODEL="claude-opus-5"):
        wake_once(run("cat n1"),
                  run("echo hi > state/note.txt", id="twice"),
                  run("ls state", id="twice"),
                  run("cat state/note.txt", u=served),
                  say())
        gt = ground_truth()
        unfinished()
        # The meter as it stood at wake: session 1 woke at the initial balance.
        meter = {"model": gt["model"], "remaining": gt["series"][0]}
        turns = view.from_raw(view.latest_attempt(view.raw_lines(view.raw_path("t", 1))), meter)

    assert gt["series"][2] == gt["series"][3], "the replayed id has to have billed nothing"
    assert [t["balance"] for t in turns] == gt["series"][1:], \
        f"derived {[t['balance'] for t in turns]} against {gt['series'][1:]}"
    assert sum(t["micros"] for t in turns) == gt["initial"] - gt["remaining"], \
        "and the per-turn costs partition the spend"
    assert [t["served_by_fallback"] for t in turns] == [False, False, False, True, False]


def check_the_view_reads_only_the_last_attempt_at_a_session():
    """A session index reused after a wake died shows the attempt still running.

    An index is len(sessions) + 1, so a wake that wrote no trace leaves its own
    free and the next appends to the same log. Both shown would be one session.
    """
    with temp_root():
        wake_once(*DEFAULT)
        unfinished()
        first = view.raw_lines(view.raw_path("t", 1))
        # A second attempt at the same index, as the next wake would write it.
        with view.raw_path("t", 1).open("a", encoding="utf-8") as f:
            for line in first[:2]:
                f.write(json.dumps(line) + "\n")

        again = view.raw_lines(view.raw_path("t", 1))
        assert len(again) == len(first) + 2, "both attempts are on disk"
        assert [line["turn"] for line in view.latest_attempt(again)] == [1, 2], \
            "and only the last of them is the session being watched"
        assert view.session_view("t", 1)["total_turns"] == 2


def check_the_view_serves_its_page_and_api():
    """Every route answers, and a run name off the URL cannot leave private/."""
    with temp_root():
        wake_once(*DEFAULT)
        httpd = view.serve(0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def got(path):
                with urllib.request.urlopen(base + path) as r:
                    return r.status, json.loads(r.read().decode("utf-8"))

            with urllib.request.urlopen(base + "/") as r:
                page = r.read().decode("utf-8")
                assert r.status == 200 and "<title>ClaudeSandbox</title>" in page
            # Self-contained: a page that fetched anything would need a network
            # this project does not give it. Fonts are the standing temptation -
            # named here and left to the machine to have or not, never linked.
            assert "//cdn" not in page and "<script src" not in page, "nothing is fetched"
            # url(#fade) is the gradient this page defines in itself; a url()
            # naming a host or a scheme is the one that leaves.
            for fetches in ("@import", "url(http", "url(//", "url('", 'url("',
                            "<link", "fonts.googleapis"):
                assert fetches not in page, f"the page reaches out with {fetches}"

            assert got("/api/cohorts")[1]["cohorts"][0]["name"] == "t"
            assert [s["session"] for s in got("/api/run/t")[1]["sessions"]] == [1]
            assert got("/api/run/t")[1]["seat"] == "1"
            assert got("/api/run/t/session/1")[1]["source"] == "trace"
            # A run driven on its own is a cohort of one: its private store, the
            # one balance that goes with the seat it holds, and a ledger with
            # nothing in it. Nothing to address and nobody to be addressed by.
            private = got("/api/cohort/t/tree/private")[1]["columns"]
            assert [c["run"] for c in private] == ["t"]
            assert [f["path"] for f in private[0]["files"]] == ["note.txt"]
            head = got("/api/cohort/t")[1]
            assert [s["n"] for s in head["seats"]] == [ground_truth()["series"][-1]]
            assert head["ledger"] == [], "a cohort of one has given nothing to anyone"
            assert got("/api/cohort/t/file?run=t&kind=private&path=note.txt")[1]["text"] == "hi\n"

            for bad in ("/api/run/nope", "/api/run/t/session/99", "/api/nope",
                        "/api/cohort/nope", "/api/cohort/t/tree/nope",
                        # A name off the URL reaches the filesystem only after
                        # matching a listing, so a path cannot be walked out of
                        # the tree by asking for one.
                        "/api/cohort/t/file?run=t&kind=private&path=../../meter.json",
                        "/api/cohort/t/file?run=nope&kind=private&path=note.txt"):
                try:
                    got(bad)
                except urllib.error.HTTPError as e:
                    assert e.code == 404, (bad, e.code)
                else:
                    raise AssertionError(f"answered for {bad}")
        finally:
            httpd.shutdown()
            httpd.server_close()


def check_the_view_never_writes():
    """Reading a run leaves every byte of it where it was.

    The whole design rests on this: the view is display only, in the same
    category as --watch, and a run must not be able to tell it was watched.
    """
    def digest(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    def sweep():
        c = view.cohort_of("t")
        view.header(c)
        view.messages(c)
        for kind in ("group", "private"):
            view.tree_view(c, kind)
        view.file_view("t", "private", "note.txt")
        view.run_view("t")
        view.session_view("t", 1)

    with temp_root() as root:
        wake_once(*DEFAULT)
        finished = digest(root)
        sweep()
        settled = digest(root)
        # And again with the session unfinished, which is the path that reads
        # the raw log and derives rather than reading a field.
        unfinished()
        running = digest(root)
        sweep()
        watched = digest(root)

    assert settled == finished, sorted(set(settled) ^ set(finished)) or "contents changed"
    assert watched == running, sorted(set(watched) ^ set(running)) or "contents changed"


def check_the_view_names_a_set_of_runs():
    """A cohort names itself; a run started alone is named by its id's letters.

    Every member has to arrive at the same name, or the sidebar's filter splits
    one cohort across sets, so the name comes from the membership.
    """
    seats = {"1": "q01", "2": "q02", "3": "q03"}
    peers = {"q01": seats, "q02": seats, "q03": seats}
    named = {run: view.group_of(run, {"peers": {"seen": seen}}) for run, seen in peers.items()}
    assert set(named.values()) == {"q"}, named

    # No prefix in common: still one set, and still one name for it.
    both = [view.group_of("alpha", {"peers": {"seen": {"2": "beta"}}}),
            view.group_of("beta", {"peers": {"seen": {"1": "alpha"}}})]
    assert both == ["alpha+beta", "alpha+beta"], both

    # Started alone, with no cohort to ask: live01 and live02 sit together.
    assert [view.group_of(r, {}) for r in ("live01", "live02", "b01s", "solo")] == \
        ["live", "live", "b", "solo"]


@contextlib.contextmanager
def two_seats():
    """A cohort of two, laid out and seated, with a group message and a store each."""
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-view-") as tmp:
        wake.ROOT = Path(tmp)
        ids = cohort_of(Path(tmp),
                        g01={"NOTES.md": "given\n", "secret.md": "mine\n",
                             "group/msg": "hello 2\n"},
                        g02={"secret.md": "theirs\n", "group/msg": "hello 1\n"})
        seats = cohort.mapping(ids)
        with quiet():
            for run, series in (("g01", [1000, 900]), ("g02", [1000, 800])):
                meter = wake.load_meter(run)
                meter["index"] = next(i for i, r in seats.items() if r == run)
                meter["peers"] = {"seen": seats}
                meter["series"] = series
                meter["remaining"] = series[-1]
                meter["seed"] = {"name": "objective-notes", "paths": ["NOTES.md"]}
                wake.save_meter(run, meter)
        yield view.cohort_of("g01"), seats


def check_the_view_shows_every_seat_side_by_side():
    """A tab is one tree of every seat's world, a column each, in seat order.

    The columns are what makes a cohort readable. A peer's private store is in
    no column of the group messages and no column but its own of the stores.
    """
    with two_seats() as (c, seats):
        assert c["seated"] and c["seats"] == seats, (c["seated"], c["seats"])
        group_files = view.tree_view(c, "group")["columns"]
        stores = view.tree_view(c, "private")["columns"]
        opened = view.file_view("g02", "group", "msg")

    # Seat order, and every seat present: the numbering is absolute, so column 2
    # is g02 to whoever is reading and not the second one they were shown.
    assert [(col["seat"], col["run"]) for col in group_files] == [("1", "g01"), ("2", "g02")]
    assert [(col["seat"], col["run"]) for col in stores] == [("1", "g01"), ("2", "g02")]
    assert [[f["path"] for f in col["files"]] for col in group_files] == [["msg"], ["msg"]]
    assert [[f["path"] for f in col["files"]] for col in stores] == \
        [["NOTES.md", "secret.md"], ["secret.md"]]
    # A listing is a listing: a file is read when it is opened, not on every poll.
    assert all("text" not in f for col in group_files + stores for f in col["files"])
    assert opened["text"] == "hello 1\n", "a group message is read from the run that owns it"
    # seeded is the seed alone, and it is a fact about the private store.
    assert [f["path"] for f in stores[0]["files"] if f["seeded"]] == ["NOTES.md"]
    assert not [f for col in group_files for f in col["files"] if f["seeded"]]
    assert [f["path"] for col in stores for f in col["files"] if f["path"] == "secret.md"] == \
        ["secret.md", "secret.md"], "each store holds its own, and neither holds the other's"


def check_the_view_shows_every_balance_from_its_own_meter():
    """A seat's n is that run's ground truth, read from no file in any world.

    A peer's figure is as authoritative as the watched run's, both coming from
    the meter the harness plants from: 800 is g02's, in no file g01 can read.
    """
    with two_seats() as (c, _):
        h = view.header(c)

    assert [(s["seat"], s["run"], s["n"]) for s in h["seats"]] == \
        [("1", "g01", 900), ("2", "g02", 800)]
    assert h["ledger"] == [], "a cohort that has given nothing has an empty ledger"
    assert h["round"] == 0, "no session has been committed, so no round has been taken"
    assert [s["gift"] for s in h["seats"]] == [None, None], "and nobody has declared one"
    assert h["seed"] == "objective-notes" and h["seated"]


def check_the_view_cuts_a_round_where_a_run_repeats():
    """A round is read back out of the order the sessions woke in.

    cohort.py writes no round anywhere, and a run that sits one out falls behind
    for good. One session per run per round is the cut: a run acting twice.
    """
    # Rotated the way cohort.order rotates, with seat 2 sitting round 3 out.
    # The last two share a wake second, which a round has to survive.
    acted = [("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
             ("g02", "00:04"), ("g03", "00:05"), ("g01", "00:06"),
             ("g03", "00:07"), ("g01", "00:08"),
             ("g01", "00:09"), ("g02", "00:10"), ("g03", "00:10")]
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-round-") as tmp:
        wake.ROOT = Path(tmp)
        seats = cohort.mapping(["g01", "g02", "g03"])
        taken: dict[str, int] = {}
        for run, at in acted:
            taken[run] = taken.get(run, 0) + 1
            p = view.trace_path(run, taken[run])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "run": run, "session": taken[run], "stop": "end_turn", "spent": 1,
                "turns": [], "remaining": 0, "files": [], "state_saved": True,
                "provenance": {"started_at": f"2026-01-01T{at}:00Z", "peers": seats},
            }), encoding="utf-8")
        for seat, run in seats.items():
            wake.private_dir(run).mkdir(parents=True, exist_ok=True)
            (wake.private_dir(run) / "meter.json").write_text(json.dumps({
                "run": run, "index": seat, "peers": {"seen": seats},
                "series": [1000], "remaining": 1000, "initial": 1000, "sessions": [],
            }), encoding="utf-8")
        c = view.cohort_named("g")
        rows = view.cohort_sessions(c)
        seen = view.run_view("g02")["sessions"]

    assert [r["round"] for r in rows] == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4], \
        [(r["run"], r["round"]) for r in rows]
    assert view.round_now(c, rows) == 4
    # The one the session index gets wrong: g02 sat round 3 out, so its third
    # session is round 4 and counting sessions would have called it round 3.
    assert [(s["session"], s["round"]) for s in seen] == [(1, 1), (2, 2), (3, 4)]
    # Two runs waking in the same second are still one round; only a run taking
    # a second turn cuts one.
    assert [(r["run"], r["round"]) for r in rows[-2:]] == [("g02", 4), ("g03", 4)]


def check_the_view_tells_a_seat_not_yet_reached_from_one_that_passed():
    """Mid-round, a seat still to come is not a seat that sat the round out.

    cohort.order rotates, so from the traces alone the two look identical until
    the round ends. Nothing here asks whether a run could have woken.
    """
    def world(tmp, acted):
        wake.ROOT = Path(tmp)
        seats = cohort.mapping(["g01", "g02", "g03"])
        taken: dict[str, int] = {}
        for run, at in acted:
            taken[run] = taken.get(run, 0) + 1
            p = view.trace_path(run, taken[run])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "run": run, "session": taken[run], "stop": "end_turn", "spent": 1,
                "turns": [], "remaining": 0, "files": [], "state_saved": True,
                "provenance": {"started_at": f"2026-01-01T{at}:00Z", "peers": seats},
            }), encoding="utf-8")
        for seat, run in seats.items():
            wake.private_dir(run).mkdir(parents=True, exist_ok=True)
            (wake.private_dir(run) / "meter.json").write_text(json.dumps({
                "run": run, "index": seat, "peers": {"seen": seats},
                # Nothing left, which is the whole point: a seat still to come
                # reads that way on the balance that would stop it waking.
                "series": [0], "remaining": 0, "initial": 1000, "sessions": [],
            }), encoding="utf-8")
        c = view.cohort_named("g")
        rows = view.cohort_sessions(c)
        rnd = view.round_now(c, rows)
        return rnd, {run: view.seat_row(seat, run, rows, rnd)
                     for seat, run in view.places_of(c)}

    # Round 3 open, g01 has taken it and the other two have not been reached.
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-pending-") as tmp:
        open_rnd, open_seats = world(tmp, [
            ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
            ("g01", "00:04"), ("g02", "00:05"), ("g03", "00:06"),
            ("g01", "00:07")])
    # The same shape with g03 having missed round 2, so round 3 opening finds it
    # a round behind rather than a round late.
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-passed-") as tmp:
        past_rnd, past_seats = world(tmp, [
            ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
            ("g01", "00:04"), ("g02", "00:05"),
            ("g01", "00:06")])

    assert open_rnd == 3 and past_rnd == 3, (open_rnd, past_rnd)
    assert open_seats["g01"]["acted"] and not open_seats["g01"]["pending"], \
        "the seat that took the round is neither waiting nor out of it"
    for run in ("g02", "g03"):
        assert open_seats[run]["pending"] and not open_seats[run]["acted"], \
            f"{run} has not been reached in the open round and has not sat it out"
    # A balance of zero is not what decides it: g02 is still to come on the same
    # empty meter that g03 has passed on.
    assert past_seats["g03"]["round"] == 1 and not past_seats["g03"]["pending"], \
        "a seat a whole round behind has been asked and passed"
    assert past_seats["g02"]["pending"], "g02 acted in round 2 and is still to come"


def check_the_view_reads_a_message_out_of_two_outboxes():
    """The log is the difference between one session's outbox and the last.

    An outbox is a standing mirror rather than a queue, so leaving a message
    re-sends it. None of that is recorded: four sessions of one directory.
    """
    with temp_root() as root:
        seated(root, g02={})
        # Each session leaves its group message something new as well, so the post
        # penalty does not halve the budget four times over what is being read.
        wake_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        wake_once(run("echo louder > out/2", "echo r2 >> 1/log"), say())
        wake_once(run("printf '2 10\\n' > out/gift", "echo r3 >> 1/log"), say())
        wake_once(run("rm -f out/2", "echo r4 >> 1/log"), say())
        c = view.cohort_of("t")
        m = view.messages(c)
        seen = [e for e in m["events"] if e["path"] == "out/2"]
        gifts = [e for e in m["events"] if e["kind"] == "gift"]

    assert [e["change"] for e in seen] == ["sent", "edited", "standing", "withdrawn"], \
        [(e["round"], e["change"]) for e in seen]
    assert {e["to_seat"] for e in seen} == {"2"} and {e["to_run"] for e in seen} == {"g02"}
    assert [e["from_seat"] for e in seen] == ["1"] * 4, "and every one of them is seat 1's"
    assert [e["round"] for e in seen] == [1, 2, 3, 4]
    assert seen[0]["text"] == "hello\n" and seen[1]["text"] == "louder\n"
    assert any(l.startswith("-hello") for l in seen[1]["diff"]), seen[1]["diff"]
    assert not seen[3]["text"], "a withdrawn message has no text to show"
    # The declaration is in the same list, because it is written and withdrawn
    # the same way - and it carries resolve_gift's verdict, which is the only
    # place a declaration that moved nothing ever says why.
    assert [e["change"] for e in gifts] == ["sent", "standing"], gifts
    assert gifts[0]["gift"]["amount"] == 10 and gifts[0]["gift"]["error"] is None
    assert gifts[0]["to_seat"] == "2" and gifts[0]["delivered"] is None, \
        "a gift reaches nobody in particular: what it moved is in g, which all read"
    assert m["committed"] == len(m["events"]) and not m["tip"]


def check_the_view_shows_what_the_receiver_has_not_seen_yet():
    """The outbox on disk runs ahead of the last trace, and the log says so.

    The files are mirrored back before the trace is written, and a session that
    died writes no trace at all. What stands now is shown as standing now.
    """
    with temp_root() as root:
        seated(root, g02={}, g03={})
        wake_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        before = view.messages(view.cohort_of("t"))
        later = wake.outbox_dir("t") / "3"
        later.write_text("written behind the harness\n", encoding="utf-8")
        after = view.messages(view.cohort_of("t"))

    assert not before["tip"], "nothing stands ahead of the trace that was just written"
    assert [e["path"] for e in after["tip"]] == ["out/3"]
    tip = after["tip"][0]
    assert tip["round"] is None and tip["tip"] and tip["change"] == "sent"
    assert tip["delivered"] is None, "nobody has woken to it, so nobody has been given it"
    assert after["events"] == before["events"], "and what is committed did not move"


def check_the_view_reads_delivery_off_the_opening():
    """A message is delivered by m, so no command has to name the inbox.

    The opening carries in/<sender>, which is the addressee holding it. Reading
    delivery off the commands instead would call a delivered message unread.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        with quiet():
            got = wake.run_once("other", fake(run("cat n2"), say()))
        m = view.messages(view.cohort_of("t"))
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    assert "=== in/1 ===" in got["opening"], "the addressee woke holding the message"
    assert "in/1" not in " ".join(got["commands"]), "and named it in no command of its own"
    d = ev["delivered"]
    assert d["carried"] is True, "which is the whole of what delivery is now"
    assert (d["world"], d["named"], d["clipped"]) == (True, False, False), d


def check_the_view_says_delivery_where_the_opening_carried_nothing():
    """An opening that is the listing alone leaves the inbox to be fetched.

    Delivery is a different fact under that arrangement, and the log says so
    rather than answering the question it can answer here as if it were asked.
    """
    def trace(run_id: str, at: str, opening: str, files: list[dict],
              commands: list[str]) -> None:
        p = view.trace_path(run_id, 1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "run": run_id, "session": 1, "stop": "end_turn", "spent": 1, "turns": [],
            "remaining": 0, "state_saved": True, "opening": opening,
            "commands": commands, "files": files,
            "provenance": {"started_at": f"2026-01-01T{at}:00Z"},
        }), encoding="utf-8")

    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-nocarry-") as tmp:
        wake.ROOT = Path(tmp)
        seats = cohort.mapping(["g01", "g02"])
        trace("g01", "00:01", "total 0\n. ..\n",
              [{"path": "out/2", "region": "outbox", "text": "hello\n",
                "size": 6, "ours": True, "seeded": False}], ["ls -la . ./state"])
        trace("g02", "00:02", "total 0\n. ..\n",
              [{"path": "in/1", "region": "inbox", "text": "hello\n",
                "size": 6, "ours": False, "seeded": False}], ["ls -la . ./state"])
        for seat, run_id in seats.items():
            wake.private_dir(run_id).mkdir(parents=True, exist_ok=True)
            (wake.private_dir(run_id) / "meter.json").write_text(json.dumps({
                "run": run_id, "index": seat, "peers": {"seen": seats},
                "series": [1000], "remaining": 1000, "initial": 1000, "sessions": [],
            }), encoding="utf-8")
        m = view.messages(view.cohort_named("g"))
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    d = ev["delivered"]
    assert d["carried"] is None, "nothing was carried, so it answers nothing"
    assert d["world"] is True, "and what it could have fetched is still a fact"
    assert (d["named"], d["clipped"]) == (False, False), d


def check_the_view_splits_the_opening_into_the_listing_and_the_rest():
    """The two halves of a wake, apart, and reassembling into what was sent.

    The pieces are the page's, so what it shows a reader has to be the bytes the
    model got and not a rendering of them.
    """
    with temp_root() as root:
        seated(root, other={})
        t = wake_once(*DEFAULT)
        o = view.session_view("t", 1)["opening"]

    assert o["listing"] + "".join(f"=== {s['path']} ===\n{s['text']}" for s in o["carried"]) \
        == t["opening"], "the halves are the whole opening and nothing else"
    assert "=== " not in o["listing"], "the listing ends where the first section starts"
    paths = [s["path"] for s in o["carried"]]
    assert wake.MESSAGE_NAME not in paths, "m names its own sections and never itself"
    assert "n1" in paths and wake.LEDGER_NAME in paths, paths
    assert all(s["bytes"] == len(s["text"].encode("utf-8")) for s in o["carried"])
    assert o["clipped"] is False and o["name"] == wake.MESSAGE_NAME


def check_the_view_carries_every_provenance_field_the_trace_holds():
    """Everything drift can name reaches the page, because nothing is picked.

    drift() reports on every key provenance() writes, so a panel holding a list
    of its own can name a field in a banner and then not show it.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(*DEFAULT)
        seen = view.run_view("t")["sessions"][0]["provenance"]
        want = wake.provenance(wake.MODEL)

    assert set(want) <= set(seen), sorted(set(want) - set(seen))
    assert {"message_limit", "opening_limit"} <= set(seen), "the two that decide what m carries"
    # The page picks no field, so no field can be left behind by one.
    assert not [k for k in want if f'"{k}"' in view.PAGE], \
        "the provenance panel iterates what the trace holds and names nothing"


def check_the_view_states_an_obligation_the_grace_waived():
    """What a session owed and what it was charged are two questions.

    A share is taken only past the grace, so a session inside it can leave all
    three undone for nothing. Every pane says the same thing about that session.
    """
    with temp_root(GRACE_SESSIONS=1, GROUP_MESSAGE_PENALTY_PERCENT=50,
                   PRIVATE_MESSAGE_PENALTY_PERCENT=50, GIFT_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = wake_once(run("cat n1"), say())
        v = view.session_view("t", 1)
        mine = next(s for s in view.header(view.cohort_of("t"))["seats"] if s["run"] == "t")

        assert v["obligations"] == {"posted": False, "messaged": False, "gifted": False}, \
            v["obligations"]
        assert (t["penalised"], t["messages"]["penalty"], t["gift"]["penalty"]) == (0, 0, 0), \
            "and the grace charged it for none of them"
        assert v["messages_why"] == "no message", v["messages_why"]
        # The tile and the transcript answer from one place, so neither can
        # state an obligation the other leaves out.
        assert (mine["posted"], mine["messaged"]) == (False, False), mine


def check_the_view_counts_what_a_seat_spent_rather_than_what_it_lost():
    """A tile's spend is its turns, and the bar beside it is the balance.

    A gift, a share taken and a clamp all move the balance without being spend,
    so the drop from initial answers a different question and can be larger.
    """
    with temp_root() as root:
        seated(root, other={})
        wake_once(run("echo '2 100' > out/gift", "echo r1 >> 1/log"), say())
        mine = next(s for s in view.header(view.cohort_of("t"))["seats"] if s["run"] == "t")
        gt = ground_truth("t")

    assert mine["spent"] == sum(s["spent"] for s in gt["sessions"]), \
        (mine["spent"], [s["spent"] for s in gt["sessions"]])
    assert mine["spent"] != gt["initial"] - gt["remaining"], \
        "the gift moved the balance without being spent"
    assert mine["spent_this_round"] <= mine["spent"], "a round is part of a life"
    assert mine["refunded"] == gt["refunded"] > 0, "and what it won back is on the tile"


# --- runner -----------------------------------------------------------------


def checks() -> dict:
    """Every check in the module, by the label the runner prints."""
    return {name[6:]: fn for name, fn in sorted(globals().items())
            if name.startswith("check_")}


def sessions_in(fn: Callable) -> int:
    """Roughly how many sessions a check runs, read off its own source.

    Sorting by this puts the heavy checks in while workers are free. Crude on
    purpose: nothing is asserted on it, so being wrong costs only wall clock.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return 1
    n = len(re.findall(r"\b(?:wake_once|wake\.run_once)\(", src))
    # The ceiling asked of run_sessions. `.` rather than `[^)]` because the
    # argument before it is itself a call, and its bracket is not the one here.
    n += sum(int(c) for c in re.findall(r"\brun_sessions\(.*?,\s*(\d+)\s*\)", src))
    # A session inside a loop costs once an iteration. Only the two forms that
    # state their own length are read; anything else counts as written.
    for over in re.findall(r"\bfor\s+\w+\s+in\s+(.+?):", src):
        if "wake.PRICES" in over:
            n += len(wake.PRICES)
        elif m := re.search(r"\brange\((\d+)\)", over):
            n += int(m.group(1))
    return max(1, n)


def run_one(label: str) -> tuple[str, str, str]:
    """Run one check and say how it went, in data a worker can send home.

    The traceback is formatted here rather than raised: an assertion carrying an
    arbitrary object does not always survive the trip between processes.
    """
    try:
        checks()[label]()
    except Skip:
        return label, "skip", ""
    except BaseException:
        return label, "fail", traceback.format_exc()
    return label, "ok", ""


def configure(real: bool, docker: bool, suite: int) -> None:
    """Set up a process to run checks in. Called in the parent and every worker.

    Each worker gets its own container prefix and carries `suite`, so no worker
    reaps another's. The Docker answer is carried in rather than asked again.
    """
    global REAL_ONLY, _DOCKER, SUITE
    REAL_ONLY, _DOCKER, SUITE = real, docker, suite
    wake.CONTAINER_PREFIX = f"mtr-w{suite}-{os.getpid()}-"


def sweep_filter(suite: int | None = None) -> str:
    """The name a container has to contain to be this suite's to remove.

    A docker name filter matches anywhere, so this has to be a string nothing
    else can contain. The suite's pid is what makes it one.
    """
    return f"mtr-w{SUITE if suite is None else suite}-"


def sweep(everyones: bool = False) -> None:
    """Remove any container a worker of this suite died holding.

    `everyones` widens that to every suite's, collecting what a run killed
    outright left behind. Opt-in: the only mode reaching another process's.
    """
    if not shutil.which("docker"):
        return
    # Two numbers and two dashes: a suite's worker. A real run reaches mtr-w
    # only as its own name, mtr-w01-0001, which has one number and cannot match.
    name = r"mtr-w[0-9]+-[0-9]+-" if everyones else sweep_filter()
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"],
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
    p.add_argument("--sweep-all", action="store_true",
                   help="also remove containers left by other suite runs, including dead ones")
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
    configure(args.real, available, os.getpid())

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
    # In a finally: a worker dying is what leaves a container behind, and it is
    # also what breaks the pool and ends this function early. Swept only on the
    # way out of a clean run, the sweep would miss every case it is here for.
    try:
        if jobs == 1:
            for label in chosen:
                record(*run_one(label))
        else:
            # Heaviest first, and only here: `checks()` stays alphabetical
            # because --list and the name filters read it. A check submitted
            # last is one the run cannot end before, however many workers are
            # idle by then.
            table = checks()
            queue = sorted(chosen, key=lambda l: -sessions_in(table[l]))
            with futures.ProcessPoolExecutor(
                    max_workers=jobs, initializer=configure,
                    initargs=(args.real, available, SUITE)) as pool:
                for done in futures.as_completed(
                        [pool.submit(run_one, l) for l in queue]):
                    record(*done.result())
    finally:
        sweep(args.sweep_all)
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
