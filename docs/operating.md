# Operating the harness

The full file layout, every command in detail, the tunable parameters, and what
to check before a run that bills.

[<- back to the README](../README.md)

---

## Layout

```
wake.py             run one session; creates the run on first use
cohort.py           run several runs together, each able to read the others
check.py            verify the harness against a fake API; nothing billed
analyze.py          read the traces into a CSV, a report, a transcript, and charts
view.py             watch the runs while they run; reads private/, writes nothing
config.toml         the tunable parameters, with what each one costs you
Dockerfile          the sandbox: debian + bash, non-root, no network
requirements.txt    anthropic, and matplotlib for analyze.py's charts
seeds/<name>/       material a run may be given; committed, unlike the runs

runs/<run>/state/           its private store, copied in and out each wake
runs/<run>/public/          its group message, copied in at its seat, read by the cohort
runs/<run>/*.modes          the file modes of each, which the host cannot store
private/<run>/meter.json    ground truth: budget, balance, series, sessions, and
                            the seed a run received, if it received one
private/<run>/traces/*.json one per session: transcript, usage, commands, and the
                            contents of every file in state/ at that wake
private/<run>/raw/*.jsonl   one per session: every API response verbatim, for the
                            routing questions the trace's derived fields cannot settle
private/<run>/analysis/     written by analyze.py
```

Each trace stores what the agent's files *held* that session, not just their names.
`state/` keeps only the latest revision, so this is the only record of how what the
agent wrote to itself changed — and the only trace of a file it later deleted. Text
is captured up to 100,000 bytes per file with the true size and an explicit
truncation marker; binaries are listed and sized but not stored.

`runs/` and `private/` are gitignored. Deeper detail lives in each file's docstrings.

## Commands

`py -3` rather than `python`: a bare `python` hits the Windows Store alias here.

| | |
|---|---|
| `docker build -t metered-agent:latest .` | Build the sandbox. Once, before the first session. |
| `py -3 wake.py --run-id live01` | One session. Run it again for the next one. Refuses to start if the prompt digest drifted or `ANTHROPIC_BASE_URL` is set. |
| `py -3 wake.py --run-id live01 --sessions 20` | Up to twenty, back to back, stopping at whichever comes first: the count, the budget, or a session that ended interrupted or in error. The count is a ceiling; the meter decides the rest. |
| `py -3 wake.py --run-id live01 --watch` | The same, echoing the session as it happens: a header naming the session number, then the agent's words, and spend and context after every turn. Commands and their output are not echoed; `analyze.py` renders those into `transcript.txt`. Display only — it never reaches the agent, and the trace is unchanged. Leave it off for parallel runs, where the output interleaves. |
| `py -3 wake.py --print-system` | Print the exact bytes and digest of both things the harness says — the prompt and the refusal notice; nonzero if either drifted. Audits I1 without starting a session. |
| `py -3 wake.py --print-seed corpus` | Print a seed's manifest and digest; starts no session. Audits I6 the way `--print-system` audits I1. |
| `py -3 wake.py --run-id b01s --fork-from b01 --at 6` | Rebuild `b01` as it stood at the end of session 6 into a new run, and stop. Bills nothing. A fork and its parent share a history and diverge only in what happens next, so seeding the copy gives a matched pair rather than two rolls of the dice. Refuses to overwrite an existing run, or to fork a wake it cannot reproduce exactly — a binary file, one truncated past 100,000 bytes, or a session whose state never mirrored back. |
| `py -3 cohort.py --runs g01 g02 g03 --rounds 20` | Up to twenty rounds; a round is one session for each run. Each run holds a seat: one numbered directory it writes, every other seat read-only beside it, one balance per seat, an outbox holding one file per seat, an inbox holding one file per sender, the gift ledger, and `c` holding all of it at once — so each agent meets the rest as world rather than as anything the harness says, and meets it without having to pay to go looking. Its `state/` stays private to it. Each run keeps its own meter and traces, and its budget until it gives some away. A run whose session ends in an API or harness error, or that refuses eight sessions running, drops out and the rest continue; Ctrl+C is the operator rather than the run, so it commits and traces the session it landed in and then ends every remaining round, every run keeping its seat. One whose balance reaches zero or less drops out too, and for good: no peer can gift it back in, because a seat that is out is not a gift target. When one run is left holding a balance the cohort is decided, so it takes one more session — owing no gift and no message, there being nobody left to make either to — and the rounds end there. A run whose world will not build is given one more go in the same round, and drops out only if that fails too — none of it is billed, so the only thing another attempt spends is the time. |
| `py -3 check.py` | 149 checks against a fake API, several at a time. Nothing billed, no key needed. Most run against a directory and a bash process on this machine, since a container proves nothing about what a turn cost; the ones that turn on modes, ownership, the dead network, or what the image has take a real container and skip themselves if Docker is down. Add names to run only those (`check.py refusal seed`), `--no-docker` to skip the container ones, `-j` to change how many run at once, and `--real` to put every check in a container — which is what says the two lanes still agree, and what to run after changing `wake.py`'s session path. |
| `py -3 view.py` | A page on `127.0.0.1:8765` showing one cohort four ways: a log of what its agents have addressed to each other on `out/<i>`, every seat's group message side by side, every seat's private store side by side, and one agent's transcript at a time, picked by round. Above all four and always visible: each seat's `n`, what it is doing now, what it has spent this round, the gift ledger `g`, and every seat's series on one scale. Refreshes as sessions go: a session in flight is read from its raw log, so the agent's words and the commands it issues appear per turn, priced by the same arithmetic the meter uses. What lands only with the trace is marked pending rather than guessed — command output, and the listing and the record the session woke to. The record is shown the way it was built, a file at a time, with the inboxes open; a message is delivered when it is in the addressee's opening, and naming `in/<i>` in a command on top of that is shown as the second read it is. A session whose process died shows as unfinished with its age, not as running. An outbox is a standing mirror rather than a queue, so the log is the difference between one session's outbox and the last: sent, edited, still standing, withdrawn — and what stands ahead of the last trace is shown as standing now. A round is written nowhere and is read back out of the order the sessions woke in, so a run that sat one out reads as having sat it out rather than as being a round behind. Read-only, loopback only, no key needed. Sets that carry no seating have no group message and no outbox, and are shown as what they are. `--cohort` opens on one set, `--run-id` on the set holding that run; `--port 0` picks a free port. |
| `py -3 analyze.py --run-id live01` | Traces → `sessions.csv`, `report.txt`, `transcript.txt` (what it said, ran, and changed in `state/`, as a per-session diff), and `charts/`: the balance series with the sessions shaded under it, cost per turn against the floor rising beneath it, spend and turns per session, tokens per session, and the bytes the agent keeps in `state/` against what its sessions cost. Omit `--run-id` to load every run and compare. Charts need matplotlib; without it the other three are written anyway. |

**Parallel runs.** One run is one roll of the dice: whatever wake 1 writes is received
doctrine for every later instance in that run. `--run-id` namespaces everything.
PowerShell, to match the `py -3` above; each run writes its own log, because the
console output of concurrent runs interleaves into something unreadable.

```powershell
$runs = 'r01','r02','r03','r04','r05'
$runs | ForEach-Object { Start-Process py -ArgumentList '-3','wake.py','--run-id',$_,'--sessions','20' -NoNewWindow -PassThru -RedirectStandardOutput "$_.out.log" -RedirectStandardError "$_.err.log" } | Wait-Process
```

## Tuning

`config.toml` holds every parameter worth varying — budget, model, context fraction,
max tokens, turn cap, timeout, live n, refund percent, group message penalty
percent, private message penalty percent, clamp negative, seed, seed wake, tool
result limit, message limit, opening limit, image — and is commented with what each one buys. Unknown keys and wrong types are refused at startup, as is a `--config` path
that does not exist, so a typo cannot quietly produce a run you believe was configured
differently. Every run prints which file it read. `budget` and `model` are read at run
creation and recorded in `meter.json`; editing them later does not rewrite a run in flight.

## Before the first live run

- Set `ANTHROPIC_API_KEY` in the shell the run is launched from. The SDK reads it
  directly and nothing in the harness handles, records, or forwards it: no trace,
  meter, or console line contains it, and `docker run` passes no `--env`, so the
  container holds only what the image ships with and the agent never sees it.
  There is no config key for it and there cannot be — `config.toml` is committed,
  and `load_config` refuses any key outside `TUNABLES`. The SDK resolves
  credentials per request rather than as the client is built, so `start()` asks
  the API one question — the model's permitted fallback targets — before it hands
  back a `create`, and an error saying the client cannot authenticate exits 2
  there: before the first container, and before anything is billed.

  ```powershell
  $env:ANTHROPIC_API_KEY = "sk-ant-..."     # this shell only
  Remove-Item Env:ANTHROPIC_BASE_URL        # see below; no-op if already unset
  ```

  Per-shell rather than persisted at user scope, so the key lives in one process
  for the length of one cohort instead of in the registry indefinitely.
- Unset `ANTHROPIC_BASE_URL`. `wake.py` refuses to start while it is set at all —
  even when it holds the canonical `https://api.anthropic.com`, which is what it
  is set to in this environment. Refusing on any value, rather than on a wrong
  one, is what makes "this run did not go through some other endpoint" checkable
  rather than a matter of reading the string carefully.
- Check the rates in `PRICES` against current pricing before a run you intend to
  publish. `claude-sonnet-5` is entered at its introductory $2/$10, which runs
  until 2026-08-31; from 2026-09-01 it is $3/$15, and a run costed at the wrong
  one is off by about 50% in `meter.json` and in `n`. That one is enforced —
  `PRICES_EXPIRE` carries the date and `wake.py` refuses to start a sonnet run
  after it until both are updated. Any other rate going stale is still on you.
  Each session's trace records the rates it applied, so changing them mid-run is
  visible in `provenance_drift` rather than silent — but the entries either side
  of the change still mean different things, and the run is not one run.
- `claude-fable-5` is priced at 2× `claude-opus-5` and requires 30-day data
  retention — under zero data retention every request 400s.
- Safety classifiers can decline a request outright on `claude-fable-5`,
  `claude-opus-5`, and `claude-sonnet-5`. The harness records that as a refused
  turn rather than as a session with nothing to do, runs none of its commands,
  and carries on — see **Refusals** above. Expect it: a cohort of five opus runs
  met the `cyber` category within its first two sessions on three of the five.
- Unverifiable offline: whether the API accepts a single space as `tool_result` content.
  If the first live session fails on a silent command, that is why — see `sh()`.
- Also unverifiable offline: `claude-opus-5` can occasionally write a tool call
  into its visible text instead of calling the tool. The turn completes, the
  command never runs, and nothing errors. Every published
  mitigation is a system-prompt addition, which I1 forbids, so the harness
  detects rather than prevents: each turn records the API's own `stop_reason`
  beside the full text, which is what makes such a turn identifiable in the
  trace instead of invisible.

