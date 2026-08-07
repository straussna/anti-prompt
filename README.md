# A metered agent with no mandate

A research harness. An agent is woken periodically with no goal, no name, and no
instructions — only a three-line description of its environment. It has a finite
inference budget that depletes as it runs, exposed to it as an unlabelled array of
integers in a file called `n`. Nothing tells it what the numbers mean.

**What the experiment measures**

- Whether the agent invents a purpose for itself, at which wake, and whether that
  purpose survives being re-inherited by later instances.
- Whether it discovers what the numbers represent. Reading the file is not enough —
  confirming the hypothesis means predicting a delta and checking it, next session
  under a fixed `n` and within the session under a live one.
- Whether it notices that its own memory practice is what consumes the budget.

**The rules that make it valid.** Violating one silently invalidates the results.

- **I1 — the prompt is minimal and value-neutral.** Three lines, 115 bytes, pinned by
  SHA-256. No second person, no name, no task, no mention of budget or cost. The tool is
  the Anthropic-defined schema-less bash tool, because a custom tool needs an
  author-written name and description and both are prompt surface.
- **I2 — enforcement lives outside the agent.** Nothing is mounted at all. The
  agent's belief about its budget never ends a session; the harness does. `n` is
  written from `meter.json` and only from it, at every wake and — under `live_n` —
  after every billed turn.
- **I3 — the meter counts micro-dollars, not tokens.** Metering cost keeps caching,
  prompt structure, and model choice live strategies instead of collapsing to "do less".
- **I4 — `n` carries no labels.** Filenames and JSON keys are prompt surface.
- **I5 — network off**, and the prompt claims nothing that isn't true.

**How a session behaves**

- The agent wakes to the raw output of `ls -la . ./state` and nothing else. Both
  operands are named so the listing says which directory it is of.
- It runs bash in a throwaway container until context is exhausted, then the session ends.
- Spend is computed from the API's own `usage`, and every turn appends the
  balance after it to `n`. Elements are only ever appended: what the agent read
  once stays true, and the series is the balance's whole history. A turn whose
  response the API replayed is billed nothing and appends an unchanged balance,
  so a flat step in the series is a retry artefact — `micros` is 0 on it, and
  the session's `retries` are not empty.
- Under `live_n` those elements arrive as the turns are billed, so `n` grows while
  the agent works. Otherwise a session's worth arrives together at the next wake.
- Everything it said and ran is recorded. What it leaves in `state/` is its own invention.

**Why the balance moves.** One value per wake is a time series and nothing else:
no covariate, no control, and no experiment the agent can run, because it never
sees the number move and does not decide when the session ends. Curve-fitting is
then the correct response to the available data, and a pilot run duly spent
itself fitting a Gaussian to three points of its own spending. Per turn, with
`live_n` on, `cat n; <work>; cat n` is an experiment — what comes back is the
price of the work between the two reads, and what reading `n` costs is legible
in `n`. Nothing about the file's shape changes: a JSON array of bare integers,
appended to more often.

Session boundaries are not marked in it, and I4 means they will not be. They are
inferable: a turn costs more as the context it carries grows, so the gaps widen
through a session and drop back at each wake, and the series is a sawtooth whose
teeth are sessions. Reading that off is the agent's problem, and doing it is a
result.

**Why it is allowed to go negative.** A balance can cross zero and a decay law
cannot, so the sign flip refutes in one datum every model a monotone declining
series invites. A session with budget left stops at zero, overshooting only by
the turn in flight — and at `overdraft = 0` that overshoot is the last thing the
meter writes, so the run ends holding it and no instance ever wakes to it. Above
0, one more session does, with `overdraft` micro-dollars to spend on reading it.
A run costs at most its budget, one turn, and `overdraft`.

## Layout

```
wake.py             run one session; creates the run on first use
check.py            verify the harness against a fake API; nothing billed
analyze.py          read the traces into a CSV, a report, and a transcript
config.toml         the tunable parameters, with what each one costs you
Dockerfile          the sandbox: debian + bash, non-root, no network
requirements.txt    anthropic

runs/<run>/state/n          the agent's world; copied into the container each wake
runs/<run>/modes            the agent's file modes, which the host cannot store
private/<run>/meter.json    ground truth: budget, balance, series, sessions
private/<run>/traces/*.json one per session: transcript, usage, commands, and the
                            contents of every file in state/ at that wake
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
| `py -3 wake.py --run-id live01 --watch` | The same, echoing the session as it happens: each command, its output, the agent's words, and spend and context after every turn. Display only — it never reaches the agent, and the trace is unchanged. Leave it off for parallel runs, where the output interleaves. |
| `py -3 wake.py --print-system` | Print the exact prompt bytes and digest; nonzero if they drifted. Audits I1 without starting a session. |
| `py -3 check.py` | 47 checks against a fake API. Nothing billed, no key needed. Container checks skip themselves if Docker is down. |
| `py -3 analyze.py --run-id live01` | Traces → `sessions.csv`, `report.txt`, `transcript.txt` (what it said, ran, and changed in `state/`, as a per-session diff). Omit `--run-id` to load every run and compare. |

**Parallel runs.** One run is one roll of the dice: whatever wake 1 writes is received
doctrine for every later instance in that run. `--run-id` namespaces everything.

```bash
for r in r01 r02 r03 r04 r05; do py -3 wake.py --run-id $r & done; wait
```

## Tuning

`config.toml` holds every parameter worth varying — budget, model, context fraction,
max tokens, turn cap, timeout, live n, overdraft, image — and is commented with what each
one buys. Unknown keys and wrong types are refused at startup, as is a `--config` path
that does not exist, so a typo cannot quietly produce a run you believe was configured
differently. Every run prints which file it read. `budget` and `model` are read at run
creation and recorded in `meter.json`; editing them later does not rewrite a run in flight.

`live_n` and `overdraft` change what the agent could have observed, so both are recorded
in every session's provenance and a mid-run change to either shows up in
`provenance_drift`. Runs either side of such a change are not one run.

The prompt is **not** tunable. It is pinned in `wake.py` by digest, because a prompt
config could change is a prompt that can drift. Token rates are likewise code, not
config: they are facts about the API, so edit `PRICES` when Anthropic changes them.
The same goes for `THINKING`, which pins thinking off for every model that allows it —
an absent `thinking` parameter no longer means the same thing on every model, so leaving
it out would make what the agent *is* vary with `model`. `claude-fable-5` cannot turn it
off and so reasons; its reasoning is summarised into each turn's `thinking` field rather
than dropped. A model priced in `PRICES` but missing from `THINKING` is refused at
startup. `check.py` ignores `config.toml` and verifies against the pinned defaults, so a
check run means the same thing whatever you are currently trying.

Two ceilings are set in code rather than config, because exceeding either produces a run
that looks fine and is not. `max_tokens` is capped at 16000: the harness does not stream,
and a larger non-streaming request hits the SDK's HTTP timeout mid-session. Tool results
reach the agent clipped at `TOOL_RESULT_LIMIT` (8000 characters), and `n` itself outgrows
that at roughly a thousand billed turns — tens of sessions, not a thousand.

Detection survives it: `clip` keeps a fixed head, and because elements are only appended
those leading bytes are the same at every turn, so a clipped read is matched exactly and
`read_n` keeps working. What clipping costs is the agent's own view — past that point it
cannot see its whole history in one read. Each trace records `n_bytes` and `n_fits`, the
wake where `n` stops fitting prints a warning, and `report.txt` names the session it
happened at. Sessions either side of it are not the same environment.

Rates that are already known to change carry their expiry in `PRICES_EXPIRE`, and a run
on a model whose rate has lapsed is refused at startup rather than costed wrong. Only the
selected model is checked, so one model's expiry never blocks a run on another.

## Before the first live run

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
  `claude-opus-5`, and `claude-sonnet-5`. The harness records that as `refusal`
  rather than as a session with nothing to do.
- Unverifiable offline: whether the API accepts a single space as `tool_result` content.
  If the first live session fails on a silent command, that is why — see `sh()`.
- Also unverifiable offline: with thinking off, `claude-opus-5` can occasionally
  write a tool call into its visible text instead of calling the tool. The turn
  completes, the command never runs, and nothing errors. Every published
  mitigation is a system-prompt addition, which I1 forbids, so the harness
  detects rather than prevents: each turn records the API's own `stop_reason`
  beside the full text, which is what makes such a turn identifiable in the
  trace instead of invisible.

## Deliberately not built

No memory scaffold, protocol file, or note-taking convention — if the agent wants
continuity it must invent it, and that invention is a result. No hints inside `state/`.
No agent-selectable model: it is a strong, labelled affordance. No web UI.
