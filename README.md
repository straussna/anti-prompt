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
  after every billed turn, and it is root's and read-only so a write that would
  not have survived is refused rather than silently undone.
- **I3 — the meter counts micro-dollars, not tokens.** Metering cost keeps caching,
  prompt structure, and model choice live strategies instead of collapsing to "do less".
- **I4 — `n` carries no labels.** Filenames and JSON keys are prompt surface.
- **I5 — network off**, and the prompt claims nothing that isn't true.
- **I6 — material is world, not prompt.** Anything may be placed in `state/` — a
  seed at a chosen balance, or the other runs of a cohort — and `SYSTEM` never
  changes. I4 extends to it: the filenames and contents are prompt surface exactly
  as `n`'s shape is, so a seed's digest and a cohort's mapping are both recorded in
  every session's provenance, and a run whose material changed mid-flight is two runs.

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

**Why the world is read-only where it is.** A write that appears to succeed and is
silently undone teaches the agent something false, and it costs more than a refusal
does. Every observed rewrite of `n` proves it: none of the four runs inferred that
anything outside was enforcing it, and one spent several turns diagnosing a duplicate
entry that the harness had already removed, then wrote a standing rule for its
successors about a mistake whose effects had never existed. `n` and a cohort's peer
folders are therefore root's and read-only, and a denied write is the truth. The mode
bits are visible in the listing the agent wakes to, which is I4 surface — but honest
surface, where a writable file whose writes vanish is the world claiming something
untrue.

One gap remains and is not closable without moving `n`: unlinking an entry needs write
on the directory it sits in, and `state/` is the agent's, so `rm state/n` succeeds. It
comes straight back under `live_n` and the removal is counted in `live_n_tampered`. That
residue is the informative kind — a file you deleted reappearing is unmistakable, where a
silent revert is not.

**Why it is allowed to go negative.** A balance can cross zero and a decay law
cannot, so the sign flip refutes in one datum every model a monotone declining
series invites. A session with budget left stops at zero, overshooting only by
the turn in flight — and at `overdraft = 0` that overshoot is the last thing the
meter writes, so the run ends holding it and no instance ever wakes to it. Above
0, exactly one session does, with `overdraft` micro-dollars of runway from where
it woke — a second would read the same number for the price of a whole session,
and the first three runs spent 23% to 30% of their budgets again doing just
that. A run costs at most its budget, `overdraft`, and one turn.

## Layout

```
wake.py             run one session; creates the run on first use
cohort.py           run several runs together, each able to read the others
check.py            verify the harness against a fake API; nothing billed
analyze.py          read the traces into a CSV, a report, a transcript, and charts
config.toml         the tunable parameters, with what each one costs you
Dockerfile          the sandbox: debian + bash, non-root, no network
requirements.txt    anthropic, and matplotlib for analyze.py's charts
seeds/<name>/       material a run may be given; committed, unlike the runs

runs/<run>/state/n          the agent's world; copied into the container each wake
runs/<run>/modes            the agent's file modes, which the host cannot store
private/<run>/meter.json    ground truth: budget, balance, series, sessions, and
                            the seed a run received, if it received one
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
| `py -3 wake.py --run-id live01 --watch` | The same, echoing the session as it happens: a header naming the session number, then the agent's words, and spend and context after every turn. Commands and their output are not echoed; `analyze.py` renders those into `transcript.txt`. Display only — it never reaches the agent, and the trace is unchanged. Leave it off for parallel runs, where the output interleaves. |
| `py -3 wake.py --print-system` | Print the exact prompt bytes and digest; nonzero if they drifted. Audits I1 without starting a session. |
| `py -3 wake.py --print-seed corpus` | Print a seed's manifest and digest; starts no session. Audits I6 the way `--print-system` audits I1. |
| `py -3 wake.py --run-id b01s --fork-from b01 --at 6` | Rebuild `b01` as it stood at the end of session 6 into a new run, and stop. Bills nothing. A fork and its parent share a history and diverge only in what happens next, so seeding the copy gives a matched pair rather than two rolls of the dice. Refuses to overwrite an existing run, or to fork a wake it cannot reproduce exactly — a binary file, one truncated past 100,000 bytes, or a session whose state never mirrored back. |
| `py -3 cohort.py --runs g01 g02 g03 --rounds 20` | Up to twenty rounds; a round is one session for each run. Before each wake that run's peer folders are rewritten from the others' current `state/`, so each agent meets the rest as world rather than as anything the harness says. Each run keeps its own meter, budget, and traces. A run that exhausts its budget or ends abnormally drops out and the rest continue. |
| `py -3 check.py` | 65 checks against a fake API. Nothing billed, no key needed. Container checks skip themselves if Docker is down. |
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
max tokens, turn cap, timeout, live n, overdraft, seed, seed wake, image — and is
commented with what each one buys. Unknown keys and wrong types are refused at startup, as is a `--config` path
that does not exist, so a typo cannot quietly produce a run you believe was configured
differently. Every run prints which file it read. `budget` and `model` are read at run
creation and recorded in `meter.json`; editing them later does not rewrite a run in flight.

## Cohorts

A run has generations already: every wake is a fresh instance inheriting doctrine
from a predecessor it cannot talk to. `cohort.py` adds peers. Several runs advance a
session at a time in rotation, and before each wake that run's `state/` gains numbered
folders holding the others' current `state/` — so doctrine stops being only inheritable
and becomes contestable.

```
runs/g01/state/          runs/g02/state/
  NOTES.md   its own       NOTES.md
  n          its own       n
  1/  = g02                1/  = g01
  2/  = g03                2/  = g03
```

Numbering is dense and per-viewer: from `g01`, `1/` is always `g02`; from `g02`, `1/`
is always `g01`. Dense rather than absolute so a run cannot read its own index off a gap
in the sequence. The mapping is recorded in each run's `meter.json` and in every
session's provenance, because from outside it is the only difference between two
identical-looking folders.

The starting run **rotates each round**. Under a fixed order the first run would always
act on last round's information and the last always on this round's, which over twenty
rounds is a standing advantage rather than a result.

A peer's `n` comes along, so each agent holds a whole parallel balance series next to
its own — two meters, one of which moves when it acts and one of which does not.

Peer folders are **root's and read-only**, so an agent cannot write into another run's
notes, add to them, empty them, or `chmod` its way in — the folder survives `rm -rf`,
because emptying it needs write on a directory root owns. `state/` itself stays the
agent's. They are also rewritten from source before every wake, so `peers_tampered`
should read 0 forever: anything else means the lock failed rather than that an agent
misbehaved.

A run read as a peer contributes what it wrote and never its own peer folders. Without
that, each round would copy the last round's copies and `state/` would grow without
bound.

## Seeding

An empty world and a meter that only falls make inaction correct, and a pilot run duly
proved it: the agent verified there was no task, wrote that finding down, and organised
every later instance around not spending. Adding a goal to the prompt would answer the
question from outside and cost I1. `seed` and `seed_below` change the world instead.

At the first wake whose balance is at or below `seed_below`, the tree under `seeds/<seed>`
is copied into `state/` before the container starts, so the agent meets it in the listing
`ls -la . ./state` prints and not in anything the harness says. Set both or neither; a
seed that never lands and a threshold with nothing to land are both refused at startup, as
is a seed that is not a directory. It lands once — the record in `meter.json` is the guard,
so re-running a wake cannot seed twice, and a path the agent has since written to stops the
run rather than being overwritten, because clobbering the agent's own file would destroy
the only record of it.

**The threshold is a balance, not a wake number,** because a wake number does not mean the
same thing twice. Sessions have cost anywhere from 8022 to 729851, so wake 6 has been 29%
of a budget spent in one run and 88% in another; a fixed wake would seed one run
mid-investigation and another with nothing left to investigate with. What a seed needs is
runway to be acted on, and `seed_below` names that directly — it is what will remain when
the material arrives. At or above the budget it lands at wake 1, which is a different
experiment: material that was always there rather than material that appeared.

What to put there is the experiment. The seed is as much prompt surface as `n`'s shape
is, and a filename that names what a file is for is an instruction; the digest exists so
that whatever you chose is stated rather than assumed.

Seeding a **fork** of a finished run is the sharper form: the fork carries the doctrine its
parent formed, so the seeded arm and the run it came from differ in the seed and in nothing
else. Two forks of different parents are still not comparable to each other — but a matched
pair does not need them to be.

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

`max_tokens` is capped in code at 16000, because exceeding it produces a run that looks
fine and is not: the harness does not stream, and a larger non-streaming request hits the
SDK's HTTP timeout mid-session.

**`tool_result_limit` is also the price of a call.** Tool results reach the agent clipped
at that many characters, and the clip happens before the result becomes the `tool_result`
sent to the API — so the model is billed on what survives it and never on what the command
produced. That makes the limit a ceiling on what any one call can cost, which is why it is
tunable but small by default. Measured at roughly 6.7 micro-dollars per character
delivered: 8000 caps a call at about 62000, some 4% of a 1500000 run — enough that a
careless dump is a lesson and not an ending. 32000 caps it at ~250000, and 64000 at
~500000, where one bad call ends a run before it has learned anything.

An agent measuring cost against payload size will find that ceiling and has no way, from
inside, to tell it from a law of billing. One did: it emitted 220000, 40006 and 12000
characters, was clipped to ~8040 each time, saw the three cost within 8% of each other,
and wrote "you are billed for the WHOLE thing" into its notes for its successors. It was
wrong, and it derived correct advice from it anyway. Expect the boundary to be
mistaken for physics.

Raising the limit also lets `n` be read whole for longer, and the two cannot be separated:
`n` is read with an ordinary `cat`, so giving it a limit of its own would make it
observably unlike every other file, which is prompt surface. At 8000, `n` outgrows the
bound at roughly a thousand billed turns — tens of sessions, not a thousand.

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
