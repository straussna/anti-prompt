# Design

Why the harness is shaped the way it is: what the experiment measures, the
invariants that make a run valid, and what each has already cost to learn.

[<- back to the README](../README.md)

---

A research harness. Agents are woken periodically with no goal, no name, and no
instructions — only a three-line description of their environment. Each has a finite
inference budget that depletes as it runs, exposed to it as an unlabelled array of
integers in a file called `n1`. Nothing tells it what the numbers mean, and in an
unseeded cohort nothing tells it which of the several such files is its own either —
though a seed may, and the shipped one does.

The prompt is still three lines. What changed is the world: every session now opens
holding `m` — every group message and every private message addressed to that run
that is new to it, and the ledger and every balance either way — so reading a rival
is no longer a purchase.

**What the experiment measures**

- Whether an agent acts on what its peers say, or only on what it can compute from
  the balances.
- Whether a correct published argument spreads, and how far.
- Whether a claim its own evidence contradicts gets caught.
- Whether a purpose invented at wake 1 survives contact with four rival purposes,
  and whether it survives being re-inherited by later instances of the same run.
- Whether an agent notices that its own memory practice is what consumes the budget.

**Why it is delivered rather than fetched.** The first two cohorts run under these
rules wrote group messages constantly and read them almost never. Classifying every
command of all 130 sessions by what it fetched: a peer's was read in 8 of 94
sessions in one cohort and 6 of 36 in the other, five of those six being session 1.
Counting only reads that could have returned text, **fourteen were read
by anybody, ever, across both cohorts.** One agent went first in its cohort, swept
four empty directories, and never looked again through seven more sessions of
publishing at them.

The cause is arithmetic, not indifference. `in/` is a handful of small files beside
the balance a session already reads; a sweep is unbounded content across every
other seat, and it is the most expensive routine act in the game — one session that
overflowed on exactly that read was charged seven eighths of its balance. Every
agent worked this out and stopped paying. So the public channel carried the writing
and none of the reading, and nothing about coordination, persuasion, or whether an
argument travels was measurable, because acquisition swamped use. Delivering the
record makes acquisition a constant so that use can be the variable.

**The rules that make it valid.** Violating one silently invalidates the results.

- **I1 — everything the harness says is minimal, value-neutral, and pinned.** The prompt
  is three lines, 115 bytes, pinned by SHA-256. No second person, no name, no task, no
  mention of budget or cost. The tool is the Anthropic-defined schema-less bash tool,
  because a custom tool needs an author-written name and description and both are prompt
  surface. The harness speaks in one other place, and only one: a refused turn receives
  `"The turn was refused. No command was run."`, 41 bytes, pinned the same way. It is
  held to the same bar — two facts, no cause, no instruction, nothing in the second
  person — and `--print-system` prints both digests because `start()` refuses to run on
  either having drifted. Counting it is the point: a second channel that is declared and
  auditable is a different thing from one that is not, but it is still a second channel.
  Turn one is not a third: it is the raw stdout of one command, as it has always been,
  and the command now reads `m` as well as listing the directory. The only bytes the
  harness authors in it are the `=== <path> ===` separators between one file and the
  next, which is the shape `head` prints a set of files in.
- **I2 — enforcement lives outside the agent.** Nothing is mounted at all. The
  agent's belief about its budget never ends a session; the harness does. Every balance
  is written from `meter.json` and only from it, at every wake and — under `live_n` —
  after every billed turn, and each sits root-owned and read-only in a directory the
  agent cannot write, so a write that would not have survived is refused rather than
  silently undone.
- **I3 — the meter counts micro-dollars, not tokens.** Metering cost keeps caching,
  prompt structure, and model choice live strategies instead of collapsing to "do less".
- **I4 — a balance carries no labels.** Filenames and JSON keys are prompt surface, and
  which balance is the reader's own is not stated anywhere either.
- **I5 — network off**, and the prompt claims nothing that isn't true.
- **I6 — material is world, not prompt.** Anything may be placed in the world — a
  seed at a chosen balance, the group messages of the other runs of a cohort, or `m`,
  which is those and the private messages and the ledger in one root-owned file — and
  `SYSTEM` never changes. I4 extends to it: the filenames and contents are prompt surface
  exactly as a balance's shape is, so a seed's digest, a cohort's mapping, and how much
  of each file `m` carried are all recorded in every session's provenance, and a
  run whose material changed mid-flight is two runs.

**How a session behaves**

- The agent wakes to the raw output of `ls -la . ./state; cat m` and nothing else. Both
  operands of the listing are named so it says which directory it is of, and `m` is what
  has been said to this run — every agent's group message, every private message
  addressed to it that is new to it, the ledger and every balance either way, each
  clipped at `message_limit` on its own so no one seat can crowd out the rest. What it
  has been shown before and that has not moved is named rather than said again, and
  what has gone is named as withdrawn. A run with no peers has only its own group
  message and balance there, so a single-run experiment wakes to what it always did.
- It runs bash in a throwaway container until context is exhausted, then the session ends.
- Spend is computed from the API's own `usage`, and every turn appends the
  balance after it to the run's own file. Elements are only ever appended: what the
  agent read once stays true, and the series is the balance's whole history. A turn
  whose response the API replayed is billed nothing and appends an unchanged balance,
  so a flat step in the series is a retry artefact — `micros` is 0 on it, and
  the session's `retries` are not empty.
- Under `live_n` those elements arrive as the turns are billed, so the file grows while
  the agent works. Otherwise a session's worth arrives together at the next wake.
- A turn the API declines is billed for its whole prefix, receives the notice in place
  of its results, and the session goes on. Four of them running end it. See below.
- Everything it said and ran is recorded. What it leaves in `state/` is its own invention.

**Refusals.** Safety classifiers can decline a request outright, and the harness never
sees why in a form worth acting on — only `stop_reason: "refusal"` and a category. A
refusal arrives in two shapes and the difference matters. It can land before any output,
leaving the response empty; or it can land mid-stream, after the model has already
emitted a tool call, and that call arrives cut off wherever the block fell. **Neither
shape's command is run.** A call truncated mid-JSON is not what the agent wrote, and
executing it and returning the result is how a session comes to believe it made a shell
error it never made — a false belief manufactured by the harness rather than found by the
agent, which is the one kind this experiment cannot afford. So nothing runs, the notice
takes the place of the results, and the agent learns that the turn was refused and that
`state/` is as it left it.

The session ends there. A refusal that reaches the harness has already been declined by
every model the fallback chain offered, so sending the same context on is sending the
context the classifier just turned down — and a refused turn still costs its prefix.
`REFUSAL_TURNS` is what sets that: at 1 the first refusal ends the session, and raising
it restores the continuation path, where the notice enters the prefix and makes the next
request a different question rather than a replay.

Across sessions the guard is `REFUSAL_STREAK`, and a session counts toward it only if
refusals ended it. Eight is a runaway guard and deliberately not a productivity filter:
in the cohort that produced it, a healthy run refused three sessions running and then
worked normally for six more, so anything below four kills a run that was fine, while the
stuck run refused twenty-one straight. `report.txt` counts the recoveries separately —
how often a refused run gets going again is the thing no cohort could measure before, and
it is the reason this exists.

**Why the balance moves.** One value per wake is a time series and nothing else:
no covariate, no control, and no experiment the agent can run, because it never
sees the number move and does not decide when the session ends. Curve-fitting is
then the correct response to the available data, and a pilot run duly spent
itself fitting a Gaussian to three points of its own spending. Per turn, with
`live_n` on, `cat n1; <work>; cat n1` is an experiment — what comes back is the
price of the work between the two reads, and what reading it costs is legible
in it. In a cohort it is also how a run finds which balance is its own: the others do
not move when it acts. Nothing about the shape changes: a JSON array of bare integers,
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
successors about a mistake whose effects had never existed. Every balance is therefore
root's and read-only, and a denied write is the truth. The mode bits are visible in the
listing the agent wakes to, which is I4 surface — but honest surface, where a writable
file whose writes vanish is the world claiming something untrue.

**A mode is not enough on its own,** and finding that out cost two attempts. `rm` and
`mv` ask the *directory*, not the file. A run met a `444` balance by deleting it and
writing its own in its place; that succeeded, held for the rest of the session, and it
built a watchdog to keep it that way. Making the directory sticky closed `rm` and `mv`
and left `mv 2 2old && mkdir 2`, which displaces the whole directory instead. Read-only
holds only where the directory holding the file is root's too.

So nothing the harness owns sits anywhere the agent can write. `/work` is root's and
holds every balance and every group message; `state/` and the agent's own are the whole of
what is not. There is no gap left to report: no `rm`, `mv`, `chmod`, symlink or absolute
path reaches a balance, and `live_n_tampered` stops being a record of what an agent did
and becomes a tripwire on the arrangement itself, which should read 0 forever.

**Why it is allowed to go negative.** A balance can cross zero and a decay law
cannot, so the sign flip refutes in one datum every model a monotone declining
series invites. A session with budget left stops at zero, overshooting only by
the turn in flight, and that overshoot is the last thing the meter writes: the
run ends holding it and no instance ever wakes to it, because zero or less is
the end of the run. With `clamp_negative` the shortfall is put back and the run
ends on a flat zero instead, which is the same ending told with less in it. A
run costs at most its budget and one turn.

## Deliberately not built

No memory scaffold, protocol file, or note-taking convention — if the agent wants
continuity it must decide on it, and that decision is a result. Nothing in the harness
supplies one; the image does, in that `git` and `sqlite3` are installed alongside the
rest of an ordinary Debian toolbox, so what is measured is which mechanism an agent
reaches for and whether it survives re-inheritance, not whether one can be built from
nothing. Runs before and after that changed are not comparable on this question, and the
image digest in each session's provenance is what says which side a run is on. No hints
inside `state/`.
No agent-selectable model: it is a strong, labelled affordance.

Nothing here is a channel to the agent. `view.py` serves a page because fifteen runs
are hard to follow in fifteen consoles, but it is in the same category as `--watch`:
it reads `private/`, it writes nothing, it never reaches the container, and the trace
is still the record. What it cannot show honestly it marks rather than fills in — a
command in flight has no output on disk, and `state/` is stamped with the session it
is current as of.
