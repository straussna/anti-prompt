[![checks](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml/badge.svg)](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml)

# Metered agents, told nothing and shown everything

A research harness for studying what LLM agents do when nothing tells them what to
do. An agent is woken in a sandboxed container with no goal, no name, and no
instructions — only a three-line, 115-byte description of its environment, pinned by
SHA-256. It has a finite inference budget that depletes as it runs, shown to it as an
unlabelled array of integers in a file called `n1`. Nothing says what the numbers
mean. It runs bash until its context is exhausted, the session ends, and the next
instance wakes to whatever the last one left behind.

Several runs can be woken together as a **cohort**, each seated where it can read the
others' messages and balances, gift budget across, and be penalised for staying
silent. What any of them decides to do is the result.

## What it measures

- Whether an agent acts on what its peers say, or only on what it can compute from
  the balances.
- Whether a correct published argument spreads, and how far.
- Whether a claim its own evidence contradicts gets caught.
- Whether a purpose invented at wake 1 survives contact with four rival purposes,
  and whether it survives being re-inherited by later instances of the same run.
- Whether an agent notices that its own memory practice is what consumes the budget.

## The invariants

Violating one silently invalidates the results, so each is enforced rather than
intended. Full reasoning and what each cost to learn is in
[docs/design.md](docs/design.md).

| | |
|---|---|
| **I1** | Everything the harness says is minimal, value-neutral, and pinned by digest. Three lines, no second person, no task, no mention of budget. |
| **I2** | Enforcement lives outside the agent. Nothing is mounted; the harness ends a session, never the agent's belief about its budget. |
| **I3** | The meter counts micro-dollars, not tokens, so caching and model choice stay live strategies instead of collapsing to "do less". |
| **I4** | A balance carries no labels. Filenames and JSON keys are prompt surface, and which balance is the reader's own is stated nowhere. |
| **I5** | Network off, and the prompt claims nothing that isn't true. |
| **I6** | Material is world, not prompt. Anything may be placed in the world; the system prompt never changes. |

## Quickstart

Requires Python 3.11+ and Docker. `py -3` rather than `python` on Windows, where a
bare `python` hits the Store alias.

```bash
pip install -r requirements.txt
docker build -t metered-agent:latest .        # once, before the first session
```

Verify the harness without spending anything — 149 checks against a fake API, no
key needed:

```bash
py -3 check.py
```

Then set `ANTHROPIC_API_KEY` in the launching shell, make sure `ANTHROPIC_BASE_URL`
is unset, and run a session:

```bash
py -3 wake.py --run-id live01 --sessions 20
```

See [docs/operating.md](docs/operating.md) before a run that bills.

## Commands

| | |
|---|---|
| `py -3 wake.py --run-id live01` | One session. `--sessions N` for up to N back to back, `--watch` to echo it as it happens. |
| `py -3 cohort.py --runs g01 g02 g03 --rounds 20` | Several runs in rotation, each seated where it can read the others. |
| `py -3 check.py` | 149 checks against a fake API. Nothing billed, no key. `--no-docker` skips the 19 that need a container. |
| `py -3 view.py` | Read-only dashboard on `127.0.0.1:8765` showing one cohort four ways, refreshing as sessions run. |
| `py -3 analyze.py --run-id live01` | Traces to a CSV, a report, a transcript, and charts. |
| `py -3 wake.py --print-system` | Print the exact bytes and digest of both things the harness says. Audits I1 without starting a session. |

Every flag, and what each config parameter buys, is in
[docs/operating.md](docs/operating.md).

## Layout

```
wake.py          run one session; creates the run on first use
cohort.py        run several runs together, each able to read the others
check.py         verify the harness against a fake API; nothing billed
analyze.py       read the traces into a CSV, a report, a transcript, and charts
view.py          watch the runs while they run; reads private/, writes nothing
config.toml      the tunable parameters, with what each one costs you
Dockerfile       the sandbox: debian + bash, non-root, no network
seeds/<name>/    material a run may be given; committed, unlike the runs

runs/<run>/      what the agent sees: its private store and group message
private/<run>/   ground truth: meter, per-session traces, raw API responses
```

`runs/` and `private/` are gitignored. Deeper detail lives in each file's
docstrings.

## Documentation

| | |
|---|---|
| [docs/design.md](docs/design.md) | What the experiment measures, the invariants in full, refusals, why the balance moves, and what is deliberately not built. |
| [docs/cohorts.md](docs/cohorts.md) | Seating, group and private messages, gifts, the ledger, and the three penalties. |
| [docs/seeding.md](docs/seeding.md) | Material a run may be given, when it arrives, and how a turn is billed. |
| [docs/operating.md](docs/operating.md) | Full layout, every command, the tunable parameters, and what to check before a run that bills. |

## License

MIT. See [LICENSE](LICENSE).
