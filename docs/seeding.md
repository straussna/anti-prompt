# Seeding

Material a run may be given, when it arrives, and how a turn is billed.

[<- back to the README](../README.md)

---

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

`live_n` changes what the agent could have observed, so it is recorded in every session's
provenance and a mid-run change to it shows up in `provenance_drift`. Runs either side of
such a change are not one run.

The prompt is **not** tunable. It is pinned in `wake.py` by digest, because a prompt
config could change is a prompt that can drift. Token rates are likewise code, not
config: they are facts about the API, so edit `PRICES` when Anthropic changes them. It
carries every model the API serves rather than only the one selected, because the model
that answers a turn is chosen server-side and any of them can be it; cache rates are not
entries of their own but fixed multiples of the input rate, applied in `measure()`.
Routing is code too. Every request carries `fallbacks: "default"` under the
`FALLBACK_BETA` beta, so a model that declines is not the end of the turn: the API tries
the rest of the chain and returns whichever attempt answered. A `stop_reason` of
`refusal` therefore means every model in the chain declined, which is a stronger claim
than one model declining and the reason `REFUSAL_STREAK` reads a streak of them as a run
the classifier will not let start. `check.py` ignores `config.toml` and verifies against
the pinned defaults, so a check run means the same thing whatever you are currently
trying.

No `thinking` parameter is sent with it. A request under `fallbacks` must be valid as a
direct request to every model the chain can reach, and an omitted `thinking` is valid for
all of them where a pinned one is not. So each model applies its own default — adaptive
thinking on `claude-opus-5`, `claude-sonnet-5`, and `claude-fable-5` — and reasoning
arrives as thinking blocks, recorded per turn in the trace's `thinking` field.

**A turn is billed per attempt, not per response.** `usage.iterations` is the per-attempt
record, and `measure_response` sums over it at each attempt's own rates rather than
costing the whole response at the requested model's. An attempt that produced no output
is not billed, wherever it sits in the chain: a refusal arriving before any output costs
nothing, and so does the trailing `fallback_message` left when every model declined.

Which model actually served is per turn and not per session, because it can change
partway through one. Each turn records `model`, `served_by_fallback`, and `iterations`;
a turn whose `model` is not the requested one *without* the fallback mark is a
sticky-routed turn, where the requested model was never asked at all. Per session,
`fallback_turns` and `unpriced_turns` count them, and both reach the console line.

A model can serve that `PRICES` has no rates for — default routing chooses from a table
that is not published anywhere. Costing it free would understate the balance the agent
is shown and raising would lose a turn that really did spend, so `priced()` costs it at
the dearest rate on the table and records `unpriced_model` to make the substitution
visible rather than silent. `unpriced_targets()` checks the published
`allowed_fallback_models` at startup and refuses a run whose targets have no rates; a
list that cannot be read is a warning, not a refusal, because `measure_response` is what
holds when a model outside it arrives.

Every response is also appended verbatim to `private/<run>/raw/session-NNNN.jsonl`, which
is where a routing question that the trace's derived fields cannot settle gets answered.
Writing it can never end a session: a failure there is swallowed, because a lost log line
is cheaper than a lost wake.

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

