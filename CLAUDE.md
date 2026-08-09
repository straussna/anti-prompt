# Running check.py

`check.py` is the verification suite. Nothing in it bills an API. Some of it
starts Docker containers, and that is the only part that is slow.

**Run the cheapest thing that answers the question you actually have.** Work down
this ladder and stop at the first rung that covers what you changed.

| what you changed | run |
|---|---|
| one behaviour, and you know its name | `py -3 check.py <name-fragment>` |
| pricing, metering, refusals, traces, seeds, forks, cohort, gifts, the ledger, messages, the post penalty, the message penalty, the gift penalty, the grace, the clamp | `py -3 check.py --no-docker` |
| anything, before handing work over | `py -3 check.py` |
| `wake.py`'s session path — the container, the shell, `load_state`/`save_state`, `run_once` | `py -3 check.py --real` |

Rough costs: a name filter is seconds, `--no-docker` about 25s, the full run
about 40s, `--real` two to four minutes.

`py -3 check.py --list` prints every name. Fragments match anywhere, and several
can be given at once: `py -3 check.py refusal fallback`.

## Why there are two lanes

Most checks are arithmetic — what a turn cost, what reached the series, which
stop a session ended on. A container proves none of that, so those sessions run
in a directory and a bash process on this machine.

What only a container can show — modes, ownership, the dead network, what the
image has and lacks — takes a real one. That includes the checks that an inbox
and the gift ledger really are root's and really do refuse every route into
them. Those are the checks that skip when Docker is down, and the reason
`--no-docker` still runs 116 of 135.

`--real` puts every check in a container. It is what says the two lanes still
agree, so run it after changing how a session is set up or torn down. It is not
the default, and it is not what to run to check an assertion you just edited.

## Things that will waste your time

Do not run the full suite repeatedly to watch a number. Run it once when the
work is done.

Docker Desktop slows down markedly after a few hundred containers. A full run
that took 40s on a fresh daemon can take two minutes later in a long session.
That is the daemon, not a regression — restart Docker rather than hunting it.

A suite run only removes containers carrying its own pid, so two runs at once
leave each other alone and no run of `check.py` can touch a live experiment.
Nothing collects what a run killed outright leaves behind: `--sweep-all` does,
and is the only mode that reaches a container this process did not make.

Two checks are wall-clock sensitive by design: `hostile_output_survives` (a 4MB
flood against a deadline) and anything setting `TIMEOUT`. Running with `-j` above
the core count makes them fail for contention rather than for cause. The default
`-j` is already sized for this machine; lower it before raising it.

# Stopping a run early

One `Ctrl+C` ends the run cleanly. It does not raise: it sets a flag that the
turn loop reads where it reads the meter floor, so the session ends the way an
exhausted budget ends it — the turn in flight finishes, its spend is committed,
the agent's trees are mirrored back, the trace is written and the container is
reaped. A cohort ends every remaining round, and every run keeps its seat, so it
can be started again from where it stopped.

The cost is latency: worst case one whole turn, which is one API call plus the
commands it asks for. Press `Ctrl+C` a second time to stop waiting — the handler
puts the default back before it returns, so the second one is the ordinary hard
stop. That abandons the session: the container leaks and the spend never reaches
`meter.json`. `SIGTERM` behaves the same way.

Nothing in the repo reaps a container left by a hard kill — `check.py`'s sweep is
scoped to its own pid and cannot match `mtr-<run>-<index>`. Remove those by hand.

# Editing wake.py

`wake.py` hashes itself at import and records the digest in every trace, and
`check_the_harness_digest_is_read_once` compares that against the file on disk.
So do not edit `wake.py` while a suite run or an experiment is in flight — the
running process will disagree with the file and the check fails for a reason
that has nothing to do with the change.
