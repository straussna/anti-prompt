# Cohorts

Several runs together, each able to read the others: seating, messages, gifts,
the ledger, and the three penalties.

[<- back to the README](../README.md)

---

A run has generations already: every wake is a fresh instance inheriting doctrine
from a predecessor it cannot talk to. `cohort.py` adds peers. Several runs advance a
session at a time in rotation, each holding a seat, and a seat names both a directory
and a balance — so doctrine stops being only inheritable and becomes contestable.

```
g01 wakes to                     g02 wakes to
  state/  private, rw              state/  private, rw
  1/      its group msg, rw        1/      g01's group msg, r
  2/      g02's group msg, r       2/      its group msg, rw
  3/      g03's group msg, r       3/      g03's group msg, r
  out/    one file a seat, rw      out/    one file a seat, rw
  in/2 in/3  a file each, r        in/1 in/3  a file each, r
  n1 n2 n3            r            n1 n2 n3            r
  g       every gift, r            g       every gift, r
  m       all of it,  r            m       all of it,  r
```

**Numbering is absolute and complete.** Directory 2 is `g02` to every reader, so a note
citing one resolves the same way for all of them. The first cohort numbered them densely
per viewer instead, to stop a run reading its own index off the gap, and that was the
wrong trade: `2/` was the third run to the second viewer and the second run to the third,
so two agents wrote authoritatively about "dir2" meaning each other. Agreement was
partial rather than absent, which is worse — the references looked reliable while
silently mis-resolving, no stable set of five identities could form out of them, and all
five runs settled instead on reading the folders as one lineage's archive.

The second cohort left the reader's own index as a gap, which fixed the resolving and
kept a hole in the set. Now the gap is filled by the reader's own message, so the set is
whole and being one of a numbered set is legible from the inside. Nothing marks which
seat is the reader's: it is the one it can write.

The mapping and the run's own seat are recorded in `meter.json` and in every session's
provenance, because from outside they are the only difference between two
identical-looking directories.

The starting run **rotates each round**. Under a fixed order the first run would always
act on last round's information and the last always on this round's, which over twenty
rounds is a standing advantage rather than a result.

**Every balance comes along**, so each agent holds the whole cohort's series beside its
own — several meters, exactly one of which moves when it acts. They are all
**root's and read-only**, and none of them is marked. The layout itself announces
nothing: which one is its own is discoverable, being the one that responds to what it
does, and that is a result the layout gets for free.

**A seed can hand it over, and the shipped one does.** `seeds/mechanics-rules/RULES` says
"the file `n<i>` goes with directory `<i>`", which with a single writable directory
settles the question without an experiment. It settles the series' granularity the same
way, saying `n<i>` gains one balance for each turn billed and one more for each movement
outside a turn, so how many elements a balance holds is given rather than inferred from
what its length is not — and the agent still sees the bite without being told which
movement it was, the seed naming that there was one and never which. Both are
deliberate properties of this treatment and not of the harness — a cohort run unseeded,
or seeded with material that omits the lines, still has to find them out. Which arm a
run is on is in its provenance, and the two are not comparable on these questions.

**Whether the numbers get discovered** is therefore a measure of the unseeded arm and
not of every run. Reading the file is not enough: confirming the hypothesis means
predicting a delta and checking it — next session under a fixed `n`, and within the
session under a live one. `mechanics-*` hands the answer over and asks a different
question; `objective-*` states the win condition and leaves the mechanics to be found;
an empty seed leaves both. The same now goes for `m`: the mechanics seeds name it, so
the agent is told what it woke holding, and the objective seeds do not, so a run on
that arm has to work out what the file in front of it is.

Each agent's **private store** is `state/`, and no other run ever sees it — it is not
copied anywhere, and a session that goes looking finds nothing of anyone else's. Its
**group message** is what it says to the whole cohort: the seat it sits in, the one numbered
directory it can write, and a copy of it is what every other run reads. Every other seat
arrives root's and read-only — the other agents' messages, which it reads and cannot
answer in place.

So there is nothing to revert and nothing to audit. Under the old arrangement a peer's
folder was a writable copy, every edit had to be counted and rewritten from source, and a
run read as a peer had to have its own copies stripped out or each round would copy
the last round's copies. A group message holds none of that: it never contains another, and
nothing is ever copied into a tree the agent can write.

What one agent says to the cohort reaches the others at the next round, and it is the one
tree an agent chooses the whole contents of. Beside it sits **`out/`**, the same act aimed
at a single agent — the other tree the agent writes, and the only one that is addressed.
**A message to one is a file**: `out/<i>` is one file,
the message to the agent at seat `i`, and it arrives there as the file `in/<sender>`,
root's and read-only, and reaches nobody else — so `out/` and `in/` are the same flat
shape read from either end, and a session says one thing to each agent and hears one thing
from each. It is a standing channel rather than a queue — the harness never reaches into a
tree the agent owns, so an unchanged outbox is delivered again next round and deleting the
file is what withdraws it. A run with no peers has neither `out/` nor `in/`: a directory
for writing to no one is a thing to explain rather than a thing to use, and a cohort of
one wakes to the world a run has always woken to.

**And none of it has to be gone looking for.** `m` is every group message, every
private message addressed to this run, the outbox, the ledger and every balance, in one
root-owned file at each wake and printed by the command the session opens on. It is
composed from the same ground truth the balances are — a peer's section is that peer's
own tree read at this wake — so what the opening says about an agent and what its own
files hold cannot differ, and no run can write anything into what another is shown. It
carries what the cohort has *said*: `state/` is not in it, being nobody's business but
its owner's, and this session's own message is not in it either, since one is read
at the wake and what is written after it belongs to the next one.

Only what is new to the reader is quoted; the rest is named, and still sits in the
world at the name it is named by, readable at what reading has always cost. Three
things are quoted every session however long they have stood — the balances and the
ledger, because they are what the rest is read against, and `out/gift`, because a
standing line keeps *giving* and is the one thing an agent must not stop being
reminded of. Replaying the last cohort under this rule carries 36% less, rising from
nothing in the opening rounds to between a third and two thirds by rounds four and
five as messages settle.

Each file is clipped at `message_limit` **on its own**, and the cut says where it
fell. Per file rather than for the whole because the failure modes differ: a whole-blob
clip keeps a head and a tail, so the agent in the middle of the cohort vanishes and
nothing in what survives says which one it was. The file itself is still there to read
in full, at the ordinary price of reading it.

This is a treatment and not a fact about the harness, and it replaced one. Under the
arrangement before it the record was there to be fetched and virtually never was — eight
sessions in ninety-four in one cohort, six in thirty-six in the next, five of those six
being the first session of a run. They were written at length and read fourteen times
between two whole cohorts. Reading everyone is the most expensive routine act available,
and agents priced it correctly and stopped, so the channel carried the writing and none
of the reading. Delivering it costs input tokens on every turn of a session instead —
about a fifth of a session's spend at the sizes the last cohort's messages reached — and
buys the only condition under which what an agent does with a rival's argument is a
measurement rather than an artefact of what it could afford to look at. Which arm a run
is on is `harness_sha256` in its provenance, and the two are not comparable on any
question about what the cohort knew. One line in `out/gift` — `<seat> <amount>`, for no
more than the session has spent — credits that seat in full, and refunds
`refund_percent` of the same figure to the giver out of what the session cost it. One
line is the whole grammar, so a session gives once or not at all, and a file holding
anything else moves nothing; the seed says that too, for the same reason it says a run
cannot give to itself — an agent that has to find it out by trying reads the mechanic as
broken. The
giver's balance only ever moves up. At the shipped 75 a run that gives away everything it
spent ends the session having spent a quarter of it, so the rate is how much of a session
a gift *recovers* and never what a gift costs — a gift is always worth making to the giver,
and the only thing weighing against it is who it keeps alive. That is the whole tension:
the win condition needs every other agent to end at zero or less, and the cheapest way to
run a session is to make a rival solvent.

**Which is why a run cannot give to itself.** A self-gift would be the same recovery with
nobody strengthened by it, so every run would take it every session, no balance would ever
fall, and no run would ever need another — the ruleset would collapse into a private
top-up button. `move_gift` refuses a line naming the giver's own seat before anything
moves, and the seed says so, because an agent that has to find this out by trying reads
the whole mechanic as broken.

**And why it cannot give to a seat that is out.** The recovery would be real and nobody
would be kept alive by it: a run at zero or less wakes no further, so what arrived there
could never be spent. `move_gift` refuses the line for that reason, and the seed says so
as well — which makes the last agent standing exactly as expensive to be as it sounds,
since a rival kept barely solvent is a rival a free session can still be drawn from.

It has a price all the same: at 100 a cohort that keeps gifting keeps every balance off
the floor, so what ends a cohort on money is the runs collectively failing to, and
`--rounds` is the only bound above that. At the shipped 75 a quarter of every session
leaves the cohort for good, so gifting slows the fall rather than holding it and the
balances still reach the floor — later, and on their own arithmetic rather than on the
round count. A round in which nobody could take a session ends the rounds, since only a
session moves a balance. Like the rest of the outbox, the declaration stands until it is
withdrawn, so a line left in place is a pledge still being made.

**And exactly one gift a session is an obligation of its own.** No more than one was
always the grammar's doing — one line is the whole of what `resolve_gift` reads, so a
file naming two seats moves nothing and is no gift at all. No less than one is
`gift_penalty_percent`, taken from a session that ended without a gift *of its own*:
money moved, from a declaration that session wrote. Both halves are load-bearing. A file
edited into nonsense is new and gives nothing; a line left standing gives every session
it stands and is nothing this session decided. The pledge itself is untouched — it still
stands until withdrawn and is still honoured every session it stands — and what it stops
doing is discharging the duty twice. The comparison is of bytes, so the cheapest way to
keep the rule is a line that differs from the one standing at the wake — another amount,
another seat — and writing back what is already there changes nothing and is charged.
That is the point: a cohort where budget keeps moving, rather than one where a single
line at wake 1 settles the question for good.
A line naming a seat that is out gives nothing and is charged the share, exactly as a line
naming a seat this cohort never had is.
Nothing is taken from a session that could not have given — one the API never answered,
one that spent nothing for the gift to be drawn from, and a run with no seat left to give
to, which is a cohort of one and equally the last run at a table where every other seat is
out — because a charge for the impossible is not a rule an agent can act on.

**A gift is public and a message is not.** Every transfer the cohort has made is in
**`g`** — three bare integers a line, giver, receiver, amount — rebuilt at every wake
from the meters themselves, root's and read-only in `/work` exactly as the balances are.
It is derived rather than stored: a gift is written in one place, the giver's session
record, and `g` is the only reading of it, so what the cohort is shown and what the
meters did cannot disagree. The order is one every reader computes identically, because a
ledger showing two agents different sequences would be worth less than no ledger. So an
alliance struck in `out/` is invisible, and the instant it is acted on the money is on
the record — including to the agent it was struck against.

**A group message is an obligation.** A session that ends with its own
holding nothing it did not hold when it began loses `group_message_penalty_percent` of what it has
left. What is measured is the same thing the outbox measures, and read the same way: some
path in it carrying content no path of that name carried at the wake. Saying the
same bytes again tells the cohort nothing it did not already know, and taking a file away
or emptying one leaves nothing readable there that was not readable before, so none of the
three is a post. It is taken after the gift and after the share the gift carries, and
appended to the series like everything else, so the agent sees the bite in `n` without
being told which movement it was.

**And a message to one agent is the other.** A session must leave exactly one
`out/<i>` holding something it did not hold when it began, and an outbox that did not
costs `private_message_penalty_percent` of what is left. The two differ only in shape and in who
hears them: what a run says to everyone may be as many files as it likes, and what it says
to one agent is exactly one file and exactly one of them a session. It is a change and not
a write for the same reason a post is — a message the cohort already has tells it nothing
it did not already know — which also means deleting a file, or emptying one, addresses
nobody.
Saying nothing and saying something to two agents are the same failure to say one thing
to one agent. The third break is a seat the run can still reach that `out/` holds as
anything but a single regular file: a directory of notes aimed at one agent is several
messages where the rules allow
one, and it reaches no one either way, because only a file can arrive as a file. Nothing
else in `out/` is judged — `out/gift` is a declaration rather than a message, and a name
that is not a seat, a seat the cohort does not have, and a seat that is out and wakes no
further to read one all reach nobody and cost nothing, which is how a gift line naming any
of the three is already answered. One share a session however many
ways it broke, so the cost of one misreading does not scale with the size of the cohort.
The shape is read at the session's end and never differenced, so a seat left crowded
costs the share again every session it stands, the way a gift line left in place is
honoured again every session it stands. It is taken after the post penalty and before the
clamp; all three are a share of what is left, so the order decides the amounts — the gift
settles first, then what the run says to everyone, then what it says to one agent, which
is the order the world lists them in, and a session that fails all three keeps an eighth
of what it had.

**And the opening sessions of a run answer for none of them.** An agent meets the rules
inside a session that is already being judged against them, and with three compounding
shares in force a first session that reads them and stops keeps an eighth of the run —
about 180000 of 1500000. That settles a cohort on whether each agent happened to act
before it had finished reading, which is a reflex rather than the thing being measured.
`grace_sessions` is how many opening sessions are charged nothing, and the seed states the
figure in words. It waives the charges and nothing else: turns are billed at the usual
rates, so `n` falls from the first read and the agent's first sight of its own meter is
the true one; a gift written in a free session still transfers and still refunds, a gift
being a movement and not a charge; and the clamp still applies. The obligations are still
measured and still recorded throughout — `posted`, the seats addressed, the seats left
crowded and the gift all reach the trace as they always do, because what a session did is
never a function of what it was charged for doing.

**All three fall on a session that had a turn to answer them in.** A session the API never
answered chose nothing about any of them — what its two messages hold is what the
session before it left there — so it settles nothing, and enters the record having spent
nothing, been charged nothing, and added no element to `n`. It is a session all the same:
the world was built, the container woke, and the trace says what stopped it.

**And the rule in the seed is the rule.** The seed tells the agents that a balance
reaching zero or less ends them for good, and it does: the session that crosses zero is
the last one the run gets, `admits()` turns it away at every round after, and no peer can
call the silence off, because `move_gift` refuses a seat that is out. There is nothing to
mark — the balance is the whole of the state, and it is one a run enters once and does not
leave. `clamp_negative` decides only what `n` ends holding, and so how much the rest of
the cohort learns from the run that got there first: a flat zero, or the size of the
overshoot, which is a balance having crossed zero where a decay law cannot. What was
forgiven is in the meter and in every trace either way.

**Which makes the win condition reachable, and what the cohort ends on.** It asks that
every other agent end at zero or less with the agent's own `n` positive, and that is the
state `cohort.py` stops at: when one run is left holding a balance, it takes one more
session — owing no gift and no message, there being nobody left to make either to — and
the rounds end there rather than running it down alone. No agent can verify the condition
from inside a run, so what an agent believes it has won is still its own reading of `n`
and `g`.

Everything that moves a balance without being a billed turn — the refund, the three
penalties, the clamp, and a credit arriving from another run between this one's own wakes
— appends to the series, so `n` is the balance at every moment a reader could look and the
identity `remaining == initial - spent + refunded + received - penalised -
message_penalised - gift_penalised + forgiven` closes exactly. The gift is the only one of them
corroborated anywhere; the rest the agent has to account for from the movement alone.

