# Why this work bypasses Grounding DINO's label post-processing

A methods note for Milestone D. Measured on the Chactun canonical split,
2026-09-16, with `IDEA-Research/grounding-dino-base`.

## The mechanism

Grounding DINO does not classify a box. It aligns box queries to text tokens by
cross-attention, and the library's post-processor then recovers a label by
thresholding per-token similarity at `text_threshold`. That threshold is a
single global gate applied across the whole prompt, which produces three
failures on a multi-category prompt:

**Token-level leakage.** One box often scores highly across adjacent or related
tokens, so the recovered label runs across phrase boundaries and merges two
categories into one string.

**Truncation.** A multi-token phrase can have only part of its tokens clear the
gate, so the label comes back as a fragment of the intended category.

**Length dependence.** This is the one that matters most for an experiment that
varies prompt wording. The gate is per token, so a longer prompt spreads
similarity across more tokens, fewer of them clear the bar, and more boxes
return an empty label and are discarded. Recall then depends on how many words
the prompt happens to contain.

## What was measured here

On four tiles, with class assignment left to the library post-processor at
`text_threshold=0.25`:

| prompt set | phrase length | boxes kept | boxes discarded |
|---|---|---|---|
| `building. platform. water reservoir.` | shortest | 4 | 23 |
| `mound. rectangular structure. circular pit.` | short | 14 | 155 |
| `ancient maya building. stone platform. reservoir.` | long | 9 | 42 |
| `small rectangular mound. raised rectangular terrace. dark circular depression.` | longest | 10 | 384 |

382 of the longest prompt's 384 discarded spans were **empty strings** -- boxes
with no token above the gate. The remaining two show the other two failures
directly: `'rectangular mound rectangular terrace'` merges two categories into
one label, and `'small rectangular mound dark circular de'` truncates mid-word
across a phrase boundary.

The experiment this note supports compares prompt wordings against each other.
Left uncorrected, the table above would have reported a recall penalty
proportional to word count as though it were a property of the prompts. It would
have produced a "short prompts win" result that is an artifact of the gate.

## What replaces it

Class assignment does not use `text_threshold` at all. Each box is scored
against every phrase's token span directly from the logits, taking the maximum
over that span, and is assigned the argmax phrase. Every box receives a class;
none is dropped for failing a gate.

A fixed score floor was then also rejected, because it is not neutral either: a
maximum over a longer token span is stochastically larger, so the same floor
admits more candidates from longer prompts. The bias simply inverts. Each arm
instead receives an identical budget of the top 100 boxes per tile, which is
COCO's own `maxDets`.

Implementation: `scripts/chactun_openvocab.py`. This departs from the thresholds
named in `milestone_d_preregistration.md`, which were frozen before the dry run
exposed the problem; the departure is recorded here rather than folded in
silently.

## Scope of this note

This concerns label mapping only -- how a box that Grounding DINO has already
proposed gets a category. It says nothing about whether the box is in the right
place, and it adds no verification of what the box contains. Both remain open.
The per-class results point at where verification would pay: prompt set 4
reached platform AP 6.88 against building AP 0.15 on the same run, so candidate
recall exists for some classes while precision does not.
