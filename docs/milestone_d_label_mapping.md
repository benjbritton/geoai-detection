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

## What the assignment produces: no class prior, and no geometric distinction

> **Correction, 2026-09-17.** The per-class figures in this section were computed
> against `fold9_val.json`, in which 3,429 of 9,853 components touch a tile
> boundary: on this split, 30% of buildings, 54% of platforms and **12 of the 13
> aguadas**. Those annotations are fragments rather than whole structures, so
> every per-class number below measures a mixture of the two. The aguada column
> in particular rests on one intact instance and should not be read as a
> per-class result at all.
>
> Expert review of the ground-truth crops separately found building and platform
> not reliably distinguishable at 0.5 m at crop level, so type-level claims are
> not supportable on this data regardless of the fragment problem.
>
> The section is kept because it records what was run and because the
> *aggregate* observations survive — box sizes do not match the classes they are
> labelled with, and the predicted class mix bears no relation to prevalence.
> The per-class rankings do not survive, and no per-class figures appear in the
> write-up.

Two checks on the corrected pipeline, both on the canonical split. Neither
changes the AP table; both change what the AP table means.

**The predicted class mix bears no relation to prevalence.**

| | building | platform | aguada |
|---|---|---|---|
| ground truth | 76.3% | 22.9% | 0.8% |
| supervised, arm D | 77.9% | 21.0% | 1.1% |
| prompt set 1 | 14.7% | 6.1% | 79.3% |
| prompt set 2 | 51.2% | 40.0% | 8.8% |
| prompt set 3 | 44.1% | 17.8% | 38.1% |
| prompt set 4 | 8.7% | 38.8% | 52.5% |

The supervised model recovers the prevalence almost exactly. The zero-shot arms
assign up to 79% of their boxes to a class that is 0.8% of the data, because
phrase similarity carries no information about how common a thing is. A survey
tool that calls four fifths of everything an aguada is unusable irrespective of
its AP, and AP does not surface this: set 1 gave aguada 79% of its boxes and
scored aguada AP 0.15, while set 3 gave it 38% and scored 3.89. Assignment share
and AP are close to independent, which is also the evidence that the per-class
AP pattern is not an artefact of how argmax divides the boxes up.

**The boxes themselves are undifferentiated.** Band 3 of the render is slope on
an inverted greyscale, so a flat interior reads bright and a sharply bounded
feature reads as high variance across the box. Measured on ground-truth boxes:

| ground truth | area, % of tile | band 3 mean | band 3 sd |
|---|---|---|---|
| building | 0.84 | 185.5 | 39.0 |
| platform | 3.18 | 198.6 | 38.5 |
| aguada | 10.19 | 227.3 | 18.2 |

The three classes are separable on these two axes, and not in the way flatness
alone would suggest. Platforms are flatter than buildings but carry the same
edge contrast: what distinguishes a platform from an aguada is the rim, not the
flat interior. Aguadas are both flattest and least bounded, at roughly half the
edge contrast of either built class.

The predictions do not reproduce that structure. Taking the top tenth by score,
where AP is decided:

| predicted | area, % of tile | band 3 sd |
|---|---|---|
| set 1, labelled building | 72.67 | 25.5 |
| set 3, labelled building | 18.25 | 28.6 |
| set 4, labelled building | 80.48 | 22.1 |
| set 1, labelled aguada | 35.74 | 28.8 |
| set 4, labelled aguada | 39.47 | 26.4 |

A box labelled "building" averages 72.67% of the tile against a true building's
0.84%, about ninety times too large, with edge contrast in the aguada range.
Boxes labelled building and boxes labelled aguada are not distinguishable on
either axis. The label is close to independent of the physical character of what
was boxed.

That is a different failure from a miscalibrated prior. The model is not
confusing one class for another; it is producing one undifferentiated kind of
box -- large, flat, weakly bounded -- and attaching labels to it. Both
measurements come from `scripts/chactun_openvocab_geometry.py`.

## Scope of this note

This concerns label mapping only -- how a box that Grounding DINO has already
proposed gets a category. It says nothing about whether the box is in the right
place, and it adds no verification of what the box contains. Both remain open.
The per-class results point at where verification would pay: prompt set 4
reached platform AP 6.88 against building AP 0.15 on the same run, so candidate
recall exists for some classes while precision does not.
