# Milestone D preregistration: open-vocabulary detection on Chactun

Written 2026-09-16, before any run over the canonical split. Both predictions
below were recorded before a single tile was scored. The point of writing them
down is that the two of us disagree about the main effect, and a result that
can be read either way afterwards is worth nothing.

## Protocol, frozen

| | |
|---|---|
| tiles | the 329 canonical validation tiles, 1765-2093 (`fold9_val.json`) |
| ground truth | 1614 annotations: 1231 building, 370 platform, 13 aguada |
| metric | bbox AP via COCOeval, the same path that scored the supervised model |
| box threshold | 0.05, so AP integrates the full precision-recall curve |
| text threshold | 0.25 |
| prompt syntax | bare nouns, period-separated, no articles |

Masks are not scored: neither open-vocabulary model emits them, so the
comparison is bbox AP on both sides.

**Arms.** Grounding DINO base on the Z8; OWL-ViT and OWLv2 on the laptop. Every
model runs all four prompt sets.

## The four prompt sets form a 2x2

This was not deliberate when the sets were drafted, and it is the most useful
thing about them. They cross *what the prompt names* against *how long the
phrase is*:

| | short phrase | long phrase |
|---|---|---|
| **semantic** (names the class) | 1. `building. platform. water reservoir.` | 2. `ancient maya building. stone platform. reservoir.` |
| **morphological** (describes the render) | 4. `mound. rectangular structure. circular pit.` | 3. `small rectangular mound. raised rectangular terrace. dark circular depression.` |

That matters because two different mechanisms could produce the same ranking,
and the 2x2 separates them. Naming could help or hurt on its own; phrase length
could help or hurt on its own. A design that only contrasted 1 against 3 would
confound the two.

## Registered predictions

Both recorded 2026-09-16, before any tile was scored.

**Ben.**

1. Grounding DINO beats OWL-ViT on sets 3 and 4 - the morphological prompts.
2. Both models score higher on sets 3 and 4 than on 1 and 2. Describing the
   shape beats naming the class.

**Claude.**

3. Sets 3 and 4 beat 1 and 2, on the reasoning that CLIP-style pretraining has
   seen "mound" and "rectangular" captioned against images and has never seen
   "ancient maya building" against a sky-view-factor raster.
4. All four land far below the supervised model. The open question is whether
   any of them clears zero.

## Ben's prediction is the finer-grained one

The two sets are not rivals. On the main effect they agree: both expect
morphological prompts to win, and prediction 4 is not in dispute. Ben's set adds
what Claude's does not have - a cross-family ordering, Grounding DINO over
OWL-ViT on sets 3 and 4 specifically. That is the sharper claim in the document
and the one the experiment can most cleanly confirm or refute, because it names
a direction between two arms rather than between two prompt styles.

The cost of agreeing on the main effect is worth stating plainly: a
preregistration whose authors agree has little power to adjudicate it. It does
not sort between two live accounts, it records what we both expected to see.

The consequence is that the third outcome listed below - short prompts win,
naming is irrelevant - carries more weight than either registered prediction,
not less. It is the only outcome neither of us is invested in, and therefore the
one most likely to be explained away if it occurs. Whoever writes up the result
should test it explicitly rather than confirm the shared prediction and stop.

## What would distinguish the mechanisms

Ben's prediction 1 has a specific basis worth recording, because it also
predicts *where* the models should diverge. Grounding DINO's text encoder
grounds arbitrary phrases; OWL-ViT was built around short class-name queries in
a `a photo of a {}` frame. If that is the operative difference, the gap between
the two families should open up on the LONG sets (2 and 3) and stay narrow on
the SHORT sets (1 and 4) — regardless of whether the phrase is semantic or
morphological.

So the 2x2 admits three distinguishable outcomes:

- **Naming wins, length irrelevant** — 1 and 2 beat 3 and 4 in both families.
  Ben's prediction 2 holds.
- **Short wins, naming irrelevant** — 1 and 4 beat 2 and 3. Neither of us
  predicted this, and it would mean the result is about phrase length rather
  than about semantics at all.
- **Describing wins** — 3 and 4 beat 1 and 2. Claude's prediction 3 holds.

Recording the second possibility now matters more than either of the first two,
because it is the one that would otherwise be reported as whichever registered
prediction it happened to resemble.

## Positive control

Established on the Z8 before any Chactun number is trusted: Grounding DINO on
SpaceNet 2 Vegas RGB returns 53-209 boxes per tile, 0-6% of them covering the
whole tile, median box area 0.7-5.2% of the tile. The same code on a Chactun
render returns 3-4 boxes with a third to a half covering the whole tile. The
laptop runs the equivalent control for OWLv2 on the COCO demo image before its
arm counts.

Without the control, "the model found nothing" and "the code is wrong" are the
same observation.
