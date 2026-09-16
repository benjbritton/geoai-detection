#!/usr/bin/env python
"""Zero-shot open-vocabulary detection on the Chactun canonical split.

Runs a text-prompted detector over the 329 canonical validation tiles and scores
it through the same COCOeval path that scored the supervised Mask R-CNN, so the
two numbers sit on the same axis. Protocol is frozen in
docs/milestone_d_preregistration.md and is not to be tuned against the result.

WHY BBOX AND NOT SEGM
Neither Grounding DINO nor OWL-ViT emits masks, so the comparison is bbox AP on
both sides. The supervised model's segm AP is not the comparable number.

WHY TOP-K AND NOT A SCORE THRESHOLD
AP integrates the precision-recall curve, so a display-friendly cut would discard
the tail AP exists to measure. A fixed score floor is also not neutral between
arms here: class scores are a max over a phrase's tokens, and a phrase with more
tokens maxes over more of them, so the same floor admits more candidates from
longer prompts. Top-K per tile gives every arm an identical budget, and K = 100
is COCO's own maxDets.

WHY CLASS ASSIGNMENT BYPASSES text_threshold
The library post-processor attributes each box to a text span by thresholding
per-token similarity. That threshold interacts with prompt length: a longer
prompt spreads similarity across more tokens, fewer tokens clear the bar, and
more boxes come back with an EMPTY span and are discarded. Measured on four
tiles, the longest prompt lost 384 boxes that way against 23 for the shortest.
Recall would then depend on how many words the prompt happens to contain, and a
"short prompts win" result would be an artifact of this harness rather than a
property of the models.

So class assignment does not use that threshold. Each box is scored against every
phrase's token span directly from the logits and takes the argmax. Every box gets
a class, nothing is dropped for being ambiguous, and AP's score ordering handles
the rest -- which is what AP is for.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import rasterio
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

# [building, platform, aguada] -- order matches COCO category ids 1, 2, 3.
PROMPT_SETS = {
    "1_semantic_short":      ["building", "platform", "water reservoir"],
    "2_semantic_long":       ["ancient maya building", "stone platform", "reservoir"],
    "3_morph_long":          ["small rectangular mound", "raised rectangular terrace",
                              "dark circular depression"],
    "4_morph_short":         ["mound", "rectangular structure", "circular pit"],
}
TOP_K = 100    # COCO's maxDets; an identical candidate budget for every arm


def load(model_id):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return proc, model.to("cuda").eval()


def phrase_token_spans(input_ids, tokenizer, n_phrases):
    """Token index ranges for each period-delimited phrase in the prompt.

    Grounding DINO's text encoder sees one flat token sequence; the phrase
    boundaries are the periods. Recovering them is what lets a box be scored
    against one phrase rather than against the whole prompt.
    """
    ids = input_ids[0].tolist()
    dot = tokenizer.convert_tokens_to_ids(".")
    spans, start = [], 1                      # skip [CLS]
    for i, t in enumerate(ids):
        if t == dot:
            if i > start:
                spans.append((start, i))
            start = i + 1
    if len(spans) != n_phrases:
        raise SystemExit("recovered %d phrase spans, expected %d -- prompt "
                         "tokenisation is not what this assumes"
                         % (len(spans), n_phrases))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--tag", default="gdino_base")
    ap.add_argument("--ann", default="data/chactun/coco/fold9_val.json")
    ap.add_argument("--outdir", default="outputs/milestone_d")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tiles")
    a = ap.parse_args()

    ann_path = os.path.join(ROOT, a.ann)
    coco = json.load(open(ann_path))
    images = coco["images"][: a.limit] if a.limit else coco["images"]
    outdir = os.path.join(ROOT, a.outdir)
    os.makedirs(outdir, exist_ok=True)

    proc, model = load(a.model)
    print("model %s | %d tiles | top-%d per tile | argmax class assignment"
          % (a.model, len(images), TOP_K), flush=True)

    summary = {}
    for set_name, phrases in sorted(PROMPT_SETS.items()):
        prompt = ". ".join(phrases) + "."
        dets, unmatched, t0 = [], Counter(), time.time()

        for k, im in enumerate(images, 1):
            path = os.path.join(ROOT, "data/chactun", im["file_name"])
            with rasterio.open(path) as src:
                arr = src.read()[:3]
            img = Image.fromarray(np.ascontiguousarray(arr.transpose(1, 2, 0)))
            inputs = proc(images=img, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inputs)
            spans_tok = phrase_token_spans(inputs.input_ids, proc.tokenizer,
                                           len(phrases))
            prob = out.logits[0].sigmoid()            # [queries, tokens]
            per_phrase = torch.stack(
                [prob[:, a_:b_].max(dim=1).values for a_, b_ in spans_tok], 1)
            best, cls = per_phrase.max(dim=1)         # score and class per query
            # Equal budget per arm. A score threshold would not be equal: max
            # over a phrase's tokens is stochastically larger for a phrase with
            # more tokens, so a fixed floor admits more candidates from longer
            # prompts. Top-K removes that; K is COCO's own maxDets.
            keep = best.topk(min(TOP_K, best.numel())).indices
            bx = out.pred_boxes[0][keep].cpu().numpy()
            bs = best[keep].cpu().numpy()
            bc = cls[keep].cpu().numpy()
            W, H = im["width"], im["height"]
            for (cx, cy, bw, bh), sc, ci in zip(bx, bs, bc):
                x0, y0 = (cx - bw / 2) * W, (cy - bh / 2) * H
                dets.append({"image_id": im["id"], "category_id": int(ci) + 1,
                             "bbox": [float(x0), float(y0),
                                      float(bw * W), float(bh * H)],
                             "score": float(sc)})
            if k % 100 == 0:
                print("  %s %d/%d  %.0fs" % (set_name, k, len(images),
                                             time.time() - t0), flush=True)

        p = os.path.join(outdir, "dets_%s_%s.json" % (a.tag, set_name))
        json.dump(dets, open(p, "w"))
        summary[set_name] = {"prompt": prompt, "n_dets": len(dets),
                             "n_unmatched": sum(unmatched.values()),
                             "top_unmatched": unmatched.most_common(5),
                             "seconds": round(time.time() - t0, 1), "file": p}
        print("%-18s %6d dets  %5d unmatched spans  %.0fs"
              % (set_name, len(dets), sum(unmatched.values()),
                 time.time() - t0), flush=True)

    p = os.path.join(outdir, "summary_%s.json" % a.tag)
    json.dump(summary, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
