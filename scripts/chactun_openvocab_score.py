#!/usr/bin/env python
"""Score every Milestone D arm through one COCOeval path.

The open-vocabulary arms and the supervised Mask R-CNN are evaluated by the same
code on the same annotations, so the numbers sit on one axis. bbox only: neither
open-vocabulary model emits masks, so segm AP is not the comparable quantity and
the supervised model's segm AP must not be quoted beside these.

Reports the 2x2 the prompt sets form -- semantic vs morphological, short vs long
-- because a ranking of four arms cannot separate those two factors on its own.
"""
import contextlib
import glob
import io as _io
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
ANN = os.path.join(ROOT, "data/chactun/coco/fold9_val.json")
DETDIR = os.path.join(ROOT, "outputs/milestone_d")
SUP = os.path.join(ROOT, "outputs/chactun_D_maskrcnn_d4_augmentation",
                   "fold9_seed0/inference/coco_instances_results.json")

FACTORS = {                      # set -> (names the class?, long phrase?)
    "1_semantic_short": ("semantic", "short"),
    "2_semantic_long":  ("semantic", "long"),
    "3_morph_long":     ("morphological", "long"),
    "4_morph_short":    ("morphological", "short"),
}
CLASSES = {1: "building", 2: "platform", 3: "aguada"}


def evaluate(coco, det_path):
    dets = json.load(open(det_path))
    dets = [d for d in dets if "bbox" in d]
    if not dets:
        return None
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes(dets)
        e = COCOeval(coco, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    per = {}
    for cid, name in CLASSES.items():
        with contextlib.redirect_stdout(_io.StringIO()):
            ec = COCOeval(coco, dt, "bbox")
            ec.params.catIds = [cid]
            ec.evaluate(); ec.accumulate(); ec.summarize()
        per[name] = 100 * ec.stats[0]
    return {"AP": 100 * e.stats[0], "AP50": 100 * e.stats[1],
            "AP75": 100 * e.stats[2], "n": len(dets), "per_class": per}


def main():
    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(ANN)

    rows = []
    for p in sorted(glob.glob(os.path.join(DETDIR, "dets_*.json"))):
        tag = os.path.basename(p)[5:-5]              # dets_<model>_<set>.json
        model, _, sname = tag.partition("_")
        if model == "gdino":                          # tag is gdino_base_<set>
            model, _, sname = tag.split("_", 2)[0] + "_" + tag.split("_", 2)[1], "_", tag.split("_", 2)[2]
        r = evaluate(coco, p)
        if r:
            rows.append((model, sname, r))

    print("Milestone D -- zero-shot open-vocabulary vs supervised, bbox AP")
    print("canonical split, 329 tiles, 1614 annotations\n")
    print("%-14s %-18s %7s %7s %7s %9s %9s %8s"
          % ("model", "prompt set", "AP", "AP50", "AP75",
             "building", "platform", "aguada"))
    print("-" * 88)
    for model, sname, r in rows:
        pc = r["per_class"]
        print("%-14s %-18s %7.2f %7.2f %7.2f %9.2f %9.2f %8.2f"
              % (model, sname, r["AP"], r["AP50"], r["AP75"],
                 pc["building"], pc["platform"], pc["aguada"]))

    if os.path.exists(SUP):
        s = evaluate(coco, SUP)
        pc = s["per_class"]
        print("-" * 88)
        print("%-14s %-18s %7.2f %7.2f %7.2f %9.2f %9.2f %8.2f"
              % ("supervised", "arm D, fold9", s["AP"], s["AP50"], s["AP75"],
                 pc["building"], pc["platform"], pc["aguada"]))

    # the 2x2
    by = {sname: r["AP"] for _, sname, r in rows if sname in FACTORS}
    if len(by) == 4:
        print("\n2x2: which factor moves AP?\n")
        print("%-16s %8s %8s   %8s" % ("", "short", "long", "row mean"))
        for naming in ("semantic", "morphological"):
            vals = {}
            for s, (nm, ln) in FACTORS.items():
                if nm == naming:
                    vals[ln] = by[s]
            print("%-16s %8.2f %8.2f   %8.2f"
                  % (naming, vals["short"], vals["long"],
                     np.mean(list(vals.values()))))
        cols = {}
        for ln in ("short", "long"):
            cols[ln] = np.mean([by[s] for s, (nm, l2) in FACTORS.items() if l2 == ln])
        print("%-16s %8.2f %8.2f" % ("col mean", cols["short"], cols["long"]))
        naming_eff = (np.mean([by[s] for s, (nm, _) in FACTORS.items()
                               if nm == "morphological"])
                      - np.mean([by[s] for s, (nm, _) in FACTORS.items()
                                 if nm == "semantic"]))
        length_eff = cols["long"] - cols["short"]
        print("\nmain effect of describing rather than naming : %+.2f AP" % naming_eff)
        print("main effect of a longer phrase               : %+.2f AP" % length_eff)

    json.dump([{"model": m, "set": s, **r} for m, s, r in rows],
              open(os.path.join(DETDIR, "scores.json"), "w"), indent=1)
    print("\nwrote", os.path.join(DETDIR, "scores.json"))


if __name__ == "__main__":
    main()
