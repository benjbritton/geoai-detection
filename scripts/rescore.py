"""Score every arm against both ground truths, full and edge-free.

3429 of 9853 components in the Chactun conversion touch a tile boundary, so a
fraction of every class is a truncated fragment rather than a whole structure.
On the canonical split that is 30% of buildings, 54% of platforms and 12 of the
13 aguadas. A number computed against fragments is not wrong, but it is not
measuring what its column heading says, so both are reported side by side.

Aguada survives edge filtering with ONE instance. Its edge-free column is
printed for completeness and means nothing; it is not a performance figure.
"""
import contextlib, glob, io as _io, json, os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = "/workspace"
D = R + "/outputs/milestone_d"
SUP = (R + "/outputs/chactun_D_maskrcnn_d4_augmentation/"
       "fold9_seed0/inference/coco_instances_results.json")
CLS = {1: "building", 2: "platform", 3: "aguada"}


def ev(coco, dets, cid=None):
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes([d for d in dets if "bbox" in d])
        e = COCOeval(coco, dt, "bbox")
        if cid:
            e.params.catIds = [cid]
        e.evaluate(); e.accumulate(); e.summarize()
    return 100 * e.stats[0]


gts = {}
for tag, f in (("full", "fold9_val.json"), ("noedge", "fold9_val_noedge.json")):
    with contextlib.redirect_stdout(_io.StringIO()):
        gts[tag] = COCO(R + "/data/chactun/coco/" + f)

files = sorted(glob.glob(D + "/dets_*.json")) + [SUP]
print("%-30s %7s %7s %6s | %8s %8s %7s"
      % ("arm", "AP", "AP-ne", "delta", "bldg-ne", "plat-ne", "agu-ne"))
print("-" * 82)
for p in files:
    dets = json.load(open(p))
    name = ("supervised arm D" if p == SUP
            else os.path.basename(p)[5:-5].replace("gdino_base_", "gdino ")
            .replace("owlv2_", "owlv2 "))
    a_full = ev(gts["full"], dets)
    a_ne = ev(gts["noedge"], dets)
    per = [ev(gts["noedge"], dets, c) for c in (1, 2, 3)]
    print("%-30s %7.2f %7.2f %+6.2f | %8.2f %8.2f %7.2f"
          % (name, a_full, a_ne, a_ne - a_full, per[0], per[1], per[2]))

print()
print("ground truth: full 1614 anns (1231/370/13); edge-free 1029 (858/170/1)")
print("the aguada edge-free column rests on ONE instance and is not a result")
