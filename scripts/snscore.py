"""Score the SpaceNet arms against the same subsample, one COCOeval path.

The ground truth is restricted to the sampled images so the evaluation set and
the detection set describe the same pictures; otherwise every unsampled image
counts as pure missed recall and every arm is dragged toward zero equally,
which flatters nothing but obscures the comparison.
"""
import contextlib, glob, io as _io, json, os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = "/workspace"
D = R + "/outputs/milestone_d"
ids = set(json.load(open(D + "/sn_image_ids.json")))

full = json.load(open(R + "/data/spacenet2/coco/pooled_val.json"))
sub = {"images": [i for i in full["images"] if i["id"] in ids],
       "annotations": [a for a in full["annotations"] if a["image_id"] in ids],
       "categories": full["categories"]}
p = D + "/sn_val_subset.json"
json.dump(sub, open(p, "w"))
print("evaluation set: %d images, %d building annotations"
      % (len(sub["images"]), len(sub["annotations"])))
print("mean %.1f buildings per image\n"
      % (len(sub["annotations"]) / len(sub["images"])))

with contextlib.redirect_stdout(_io.StringIO()):
    coco = COCO(p)

print("%-8s %-12s %8s %8s %8s %10s"
      % ("model", "prompt", "AP", "AP50", "AP75", "med area"))
print("-" * 60)
rows = []
for f in sorted(glob.glob(D + "/sn_dets_*.json")):
    tag = os.path.basename(f)[8:-5]
    model, _, prompt = tag.partition("_")
    dets = json.load(open(f))
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes(dets)
        e = COCOeval(coco, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    ar = np.median([d["bbox"][2] * d["bbox"][3] for d in dets]) / (650.0 * 650) * 100
    rows.append((model, prompt, 100 * e.stats[0], 100 * e.stats[1],
                 100 * e.stats[2], ar))
    print("%-8s %-12s %8.2f %8.2f %8.2f %9.2f%%"
          % (model, prompt, 100 * e.stats[0], 100 * e.stats[1],
             100 * e.stats[2], ar))

gt = np.median([a["bbox"][2] * a["bbox"][3] for a in sub["annotations"]]) / (650.0 * 650) * 100
print("\nground-truth median building area: %.2f%% of tile" % gt)
print("\nregistered prediction was that 'rooftop' beats 'building'.")
for m in ("gdino", "owlv2"):
    d = {p_: ap for mm, p_, ap, _, _, _ in rows if mm == m}
    if "building" in d and "rooftop" in d:
        print("  %-6s building %.2f  rooftop %.2f  -> %s"
              % (m, d["building"], d["rooftop"],
                 "HELD" if d["rooftop"] > d["building"] else "FAILED"))
