"""Test the reading that aguada-assigned boxes are flat and weakly bounded.

Band 3 of the Chactun render is slope on an inverted greyscale: flat ground is
bright, steep ground is dark. So a flat interior means a HIGH band-3 mean, and a
weakly bounded feature means LOW variance across the box -- no rim to create
contrast. If the reading holds, boxes assigned to aguada should be larger,
brighter in band 3, and lower in variance than boxes assigned to building.

Compares against the ground-truth boxes of each class, which is the reference
the prediction should match if the assignment were meaningful.
"""
import glob, json, os
from collections import defaultdict

import numpy as np
import rasterio

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
R = _os.path.dirname(_HERE)
NAME = {1: "building", 2: "platform", 3: "aguada"}


def stats_for(boxes_by_img, images, label):
    area, bright, var = [], [], []
    for im in images:
        bs = boxes_by_img.get(im["id"])
        if not bs:
            continue
        with rasterio.open(os.path.join(R, "data/chactun", im["file_name"])) as s:
            slope = s.read(3).astype(np.float32)     # inverted greyscale slope
        H, W = slope.shape
        for x, y, w, h in bs:
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x + w)), min(H, int(y + h))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            crop = slope[y0:y1, x0:x1]
            area.append((x1 - x0) * (y1 - y0) / (W * H))
            bright.append(crop.mean())
            var.append(crop.std())
    if not area:
        return None
    return (100 * np.mean(area), np.mean(bright), np.mean(var), len(area))


coco = json.load(open(os.path.join(R, "data/chactun/coco/fold9_val.json")))
images = coco["images"]

print("%-28s %8s %9s %9s %8s" % ("", "area%", "band3 mean", "band3 sd", "n"))
print("-" * 68)
print("GROUND TRUTH")
for cid in (1, 2, 3):
    by = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == cid:
            by[a["image_id"]].append(a["bbox"])
    r = stats_for(by, images, NAME[cid])
    if r:
        print("  %-26s %8.2f %9.1f %9.1f %8d" % (NAME[cid], *r))

for p in sorted(glob.glob(os.path.join(R, "outputs/milestone_d/dets_*.json"))):
    tag = os.path.basename(p)[len("dets_gdino_base_"):-5]
    dets = json.load(open(p))
    # top tenth by score: where AP is actually decided
    dets = sorted(dets, key=lambda d: -d["score"])[: len(dets) // 10]
    print("PREDICTED  %s  (top 10%% by score)" % tag)
    for cid in (1, 2, 3):
        by = defaultdict(list)
        for d in dets:
            if d["category_id"] == cid:
                by[d["image_id"]].append(d["bbox"])
        r = stats_for(by, images, NAME[cid])
        if r:
            print("  %-26s %8.2f %9.1f %9.1f %8d" % (NAME[cid], *r))
