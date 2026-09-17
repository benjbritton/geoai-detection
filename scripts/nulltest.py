"""Size-matched random-box null: is any arm above chance overlap?

Each detection is replaced by a box of the SAME area and aspect ratio, placed
uniformly at random inside the SAME image, keeping its score and its label.
That holds constant everything except where the box sits: the size distribution,
the class mix, the score ordering, the per-image budget, the image assignment.
Only localisation is destroyed.

An arm whose real AP falls inside this distribution is not detecting. It is
emitting boxes whose sizes happen to overlap objects of similar size, which a
random placement of the same boxes reproduces.

This is the test the earlier image-shuffle null could not do: that one moved
detections to other images, where tiny objects meant the null was 0.00 +- 0.00
and had no discriminative power at all.
"""
import contextlib, glob, io as _io, json, os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = "/workspace"
D = R + "/outputs/milestone_d"
DRAWS = 20
rng = np.random.default_rng(0)


def ap(coco, dets):
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes([dict(d) for d in dets])
        e = COCOeval(coco, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    return 100 * e.stats[0]


def scatter(dets, wh):
    out = []
    for d in dets:
        W, H = wh[d["image_id"]]
        x, y, w, h = d["bbox"]
        nx = float(rng.uniform(0, max(W - w, 1e-6)))
        ny = float(rng.uniform(0, max(H - h, 1e-6)))
        e = dict(d); e["bbox"] = [nx, ny, w, h]
        out.append(e)
    return out


for label, annf, pat in (
        ("CHACTUN", R + "/data/chactun/coco/fold9_val.json", D + "/dets_*.json"),
        ("SPACENET", D + "/sn_val_subset.json", D + "/sn_dets_*.json")):
    gt = json.load(open(annf))
    wh = {i["id"]: (i["width"], i["height"]) for i in gt["images"]}
    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(annf)
    print("\n=== %s ===" % label)
    print("%-26s %8s %16s %9s %s"
          % ("arm", "real AP", "null mean +- sd", "z", "verdict"))
    print("-" * 74)
    for f in sorted(glob.glob(pat)):
        dets = json.load(open(f))
        name = os.path.basename(f).replace("dets_", "").replace("sn_", "")[:-5]
        real = ap(coco, dets)
        nulls = [ap(coco, scatter(dets, wh)) for _ in range(DRAWS)]
        m, s = float(np.mean(nulls)), float(np.std(nulls))
        z = (real - m) / s if s > 1e-9 else float("inf")
        verdict = "above chance" if real > m + 3 * s else "INSIDE NULL"
        print("%-26s %8.2f %8.2f +- %5.2f %9.1f  %s"
              % (name, real, m, s, z, verdict))
