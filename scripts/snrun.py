"""Zero-shot open-vocabulary detection on SpaceNet 2, scored like everything else.

Same protocol as the Chactun arms: top-100 boxes per image, no score threshold,
COCOeval bbox against the same ground truth the supervised model was scored on.
This is the second modality -- ordinary overhead optical rather than a lidar
derivative -- so it separates "fails on lidar" from "fails on aerial views".

The prompt set is chosen to test the viewpoint account directly. "building"
names the object as captions name it, from the street. "rooftop" names what is
actually in a nadir image of the same object. If the viewpoint account is right,
rooftop should win, and that is a prediction registered before the run.

Stratified subsample: AP over ~12000 annotations is stable well before 2118
images, and the full set would cost an hour for no additional certainty.
"""
import json, os, random, time
import numpy as np, rasterio, torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

R = "/workspace"
ANN = R + "/data/spacenet2/coco/pooled_val.json"
OUT = R + "/outputs/milestone_d"
PROMPTS = [("building", "building"), ("rooftop", "rooftop"),
           ("house", "house"), ("small_rect", "small rectangular building")]
TOP_K, N_IMG = 100, 600
MODELS = [("gdino", "IDEA-Research/grounding-dino-base"),
          ("owlv2", "google/owlv2-base-patch16-ensemble")]

coco = json.load(open(ANN))
random.seed(0)
imgs = sorted(coco["images"], key=lambda i: i["id"])
random.shuffle(imgs)
imgs = imgs[:N_IMG]
print("%d images sampled of %d" % (len(imgs), len(coco["images"])), flush=True)


def load(path):
    with rasterio.open(path) as s:
        a = s.read()[:3].astype(np.float32)
    if a.max() > 255:
        o = np.zeros_like(a, np.uint8)
        for i in range(3):
            lo, hi = np.percentile(a[i], (2, 98))
            o[i] = np.clip((a[i] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        a = o
    return Image.fromarray(np.ascontiguousarray(
        a.astype(np.uint8).transpose(1, 2, 0))).convert("RGB")


for mtag, mid in MODELS:
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
    for ptag, text in PROMPTS:
        dets, t0 = [], time.time()
        for k, im in enumerate(imgs, 1):
            img = load(os.path.join(R, "data/spacenet2", im["file_name"]))
            inp = proc(images=img, text=text + ".", return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
            sc = out.logits[0].sigmoid().max(dim=1).values
            keep = sc.topk(min(TOP_K, sc.numel())).indices
            bx = out.pred_boxes[0][keep].cpu().numpy()
            bs = sc[keep].cpu().numpy()
            W, H = im["width"], im["height"]
            for (cx, cy, w, h), s in zip(bx, bs):
                dets.append({"image_id": im["id"], "category_id": 1,
                             "bbox": [float((cx - w / 2) * W), float((cy - h / 2) * H),
                                      float(w * W), float(h * H)],
                             "score": float(s)})
            if k % 200 == 0:
                print("  %s/%s %d/%d %.0fs" % (mtag, ptag, k, len(imgs),
                                               time.time() - t0), flush=True)
        p = "%s/sn_dets_%s_%s.json" % (OUT, mtag, ptag)
        json.dump(dets, open(p, "w"))
        print("%-6s %-11s %6d dets  %.0fs" % (mtag, ptag, len(dets),
                                              time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()

json.dump([i["id"] for i in imgs], open(OUT + "/sn_image_ids.json", "w"))
print("done")
