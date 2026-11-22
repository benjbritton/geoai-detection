"""Gradio demo: SpaceNet 2 building footprints with Mask R-CNN R50-FPN."""

import json
import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO = "benjbritton/spacenet2-building-detector"
EXAMPLES = os.path.join(HERE, "examples")
GROUND_TRUTH = json.load(open(os.path.join(EXAMPLES, "ground_truth.json")))
DEFAULT_SCORE = 0.544
PRED_COLOR = (255, 214, 0)     # yellow: predictions
GT_COLOR = (0, 200, 255)       # cyan: SpaceNet footprints

torch.set_num_threads(os.cpu_count() or 2)


def weights_path():
    local = os.path.join(HERE, "model_final.pth")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, "model_final.pth")


cfg = get_cfg()
cfg.merge_from_file(os.path.join(HERE, "config.yaml"))
cfg.MODEL.WEIGHTS = weights_path()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05     # filter by the slider instead
predictor = DefaultPredictor(cfg)


def percentile_stretch(arr, low=2.0, high=98.0):
    out = np.empty(arr.shape[:2] + (3,), np.uint8)
    for c in range(3):
        band = arr[..., c]
        v = band[band > 0]
        lo, hi = (0.0, 1.0) if v.size == 0 else (float(np.percentile(v, low)),
                                                 float(np.percentile(v, high)))
        out[..., c] = np.clip((band.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)),
                              0, 255).astype(np.uint8)
    return out


def gt_masks(key, shape):
    masks = []
    for parts in GROUND_TRUTH.get(key, []):      # one building = one or more parts
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32) for p in parts], 1)
        if m.any():
            masks.append(m.astype(bool))
    return masks


def match_f1(pred, gt, iou_thr=0.5):
    """Greedy one-to-one matching at mask IoU >= 0.5, highest score first."""
    used, tp = set(), 0
    for p in pred:
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if j in used:
                continue
            inter = np.logical_and(p, g).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(p, g).sum()
            if iou > best:
                best, best_j = iou, j
        if best_j is not None and best >= iou_thr:
            used.add(best_j)
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def outline(img, masks, color, thickness=2):
    for m in masks:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness)


def run(image, key, score, stretch, show_gt):
    if image is None:
        return None, "Choose an example tile or upload an image."
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8 or stretch:
        rgb = percentile_stretch(rgb)
    rgb = np.ascontiguousarray(rgb)

    inst = predictor(rgb[:, :, ::-1])["instances"].to("cpu")
    keep = inst.scores >= score
    scores = inst.scores[keep].numpy()
    order = np.argsort(-scores)
    masks = list(inst.pred_masks[keep].numpy()[order])

    out = rgb.copy()
    gt = gt_masks(key, rgb.shape[:2]) if key in GROUND_TRUTH else None
    if gt is not None and show_gt:
        outline(out, gt, GT_COLOR, 1)
    outline(out, masks, PRED_COLOR, 2)

    lines = [f"**{len(masks)} buildings detected** at score ≥ {score:.3f}"]
    if gt is not None:
        tp, fp, fn, p, r, f1 = match_f1(masks, gt)
        lines.append(f"SpaceNet footprints on this tile: {len(gt)}  \n"
                     f"Matched at IoU ≥ 0.5: {tp} · precision {p:.3f} · recall {r:.3f} · "
                     f"**F1 {f1:.3f}**")
        if show_gt:
            lines.append("Yellow: predictions · cyan: SpaceNet footprints")
    return Image.fromarray(out), "\n\n".join(lines)


def example_list():
    order = ["Vegas", "Paris", "Shanghai", "Khartoum"]
    keys = sorted(GROUND_TRUTH, key=lambda k: (order.index(k.split("_")[0]),
                                               int(k.split("img")[1])))
    return [[os.path.join(EXAMPLES, k + ".png"), k] for k in keys]


with gr.Blocks(title="SpaceNet 2 building detector") as demo:
    gr.Markdown(
        "# SpaceNet 2 building detector\n"
        "Mask R-CNN R50-FPN trained on all four SpaceNet 2 cities (Las Vegas, Paris, "
        "Shanghai, Khartoum) at 30 cm. Macro F1 **0.7447** on the validation split at "
        "score threshold 0.544. "
        "[Model card](https://huggingface.co/benjbritton/spacenet2-building-detector) · "
        "[Code](https://github.com/benjbritton/geoai-detection)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Input tile", type="numpy", image_mode="RGB")
            key = gr.Textbox(visible=False)
            score = gr.Slider(0.05, 0.95, value=DEFAULT_SCORE, step=0.001,
                              label="Score threshold (0.544 = reported operating point)")
            stretch = gr.Checkbox(False, label="Apply 2–98% percentile stretch "
                                               "(for low-contrast uploads)")
            show_gt = gr.Checkbox(True, label="Show SpaceNet footprints (example tiles)")
            go = gr.Button("Detect buildings", variant="primary")
        with gr.Column():
            result = gr.Image(label="Detections", type="pil")
            summary = gr.Markdown()
    gr.Examples(example_list(), inputs=[image, key], label="Validation tiles, 3 per city",
                examples_per_page=12)
    gr.Markdown(
        "Example tiles: SpaceNet 2 validation split, not seen in training, prepared with "
        "the training stretch. Uploads work best with roughly 30–50 cm RGB overhead "
        "imagery. Imagery and footprints: The SpaceNet Dataset by SpaceNet Partners, "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).")

    image.upload(lambda: "", outputs=key)
    go.click(run, [image, key, score, stretch, show_gt], [result, summary])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
