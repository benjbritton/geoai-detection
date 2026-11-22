# SpaceNet 2 building detector: interactive demo

A Gradio app that runs the SpaceNet 2 Mask R-CNN R50-FPN building detector
([model card](https://huggingface.co/benjbritton/spacenet2-building-detector))
on twelve validation tiles, three per city, or on an uploaded overhead RGB
image. It outlines each detected building and, on the example tiles, shows the
SpaceNet footprints and the tile's F1 at IoU ≥ 0.5.

## Run the published image

```bash
docker run -p 7860:7860 ghcr.io/benjbritton/spacenet2-building-detector:v1.0
```

Then open http://localhost:7860. The image runs on CPU, about 3 seconds per
tile.

## Build it yourself

```bash
cd demo
docker build -t spacenet2-building-detector .
docker run -p 7860:7860 spacenet2-building-detector
```

The build installs PyTorch 2.5.1 (CPU), detectron2 at commit `a2f4a877` (the
training commit) and Gradio 4.44.1. `fetch_assets.py` then downloads the model
weights and the example tiles from the Hugging Face model repository.

## Files

| file | |
|---|---|
| `app.py` | Gradio interface, inference and per-tile F1 scoring |
| `Dockerfile` | CPU image definition |
| `requirements.txt` | pinned Python dependencies |
| `fetch_assets.py` | downloads weights and example tiles at build time |
| `config.yaml` | self-contained detectron2 config for the model |

## Data

The example tiles and footprints are derived from the SpaceNet 2 validation
split and are distributed with the model weights on Hugging Face under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): The SpaceNet
Dataset by SpaceNet Partners. Van Etten, A., Lindenbaum, D., & Bacastow, T. M.
(2018). SpaceNet: A Remote Sensing Dataset and Challenge Series.
arXiv:1807.01232. They are not stored in this repository.

**Footprint counts.** Each example tile's footprints come from its SpaceNet
labels, one entry per building. One Shanghai_img205 footprint is a sub-pixel
sliver on the tile's right edge (x 649.9–650.0 px) that cannot be rasterized,
so it is omitted: that tile has 41 footprints against 42 in its source GeoJSON.
