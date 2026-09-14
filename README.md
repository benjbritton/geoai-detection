# geoai-detection

Multi-class detection of ancient Maya structures in airborne lidar, and a
SpaceNet 2 building-footprint baseline. Detectron2 / PyTorch. FA26 Independent
Study, University of Cincinnati, Geography & GIS.

Two projects share this repository, a container, and one continuous lab
notebook. The notebook is the point: it runs unbroken across both, and the
reasoning that came out of the first is what made the second's central result
legible.

---

## 1. Chactún — three-class detection in airborne lidar

Building, platform and aguada across 2,094 tiles / 120.6 km² of the central
Yucatán Peninsula ([Kokalj et al. 2023](https://doi.org/10.1038/s41597-023-02455-x),
CC BY 4.0). Mask R-CNN R50-FPN in detectron2.

**A controlled six-arm ablation, 36 cross-validated training runs.** One
intervention moved the problem and four did not:

| arm | segm AP | vs control | 95% CI | *p* |
|---|---|---|---|---|
| **D — D4 augmentation** | **42.87 ± 2.80** | **+4.16** | **[+2.70, +5.61]** | **0.0014** |
| F — 960 px input | 38.81 ± 2.77 | +0.10 | [−1.25, +1.44] | 0.849 |
| A — control | 38.71 ± 2.45 | — | — | — |
| E — repeat sampling | 38.65 ± 2.31 | −0.06 | [−1.65, +1.53] | 0.922 |
| C — cascade head | 38.58 ± 2.71 | −0.13 | [−1.66, +1.41] | 0.829 |
| B — shifted anchors | 37.91 ± 2.67 | −0.80 | [−2.13, +0.53] | 0.169 |

Four of five intervals contain zero, which is what "within noise" means stated
properly. The gain came from the data pipeline, not the architecture, at no
additional compute — and it **replicated at +4.17 on the dataset's own
challenge split**, which shares no design decision with the folds that produced
it.

**What five folds can support.** With n = 5 the exact sign-flip permutation test
cannot return a two-sided *p* below 0.0625 whatever the effect, and this result
sits on that floor with every fold positive. The claim rests on the interval,
the effect size (*dz* = 3.55) and the independent replication — not on a
corrected *p*-value.

**Scored against the published field.** Reproducing the ECML PKDD 2021 Chactún
benchmark protocol gives 0.797 semantic IoU on its held-out split, against a
published field of 0.811–0.834. A semantic-segmentation baseline trained at
matched compute then decomposes that gap: architecture formulation accounts for
0.012 — essentially the whole distance to eighth place — and ensembling with
test-time augmentation for the 0.025 beyond it.

**Portability.** Taken to an independent lidar survey, the detector's behaviour
turned out to be governed by whether the input rendering could be reproduced.
The recipe was published in a table of the dataset paper rather than in the
dataset. Counts and quality then ran in opposite directions, which is the part
worth reading.

→ [`posts/Chactun_Multiclass_M2_blog_090226v17.md`](posts/Chactun_Multiclass_M2_blog_090226v17.md)
 · [W&B](https://wandb.ai/benjbritton-geoai/chactun-multiclass) (40 runs)

---

## 2. SpaceNet 2 — multi-city building footprints

Mask R-CNN R50-FPN, COCO-pretrained, trained across all four AOIs and scored
with the SpaceNet F1 metric.

| | macro F1 | Vegas | Paris | Shanghai | Khartoum |
|---|---|---|---|---|---|
| **this work** (3 seeds) | **0.7459** ± 0.0012 | 0.8948 | 0.7787 | 0.6848 | 0.6254 |
| XD_XD, 2017 winner | 0.6930 | 0.885 | 0.745 | 0.597 | 0.544 |
| YOLT baseline | 0.6000 | | | | |
| modified MNC baseline | 0.5700 | | | | |

Every city is scored at **one** threshold, 0.544, selected on the training split
and never on the data being reported. Tuning per city on the scored set gives
0.7462 — better, and not a result.

**This is not a claim of beating the 2017 winner.** Those scores are on the
competition's withheld test set; these are on a validation split carved from the
training data, with a random tile split that is spatially autocorrelated (worth
about 0.5% on the pooled metric, measured) and IoU on rasterised masks rather
than georeferenced polygons. What it does establish is that the pipeline lands
near published results and reproduces the per-city difficulty ordering exactly.

**A hypothesis measured, then refuted.** Roof-to-ground hue separation predicts
inter-city difficulty — 2.3° in Khartoum against 29.4° in Vegas, ordering the
cities exactly. A three-seed grayscale ablation then showed the detector does
not use it: removing colour costs 0.61% of pooled F1 and the Vegas–Khartoum gap
does not close.

→ [`posts/2026-08-30-hue-predicts-which-cities-are-hard.md`](posts/2026-08-30-hue-predicts-which-cities-are-hard.md)
 · [`posts/2026-09-03-replicating-the-grayscale-ablation.md`](posts/2026-09-03-replicating-the-grayscale-ablation.md)
 · [W&B](https://wandb.ai/benjbritton-geoai/benjbritton_FA26) (14 runs)

---

## Read this first

| | |
|---|---|
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | The actual record, unbroken across both projects. What was done, what it cost, what turned out to be wrong. Written as work happened |
| [`REPRODUCE.md`](REPRODUCE.md) | Every result and the literal command that produced it, with expected values |

The notebook is the substance. Several claims made in it were later refuted by
further measurement, and the refutations are kept in place rather than edited
out — including one that had to refute its own correction.

## Requirements

- WSL2 Ubuntu 24.04, Docker Engine, NVIDIA Container Toolkit
- Image `m2/detectron2:cu124-torch251`, built from `docker/Dockerfile.detectron2`
- ~26 GB for SpaceNet 2 (PS-RGB plus building footprints, all four AOIs)
- An NVIDIA GPU. Developed on an RTX 2080 Ti (11 GB), current results on an
  RTX A5000 (24 GB). A full run is about 1:52
- A W&B API key, optional -- `--offline` or `--no-wandb` work without one

Keep this repo, `data/` and `outputs/` on the WSL ext4 filesystem, **not** under
`/mnt/c`. Cross-filesystem I/O to the Windows drive is slow for the many-small-file
access pattern training uses.

## Quick start

```bash
docker build -t m2/detectron2:cu124-torch251 -f docker/Dockerfile.detectron2 docker/
./scripts/run.sh python scripts/verify_gpu.py
./scripts/run.sh python scripts/train_spacenet.py --seed 0
```

`scripts/run.sh` wraps `docker run` with the GPU, bind mount, host UID/GID and
W&B credential wired up, so nothing is installed on the host. Full sequence,
including data preparation, in [`REPRODUCE.md`](REPRODUCE.md).

## Layout

| Path | Purpose |
|---|---|
| `src/detlab/datasets/spacenet.py` | SN2 registration and the 16-bit to 8-bit load path |
| `src/detlab/datasets/geojson_to_coco.py` | Footprint GeoJSON to COCO, with the geometry fixes real data forced |
| `src/detlab/spacenet_f1.py` | SpaceNet F1 at IoU 0.5, greedy matching, score-threshold sweep |
| `src/detlab/trainer.py` | DefaultTrainer subclass: COCO eval + SpaceNet F1 + W&B + best-checkpointing |
| `src/detlab/wandb_writer.py` | W&B EventWriter. detectron2 ships no W&B support |
| `scripts/train_spacenet.py` | The training entry point |
| `scripts/score_f1.py`, `f1_report.py` | Score a finished run without re-running inference |
| `scripts/city_separability.py`, `city_hue.py`, `factor_attribution.py`, `f1_by_size.py`, `iou_sweep.py` | The per-city difficulty analysis |
| `scripts/export_predictions_geojson.py`, `overlay_geotiff.py` | Predictions as GIS-ready vectors and georeferenced overlays |
| `src/detlab/datasets/chactun.py` | Chactun registration, folds, and the D4 dihedral mapper |
| `src/detlab/datasets/masks_to_coco.py` | Semantic masks to instance COCO, with the inverted-polarity and watershed findings in its docstring |
| `src/detlab/wandb_registry.py` | Resolves which W&B project a run belongs to, so identity is declared rather than inherited |
| `scripts/train_chactun.py`, `run_chactun_matrix.sh` | The six-arm ablation |
| `scripts/chactun_semantic_iou.py` | Rescores instance predictions under all three semantic-IoU conventions |
| `scripts/chactun_intervals.py`, `chactun_permutation.py` | Confidence intervals, exact comparison family, distribution-free floor |
| `scripts/train_chactun_semantic.py` | DeepLabV3-R50 baseline that decomposes the leaderboard gap |
| `scripts/dem_to_g1bands.py`, `run_on_gliht.py` | Cross-survey transfer: the published stretch, and inference on G-LiHT |
| `posts/` | The write-ups. Markdown only; renders are regenerable |
| `configs/` | Overrides layered on a model-zoo base config, plus the split files |
| `docker/` | Image definition, and `environment.lock.txt` -- the resolved package set the results were produced with |

`data/`, `outputs/`, `wandb/` and checkpoints are gitignored. Every artefact is
regenerable from `REPRODUCE.md`; the numbers are transcribed into the notebook.

## Notes

- **The split file, not the seed, is the authority.** `configs/spacenet2_split.json`
  fixes train/val membership. `cfg.SEED` varies between runs to measure variance;
  the split must not, or seed variance and split variance become inseparable.
- **`--seed` is wired to `seed_all_rng()`, not `cfg.SEED`.** detectron2 reads
  `cfg.SEED` only inside `default_setup()`, which this script never calls, so a
  flag wired to the config value would have looked correct and done nothing.
- **No 8-bit files are written.** SN2 tiles are 11-bit data in a 16-bit
  container, occupying about 3% of the nominal range; a naive divide-by-256
  yields a tile whose maximum value is 6 of 255. The stretch happens in the
  mapper at load time, so the georeferenced UInt16 GeoTIFFs stay the only copy.
- **`FILTER_EMPTY_ANNOTATIONS: False` is required.** 2069 of 10592 tiles contain
  no buildings and detectron2 drops such images by default -- 20% of the dataset,
  and 45% of Paris, would vanish silently.
- **fp16, not bf16.** bf16 and TF32 are Ampere-only; fp16 runs on both cards the
  project has used, so configs survived the GPU swap unchanged.
- **`numpy<2` is pinned in the image** via `PIP_CONSTRAINT`. detectron2 predates
  numpy 2 and the breakage surfaces in COCO evaluation, not at import.

## Data attribution and licensing

**SpaceNet 2 (Building Detection v2)** -- imagery and building footprint labels.

> The SpaceNet Dataset by SpaceNet Partners is licensed under a
> [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Cite as:

> Van Etten, A., Lindenbaum, D., & Bacastow, T.M. (2018). SpaceNet: A Remote
> Sensing Dataset and Challenge Series. *arXiv:1807.01232*.

Accessed from the [SpaceNet AWS Open Data registry][aws] on 2026-08-27
(requester-pays bucket; PS-RGB and `geojson_buildings` for all four AOIs).

**What ShareAlike means here.** CC BY-SA obligations attach to material *derived
from the dataset*, not to independently written code. In this repo:

| | derivative? |
|---|---|
| `data/spacenet2/coco/*.json` (converted footprints) | yes -- a reformatting of the labels |
| exported prediction vectors, overlay rasters, figures | yes -- derived from the imagery |
| `configs/spacenet2_split*.json` (filename lists) | membership only, no dataset content |
| `scripts/`, `src/`, `docker/` | no -- independent code |

None of the derivative material is tracked in git (`data/` and `outputs/` are
ignored), so the repository as published contains no SpaceNet-derived content.
**Anything derived that does get published -- overlay figures in a blog post, a
released set of predicted footprints -- carries the attribution above and the
ShareAlike term with it.**

**Trained model weights** are treated here as a derivative of the CC BY-SA
imagery, and would carry the ShareAlike obligation if released. See `NOTICE`.

Whether that is *legally* required is genuinely unsettled — there is no
settled answer on whether model weights are a derivative work of training data.
This is a deliberately conservative posture rather than a claim about the law:
the cost of assuming ShareAlike applies is releasing under ShareAlike, and the
cost of assuming it does not is a licence violation. No weights are tracked in
this repository.

## Third-party components

None are vendored into this repository; all are fetched at build or run time.
Listed so the obligations are visible rather than implicit.

| component | licence | how it is used |
|---|---|---|
| [detectron2](https://github.com/facebookresearch/detectron2) (Meta), commit `a2f4a877` | Apache-2.0 | cloned into the image; `src/detlab/` extends its documented APIs (`EventWriter`, `DefaultTrainer`, `DatasetEvaluator`) |
| detectron2 model zoo, `mask_rcnn_R_50_FPN_3x` COCO weights | Apache-2.0 | initialisation for every run |
| [PyTorch](https://github.com/pytorch/pytorch) 2.5.1 and the `pytorch/pytorch` CUDA base image | BSD-3-Clause | base image |
| [pycocotools](https://github.com/ppwwyyxx/cocoapi) | BSD-2-Clause | COCO evaluation and RLE mask handling |
| rasterio, shapely, pyproj, OpenCV, numpy | BSD / MIT / Apache-2.0 | geospatial and array stack; see `docker/environment.lock.txt` |
| [balloon dataset](https://github.com/matterport/Mask_RCNN/releases) | see note | Milestone A only, not used for any SpaceNet result |

`src/detlab/wandb_writer.py` deliberately mirrors the structure of detectron2's
`TensorboardXWriter` (`events.py:141`) so the two read alike, but it is
independently written against the public `EventWriter` interface rather than
copied.

Note on the balloon dataset: it is distributed through the releases of
matterport/Mask_RCNN, an MIT-licensed repository, but that project does not state
separate terms for the images themselves. It was used for the Milestone A
plumbing test and contributes to no result reported here.

## Licence

Code and written record: [MIT](LICENSE), (c) 2026 Benjamin Britton.

The MIT grant is scoped: it covers the code and the written record, not the SpaceNet 2
dataset or anything derived from it, which stays CC BY-SA 4.0 with ShareAlike intact.
[NOTICE](NOTICE) sets out exactly what is and is not covered.

The MIT grant does not extend to the SpaceNet 2 dataset or anything derived from
it, which remains CC BY-SA 4.0 -- see above.

[cc-by-sa]: https://creativecommons.org/licenses/by-sa/4.0/
[aws]: https://registry.opendata.aws/spacenet/
