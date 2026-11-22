"""Fetch the model weights and example tiles from the Hugging Face model repo
at image build time, unless they are already in the build context.

The example tiles are SpaceNet-derived (CC BY-SA 4.0) and live with the
weights in the model repo, not in the MIT-licensed code repository.
"""

import os

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "benjbritton/spacenet2-building-detector"
HERE = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(HERE, "model_final.pth")):
    hf_hub_download(REPO, "model_final.pth", local_dir=HERE)
    print("downloaded model_final.pth")

if not os.path.exists(os.path.join(HERE, "examples", "ground_truth.json")):
    snapshot_download(REPO, allow_patterns=["examples/*"], local_dir=HERE)
    print("downloaded examples/")
