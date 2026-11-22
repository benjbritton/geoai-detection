FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/benjbritton/geoai-detection"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Same torch release the model was trained with (2.5.1), CPU build.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# detectron2 at the commit used for training; needs torch already installed.
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git@a2f4a877"

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user . /home/user/app

# Weights and example tiles come from the Hugging Face model repo unless they
# are already present in the build context.
RUN python fetch_assets.py

EXPOSE 7860
CMD ["python", "app.py"]
