# One image definition for every variant.
#   MODEL=repack|original        which directory under models/ is baked in (scripts/prepare_models.sh)
#   TORCH_INDEX                  https://download.pytorch.org/whl/cpu for CPU, .../whl/cu130 for the L4
FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG MODEL=repack
RUN pip install --no-cache-dir torch==2.14.1 --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir transformers==5.19.0 sentence-transformers==6.1.0 \
        safetensors fastapi uvicorn
# The EmbeddingGemma2 processor imports PIL and torchvision even for text.
RUN pip install --no-cache-dir pillow torchvision==0.29.1 --index-url ${TORCH_INDEX} \
        --extra-index-url https://pypi.org/simple
COPY models/${MODEL}/ /model/
COPY app/ /app/
WORKDIR /app
ENV MODEL_DIR=/model MODEL_VARIANT=${MODEL} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
